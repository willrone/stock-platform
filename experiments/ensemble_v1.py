#!/usr/bin/env python3
"""
三模型集成预测 v1 - LightGBM + XGBoost + CatBoost
基于 lgbm_stock_prediction_v4_final.py 的数据处理和特征工程
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path('/Users/ronghui/Projects/willrone/data/parquet/stock_data')

def compute_cross_sectional_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """计算截面特征（每日排名）"""
    df = df_all.copy()
    
    for col in ['return_1d', 'return_5d', 'return_20d', 'volume', 'volatility_20']:
        if col in df.columns:
            df[f'{col}_rank'] = df.groupby('date')[col].rank(pct=True)
    
    df['market_up_ratio'] = df.groupby('date')['return_1d'].transform(
        lambda x: (x > 0).sum() / len(x)
    )
    
    market_return_5d = df.groupby('date')['return_5d'].transform('mean')
    market_return_20d = df.groupby('date')['return_20d'].transform('mean')
    
    df['relative_strength_5d'] = df['return_5d'] - market_return_5d
    df['relative_strength_20d'] = df['return_20d'] - market_return_20d
    df['relative_momentum'] = df['relative_strength_5d'] - df['relative_strength_20d']
    
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标特征"""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_ = df['open']
    
    # 基础收益
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'return_{period}d'] = close.pct_change(period)
    
    # 动量因子
    df['momentum_short'] = df['return_5d'] - df['return_10d']
    df['momentum_long'] = df['return_10d'] - df['return_20d']
    df['momentum_reversal'] = -df['return_1d']
    
    returns = close.pct_change()
    for period in [5, 10, 20]:
        up_days = (returns > 0).rolling(period).sum()
        df[f'momentum_strength_{period}'] = up_days / period
    
    # 移动平均
    for window in [5, 10, 20, 60]:
        ma = close.rolling(window).mean()
        df[f'ma_ratio_{window}'] = close / ma - 1
        df[f'ma_slope_{window}'] = ma.pct_change(5)
    
    df['ma_5'] = close.rolling(5).mean()
    df['ma_10'] = close.rolling(10).mean()
    df['ma_20'] = close.rolling(20).mean()
    
    df['ma_alignment'] = (
        (df['ma_5'] > df['ma_10']).astype(int) +
        (df['ma_10'] > df['ma_20']).astype(int)
    )
    
    # 波动率因子
    for window in [5, 20, 60]:
        df[f'volatility_{window}'] = returns.rolling(window).std()
    
    df['vol_regime'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)
    df['volatility_skew'] = returns.rolling(20).skew()
    
    # 成交量特征
    vol_ma20 = volume.rolling(20).mean()
    vol_ma5 = volume.rolling(5).mean()
    
    df['vol_ratio'] = volume / (vol_ma20 + 1)
    df['vol_ma_ratio'] = vol_ma5 / (vol_ma20 + 1)
    df['vol_std'] = volume.rolling(20).std() / (vol_ma20 + 1)
    
    df['price_up'] = (close > close.shift(1)).astype(int)
    df['vol_up'] = (volume > volume.shift(1)).astype(int)
    df['vol_price_diverge'] = (df['price_up'] != df['vol_up']).astype(int)
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    for period in [6, 14]:
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    df['rsi_diff'] = df['rsi_6'] - df['rsi_14']
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_hist_slope'] = df['macd_hist'].diff(3)
    
    # 布林带
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_position'] = (close - bb_mid) / (2 * bb_std + 1e-10)
    df['bb_width'] = 4 * bb_std / (bb_mid + 1e-10)
    
    # 价格形态
    df['body'] = (close - open_) / (open_ + 1e-10)
    df['wick_upper'] = (high - np.maximum(close, open_)) / (high - low + 1e-10)
    df['wick_lower'] = (np.minimum(close, open_) - low) / (high - low + 1e-10)
    df['range_pct'] = (high - low) / (close + 1e-10)
    
    df['consecutive_up'] = (close > close.shift(1)).rolling(5).sum()
    df['consecutive_down'] = (close < close.shift(1)).rolling(5).sum()
    
    # 价格位置
    for window in [20, 60]:
        high_n = high.rolling(window).max()
        low_n = low.rolling(window).min()
        df[f'price_pos_{window}'] = (close - low_n) / (high_n - low_n + 1e-10)
        df[f'dist_high_{window}'] = (high_n - close) / (close + 1e-10)
        df[f'dist_low_{window}'] = (close - low_n) / (close + 1e-10)
    
    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / (close + 1e-10)
    
    # 趋势强度
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-10)
    df['di_diff'] = plus_di - minus_di
    df['adx'] = (plus_di - minus_di).abs().rolling(14).mean()
    
    # 未来收益
    df['future_return'] = close.shift(-1) / close - 1
    df['label'] = (df['future_return'] > 0.003).astype(int)
    
    return df

def load_data(n_stocks=350, start_date='2018-01-01', end_date='2025-01-01'):
    """加载数据"""
    print(f"加载数据: {start_date} - {end_date}")
    
    stock_files = list(DATA_DIR.glob('*.parquet'))
    
    valid = []
    for f in stock_files:
        try:
            df = pd.read_parquet(f)
            df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
            if len(df) >= 200:
                valid.append((f, df['volume'].mean()))
        except:
            continue
    
    valid.sort(key=lambda x: x[1], reverse=True)
    selected = valid[:n_stocks]
    print(f"选择 {len(selected)} 只股票")
    
    all_data = []
    for f, _ in selected:
        df = pd.read_parquet(f)
        df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
        df = compute_features(df)
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    print(f"总数据: {len(data)} 条")
    
    print("计算截面特征...")
    data = compute_cross_sectional_features(data)
    
    print(f"标签分布: {data['label'].value_counts().to_dict()}")
    
    return data

def get_feature_cols():
    """特征列"""
    return [
        'return_1d', 'return_2d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
        'momentum_short', 'momentum_long', 'momentum_reversal',
        'momentum_strength_5', 'momentum_strength_10', 'momentum_strength_20',
        'relative_strength_5d', 'relative_strength_20d', 'relative_momentum',
        'return_1d_rank', 'return_5d_rank', 'return_20d_rank', 'volume_rank', 'volatility_20_rank',
        'market_up_ratio',
        'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_60',
        'ma_slope_5', 'ma_slope_10', 'ma_slope_20', 'ma_alignment',
        'volatility_5', 'volatility_20', 'volatility_60', 'vol_regime', 'volatility_skew',
        'vol_ratio', 'vol_ma_ratio', 'vol_std', 'vol_price_diverge',
        'rsi_6', 'rsi_14', 'rsi_diff',
        'macd', 'macd_signal', 'macd_hist', 'macd_hist_slope',
        'bb_position', 'bb_width',
        'body', 'wick_upper', 'wick_lower', 'range_pct',
        'consecutive_up', 'consecutive_down',
        'price_pos_20', 'price_pos_60',
        'dist_high_20', 'dist_low_20', 'dist_high_60', 'dist_low_60',
        'atr_pct',
        'di_diff', 'adx',
    ]


class EnsembleModel:
    """三模型集成"""
    
    def __init__(self, weights=None):
        self.lgb_model = None
        self.xgb_model = None
        self.cat_model = None
        self.weights = weights or [0.4, 0.35, 0.25]  # LGB, XGB, CAT
        
    def train_lgb(self, X_train, y_train, X_val, y_val, params=None):
        """训练 LightGBM"""
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 45,
                'max_depth': 6,
                'learning_rate': 0.015,
                'feature_fraction': 0.65,
                'bagging_fraction': 0.65,
                'bagging_freq': 5,
                'min_child_samples': 100,
                'reg_alpha': 0.6,
                'reg_lambda': 0.6,
                'verbose': -1,
                'seed': SEED,
            }
        
        train_data = lgb.Dataset(X_train, y_train)
        val_data = lgb.Dataset(X_val, y_val, reference=train_data)
        
        self.lgb_model = lgb.train(
            params, train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
        )
        
        return self.lgb_model
    
    def train_xgb(self, X_train, y_train, X_val, y_val, params=None):
        """训练 XGBoost"""
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.015,
                'subsample': 0.65,
                'colsample_bytree': 0.65,
                'min_child_weight': 100,
                'reg_alpha': 0.6,
                'reg_lambda': 0.6,
                'seed': SEED,
                'verbosity': 0,
            }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        self.xgb_model = xgb.train(
            params, dtrain,
            num_boost_round=2000,
            evals=[(dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=200
        )
        
        return self.xgb_model
    
    def train_catboost(self, X_train, y_train, X_val, y_val, params=None):
        """训练 CatBoost"""
        if params is None:
            params = {
                'iterations': 2000,
                'depth': 6,
                'learning_rate': 0.015,
                'l2_leaf_reg': 3,
                'random_seed': SEED,
                'verbose': 200,
                'early_stopping_rounds': 100,
                'eval_metric': 'AUC',
            }
        
        self.cat_model = CatBoostClassifier(**params)
        self.cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        return self.cat_model
    
    def train_all(self, X_train, y_train, X_val, y_val):
        """训练所有模型"""
        print("\n" + "="*60)
        print("训练 LightGBM...")
        print("="*60)
        self.train_lgb(X_train, y_train, X_val, y_val)
        
        print("\n" + "="*60)
        print("训练 XGBoost...")
        print("="*60)
        self.train_xgb(X_train, y_train, X_val, y_val)
        
        print("\n" + "="*60)
        print("训练 CatBoost...")
        print("="*60)
        self.train_catboost(X_train, y_train, X_val, y_val)
    
    def predict(self, X):
        """集成预测"""
        pred_lgb = self.lgb_model.predict(X)
        pred_xgb = self.xgb_model.predict(xgb.DMatrix(X))
        pred_cat = self.cat_model.predict_proba(X)[:, 1]
        
        # 加权平均
        pred_ensemble = (
            self.weights[0] * pred_lgb +
            self.weights[1] * pred_xgb +
            self.weights[2] * pred_cat
        )
        
        return pred_ensemble, pred_lgb, pred_xgb, pred_cat
    
    def optimize_weights(self, X_val, y_val):
        """优化集成权重"""
        pred_lgb = self.lgb_model.predict(X_val)
        pred_xgb = self.xgb_model.predict(xgb.DMatrix(X_val))
        pred_cat = self.cat_model.predict_proba(X_val)[:, 1]
        
        def objective(trial):
            w1 = trial.suggest_float('w_lgb', 0.2, 0.6)
            w2 = trial.suggest_float('w_xgb', 0.2, 0.5)
            w3 = 1 - w1 - w2
            if w3 < 0.1:
                return 0.5
            
            pred = w1 * pred_lgb + w2 * pred_xgb + w3 * pred_cat
            return roc_auc_score(y_val, pred)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, show_progress_bar=True)
        
        best = study.best_params
        w3 = 1 - best['w_lgb'] - best['w_xgb']
        self.weights = [best['w_lgb'], best['w_xgb'], w3]
        
        print(f"\n优化后权重: LGB={self.weights[0]:.3f}, XGB={self.weights[1]:.3f}, CAT={self.weights[2]:.3f}")
        print(f"验证集 AUC: {study.best_value:.4f}")
        
        return self.weights


def train_and_evaluate(data):
    """训练和评估集成模型"""
    feature_cols = get_feature_cols()
    
    df = data.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ['label', 'future_return'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"\n有效数据: {len(df)} 条")
    print(f"正样本比例: {df['label'].mean():.4f}")
    
    # 时间划分
    train_end = '2023-07-01'
    val_end = '2024-01-01'
    
    train = df[df['date'] < train_end]
    val = df[(df['date'] >= train_end) & (df['date'] < val_end)]
    test = df[df['date'] >= val_end]
    
    print(f"\n训练集: {len(train)} ({train['date'].min().date()} - {train['date'].max().date()})")
    print(f"验证集: {len(val)} ({val['date'].min().date()} - {val['date'].max().date()})")
    print(f"测试集: {len(test)} ({test['date'].min().date()} - {test['date'].max().date()})")
    
    X_train, y_train = train[feature_cols], train['label']
    X_val, y_val = val[feature_cols], val['label']
    X_test, y_test = test[feature_cols], test['label']
    
    # 训练集成模型
    ensemble = EnsembleModel()
    ensemble.train_all(X_train, y_train, X_val, y_val)
    
    # 优化权重
    print("\n" + "="*60)
    print("优化集成权重...")
    print("="*60)
    ensemble.optimize_weights(X_val, y_val)
    
    # 预测
    pred_train, _, _, _ = ensemble.predict(X_train)
    pred_val, _, _, _ = ensemble.predict(X_val)
    pred_test, pred_lgb, pred_xgb, pred_cat = ensemble.predict(X_test)
    
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    
    # 各模型单独评估
    print("\n单模型测试集 AUC:")
    print(f"  LightGBM: {roc_auc_score(y_test, pred_lgb):.4f}")
    print(f"  XGBoost:  {roc_auc_score(y_test, pred_xgb):.4f}")
    print(f"  CatBoost: {roc_auc_score(y_test, pred_cat):.4f}")
    
    print("\n集成模型:")
    for name, pred, y in [('训练', pred_train, y_train), ('验证', pred_val, y_val), ('测试', pred_test, y_test)]:
        auc = roc_auc_score(y, pred)
        acc = accuracy_score(y, (pred > 0.5).astype(int))
        print(f"  {name}集 - AUC: {auc:.4f}, Acc: {acc:.4f}")
    
    # 分组分析
    print("\n" + "="*60)
    print("测试集分组分析")
    print("="*60)
    
    result = pd.DataFrame({
        'date': test['date'].values,
        'ts_code': test['ts_code'].values,
        'prob': pred_test,
        'prob_lgb': pred_lgb,
        'prob_xgb': pred_xgb,
        'prob_cat': pred_cat,
        'label': y_test.values,
        'return': test['future_return'].values
    })
    
    result['decile'] = pd.qcut(result['prob'], 10, labels=False, duplicates='drop')
    
    print("\n按预测概率十分位:")
    print(f"{'分位':>6} {'样本数':>8} {'平均概率':>10} {'实际涨比例':>12} {'平均收益':>10}")
    print("-" * 50)
    
    for d in sorted(result['decile'].unique()):
        g = result[result['decile'] == d]
        print(f"{d:>6} {len(g):>8} {g['prob'].mean():>10.4f} {g['label'].mean():>12.2%} {g['return'].mean()*100:>9.3f}%")
    
    # 策略回测
    print("\n" + "="*60)
    print("策略回测（考虑交易成本 0.1%）")
    print("="*60)
    
    result['date_str'] = result['date'].astype(str)
    
    TRANSACTION_COST = 0.001
    
    backtest_results = {}
    
    for top_n in [5, 10, 20, 30]:
        # 每日选 top_n
        daily_top_list = []
        for date_str, group in result.groupby('date_str'):
            top_stocks = group.nlargest(top_n, 'prob')
            daily_top_list.append(top_stocks)
        daily_top = pd.concat(daily_top_list, ignore_index=True)
        
        daily_returns = daily_top.groupby('date_str')['return'].mean() - TRANSACTION_COST
        
        total_days = len(daily_returns)
        win_days = (daily_returns > 0).sum()
        win_rate = win_days / total_days
        avg_daily = daily_returns.mean()
        total_return = (1 + daily_returns).prod() - 1
        sharpe = daily_returns.mean() / (daily_returns.std() + 1e-10) * np.sqrt(252)
        
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        backtest_results[top_n] = {
            'days': total_days,
            'win_rate': win_rate,
            'avg_daily': avg_daily,
            'annual_return': avg_daily * 252,
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd
        }
        
        print(f"\n每日选 Top {top_n}:")
        print(f"  交易天数: {total_days}")
        print(f"  日胜率: {win_rate:.2%}")
        print(f"  日均收益: {avg_daily*100:.3f}%")
        print(f"  年化收益: {avg_daily*252*100:.1f}%")
        print(f"  累计收益: {total_return*100:.1f}%")
        print(f"  夏普比率: {sharpe:.2f}")
        print(f"  最大回撤: {max_dd*100:.1f}%")
    
    return ensemble, result, backtest_results


if __name__ == '__main__':
    print("="*60)
    print("三模型集成预测 v1 - LightGBM + XGBoost + CatBoost")
    print("="*60)
    
    data = load_data(n_stocks=350, start_date='2018-01-01', end_date='2025-01-01')
    
    ensemble, result, backtest_results = train_and_evaluate(data)
    
    # 保存结果
    result.to_csv('/Users/ronghui/Projects/willrone/experiments/ensemble_v1_results.csv', index=False)
    print("\n结果已保存到: ensemble_v1_results.csv")
    
    # 生成报告
    report = """# 集成模型 v1 对比报告

## 基线 (LightGBM v4.1)
- 测试集 AUC: 0.5551
- Top 5 夏普比率: 7.22
- Top 5 年化收益: 375.3%
- 最大回撤: -16.5%

## 集成模型 v1 (LightGBM + XGBoost + CatBoost)
"""
    
    top5 = backtest_results[5]
    report += f"""
### 测试集性能
- Top 5 夏普比率: {top5['sharpe']:.2f}
- Top 5 年化收益: {top5['annual_return']*100:.1f}%
- 最大回撤: {top5['max_drawdown']*100:.1f}%
- 日胜率: {top5['win_rate']:.2%}

### 各 Top N 策略对比
| Top N | 夏普比率 | 年化收益 | 最大回撤 | 日胜率 |
|-------|---------|---------|---------|--------|
"""
    
    for n in [5, 10, 20, 30]:
        r = backtest_results[n]
        report += f"| {n} | {r['sharpe']:.2f} | {r['annual_return']*100:.1f}% | {r['max_drawdown']*100:.1f}% | {r['win_rate']:.2%} |\n"
    
    report += f"""
### 改进分析
- 夏普比率变化: {top5['sharpe']:.2f} vs 7.22 (基线)
- 年化收益变化: {top5['annual_return']*100:.1f}% vs 375.3% (基线)
- 回撤变化: {top5['max_drawdown']*100:.1f}% vs -16.5% (基线)

### 结论
集成模型通过结合三种不同的梯度提升算法，利用模型多样性来提升预测稳定性。
"""
    
    with open('/Users/ronghui/Projects/willrone/experiments/ensemble_report.md', 'w') as f:
        f.write(report)
    
    print("报告已保存到: ensemble_report.md")
