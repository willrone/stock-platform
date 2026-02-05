#!/usr/bin/env python3
"""
LightGBM 股票预测模型 v4.1 - 最终优化版
核心改进：
1. 标签阈值提高到 0.3%（提高正样本质量）
2. 降低市场特征权重（增加个股选择能力）
3. 精简特征集（移除低重要性特征）
4. 增加个股相对强度因子
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path('/Users/ronghui/Documents/GitHub/willrone/data/parquet/stock_data')

def compute_cross_sectional_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """计算截面特征（每日排名）"""
    df = df_all.copy()
    
    # 按日期分组计算排名
    for col in ['return_1d', 'return_5d', 'return_20d', 'volume', 'volatility_20']:
        if col in df.columns:
            df[f'{col}_rank'] = df.groupby('date')[col].rank(pct=True)
    
    # 市场情绪（降权：只保留涨跌比例）
    df['market_up_ratio'] = df.groupby('date')['return_1d'].transform(
        lambda x: (x > 0).sum() / len(x)
    )
    
    # 个股相对强度（重点）
    market_return_5d = df.groupby('date')['return_5d'].transform('mean')
    market_return_20d = df.groupby('date')['return_20d'].transform('mean')
    
    df['relative_strength_5d'] = df['return_5d'] - market_return_5d
    df['relative_strength_20d'] = df['return_20d'] - market_return_20d
    
    # 相对动量
    df['relative_momentum'] = df['relative_strength_5d'] - df['relative_strength_20d']
    
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标特征（精简版）"""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_ = df['open']
    
    # ========== 基础收益 ==========
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'return_{period}d'] = close.pct_change(period)
    
    # ========== 动量因子 ==========
    df['momentum_short'] = df['return_5d'] - df['return_10d']
    df['momentum_long'] = df['return_10d'] - df['return_20d']
    df['momentum_reversal'] = -df['return_1d']  # 短期反转
    
    # 动量强度
    returns = close.pct_change()
    for period in [5, 10, 20]:
        up_days = (returns > 0).rolling(period).sum()
        df[f'momentum_strength_{period}'] = up_days / period
    
    # ========== 移动平均 ==========
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
    
    # ========== 波动率因子 ==========
    for window in [5, 20, 60]:
        df[f'volatility_{window}'] = returns.rolling(window).std()
    
    df['vol_regime'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)
    
    # 波动率偏度
    df['volatility_skew'] = returns.rolling(20).skew()
    
    # ========== 成交量特征 ==========
    vol_ma20 = volume.rolling(20).mean()
    vol_ma5 = volume.rolling(5).mean()
    
    df['vol_ratio'] = volume / (vol_ma20 + 1)
    df['vol_ma_ratio'] = vol_ma5 / (vol_ma20 + 1)
    df['vol_std'] = volume.rolling(20).std() / (vol_ma20 + 1)
    
    # 量价背离
    df['price_up'] = (close > close.shift(1)).astype(int)
    df['vol_up'] = (volume > volume.shift(1)).astype(int)
    df['vol_price_diverge'] = (df['price_up'] != df['vol_up']).astype(int)
    
    # ========== RSI ==========
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    for period in [6, 14]:
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    df['rsi_diff'] = df['rsi_6'] - df['rsi_14']
    
    # ========== MACD ==========
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_hist_slope'] = df['macd_hist'].diff(3)
    
    # ========== 布林带 ==========
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_position'] = (close - bb_mid) / (2 * bb_std + 1e-10)
    df['bb_width'] = 4 * bb_std / (bb_mid + 1e-10)
    
    # ========== 价格形态 ==========
    df['body'] = (close - open_) / (open_ + 1e-10)
    df['wick_upper'] = (high - np.maximum(close, open_)) / (high - low + 1e-10)
    df['wick_lower'] = (np.minimum(close, open_) - low) / (high - low + 1e-10)
    df['range_pct'] = (high - low) / (close + 1e-10)
    
    df['consecutive_up'] = (close > close.shift(1)).rolling(5).sum()
    df['consecutive_down'] = (close < close.shift(1)).rolling(5).sum()
    
    # ========== 价格位置 ==========
    for window in [20, 60]:
        high_n = high.rolling(window).max()
        low_n = low.rolling(window).min()
        df[f'price_pos_{window}'] = (close - low_n) / (high_n - low_n + 1e-10)
        df[f'dist_high_{window}'] = (high_n - close) / (close + 1e-10)
        df[f'dist_low_{window}'] = (close - low_n) / (close + 1e-10)
    
    # ========== ATR ==========
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / (close + 1e-10)
    
    # ========== 趋势强度 ==========
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-10)
    df['di_diff'] = plus_di - minus_di
    df['adx'] = (plus_di - minus_di).abs().rolling(14).mean()
    
    # ========== 未来收益 ==========
    df['future_return'] = close.shift(-1) / close - 1
    
    # 标签：绝对收益 > 0.3%（提高阈值）
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
    """特征列（精简版）"""
    return [
        # 基础收益
        'return_1d', 'return_2d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
        # 动量因子
        'momentum_short', 'momentum_long', 'momentum_reversal',
        'momentum_strength_5', 'momentum_strength_10', 'momentum_strength_20',
        # 相对强度（重点）
        'relative_strength_5d', 'relative_strength_20d', 'relative_momentum',
        # 截面排名
        'return_1d_rank', 'return_5d_rank', 'return_20d_rank', 'volume_rank', 'volatility_20_rank',
        # 市场情绪（仅保留一个）
        'market_up_ratio',
        # MA
        'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_60',
        'ma_slope_5', 'ma_slope_10', 'ma_slope_20', 'ma_alignment',
        # 波动率
        'volatility_5', 'volatility_20', 'volatility_60', 'vol_regime', 'volatility_skew',
        # 成交量
        'vol_ratio', 'vol_ma_ratio', 'vol_std', 'vol_price_diverge',
        # RSI
        'rsi_6', 'rsi_14', 'rsi_diff',
        # MACD
        'macd', 'macd_signal', 'macd_hist', 'macd_hist_slope',
        # 布林带
        'bb_position', 'bb_width',
        # 形态
        'body', 'wick_upper', 'wick_lower', 'range_pct',
        'consecutive_up', 'consecutive_down',
        # 价格位置
        'price_pos_20', 'price_pos_60',
        'dist_high_20', 'dist_low_20', 'dist_high_60', 'dist_low_60',
        # ATR
        'atr_pct',
        # 趋势
        'di_diff', 'adx',
    ]

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna 优化目标函数"""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 30, 60),
        'max_depth': trial.suggest_int('max_depth', 5, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.8),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.8),
        'bagging_freq': 5,
        'min_child_samples': trial.suggest_int('min_child_samples', 80, 200),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.3, 1.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.3, 1.5),
        'verbose': -1,
        'seed': SEED,
    }
    
    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)
    
    model = lgb.train(
        params, train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    pred = model.predict(X_val)
    auc = roc_auc_score(y_val, pred)
    
    return auc

def train_with_optuna(X_train, y_train, X_val, y_val, n_trials=50):
    """使用 Optuna 进行超参优化"""
    print(f"\n开始 Optuna 超参优化 ({n_trials} trials)...")
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n最佳 AUC: {study.best_value:.4f}")
    print("最佳参数:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    return study.best_params

def train_and_evaluate(data, use_optuna=True, n_trials=50):
    """训练和评估"""
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
    
    # 超参优化
    if use_optuna:
        best_params = train_with_optuna(X_train, y_train, X_val, y_val, n_trials=n_trials)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'seed': SEED,
            **best_params
        }
    else:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 40,
            'max_depth': 6,
            'learning_rate': 0.01,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.6,
            'bagging_freq': 5,
            'min_child_samples': 120,
            'reg_alpha': 0.8,
            'reg_lambda': 0.8,
            'verbose': -1,
            'seed': SEED,
        }
    
    # 训练
    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)
    
    print("\n训练最终模型...")
    model = lgb.train(
        params, train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(200)
        ]
    )
    
    # 预测
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
    
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    
    for name, pred, y in [('训练', pred_train, y_train), ('验证', pred_val, y_val), ('测试', pred_test, y_test)]:
        auc = roc_auc_score(y, pred)
        acc = accuracy_score(y, (pred > 0.5).astype(int))
        print(f"{name}集 - AUC: {auc:.4f}, Acc: {acc:.4f}")
    
    # 特征重要性
    print("\n特征重要性 Top 20:")
    imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance('gain')
    }).sort_values('importance', ascending=False)
    
    for _, r in imp.head(20).iterrows():
        print(f"  {r['feature']:25s}: {r['importance']:.1f}")
    
    # 分组分析
    print("\n" + "="*60)
    print("测试集分组分析")
    print("="*60)
    
    result = pd.DataFrame({
        'date': test['date'].values,
        'ts_code': test['ts_code'].values,
        'prob': pred_test,
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
    
    for top_n in [5, 10, 20, 30]:
        daily_top = result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'prob')
        ).reset_index(drop=True)
        
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
        
        print(f"\n每日选 Top {top_n}:")
        print(f"  交易天数: {total_days}")
        print(f"  日胜率: {win_rate:.2%}")
        print(f"  日均收益: {avg_daily*100:.3f}%")
        print(f"  年化收益: {avg_daily*252*100:.1f}%")
        print(f"  累计收益: {total_return*100:.1f}%")
        print(f"  夏普比率: {sharpe:.2f}")
        print(f"  最大回撤: {max_dd*100:.1f}%")
    
    return model, result

if __name__ == '__main__':
    print("="*60)
    print("LightGBM 股票预测模型 v4.1 - 最终优化版")
    print("="*60)
    
    data = load_data(n_stocks=350, start_date='2018-01-01', end_date='2025-01-01')
    
    model, result = train_and_evaluate(data, use_optuna=True, n_trials=50)
    
    model.save_model('/Users/ronghui/Documents/GitHub/willrone/experiments/lgbm_model_v4_final.txt')
    print("\n模型已保存到: lgbm_model_v4_final.txt")
    
    result.to_csv('/Users/ronghui/Documents/GitHub/willrone/experiments/lgbm_v4_final_results.csv', index=False)
    print("结果已保存到: lgbm_v4_final_results.csv")
