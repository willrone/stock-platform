#!/usr/bin/env python3
"""
LightGBM v6 - 追求稳定性
目标：降低最大回撤，提高夏普比率稳定性

改进：
1. 更强正则化 + 更少叶子节点
2. 波动率调整仓位（高波动降仓）
3. 动态止损（连续亏损暂停）
4. 更长训练窗口（36个月）
5. 特征稳定性筛选
6. 多模型集成
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import optuna
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/Users/ronghui/Projects/willrone/data/parquet/stock_data')
np.random.seed(42)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标特征（精简稳定版）"""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_ = df['open']
    
    # 收益率
    for period in [1, 3, 5, 10, 20]:
        df[f'return_{period}d'] = close.pct_change(period)
    
    # 动量（精简）
    df['momentum_1m'] = close.pct_change(20)
    df['momentum_accel'] = df['momentum_1m'] - df['momentum_1m'].shift(5)
    
    # MA（精简）
    for window in [5, 20, 60]:
        ma = close.rolling(window).mean()
        df[f'ma_ratio_{window}'] = close / ma - 1
    
    df['ma_5'] = close.rolling(5).mean()
    df['ma_20'] = close.rolling(20).mean()
    df['ma_60'] = close.rolling(60).mean()
    df['ma_score'] = (
        (df['ma_5'] > df['ma_20']).astype(float) +
        (df['ma_20'] > df['ma_60']).astype(float)
    ) / 2
    
    # 成交量（精简）
    vol_ma20 = volume.rolling(20).mean()
    df['vol_ratio'] = volume / (vol_ma20 + 1)
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_deviation'] = df['rsi_14'] - 50
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = (ema12 - ema26) / close
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 布林带
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_position'] = (close - bb_mid) / (2 * bb_std + 1e-10)
    
    # 波动率
    returns = close.pct_change()
    df['volatility_20'] = returns.rolling(20).std()
    df['volatility_60'] = returns.rolling(60).std()
    df['vol_regime'] = df['volatility_20'] / (df['volatility_60'] + 1e-10)
    
    # 价格位置
    high_60 = high.rolling(60).max()
    low_60 = low.rolling(60).min()
    df['price_pos_60'] = (close - low_60) / (high_60 - low_60 + 1e-10)
    
    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr_pct'] = tr.rolling(14).mean() / (close + 1e-10)
    
    df['future_return'] = close.shift(-1) / close - 1
    
    return df

def add_cross_sectional_features(data: pd.DataFrame) -> pd.DataFrame:
    """添加截面特征"""
    # 排名特征（精简）
    for col in ['return_5d', 'momentum_1m', 'vol_ratio', 'rsi_14']:
        if col in data.columns:
            data[f'{col}_rank'] = data.groupby('date')[col].rank(pct=True)
    
    # 相对强度
    market_return_5d = data.groupby('date')['return_5d'].transform('mean')
    data['relative_strength'] = data['return_5d'] - market_return_5d
    
    # 市场状态
    daily_stats = data.groupby('date').agg({
        'return_1d': 'mean',
        'volatility_20': 'mean',
    }).reset_index()
    daily_stats.columns = ['date', 'market_return', 'market_vol']
    daily_stats = daily_stats.sort_values('date')
    
    # 市场趋势（更平滑）
    daily_stats['market_ma10'] = daily_stats['market_return'].rolling(10).mean()
    daily_stats['market_ma30'] = daily_stats['market_return'].rolling(30).mean()
    daily_stats['market_trend'] = (daily_stats['market_ma10'] > daily_stats['market_ma30']).astype(int)
    
    # 市场波动率状态
    daily_stats['market_vol_ma'] = daily_stats['market_vol'].rolling(20).mean()
    daily_stats['high_vol'] = (daily_stats['market_vol'] > daily_stats['market_vol_ma'] * 1.2).astype(int)
    
    data = data.merge(daily_stats[['date', 'market_trend', 'high_vol', 'market_vol']], on='date', how='left')
    
    # 标签
    daily_mean = data.groupby('date')['future_return'].transform('mean')
    data['excess_return'] = data['future_return'] - daily_mean
    data['label'] = (data['excess_return'] > 0.002).astype(int)  # 超额 > 0.2%
    
    return data

def load_data(n_stocks=450, start_date='2017-01-01', end_date='2025-01-01'):
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
    data = add_cross_sectional_features(data)
    print(f"总数据: {len(data)} 条")
    return data

def get_feature_cols():
    """精简特征集"""
    return [
        'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
        'momentum_1m', 'momentum_accel',
        'ma_ratio_5', 'ma_ratio_20', 'ma_ratio_60', 'ma_score',
        'vol_ratio',
        'rsi_14', 'rsi_deviation',
        'macd', 'macd_signal', 'macd_hist',
        'bb_position',
        'volatility_20', 'vol_regime',
        'price_pos_60',
        'atr_pct',
        'relative_strength',
        'return_5d_rank', 'momentum_1m_rank', 'vol_ratio_rank', 'rsi_14_rank',
    ]

def optuna_stable(X_train, y_train, X_val, y_val, n_trials=50):
    """Optuna 优化（偏向稳定性）"""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            # 更保守的参数范围
            'num_leaves': trial.suggest_int('num_leaves', 8, 31),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.7),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.7),
            'bagging_freq': 5,
            # 更强正则化
            'min_child_samples': trial.suggest_int('min_child_samples', 100, 300),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 3.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.01, 0.1),
            'verbose': -1,
            'seed': 42,
        }
        
        train_data = lgb.Dataset(X_train, y_train)
        val_data = lgb.Dataset(X_val, y_val, reference=train_data)
        
        model = lgb.train(
            params, train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        
        pred = model.predict(X_val)
        return roc_auc_score(y_val, pred)
    
    print(f"\nOptuna 稳定性优化 ({n_trials} trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"最佳 AUC: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")
    
    return study.best_params

def backtest_with_risk_control(data, params):
    """带风控的回测"""
    feature_cols = get_feature_cols()
    label_col = 'label'
    
    df = data.replace([np.inf, -np.inf], np.nan)
    required = [c for c in feature_cols if c in df.columns] + [label_col, 'future_return', 'excess_return', 'market_trend', 'high_vol', 'market_vol']
    df = df.dropna(subset=required)
    df = df.sort_values('date').reset_index(drop=True)
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"有效数据: {len(df)} 条")
    print(f"特征数: {len(feature_cols)}")
    
    df['date_dt'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date_dt'].dt.to_period('M')
    
    all_months = sorted(df['year_month'].unique())
    
    # 更长训练窗口
    train_months = 36
    test_months = 3
    
    start_test_idx = None
    for i, m in enumerate(all_months):
        if m >= pd.Period('2022-01'):
            start_test_idx = i
            break
    
    if start_test_idx is None or start_test_idx < train_months:
        print("数据不足")
        return None
    
    all_results = []
    
    print(f"\n滚动训练 (训练{train_months}月, 测试{test_months}月)")
    print("="*70)
    
    test_idx = start_test_idx
    while test_idx + test_months <= len(all_months):
        train_start = all_months[test_idx - train_months]
        train_end = all_months[test_idx - 1]
        test_start = all_months[test_idx]
        test_end = all_months[min(test_idx + test_months - 1, len(all_months) - 1)]
        
        train_mask = (df['year_month'] >= train_start) & (df['year_month'] <= train_end)
        test_mask = (df['year_month'] >= test_start) & (df['year_month'] <= test_end)
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(train_df) < 1000 or len(test_df) < 100:
            test_idx += test_months
            continue
        
        X_train, y_train = train_df[feature_cols], train_df[label_col]
        X_test, y_test = test_df[feature_cols], test_df[label_col]
        
        train_data = lgb.Dataset(X_train, y_train)
        
        model = lgb.train(
            params, train_data,
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        pred = model.predict(X_test)
        
        result = pd.DataFrame({
            'date': test_df['date'].values,
            'ts_code': test_df['ts_code'].values,
            'prob': pred,
            'label': y_test.values,
            'return': test_df['future_return'].values,
            'excess_return': test_df['excess_return'].values,
            'market_trend': test_df['market_trend'].values,
            'high_vol': test_df['high_vol'].values,
            'market_vol': test_df['market_vol'].values,
        })
        
        all_results.append(result)
        
        auc = roc_auc_score(y_test, pred)
        print(f"  {test_start} - {test_end}: AUC={auc:.4f}")
        
        test_idx += test_months
    
    return pd.concat(all_results, ignore_index=True)

def run_strategies(result):
    """运行多种风控策略"""
    print("\n" + "="*70)
    print("策略回测（含风控）")
    print("="*70)
    
    COST = 0.001
    
    result['date_str'] = result['date'].astype(str)
    result['date_dt'] = pd.to_datetime(result['date'])
    result['year'] = result['date_dt'].dt.year
    
    def calc_metrics(daily_returns, name):
        total_days = len(daily_returns)
        win_rate = (daily_returns > 0).mean()
        avg_daily = daily_returns.mean()
        total_return = (1 + daily_returns).prod() - 1
        sharpe = avg_daily / (daily_returns.std() + 1e-10) * np.sqrt(252)
        cumsum = daily_returns.cumsum()
        max_dd = (cumsum - cumsum.cummax()).min()
        
        print(f"\n{name}:")
        print(f"  整体: 天数={total_days}, 胜率={win_rate:.1%}, 日均={avg_daily*100:.3f}%, "
              f"累计={total_return*100:.1f}%, 夏普={sharpe:.2f}, 回撤={max_dd*100:.1f}%")
        
        return {'sharpe': sharpe, 'max_dd': max_dd, 'total_return': total_return}
    
    # 策略1: 基础择时
    print("\n【策略1】基础择时 (趋势向上时做多)")
    for top_n in [15, 20, 30]:
        daily_top = result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'prob')
        ).reset_index(drop=True)
        
        daily_agg = daily_top.groupby('date_str').agg({
            'return': 'mean',
            'market_trend': 'first',
        }).reset_index()
        
        daily_agg['strategy_return'] = daily_agg.apply(
            lambda x: x['return'] - 2 * COST if x['market_trend'] == 1 else 0,
            axis=1
        )
        
        calc_metrics(daily_agg['strategy_return'], f"择时 Top {top_n}")
        
        # 分年度
        daily_agg['year'] = pd.to_datetime(daily_agg['date_str']).dt.year
        for year in sorted(daily_agg['year'].unique()):
            year_data = daily_agg[daily_agg['year'] == year]
            y_ret = year_data['strategy_return']
            y_total = (1 + y_ret).prod() - 1
            y_sharpe = y_ret.mean() / (y_ret.std() + 1e-10) * np.sqrt(252)
            y_dd = (y_ret.cumsum() - y_ret.cumsum().cummax()).min()
            print(f"    {year}: 年收益={y_total*100:.1f}%, 夏普={y_sharpe:.2f}, 回撤={y_dd*100:.1f}%")
    
    # 策略2: 择时 + 波动率调仓
    print("\n【策略2】择时 + 波动率调仓 (高波动时半仓)")
    for top_n in [15, 20, 30]:
        daily_top = result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'prob')
        ).reset_index(drop=True)
        
        daily_agg = daily_top.groupby('date_str').agg({
            'return': 'mean',
            'market_trend': 'first',
            'high_vol': 'first',
        }).reset_index()
        
        def calc_return(row):
            if row['market_trend'] == 0:
                return 0
            position = 0.5 if row['high_vol'] == 1 else 1.0
            return (row['return'] - 2 * COST) * position
        
        daily_agg['strategy_return'] = daily_agg.apply(calc_return, axis=1)
        
        calc_metrics(daily_agg['strategy_return'], f"择时+波动调仓 Top {top_n}")
        
        daily_agg['year'] = pd.to_datetime(daily_agg['date_str']).dt.year
        for year in sorted(daily_agg['year'].unique()):
            year_data = daily_agg[daily_agg['year'] == year]
            y_ret = year_data['strategy_return']
            y_total = (1 + y_ret).prod() - 1
            y_sharpe = y_ret.mean() / (y_ret.std() + 1e-10) * np.sqrt(252)
            y_dd = (y_ret.cumsum() - y_ret.cumsum().cummax()).min()
            print(f"    {year}: 年收益={y_total*100:.1f}%, 夏普={y_sharpe:.2f}, 回撤={y_dd*100:.1f}%")
    
    # 策略3: 择时 + 波动率调仓 + 止损
    print("\n【策略3】择时 + 波动调仓 + 动态止损 (连续3天亏损暂停1天)")
    for top_n in [15, 20, 30]:
        daily_top = result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'prob')
        ).reset_index(drop=True)
        
        daily_agg = daily_top.groupby('date_str').agg({
            'return': 'mean',
            'market_trend': 'first',
            'high_vol': 'first',
        }).reset_index().sort_values('date_str')
        
        # 计算连续亏损
        daily_agg['raw_return'] = daily_agg['return'] - 2 * COST
        daily_agg['is_loss'] = (daily_agg['raw_return'] < 0).astype(int)
        daily_agg['consecutive_loss'] = daily_agg['is_loss'].rolling(3, min_periods=1).sum()
        daily_agg['pause'] = (daily_agg['consecutive_loss'].shift(1) >= 3).fillna(False).astype(int)
        
        def calc_return_with_stop(row):
            if row['market_trend'] == 0 or row['pause'] == 1:
                return 0
            position = 0.5 if row['high_vol'] == 1 else 1.0
            return row['raw_return'] * position
        
        daily_agg['strategy_return'] = daily_agg.apply(calc_return_with_stop, axis=1)
        
        calc_metrics(daily_agg['strategy_return'], f"全风控 Top {top_n}")
        
        daily_agg['year'] = pd.to_datetime(daily_agg['date_str']).dt.year
        for year in sorted(daily_agg['year'].unique()):
            year_data = daily_agg[daily_agg['year'] == year]
            y_ret = year_data['strategy_return']
            y_total = (1 + y_ret).prod() - 1
            y_sharpe = y_ret.mean() / (y_ret.std() + 1e-10) * np.sqrt(252)
            y_dd = (y_ret.cumsum() - y_ret.cumsum().cummax()).min()
            print(f"    {year}: 年收益={y_total*100:.1f}%, 夏普={y_sharpe:.2f}, 回撤={y_dd*100:.1f}%")
    
    # 策略4: 更保守的择时（趋势+低波动）
    print("\n【策略4】保守择时 (趋势向上 且 低波动时做多)")
    for top_n in [15, 20, 30]:
        daily_top = result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'prob')
        ).reset_index(drop=True)
        
        daily_agg = daily_top.groupby('date_str').agg({
            'return': 'mean',
            'market_trend': 'first',
            'high_vol': 'first',
        }).reset_index()
        
        daily_agg['strategy_return'] = daily_agg.apply(
            lambda x: x['return'] - 2 * COST if (x['market_trend'] == 1 and x['high_vol'] == 0) else 0,
            axis=1
        )
        
        calc_metrics(daily_agg['strategy_return'], f"保守择时 Top {top_n}")
        
        daily_agg['year'] = pd.to_datetime(daily_agg['date_str']).dt.year
        for year in sorted(daily_agg['year'].unique()):
            year_data = daily_agg[daily_agg['year'] == year]
            y_ret = year_data['strategy_return']
            y_total = (1 + y_ret).prod() - 1
            y_sharpe = y_ret.mean() / (y_ret.std() + 1e-10) * np.sqrt(252)
            y_dd = (y_ret.cumsum() - y_ret.cumsum().cummax()).min()
            active = ((year_data['market_trend'] == 1) & (year_data['high_vol'] == 0)).sum()
            print(f"    {year}: 交易天数={active}, 年收益={y_total*100:.1f}%, 夏普={y_sharpe:.2f}, 回撤={y_dd*100:.1f}%")

def main():
    # 加载数据（更长时间范围）
    data = load_data(n_stocks=450, start_date='2017-01-01', end_date='2025-01-01')
    
    # Optuna 优化
    df = data.replace([np.inf, -np.inf], np.nan)
    feature_cols = get_feature_cols()
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    opt_train = df[(df['date'] >= '2019-01-01') & (df['date'] < '2021-01-01')]
    opt_val = df[(df['date'] >= '2021-01-01') & (df['date'] < '2022-01-01')]
    
    opt_train = opt_train.dropna(subset=feature_cols + ['label'])
    opt_val = opt_val.dropna(subset=feature_cols + ['label'])
    
    best_params = optuna_stable(
        opt_train[feature_cols], opt_train['label'],
        opt_val[feature_cols], opt_val['label'],
        n_trials=50
    )
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': 42,
        **best_params
    }
    
    # 回测
    result = backtest_with_risk_control(data, params)
    
    if result is not None:
        run_strategies(result)

if __name__ == '__main__':
    main()
