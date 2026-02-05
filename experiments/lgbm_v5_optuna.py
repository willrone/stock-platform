#!/usr/bin/env python3
"""
LightGBM v5 + Optuna 优化参数
- 使用子代理优化的参数
- 3年滚动训练回测（2022-2024）
- 多策略对比
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import optuna
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/Users/ronghui/Documents/GitHub/willrone/data/parquet/stock_data')
np.random.seed(42)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标特征"""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_ = df['open']
    
    # 收益率
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'return_{period}d'] = close.pct_change(period)
    
    # 动量
    df['momentum_1m'] = close.pct_change(20)
    df['momentum_3m'] = close.pct_change(60)
    df['momentum_accel'] = df['momentum_1m'] - df['momentum_1m'].shift(5)
    df['reversal_5d'] = -df['return_5d']
    
    # 动量强度
    returns = close.pct_change()
    for period in [5, 10, 20]:
        up_days = (returns > 0).rolling(period).sum()
        df[f'momentum_strength_{period}'] = up_days / period
    
    # MA
    for window in [5, 10, 20, 60]:
        ma = close.rolling(window).mean()
        df[f'ma_ratio_{window}'] = close / ma - 1
        df[f'ma_slope_{window}'] = ma.pct_change(5)
    
    df['ma_5'] = close.rolling(5).mean()
    df['ma_10'] = close.rolling(10).mean()
    df['ma_20'] = close.rolling(20).mean()
    df['ma_60'] = close.rolling(60).mean()
    df['ma_score'] = (
        (df['ma_5'] > df['ma_10']).astype(float) +
        (df['ma_10'] > df['ma_20']).astype(float) +
        (df['ma_20'] > df['ma_60']).astype(float)
    ) / 3
    
    # 成交量
    vol_ma5 = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()
    df['vol_ratio'] = volume / (vol_ma20 + 1)
    df['vol_trend'] = vol_ma5 / (vol_ma20 + 1)
    df['vol_std'] = volume.rolling(20).std() / (vol_ma20 + 1)
    
    price_change = close.pct_change()
    vol_change = volume.pct_change()
    df['vol_price_corr'] = price_change.rolling(10).corr(vol_change)
    
    # 量价背离
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
    
    df['rsi_deviation'] = df['rsi_14'] - 50
    df['rsi_diff'] = df['rsi_6'] - df['rsi_14']
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = (ema12 - ema26) / close
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_hist_slope'] = df['macd_hist'].diff(3)
    
    # 布林带
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_position'] = (close - bb_mid) / (2 * bb_std + 1e-10)
    df['bb_width'] = 4 * bb_std / (bb_mid + 1e-10)
    
    # 波动率
    df['volatility_5'] = close.pct_change().rolling(5).std()
    df['volatility_20'] = close.pct_change().rolling(20).std()
    df['volatility_60'] = close.pct_change().rolling(60).std()
    df['vol_regime'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)
    df['volatility_skew'] = returns.rolling(20).skew()
    
    # 形态
    df['body'] = (close - open_) / (open_ + 1e-10)
    df['wick_ratio'] = (high - low) / (close + 1e-10)
    df['wick_upper'] = (high - np.maximum(close, open_)) / (high - low + 1e-10)
    df['wick_lower'] = (np.minimum(close, open_) - low) / (high - low + 1e-10)
    df['up_streak'] = (close > close.shift(1)).rolling(5).sum()
    df['down_streak'] = (close < close.shift(1)).rolling(5).sum()
    
    # 价格位置
    for window in [20, 60]:
        high_n = high.rolling(window).max()
        low_n = low.rolling(window).min()
        df[f'price_pos_{window}'] = (close - low_n) / (high_n - low_n + 1e-10)
    
    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr_pct'] = tr.rolling(14).mean() / (close + 1e-10)
    
    # ADX
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-10)
    df['di_diff'] = (plus_di - minus_di) / 100
    
    df['future_return'] = close.shift(-1) / close - 1
    
    return df

def add_cross_sectional_features(data: pd.DataFrame) -> pd.DataFrame:
    """添加截面特征"""
    # 排名特征
    rank_cols = ['return_1d', 'return_5d', 'return_20d', 'momentum_1m', 'vol_ratio', 'volatility_20', 'rsi_14']
    for col in rank_cols:
        if col in data.columns:
            data[f'{col}_rank'] = data.groupby('date')[col].rank(pct=True)
    
    # 市场情绪
    data['market_up_ratio'] = data.groupby('date')['return_1d'].transform(
        lambda x: (x > 0).sum() / len(x)
    )
    
    # 相对强度（重点特征）
    market_return_5d = data.groupby('date')['return_5d'].transform('mean')
    market_return_20d = data.groupby('date')['return_20d'].transform('mean')
    data['relative_strength_5d'] = data['return_5d'] - market_return_5d
    data['relative_strength_20d'] = data['return_20d'] - market_return_20d
    data['relative_momentum'] = data['relative_strength_5d'] - data['relative_strength_20d']
    
    # 市场状态
    daily_stats = data.groupby('date').agg({
        'return_1d': 'mean',
        'volatility_20': 'mean',
    }).reset_index()
    daily_stats.columns = ['date', 'market_return_1d', 'market_vol']
    daily_stats = daily_stats.sort_values('date')
    daily_stats['market_ma5'] = daily_stats['market_return_1d'].rolling(5).mean()
    daily_stats['market_ma20'] = daily_stats['market_return_1d'].rolling(20).mean()
    daily_stats['market_trend'] = (daily_stats['market_ma5'] > daily_stats['market_ma20']).astype(int)
    daily_stats['market_momentum'] = daily_stats['market_return_1d'].rolling(20).sum()
    
    data = data.merge(daily_stats[['date', 'market_trend', 'market_momentum']], on='date', how='left')
    
    # 标签
    daily_mean = data.groupby('date')['future_return'].transform('mean')
    data['market_return'] = daily_mean
    data['excess_return'] = data['future_return'] - daily_mean
    
    # 标签：超额收益 > 0.3%（Optuna 版本的阈值）
    data['label_relative'] = (data['excess_return'] > 0.003).astype(int)
    
    return data

def load_data(n_stocks=450, start_date='2018-01-01', end_date='2025-01-01'):
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
    base = [
        'return_1d', 'return_2d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
        'momentum_1m', 'momentum_3m', 'momentum_accel', 'reversal_5d',
        'momentum_strength_5', 'momentum_strength_10', 'momentum_strength_20',
        'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_60', 'ma_score',
        'ma_slope_5', 'ma_slope_10', 'ma_slope_20',
        'vol_ratio', 'vol_trend', 'vol_std', 'vol_price_corr', 'vol_price_diverge',
        'rsi_6', 'rsi_14', 'rsi_deviation', 'rsi_diff',
        'macd', 'macd_signal', 'macd_hist', 'macd_hist_slope',
        'bb_position', 'bb_width',
        'volatility_5', 'volatility_20', 'volatility_60', 'vol_regime', 'volatility_skew',
        'body', 'wick_ratio', 'wick_upper', 'wick_lower', 'up_streak', 'down_streak',
        'price_pos_20', 'price_pos_60',
        'atr_pct', 'di_diff',
        'market_trend', 'market_momentum', 'market_up_ratio',
        'relative_strength_5d', 'relative_strength_20d', 'relative_momentum',
    ]
    rank = ['return_1d_rank', 'return_5d_rank', 'return_20d_rank', 'momentum_1m_rank',
            'vol_ratio_rank', 'volatility_20_rank', 'rsi_14_rank']
    return base + rank

def optuna_optimize(X_train, y_train, X_val, y_val, n_trials=50):
    """Optuna 超参优化"""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.8),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.8),
            'bagging_freq': 5,
            'min_child_samples': trial.suggest_int('min_child_samples', 50, 200),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2.0),
            'verbose': -1,
            'seed': 42,
        }
        
        train_data = lgb.Dataset(X_train, y_train)
        val_data = lgb.Dataset(X_val, y_val, reference=train_data)
        
        model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        
        pred = model.predict(X_val)
        return roc_auc_score(y_val, pred)
    
    print(f"\nOptuna 超参搜索 ({n_trials} trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"最佳 AUC: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")
    
    return study.best_params

def rolling_train_backtest(data, use_optuna=True):
    """滚动训练回测"""
    feature_cols = get_feature_cols()
    label_col = 'label_relative'
    
    df = data.replace([np.inf, -np.inf], np.nan)
    required = [c for c in feature_cols if c in df.columns] + [label_col, 'future_return', 'excess_return', 'market_trend']
    df = df.dropna(subset=required)
    df = df.sort_values('date').reset_index(drop=True)
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"有效数据: {len(df)} 条")
    print(f"特征数: {len(feature_cols)}")
    print(f"正样本比例: {df[label_col].mean():.4f}")
    
    df['date_dt'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date_dt'].dt.to_period('M')
    
    all_months = sorted(df['year_month'].unique())
    
    # 先用 2020-2021 数据做 Optuna 优化
    if use_optuna:
        opt_train = df[(df['date_dt'] >= '2020-01-01') & (df['date_dt'] < '2021-07-01')]
        opt_val = df[(df['date_dt'] >= '2021-07-01') & (df['date_dt'] < '2022-01-01')]
        
        if len(opt_train) > 1000 and len(opt_val) > 500:
            best_params = optuna_optimize(
                opt_train[feature_cols], opt_train[label_col],
                opt_val[feature_cols], opt_val[label_col],
                n_trials=50
            )
        else:
            best_params = None
    else:
        best_params = None
    
    # 最终参数
    if best_params:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'seed': 42,
            **best_params
        }
    else:
        # 默认参数（来自子代理优化结果）
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 7,
            'learning_rate': 0.015,
            'feature_fraction': 0.65,
            'bagging_fraction': 0.65,
            'bagging_freq': 5,
            'min_child_samples': 168,
            'reg_alpha': 0.918,
            'reg_lambda': 1.2,
            'verbose': -1,
            'seed': 42,
        }
    
    print(f"\n使用参数: {params}")
    
    # 滚动训练
    train_months = 24
    test_months = 3
    
    start_test_idx = None
    for i, m in enumerate(all_months):
        if m >= pd.Period('2022-01'):
            start_test_idx = i
            break
    
    all_results = []
    
    print(f"\n滚动训练回测 (训练{train_months}月, 测试{test_months}月)")
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
            num_boost_round=500,
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
        })
        
        all_results.append(result)
        
        auc = roc_auc_score(y_test, pred)
        print(f"  {test_start} - {test_end}: AUC={auc:.4f}, 样本={len(test_df)}")
        
        test_idx += test_months
    
    full_result = pd.concat(all_results, ignore_index=True)
    
    # 回测分析
    print("\n" + "="*70)
    print("回测结果")
    print("="*70)
    
    COST = 0.001
    
    full_result['date_str'] = full_result['date'].astype(str)
    full_result['date_dt'] = pd.to_datetime(full_result['date'])
    full_result['year'] = full_result['date_dt'].dt.year
    
    # 十分位分析
    full_result['decile'] = pd.qcut(full_result['prob'], 10, labels=False, duplicates='drop')
    
    print("\n十分位分析:")
    print(f"{'分位':>6} {'样本数':>10} {'标签命中':>10} {'绝对收益':>12} {'超额收益':>12}")
    print("-" * 55)
    
    for d in sorted(full_result['decile'].unique()):
        g = full_result[full_result['decile'] == d]
        print(f"{d:>6} {len(g):>10} {g['label'].mean():>10.2%} "
              f"{g['return'].mean()*100:>11.3f}% {g['excess_return'].mean()*100:>11.3f}%")
    
    # 策略回测
    print("\n" + "="*70)
    print("策略回测（扣除 0.2% 交易成本）")
    print("="*70)
    
    for top_n in [10, 20, 30, 50]:
        print(f"\n【Top {top_n}】")
        
        daily_top = full_result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'prob')
        ).reset_index(drop=True)
        
        daily_returns = daily_top.groupby('date_str')['return'].mean() - 2 * COST
        
        total_days = len(daily_returns)
        win_rate = (daily_returns > 0).mean()
        avg_daily = daily_returns.mean()
        total_return = (1 + daily_returns).prod() - 1
        sharpe = avg_daily / (daily_returns.std() + 1e-10) * np.sqrt(252)
        cumsum = daily_returns.cumsum()
        max_dd = (cumsum - cumsum.cummax()).min()
        
        print(f"  整体: 天数={total_days}, 胜率={win_rate:.1%}, 日均={avg_daily*100:.2f}%, "
              f"累计={total_return*100:.1f}%, 夏普={sharpe:.2f}, 回撤={max_dd*100:.1f}%")
        
        daily_top['year'] = pd.to_datetime(daily_top['date']).dt.year
        for year in sorted(daily_top['year'].unique()):
            year_data = daily_top[daily_top['year'] == year]
            year_returns = year_data.groupby(year_data['date'].astype(str))['return'].mean() - 2 * COST
            if len(year_returns) > 0:
                y_total = (1 + year_returns).prod() - 1
                y_sharpe = year_returns.mean() / (year_returns.std() + 1e-10) * np.sqrt(252)
                y_dd = (year_returns.cumsum() - year_returns.cumsum().cummax()).min()
                print(f"    {year}: 年收益={y_total*100:.1f}%, 夏普={y_sharpe:.2f}, 回撤={y_dd*100:.1f}%")
    
    # 择时策略
    print("\n【择时策略】(趋势向上时做多)")
    for top_n in [20, 30]:
        daily_top = full_result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'prob')
        ).reset_index(drop=True)
        
        daily_agg = daily_top.groupby('date_str').agg({
            'return': 'mean',
            'market_trend': 'first'
        }).reset_index()
        
        daily_agg['strategy_return'] = daily_agg.apply(
            lambda x: x['return'] - 2 * COST if x['market_trend'] == 1 else 0,
            axis=1
        )
        
        daily_returns = daily_agg['strategy_return']
        
        total_days = len(daily_returns)
        active_days = (daily_agg['market_trend'] == 1).sum()
        win_rate = (daily_returns[daily_returns != 0] > 0).mean() if (daily_returns != 0).any() else 0
        avg_daily = daily_returns.mean()
        total_return = (1 + daily_returns).prod() - 1
        sharpe = avg_daily / (daily_returns.std() + 1e-10) * np.sqrt(252)
        cumsum = daily_returns.cumsum()
        max_dd = (cumsum - cumsum.cummax()).min()
        
        print(f"\n  择时 Top {top_n}: 总天数={total_days}, 交易天数={active_days}, "
              f"胜率={win_rate:.1%}, 日均={avg_daily*100:.2f}%, "
              f"累计={total_return*100:.1f}%, 夏普={sharpe:.2f}, 回撤={max_dd*100:.1f}%")
        
        daily_agg['year'] = pd.to_datetime(daily_agg['date_str']).dt.year
        for year in sorted(daily_agg['year'].unique()):
            year_data = daily_agg[daily_agg['year'] == year]
            year_returns = year_data['strategy_return']
            if len(year_returns) > 0:
                y_total = (1 + year_returns).prod() - 1
                y_sharpe = year_returns.mean() / (year_returns.std() + 1e-10) * np.sqrt(252)
                y_active = (year_data['market_trend'] == 1).sum()
                print(f"    {year}: 交易天数={y_active}, 年收益={y_total*100:.1f}%, 夏普={y_sharpe:.2f}")
    
    return full_result

if __name__ == '__main__':
    data = load_data(n_stocks=450, start_date='2018-01-01', end_date='2025-01-01')
    result = rolling_train_backtest(data, use_optuna=True)
