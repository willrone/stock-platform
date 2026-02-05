#!/usr/bin/env python3
"""
LightGBM v7 - 训练/回测股票分离
关键改进：训练集和回测集使用不同的股票，避免数据泄露

股票分配：
- 训练股票：300只（按成交量排名 1-300）
- 回测股票：150只（按成交量排名 301-450）
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
    for period in [1, 3, 5, 10, 20]:
        df[f'return_{period}d'] = close.pct_change(period)
    
    # 动量
    df['momentum_1m'] = close.pct_change(20)
    df['momentum_accel'] = df['momentum_1m'] - df['momentum_1m'].shift(5)
    
    # MA
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
    
    # 成交量
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
    for col in ['return_5d', 'momentum_1m', 'vol_ratio', 'rsi_14']:
        if col in data.columns:
            data[f'{col}_rank'] = data.groupby('date')[col].rank(pct=True)
    
    market_return_5d = data.groupby('date')['return_5d'].transform('mean')
    data['relative_strength'] = data['return_5d'] - market_return_5d
    
    daily_stats = data.groupby('date').agg({
        'return_1d': 'mean',
        'volatility_20': 'mean',
    }).reset_index()
    daily_stats.columns = ['date', 'market_return', 'market_vol']
    daily_stats = daily_stats.sort_values('date')
    
    daily_stats['market_ma10'] = daily_stats['market_return'].rolling(10).mean()
    daily_stats['market_ma30'] = daily_stats['market_return'].rolling(30).mean()
    daily_stats['market_trend'] = (daily_stats['market_ma10'] > daily_stats['market_ma30']).astype(int)
    
    daily_stats['market_vol_ma'] = daily_stats['market_vol'].rolling(20).mean()
    daily_stats['high_vol'] = (daily_stats['market_vol'] > daily_stats['market_vol_ma'] * 1.2).astype(int)
    
    data = data.merge(daily_stats[['date', 'market_trend', 'high_vol', 'market_vol']], on='date', how='left')
    
    daily_mean = data.groupby('date')['future_return'].transform('mean')
    data['excess_return'] = data['future_return'] - daily_mean
    data['label'] = (data['excess_return'] > 0.002).astype(int)
    
    return data

def load_split_data(start_date='2017-01-01', end_date='2025-01-01'):
    """
    加载数据，分离训练股票和回测股票
    
    返回:
        train_stocks_data: 训练用股票数据（300只，成交量排名 1-300）
        test_stocks_data: 回测用股票数据（150只，成交量排名 301-450）
    """
    print(f"加载数据: {start_date} - {end_date}")
    print("="*60)
    
    stock_files = list(DATA_DIR.glob('*.parquet'))
    
    # 筛选有效股票并按成交量排序
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
    
    # 分配股票
    n_train = 300  # 训练股票数量
    n_test = 150   # 回测股票数量
    
    train_files = valid[:n_train]
    test_files = valid[n_train:n_train + n_test]
    
    print(f"训练股票: {len(train_files)} 只 (成交量排名 1-{n_train})")
    print(f"回测股票: {len(test_files)} 只 (成交量排名 {n_train+1}-{n_train+n_test})")
    
    # 加载训练股票数据
    print("\n加载训练股票...")
    train_data = []
    for f, _ in train_files:
        df = pd.read_parquet(f)
        df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
        df = compute_features(df)
        train_data.append(df)
    train_data = pd.concat(train_data, ignore_index=True)
    train_data = add_cross_sectional_features(train_data)
    print(f"训练数据: {len(train_data)} 条")
    
    # 加载回测股票数据
    print("\n加载回测股票...")
    test_data = []
    for f, _ in test_files:
        df = pd.read_parquet(f)
        df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
        df = compute_features(df)
        test_data.append(df)
    test_data = pd.concat(test_data, ignore_index=True)
    test_data = add_cross_sectional_features(test_data)
    print(f"回测数据: {len(test_data)} 条")
    
    return train_data, test_data

def get_feature_cols():
    """特征列"""
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

def optuna_optimize(X_train, y_train, X_val, y_val, n_trials=50):
    """Optuna 超参优化"""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 8, 31),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.7),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.7),
            'bagging_freq': 5,
            'min_child_samples': trial.suggest_int('min_child_samples', 100, 300),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 3.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.01, 0.1),
            'verbose': -1,
            'seed': 42,
        }
        
        train_set = lgb.Dataset(X_train, y_train)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)
        
        model = lgb.train(
            params, train_set,
            num_boost_round=300,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        
        pred = model.predict(X_val)
        return roc_auc_score(y_val, pred)
    
    print(f"\nOptuna 优化 ({n_trials} trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"最佳 AUC: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")
    
    return study.best_params

def rolling_train_and_backtest(train_data, test_data, params):
    """
    滚动训练（用训练股票）+ 回测（用回测股票）
    
    关键：模型在训练股票上学习，在完全不同的回测股票上验证
    """
    feature_cols = get_feature_cols()
    label_col = 'label'
    
    # 清洗数据
    for df in [train_data, test_data]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    required = [c for c in feature_cols if c in train_data.columns] + [label_col, 'future_return', 'excess_return', 'market_trend', 'high_vol']
    train_data = train_data.dropna(subset=required)
    test_data = test_data.dropna(subset=required)
    
    feature_cols = [c for c in feature_cols if c in train_data.columns]
    
    print(f"\n有效训练数据: {len(train_data)} 条")
    print(f"有效回测数据: {len(test_data)} 条")
    print(f"特征数: {len(feature_cols)}")
    
    train_data['date_dt'] = pd.to_datetime(train_data['date'])
    train_data['year_month'] = train_data['date_dt'].dt.to_period('M')
    test_data['date_dt'] = pd.to_datetime(test_data['date'])
    test_data['year_month'] = test_data['date_dt'].dt.to_period('M')
    
    all_months = sorted(train_data['year_month'].unique())
    
    train_months = 36  # 训练窗口
    test_months = 3    # 测试窗口
    
    # 找到 2022-01 开始的位置
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
    print("训练股票 → 模型 → 回测股票（完全分离）")
    print("="*70)
    
    test_idx = start_test_idx
    while test_idx + test_months <= len(all_months):
        train_start = all_months[test_idx - train_months]
        train_end = all_months[test_idx - 1]
        test_start = all_months[test_idx]
        test_end = all_months[min(test_idx + test_months - 1, len(all_months) - 1)]
        
        # 训练数据：从训练股票池
        train_mask = (train_data['year_month'] >= train_start) & (train_data['year_month'] <= train_end)
        train_df = train_data[train_mask]
        
        # 回测数据：从回测股票池（完全不同的股票！）
        backtest_mask = (test_data['year_month'] >= test_start) & (test_data['year_month'] <= test_end)
        backtest_df = test_data[backtest_mask]
        
        if len(train_df) < 1000 or len(backtest_df) < 100:
            test_idx += test_months
            continue
        
        X_train, y_train = train_df[feature_cols], train_df[label_col]
        X_backtest = backtest_df[feature_cols]
        y_backtest = backtest_df[label_col]
        
        train_set = lgb.Dataset(X_train, y_train)
        
        model = lgb.train(
            params, train_set,
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        pred = model.predict(X_backtest)
        
        result = pd.DataFrame({
            'date': backtest_df['date'].values,
            'ts_code': backtest_df['ts_code'].values,
            'prob': pred,
            'label': y_backtest.values,
            'return': backtest_df['future_return'].values,
            'excess_return': backtest_df['excess_return'].values,
            'market_trend': backtest_df['market_trend'].values,
            'high_vol': backtest_df['high_vol'].values,
        })
        
        all_results.append(result)
        
        auc = roc_auc_score(y_backtest, pred)
        print(f"  {test_start} - {test_end}: AUC={auc:.4f} (训练{len(train_df)}条 → 回测{len(backtest_df)}条)")
        
        test_idx += test_months
    
    return pd.concat(all_results, ignore_index=True)

def run_strategies(result):
    """运行多种风控策略"""
    print("\n" + "="*70)
    print("策略回测（训练/回测股票完全分离）")
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
    
    def print_yearly(daily_agg, name):
        daily_agg['year'] = pd.to_datetime(daily_agg['date_str']).dt.year
        for year in sorted(daily_agg['year'].unique()):
            year_data = daily_agg[daily_agg['year'] == year]
            y_ret = year_data['strategy_return']
            y_total = (1 + y_ret).prod() - 1
            y_sharpe = y_ret.mean() / (y_ret.std() + 1e-10) * np.sqrt(252)
            y_dd = (y_ret.cumsum() - y_ret.cumsum().cummax()).min()
            print(f"    {year}: 年收益={y_total*100:.1f}%, 夏普={y_sharpe:.2f}, 回撤={y_dd*100:.1f}%")
    
    # 策略1: 基础择时
    print("\n【策略1】基础择时 (趋势向上时做多)")
    for top_n in [10, 15, 20]:
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
        print_yearly(daily_agg, f"择时 Top {top_n}")
    
    # 策略2: 全风控（择时 + 波动调仓 + 止损）
    print("\n【策略2】全风控 (择时 + 波动调仓 + 动态止损)")
    for top_n in [10, 15, 20]:
        daily_top = result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'prob')
        ).reset_index(drop=True)
        
        daily_agg = daily_top.groupby('date_str').agg({
            'return': 'mean',
            'market_trend': 'first',
            'high_vol': 'first',
        }).reset_index()
        daily_agg = daily_agg.sort_values('date_str').reset_index(drop=True)
        
        # 计算带风控的收益
        strategy_returns = []
        consecutive_loss = 0
        pause_days = 0
        
        for i, row in daily_agg.iterrows():
            if pause_days > 0:
                strategy_returns.append(0)
                pause_days -= 1
                continue
            
            if row['market_trend'] == 0:
                strategy_returns.append(0)
                consecutive_loss = 0
                continue
            
            # 波动率调仓
            position = 0.5 if row['high_vol'] == 1 else 1.0
            ret = row['return'] * position - 2 * COST * position
            strategy_returns.append(ret)
            
            # 止损逻辑
            if ret < 0:
                consecutive_loss += 1
                if consecutive_loss >= 3:
                    pause_days = 1
                    consecutive_loss = 0
            else:
                consecutive_loss = 0
        
        daily_agg['strategy_return'] = strategy_returns
        calc_metrics(daily_agg['strategy_return'], f"全风控 Top {top_n}")
        print_yearly(daily_agg, f"全风控 Top {top_n}")
    
    # 策略3: 纯多头（无择时，作为对照）
    print("\n【策略3】纯多头 (无择时，对照组)")
    for top_n in [10, 15, 20]:
        daily_top = result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'prob')
        ).reset_index(drop=True)
        
        daily_agg = daily_top.groupby('date_str').agg({
            'return': 'mean',
        }).reset_index()
        
        daily_agg['strategy_return'] = daily_agg['return'] - 2 * COST
        
        calc_metrics(daily_agg['strategy_return'], f"纯多头 Top {top_n}")
        print_yearly(daily_agg, f"纯多头 Top {top_n}")

def main():
    print("="*70)
    print("LightGBM v7 - 训练/回测股票完全分离")
    print("="*70)
    print("\n⚠️ 关键改进：训练股票和回测股票完全不同！")
    print("   - 训练股票：成交量排名 1-300")
    print("   - 回测股票：成交量排名 301-450")
    print("   - 这样可以验证模型的真实泛化能力\n")
    
    # 加载分离的数据
    train_data, test_data = load_split_data()
    
    # 准备 Optuna 优化数据（用训练股票的一部分）
    feature_cols = get_feature_cols()
    label_col = 'label'
    
    train_clean = train_data.replace([np.inf, -np.inf], np.nan)
    required = [c for c in feature_cols if c in train_clean.columns] + [label_col]
    train_clean = train_clean.dropna(subset=required)
    feature_cols = [c for c in feature_cols if c in train_clean.columns]
    
    # 用 2019-2021 数据做 Optuna 优化
    train_clean['date_dt'] = pd.to_datetime(train_clean['date'])
    optuna_data = train_clean[(train_clean['date_dt'] >= '2019-01-01') & (train_clean['date_dt'] < '2022-01-01')]
    
    # 分训练/验证
    split_date = '2021-01-01'
    opt_train = optuna_data[optuna_data['date_dt'] < split_date]
    opt_val = optuna_data[optuna_data['date_dt'] >= split_date]
    
    X_train = opt_train[feature_cols]
    y_train = opt_train[label_col]
    X_val = opt_val[feature_cols]
    y_val = opt_val[label_col]
    
    print(f"\nOptuna 数据: 训练 {len(X_train)} 条, 验证 {len(X_val)} 条")
    
    # Optuna 优化
    best_params = optuna_optimize(X_train, y_train, X_val, y_val, n_trials=50)
    
    # 构建完整参数
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': 42,
        **best_params
    }
    
    # 滚动训练 + 回测（股票分离）
    result = rolling_train_and_backtest(train_data, test_data, params)
    
    if result is not None:
        run_strategies(result)
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)

if __name__ == '__main__':
    main()
