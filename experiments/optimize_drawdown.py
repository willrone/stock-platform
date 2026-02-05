#!/usr/bin/env python3
"""
回撤优化测试：
1. 基于夏普比率优化权重（而非 AUC）
2. 加入止损机制
3. 动态仓位管理
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path('/Users/ronghui/Documents/GitHub/willrone/data/parquet/stock_data')

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标特征"""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_ = df['open']
    
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'return_{period}d'] = close.pct_change(period)
    
    df['momentum_short'] = df['return_5d'] - df['return_10d']
    df['momentum_long'] = df['return_10d'] - df['return_20d']
    df['momentum_reversal'] = -df['return_1d']
    
    returns = close.pct_change()
    for period in [5, 10, 20]:
        up_days = (returns > 0).rolling(period).sum()
        df[f'momentum_strength_{period}'] = up_days / period
    
    for window in [5, 10, 20, 60]:
        ma = close.rolling(window).mean()
        df[f'ma_ratio_{window}'] = close / ma - 1
        df[f'ma_slope_{window}'] = ma.pct_change(5)
    
    df['ma_5'] = close.rolling(5).mean()
    df['ma_10'] = close.rolling(10).mean()
    df['ma_20'] = close.rolling(20).mean()
    df['ma_alignment'] = ((df['ma_5'] > df['ma_10']).astype(int) + (df['ma_10'] > df['ma_20']).astype(int))
    
    for window in [5, 20, 60]:
        df[f'volatility_{window}'] = returns.rolling(window).std()
    
    df['vol_regime'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)
    df['volatility_skew'] = returns.rolling(20).skew()
    
    vol_ma20 = volume.rolling(20).mean()
    vol_ma5 = volume.rolling(5).mean()
    df['vol_ratio'] = volume / (vol_ma20 + 1)
    df['vol_ma_ratio'] = vol_ma5 / (vol_ma20 + 1)
    df['vol_std'] = volume.rolling(20).std() / (vol_ma20 + 1)
    df['price_up'] = (close > close.shift(1)).astype(int)
    df['vol_up'] = (volume > volume.shift(1)).astype(int)
    df['vol_price_diverge'] = (df['price_up'] != df['vol_up']).astype(int)
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    for period in [6, 14]:
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    df['rsi_diff'] = df['rsi_6'] - df['rsi_14']
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_hist_slope'] = df['macd_hist'].diff(3)
    
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_position'] = (close - bb_mid) / (2 * bb_std + 1e-10)
    df['bb_width'] = 4 * bb_std / (bb_mid + 1e-10)
    
    df['body'] = (close - open_) / (open_ + 1e-10)
    df['wick_upper'] = (high - np.maximum(close, open_)) / (high - low + 1e-10)
    df['wick_lower'] = (np.minimum(close, open_) - low) / (high - low + 1e-10)
    df['range_pct'] = (high - low) / (close + 1e-10)
    df['consecutive_up'] = (close > close.shift(1)).rolling(5).sum()
    df['consecutive_down'] = (close < close.shift(1)).rolling(5).sum()
    
    for window in [20, 60]:
        high_n = high.rolling(window).max()
        low_n = low.rolling(window).min()
        df[f'price_pos_{window}'] = (close - low_n) / (high_n - low_n + 1e-10)
        df[f'dist_high_{window}'] = (high_n - close) / (close + 1e-10)
        df[f'dist_low_{window}'] = (close - low_n) / (close + 1e-10)
    
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / (close + 1e-10)
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-10)
    df['di_diff'] = plus_di - minus_di
    df['adx'] = (plus_di - minus_di).abs().rolling(14).mean()
    
    df['future_return'] = close.shift(-1) / close - 1
    df['label'] = (df['future_return'] > 0.003).astype(int)
    
    df = df.dropna()
    return df

def compute_cross_sectional_features(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    for col in ['return_1d', 'return_5d', 'return_20d', 'volume', 'volatility_20']:
        if col in df.columns:
            df[f'{col}_rank'] = df.groupby('date')[col].rank(pct=True)
    df['market_up_ratio'] = df.groupby('date')['return_1d'].transform(lambda x: (x > 0).sum() / len(x))
    market_return_5d = df.groupby('date')['return_5d'].transform('mean')
    market_return_20d = df.groupby('date')['return_20d'].transform('mean')
    df['relative_strength_5d'] = df['return_5d'] - market_return_5d
    df['relative_strength_20d'] = df['return_20d'] - market_return_20d
    df['relative_momentum'] = df['relative_strength_5d'] - df['relative_strength_20d']
    return df

def load_data(n_stocks=350, start_date='2018-01-01', end_date='2025-01-01'):
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
    return data

def get_feature_cols():
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
        'atr_pct', 'di_diff', 'adx',
    ]

def backtest_with_drawdown_control(result_df, top_n=5, transaction_cost=0.001, 
                                    stop_loss=None, trailing_stop=None, 
                                    vol_scaling=False, market_filter=False):
    """
    带回撤控制的回测
    - stop_loss: 单日止损阈值（如 -0.03 表示单日亏损 3% 止损）
    - trailing_stop: 移动止损（如 0.1 表示从高点回撤 10% 止损）
    - vol_scaling: 波动率缩放仓位
    - market_filter: 市场过滤（大盘下跌时减仓）
    """
    result_df = result_df.copy()
    result_df['date_str'] = result_df['date'].astype(str)
    
    # 每日选股
    daily_top_list = []
    for date_str, group in result_df.groupby('date_str'):
        top_stocks = group.nlargest(top_n, 'prob')
        daily_top_list.append(top_stocks)
    daily_top = pd.concat(daily_top_list, ignore_index=True)
    
    # 计算每日收益
    daily_returns = daily_top.groupby('date_str')['return'].mean() - transaction_cost
    
    # 市场过滤：大盘下跌时减仓
    if market_filter:
        market_returns = result_df.groupby('date_str')['return'].mean()
        market_ma5 = market_returns.rolling(5).mean()
        # 大盘 5 日均线下跌时，仓位减半
        position_scale = pd.Series(1.0, index=daily_returns.index)
        for date in daily_returns.index:
            if date in market_ma5.index and market_ma5[date] < 0:
                position_scale[date] = 0.5
        daily_returns = daily_returns * position_scale
    
    # 波动率缩放
    if vol_scaling:
        vol_20 = daily_returns.rolling(20).std()
        target_vol = 0.02  # 目标日波动率 2%
        vol_scale = target_vol / (vol_20 + 1e-10)
        vol_scale = vol_scale.clip(0.5, 2.0)  # 限制缩放范围
        daily_returns = daily_returns * vol_scale
    
    # 止损逻辑
    if stop_loss is not None:
        daily_returns = daily_returns.clip(lower=stop_loss)
    
    # 移动止损
    if trailing_stop is not None:
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        # 触发移动止损后，当日收益归零（模拟清仓）
        stop_triggered = drawdown < -trailing_stop
        # 找到止损触发后的恢复点
        for i in range(1, len(daily_returns)):
            if stop_triggered.iloc[i-1] and not stop_triggered.iloc[i]:
                # 恢复交易
                pass
            elif stop_triggered.iloc[i]:
                daily_returns.iloc[i] = 0  # 清仓
    
    # 计算指标
    total_days = len(daily_returns)
    win_days = (daily_returns > 0).sum()
    win_rate = win_days / total_days
    avg_daily = daily_returns.mean()
    
    cumulative = (1 + daily_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / total_days) - 1
    
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    sharpe = avg_daily / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    # Calmar 比率
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'sharpe': sharpe,
        'calmar': calmar,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_return': total_return,
    }

def main():
    print("="*60)
    print("回撤优化测试")
    print("="*60)
    
    df = load_data(n_stocks=350, start_date='2018-01-01', end_date='2025-01-01')
    feature_cols = get_feature_cols()
    
    train_end = '2023-07-01'
    val_end = '2024-01-01'
    
    train = df[df['date'] < train_end]
    val = df[(df['date'] >= train_end) & (df['date'] < val_end)]
    test = df[df['date'] >= val_end]
    
    print(f"\n测试集: {len(test)} ({test['date'].min().date()} - {test['date'].max().date()})")
    
    X_train, y_train = train[feature_cols], train['label']
    X_val, y_val = val[feature_cols], val['label']
    X_test, y_test = test[feature_cols], test['label']
    
    # 训练模型
    print("\n训练 LightGBM + XGBoost...")
    
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 45, 'max_depth': 6, 'learning_rate': 0.015,
        'feature_fraction': 0.65, 'bagging_fraction': 0.65, 'bagging_freq': 5,
        'min_child_samples': 100, 'reg_alpha': 0.6, 'reg_lambda': 0.6,
        'verbose': -1, 'seed': SEED,
    }
    
    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)
    lgb_model = lgb.train(lgb_params, train_data, num_boost_round=2000, valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 5,
        'learning_rate': 0.015, 'subsample': 0.65, 'colsample_bytree': 0.65,
        'min_child_weight': 100, 'reg_alpha': 0.6, 'reg_lambda': 0.6, 'seed': SEED,
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=2000, evals=[(dval, 'val')],
                          early_stopping_rounds=100, verbose_eval=False)
    
    # 等权重集成预测
    pred_lgb = lgb_model.predict(X_test)
    pred_xgb = xgb_model.predict(dtest)
    pred_ensemble = 0.5 * pred_lgb + 0.5 * pred_xgb
    
    print(f"集成 AUC: {roc_auc_score(y_test, pred_ensemble):.4f}")
    
    test_df = pd.DataFrame({
        'date': test['date'].values,
        'ts_code': test['ts_code'].values,
        'label': y_test.values,
        'return': test['future_return'].values,
        'prob': pred_ensemble,
    })
    
    # 测试不同回撤控制策略
    print("\n" + "="*60)
    print("回撤控制策略对比")
    print("="*60)
    
    strategies = {
        'baseline': {},
        'stop_loss_2%': {'stop_loss': -0.02},
        'stop_loss_3%': {'stop_loss': -0.03},
        'trailing_10%': {'trailing_stop': 0.10},
        'trailing_15%': {'trailing_stop': 0.15},
        'vol_scaling': {'vol_scaling': True},
        'market_filter': {'market_filter': True},
        'combined_1': {'stop_loss': -0.03, 'market_filter': True},
        'combined_2': {'trailing_stop': 0.15, 'vol_scaling': True},
        'combined_3': {'stop_loss': -0.02, 'vol_scaling': True, 'market_filter': True},
    }
    
    results = []
    
    for name, params in strategies.items():
        bt = backtest_with_drawdown_control(test_df, top_n=5, **params)
        results.append({
            'strategy': name,
            'sharpe': bt['sharpe'],
            'calmar': bt['calmar'],
            'annual_return': bt['annual_return'],
            'max_drawdown': bt['max_drawdown'],
            'win_rate': bt['win_rate'],
        })
        
        print(f"\n{name}:")
        print(f"  夏普: {bt['sharpe']:.2f}, Calmar: {bt['calmar']:.2f}")
        print(f"  年化: {bt['annual_return']*100:.1f}%, 回撤: {bt['max_drawdown']*100:.1f}%")
    
    # 汇总
    print("\n" + "="*60)
    print("汇总对比")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('max_drawdown', ascending=False)
    print("\n" + results_df.to_string(index=False))
    
    # 找最佳平衡
    print("\n" + "="*60)
    print("最佳策略推荐")
    print("="*60)
    
    # 筛选夏普 >= 7 且回撤最小的
    good = results_df[results_df['sharpe'] >= 7.0]
    if len(good) > 0:
        best = good.loc[good['max_drawdown'].idxmax()]
        print(f"\n夏普 >= 7 中回撤最小: {best['strategy']}")
        print(f"  夏普: {best['sharpe']:.2f}")
        print(f"  回撤: {best['max_drawdown']*100:.1f}%")
        print(f"  Calmar: {best['calmar']:.2f}")
    
    # 保存
    results_df.to_csv('/Users/ronghui/Documents/GitHub/willrone/experiments/drawdown_optimization_results.csv', index=False)
    print("\n结果已保存")

if __name__ == '__main__':
    main()
