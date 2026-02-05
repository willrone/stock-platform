#!/usr/bin/env python3
"""
LightGBM v13: 自适应择时版
- 根据模型置信度动态调整仓位
- 根据市场状态（趋势/波动）决定是否交易
- 加入止损机制
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/Users/ronghui/Documents/GitHub/willrone/data/parquet/stock_data')

print("=" * 70)
print("LightGBM v13: 自适应择时版")
print("=" * 70)

# ============================================================
# 1. 加载和准备数据（复用 v12 的逻辑）
# ============================================================
print("\n[1/5] 加载数据...")

all_files = list(DATA_DIR.glob('*.parquet'))
dfs = []
for f in all_files:
    df = pd.read_parquet(f)
    df['code'] = f.stem
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(['code', 'date']).reset_index(drop=True)

# 只保留 2023-2024
data = data[(data['date'] >= '2023-01-01') & (data['date'] <= '2024-12-31')]

# 活跃股票
stock_counts = data.groupby('code').size()
active_stocks = stock_counts[stock_counts >= 400].index.tolist()
data = data[data['code'].isin(active_stocks)]

print(f"  股票数: {len(active_stocks)}, 数据量: {len(data):,}")

# ============================================================
# 2. 计算因子
# ============================================================
print("\n[2/5] 计算因子...")

def calc_factors(df):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    features = {}
    
    for d in [1, 5, 10, 20]:
        features[f'return_{d}d'] = pd.Series(close).pct_change(d).values
    
    for d in [5, 10, 20]:
        features[f'volatility_{d}d'] = pd.Series(close).pct_change().rolling(d).std().values
    
    for d in [5, 10, 20, 60]:
        ma = pd.Series(close).rolling(d).mean().values
        features[f'ma_bias_{d}d'] = (close - ma) / (ma + 1e-10)
    
    for d in [6, 12, 24]:
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(d).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(d).mean()
        features[f'rsi_{d}'] = (gain / (gain + loss + 1e-10)).values
    
    ema12 = pd.Series(close).ewm(span=12).mean()
    ema26 = pd.Series(close).ewm(span=26).mean()
    features['macd'] = (ema12 - ema26).values
    features['macd_signal'] = (ema12 - ema26).ewm(span=9).mean().values
    
    ma20 = pd.Series(close).rolling(20).mean()
    std20 = pd.Series(close).rolling(20).std()
    features['bb_position'] = ((close - ma20) / (2 * std20 + 1e-10)).values
    
    for d in [5, 10, 20]:
        vol_ma = pd.Series(volume).rolling(d).mean().values
        features[f'vol_ratio_{d}d'] = volume / (vol_ma + 1e-10)
    
    for d in [5, 10, 20]:
        features[f'momentum_{d}d'] = (close / (np.roll(close, d) + 1e-10) - 1)
    
    for d in [10, 20]:
        roll_high = pd.Series(high).rolling(d).max().values
        roll_low = pd.Series(low).rolling(d).min().values
        features[f'high_pos_{d}d'] = (close - roll_low) / (roll_high - roll_low + 1e-10)
    
    return pd.DataFrame(features)

factor_dfs = []
for code in active_stocks:
    stock_data = data[data['code'] == code].copy()
    factors = calc_factors(stock_data)
    factors['code'] = code
    factors['date'] = stock_data['date'].values
    factors['close'] = stock_data['close'].values
    factor_dfs.append(factors)

factor_data = pd.concat(factor_dfs, ignore_index=True)
feature_cols = [c for c in factor_data.columns if c not in ['code', 'date', 'close']]

# 标签
factor_data = factor_data.sort_values(['code', 'date'])
factor_data['future_return'] = factor_data.groupby('code')['close'].transform(lambda x: x.shift(-5) / x - 1)
factor_data['label'] = factor_data.groupby('date')['future_return'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-10))

for col in feature_cols:
    factor_data[col] = factor_data.groupby('date')[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-10))

factor_data = factor_data.dropna(subset=feature_cols + ['label'])

print(f"  因子数: {len(feature_cols)}")

# ============================================================
# 3. 计算市场状态指标
# ============================================================
print("\n[3/5] 计算市场状态...")

# 市场整体指标（用所有股票的平均）
market_data = factor_data.groupby('date').agg({
    'future_return': 'mean',
    'close': 'mean'
}).reset_index()

market_data['market_return_5d'] = market_data['future_return'].rolling(5).mean()
market_data['market_return_20d'] = market_data['future_return'].rolling(20).mean()
market_data['market_volatility'] = market_data['future_return'].rolling(20).std()

# 市场趋势：20日均线 vs 60日均线
market_data['ma20'] = market_data['close'].rolling(20).mean()
market_data['ma60'] = market_data['close'].rolling(60).mean()
market_data['trend'] = (market_data['ma20'] > market_data['ma60']).astype(int)

# 市场波动状态：高波动 = 波动率 > 中位数
vol_median = market_data['market_volatility'].median()
market_data['high_vol'] = (market_data['market_volatility'] > vol_median).astype(int)

market_state = market_data.set_index('date')[['trend', 'high_vol', 'market_volatility']].to_dict('index')

print(f"  趋势向上天数: {market_data['trend'].sum()}")
print(f"  高波动天数: {market_data['high_vol'].sum()}")

# ============================================================
# 4. 模型参数
# ============================================================
params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 100,
    'lambda_l2': 200,
    'min_data_in_leaf': 100,
    'verbose': -1,
    'seed': 42,
}

# 交易成本
TOTAL_COST = 0.0036  # 0.36%

# ============================================================
# 5. 滚动回测（带自适应择时）
# ============================================================
print("\n[4/5] 滚动回测...")

months = pd.date_range('2023-07-01', '2024-12-01', freq='MS')
all_preds = []

for m_start in months:
    m_end = m_start + pd.DateOffset(months=1)
    train_start = m_start - pd.DateOffset(months=12)
    
    train_mask = (factor_data['date'] >= train_start) & (factor_data['date'] < m_start)
    test_mask = (factor_data['date'] >= m_start) & (factor_data['date'] < m_end)
    
    if train_mask.sum() < 5000 or test_mask.sum() == 0:
        continue
    
    X_tr = factor_data.loc[train_mask, feature_cols].values
    y_tr = factor_data.loc[train_mask, 'label'].values
    X_te = factor_data.loc[test_mask, feature_cols].values
    
    train_set = lgb.Dataset(X_tr, label=y_tr)
    model = lgb.train(params, train_set, num_boost_round=200, callbacks=[lgb.log_evaluation(0)])
    
    pred = model.predict(X_te)
    
    m_data = factor_data.loc[test_mask, ['date', 'code', 'future_return', 'label']].copy()
    m_data['pred'] = pred
    all_preds.append(m_data)

rolling_data = pd.concat(all_preds, ignore_index=True)

# IC
rolling_ic = rolling_data.groupby('date').apply(lambda x: x['pred'].corr(x['label'])).mean()
print(f"  滚动 IC: {rolling_ic:.4f}")

# ============================================================
# 回测函数
# ============================================================
def backtest_adaptive(df, top_n, market_state, strategy='adaptive'):
    """
    自适应回测
    strategy:
    - 'always': 始终满仓
    - 'trend': 趋势向上时满仓，否则空仓
    - 'adaptive': 根据趋势和波动调整仓位
    - 'confidence': 根据预测置信度调整仓位
    """
    dates = sorted(df['date'].unique())
    daily_returns = []
    prev_holdings = set()
    consecutive_loss = 0
    
    for date in dates:
        state = market_state.get(date, {'trend': 1, 'high_vol': 0})
        
        # 决定仓位
        if strategy == 'always':
            position = 1.0
        elif strategy == 'trend':
            position = 1.0 if state['trend'] == 1 else 0.0
        elif strategy == 'adaptive':
            # 趋势向上 + 低波动 = 满仓
            # 趋势向上 + 高波动 = 半仓
            # 趋势向下 = 空仓
            if state['trend'] == 0:
                position = 0.0
            elif state['high_vol'] == 1:
                position = 0.5
            else:
                position = 1.0
        elif strategy == 'confidence':
            # 根据当日预测分散度调整
            day_data = df[df['date'] == date]
            pred_std = day_data['pred'].std()
            # 预测越分散，置信度越高
            position = min(1.0, pred_std / 1.5)
        else:
            position = 1.0
        
        # 止损：连续亏损 5 天，暂停 3 天
        if consecutive_loss >= 5:
            position = 0.0
            consecutive_loss -= 1  # 逐步恢复
        
        if position == 0:
            daily_returns.append({
                'date': date,
                'net_return': 0,
                'position': 0,
                'year': date.year
            })
            continue
        
        day_data = df[df['date'] == date]
        if len(day_data) < top_n:
            continue
        
        # 选 Top N
        top_stocks = day_data.nlargest(top_n, 'pred')['code'].tolist()
        current_holdings = set(top_stocks)
        
        # 换手
        if prev_holdings:
            new_stocks = current_holdings - prev_holdings
            sold_stocks = prev_holdings - current_holdings
            turnover = (len(new_stocks) + len(sold_stocks)) / (2 * top_n)
        else:
            turnover = 1.0
        
        # 收益
        day_return = day_data[day_data['code'].isin(top_stocks)]['future_return'].mean()
        if pd.isna(day_return):
            day_return = 0
        
        # 按仓位调整收益
        day_return *= position
        
        # 扣除成本
        cost = turnover * TOTAL_COST * position
        net_return = day_return - cost
        
        # 更新连续亏损
        if net_return < 0:
            consecutive_loss += 1
        else:
            consecutive_loss = 0
        
        daily_returns.append({
            'date': date,
            'net_return': net_return,
            'position': position,
            'year': date.year
        })
        
        prev_holdings = current_holdings
    
    return pd.DataFrame(daily_returns)

def calc_metrics(result_df):
    if len(result_df) == 0:
        return {}
    
    returns = result_df['net_return']
    cum_return = (1 + returns).prod() - 1
    sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
    
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    max_dd = ((cum_returns - rolling_max) / rolling_max).min()
    
    return {
        'cum_return': cum_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'avg_position': result_df['position'].mean()
    }

# ============================================================
# 对比不同策略
# ============================================================
print("\n[5/5] 策略对比...")

strategies = ['always', 'trend', 'adaptive']
top_n = 20

print(f"\n  Top {top_n} 不同策略对比:")
print("-" * 70)
print(f"  {'策略':<12} {'累计收益':>10} {'夏普':>8} {'回撤':>8} {'平均仓位':>10} {'2023':>10} {'2024':>10}")
print("-" * 70)

for strategy in strategies:
    result = backtest_adaptive(rolling_data, top_n, market_state, strategy)
    metrics = calc_metrics(result)
    
    if not metrics:
        continue
    
    # 分年
    y2023 = result[result['year'] == 2023]
    y2024 = result[result['year'] == 2024]
    
    ret_2023 = (1 + y2023['net_return']).prod() - 1 if len(y2023) > 0 else 0
    ret_2024 = (1 + y2024['net_return']).prod() - 1 if len(y2024) > 0 else 0
    
    print(f"  {strategy:<12} {metrics['cum_return']*100:>+9.1f}% {metrics['sharpe']:>8.2f} "
          f"{metrics['max_dd']*100:>7.1f}% {metrics['avg_position']*100:>9.1f}% "
          f"{ret_2023*100:>+9.1f}% {ret_2024*100:>+9.1f}%")

print("-" * 70)

# 最佳策略详细结果
print("\n  最佳策略（adaptive）详细结果:")
print()

for top_n in [10, 20, 30, 50]:
    result = backtest_adaptive(rolling_data, top_n, market_state, 'adaptive')
    metrics = calc_metrics(result)
    
    if not metrics:
        continue
    
    y2023 = result[result['year'] == 2023]
    y2024 = result[result['year'] == 2024]
    
    ret_2023 = (1 + y2023['net_return']).prod() - 1 if len(y2023) > 0 else 0
    ret_2024 = (1 + y2024['net_return']).prod() - 1 if len(y2024) > 0 else 0
    
    sharpe_2023 = y2023['net_return'].mean() / (y2023['net_return'].std() + 1e-10) * np.sqrt(252) if len(y2023) > 0 else 0
    sharpe_2024 = y2024['net_return'].mean() / (y2024['net_return'].std() + 1e-10) * np.sqrt(252) if len(y2024) > 0 else 0
    
    print(f"  Top {top_n}:")
    print(f"    整体: 收益={metrics['cum_return']*100:+.1f}%, 夏普={metrics['sharpe']:.2f}, 回撤={metrics['max_dd']*100:.1f}%")
    print(f"    2023: 收益={ret_2023*100:+.1f}%, 夏普={sharpe_2023:.2f}")
    print(f"    2024: 收益={ret_2024*100:+.1f}%, 夏普={sharpe_2024:.2f}")
    print()

# 基准
print("  基准对比:")
benchmark = rolling_data.groupby('date')['future_return'].mean()
bench_cum = (1 + benchmark).prod() - 1
bench_sharpe = benchmark.mean() / (benchmark.std() + 1e-10) * np.sqrt(252)
print(f"    等权基准: 收益={bench_cum*100:+.1f}%, 夏普={bench_sharpe:.2f}")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
