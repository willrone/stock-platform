#!/usr/bin/env python3
"""
LightGBM v12: 严格验证版
- 只用 2023-2024 数据（数据完整）
- 训练: 2023-01 ~ 2024-06
- 验证: 2024-07 ~ 2024-09
- 测试: 2024-10 ~ 2024-12（完全样本外）
- 加入交易成本
- 多种策略对比
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/Users/ronghui/Projects/willrone/data/parquet/stock_data')

print("=" * 70)
print("LightGBM v12: 严格验证版")
print("=" * 70)

# ============================================================
# 1. 加载数据（只用 2023-2024）
# ============================================================
print("\n[1/7] 加载数据...")

all_files = list(DATA_DIR.glob('*.parquet'))
dfs = []
for f in all_files:
    df = pd.read_parquet(f)
    df['code'] = f.stem
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(['code', 'date']).reset_index(drop=True)

# 只保留 2023-2024 数据
data = data[(data['date'] >= '2023-01-01') & (data['date'] <= '2024-12-31')]

# 筛选活跃股票（在整个期间都有数据的）
stock_counts = data.groupby('code').size()
active_stocks = stock_counts[stock_counts >= 400].index.tolist()  # 至少 400 天数据
data = data[data['code'].isin(active_stocks)]

print(f"  时间范围: {data['date'].min().date()} ~ {data['date'].max().date()}")
print(f"  股票数: {len(active_stocks)}")
print(f"  数据量: {len(data):,}")

# ============================================================
# 2. 计算因子
# ============================================================
print("\n[2/7] 计算因子...")

def calc_factors(df):
    """计算技术因子"""
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    features = {}
    
    # 收益率
    for d in [1, 5, 10, 20]:
        features[f'return_{d}d'] = pd.Series(close).pct_change(d).values
    
    # 波动率
    for d in [5, 10, 20]:
        features[f'volatility_{d}d'] = pd.Series(close).pct_change().rolling(d).std().values
    
    # 均线偏离
    for d in [5, 10, 20, 60]:
        ma = pd.Series(close).rolling(d).mean().values
        features[f'ma_bias_{d}d'] = (close - ma) / (ma + 1e-10)
    
    # RSI
    for d in [6, 12, 24]:
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(d).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(d).mean()
        features[f'rsi_{d}'] = (gain / (gain + loss + 1e-10)).values
    
    # MACD
    ema12 = pd.Series(close).ewm(span=12).mean()
    ema26 = pd.Series(close).ewm(span=26).mean()
    features['macd'] = (ema12 - ema26).values
    features['macd_signal'] = (ema12 - ema26).ewm(span=9).mean().values
    
    # 布林带
    ma20 = pd.Series(close).rolling(20).mean()
    std20 = pd.Series(close).rolling(20).std()
    features['bb_position'] = ((close - ma20) / (2 * std20 + 1e-10)).values
    
    # 成交量
    for d in [5, 10, 20]:
        vol_ma = pd.Series(volume).rolling(d).mean().values
        features[f'vol_ratio_{d}d'] = volume / (vol_ma + 1e-10)
    
    # 动量
    for d in [5, 10, 20]:
        features[f'momentum_{d}d'] = (close / (np.roll(close, d) + 1e-10) - 1)
    
    # 高低点位置
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
print(f"  因子数: {len(feature_cols)}")

# ============================================================
# 3. 准备标签
# ============================================================
print("\n[3/7] 准备标签...")

factor_data = factor_data.sort_values(['code', 'date'])

# 未来 5 日收益
factor_data['future_return'] = factor_data.groupby('code')['close'].transform(
    lambda x: x.shift(-5) / x - 1
)

# 截面标准化
factor_data['label'] = factor_data.groupby('date')['future_return'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-10)
)

# 特征标准化
for col in feature_cols:
    factor_data[col] = factor_data.groupby('date')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )

# 删除缺失值
factor_data = factor_data.dropna(subset=feature_cols + ['label'])

# 划分数据集
train_data = factor_data[(factor_data['date'] >= '2023-01-01') & (factor_data['date'] < '2024-07-01')]
valid_data = factor_data[(factor_data['date'] >= '2024-07-01') & (factor_data['date'] < '2024-10-01')]
test_data = factor_data[factor_data['date'] >= '2024-10-01']

print(f"  训练集: {len(train_data):,} ({train_data['date'].min().date()} ~ {train_data['date'].max().date()})")
print(f"  验证集: {len(valid_data):,} ({valid_data['date'].min().date()} ~ {valid_data['date'].max().date()})")
print(f"  测试集: {len(test_data):,} ({test_data['date'].min().date()} ~ {test_data['date'].max().date()})")

# ============================================================
# 4. 训练模型
# ============================================================
print("\n[4/7] 训练模型...")

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

X_train = train_data[feature_cols].values
y_train = train_data['label'].values
X_valid = valid_data[feature_cols].values
y_valid = valid_data['label'].values
X_test = test_data[feature_cols].values

train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

model = lgb.train(
    params,
    train_set,
    num_boost_round=500,
    valid_sets=[valid_set],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

print(f"  最佳迭代: {model.best_iteration}")

# ============================================================
# 5. 评估 IC
# ============================================================
print("\n[5/7] 评估 IC...")

# 验证集 IC
valid_data = valid_data.copy()
valid_data['pred'] = model.predict(X_valid)
valid_ic = valid_data.groupby('date').apply(lambda x: x['pred'].corr(x['label'])).mean()

# 测试集 IC
test_data = test_data.copy()
test_data['pred'] = model.predict(X_test)
test_ic = test_data.groupby('date').apply(lambda x: x['pred'].corr(x['label'])).mean()

print(f"  验证集 IC: {valid_ic:.4f}")
print(f"  测试集 IC: {test_ic:.4f}")

# ============================================================
# 6. 回测（带交易成本）
# ============================================================
print("\n[6/7] 回测（测试集 2024-10 ~ 2024-12）...")

# 交易成本
COMMISSION = 0.0003  # 手续费 0.03%
SLIPPAGE = 0.001     # 滑点 0.1%
STAMP_TAX = 0.001    # 印花税 0.1%
TOTAL_COST = COMMISSION * 2 + SLIPPAGE * 2 + STAMP_TAX  # ~0.36%

def backtest(df, top_n, use_cost=True):
    """回测"""
    dates = sorted(df['date'].unique())
    daily_returns = []
    prev_holdings = set()
    
    for date in dates:
        day_data = df[df['date'] == date]
        if len(day_data) < top_n:
            continue
        
        # 选 Top N
        top_stocks = day_data.nlargest(top_n, 'pred')['code'].tolist()
        current_holdings = set(top_stocks)
        
        # 换手率
        if prev_holdings:
            new_stocks = current_holdings - prev_holdings
            sold_stocks = prev_holdings - current_holdings
            turnover = (len(new_stocks) + len(sold_stocks)) / (2 * top_n)
        else:
            turnover = 1.0  # 第一天全买入
        
        # 收益
        day_return = day_data[day_data['code'].isin(top_stocks)]['future_return'].mean()
        if pd.isna(day_return):
            day_return = 0
        
        # 扣除成本
        cost = turnover * TOTAL_COST if use_cost else 0
        net_return = day_return - cost
        
        daily_returns.append({
            'date': date,
            'gross_return': day_return,
            'net_return': net_return,
            'turnover': turnover
        })
        
        prev_holdings = current_holdings
    
    return pd.DataFrame(daily_returns)

def calc_metrics(result_df, return_col='net_return'):
    """计算指标"""
    if len(result_df) == 0:
        return {}
    
    returns = result_df[return_col]
    cum_return = (1 + returns).prod() - 1
    sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
    
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    max_dd = ((cum_returns - rolling_max) / rolling_max).min()
    
    win_rate = (returns > 0).mean()
    
    return {
        'cum_return': cum_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'avg_turnover': result_df['turnover'].mean()
    }

print(f"\n  交易成本: {TOTAL_COST*100:.2f}%/次")
print()

for top_n in [5, 10, 20, 30, 50]:
    result = backtest(test_data, top_n)
    metrics = calc_metrics(result)
    
    if not metrics:
        continue
    
    print(f"  Top {top_n}:")
    print(f"    净收益: {metrics['cum_return']*100:+.1f}%")
    print(f"    夏普: {metrics['sharpe']:.2f}")
    print(f"    回撤: {metrics['max_dd']*100:.1f}%")
    print(f"    胜率: {metrics['win_rate']*100:.1f}%")
    print(f"    换手: {metrics['avg_turnover']*100:.1f}%")
    print()

# ============================================================
# 7. 滚动回测（2023-07 ~ 2024-12）
# ============================================================
print("\n[7/7] 滚动回测（2023-07 ~ 2024-12）...")

# 每月滚动训练
months = pd.date_range('2023-07-01', '2024-12-01', freq='MS')
all_preds = []

for m_start in months:
    m_end = m_start + pd.DateOffset(months=1)
    
    # 训练数据：该月之前 12 个月
    train_start = m_start - pd.DateOffset(months=12)
    train_mask = (factor_data['date'] >= train_start) & (factor_data['date'] < m_start)
    test_mask = (factor_data['date'] >= m_start) & (factor_data['date'] < m_end)
    
    if train_mask.sum() < 5000 or test_mask.sum() == 0:
        continue
    
    X_tr = factor_data.loc[train_mask, feature_cols].values
    y_tr = factor_data.loc[train_mask, 'label'].values
    X_te = factor_data.loc[test_mask, feature_cols].values
    
    train_set = lgb.Dataset(X_tr, label=y_tr)
    m = lgb.train(params, train_set, num_boost_round=200, callbacks=[lgb.log_evaluation(0)])
    
    pred = m.predict(X_te)
    
    m_data = factor_data.loc[test_mask, ['date', 'code', 'future_return', 'label']].copy()
    m_data['pred'] = pred
    all_preds.append(m_data)

rolling_data = pd.concat(all_preds, ignore_index=True)

# 计算滚动 IC
rolling_ic = rolling_data.groupby('date').apply(lambda x: x['pred'].corr(x['label'])).mean()
print(f"  滚动 IC: {rolling_ic:.4f}")

# 滚动回测
print(f"\n  滚动回测结果（2023-07 ~ 2024-12）:")
print()

for top_n in [10, 20, 30]:
    result = backtest(rolling_data, top_n)
    
    if len(result) == 0:
        continue
    
    metrics = calc_metrics(result)
    
    # 分年统计
    result['year'] = pd.to_datetime(result['date']).dt.year
    
    print(f"  Top {top_n}:")
    print(f"    整体: 净收益={metrics['cum_return']*100:+.1f}%, 夏普={metrics['sharpe']:.2f}, 回撤={metrics['max_dd']*100:.1f}%")
    
    for year in [2023, 2024]:
        year_result = result[result['year'] == year]
        if len(year_result) == 0:
            continue
        year_metrics = calc_metrics(year_result)
        print(f"    {year}: 净收益={year_metrics['cum_return']*100:+.1f}%, 夏普={year_metrics['sharpe']:.2f}")
    print()

# 基准对比
print("  基准对比（等权持有所有股票）:")
benchmark = rolling_data.groupby('date')['future_return'].mean()
bench_cum = (1 + benchmark).prod() - 1
bench_sharpe = benchmark.mean() / (benchmark.std() + 1e-10) * np.sqrt(252)
print(f"    等权基准: 收益={bench_cum*100:+.1f}%, 夏普={bench_sharpe:.2f}")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
