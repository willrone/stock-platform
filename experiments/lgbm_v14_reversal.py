#!/usr/bin/env python3
"""
LightGBM v14: 反转因子版
- 针对 A 股反转效应设计因子
- 过去跌的股票，未来可能涨（超跌反弹）
- 过去涨的股票，未来可能跌（获利回吐）
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/Users/ronghui/Documents/GitHub/willrone/data/parquet/stock_data')

print("=" * 70)
print("LightGBM v14: 反转因子版（针对 A 股特性）")
print("=" * 70)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1/6] 加载数据...")

all_files = list(DATA_DIR.glob('*.parquet'))
dfs = [pd.read_parquet(f).assign(code=f.stem) for f in all_files]
data = pd.concat(dfs, ignore_index=True)
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(['code', 'date']).reset_index(drop=True)

data = data[(data['date'] >= '2023-01-01') & (data['date'] <= '2024-12-31')]

stock_counts = data.groupby('code').size()
active_stocks = stock_counts[stock_counts >= 400].index.tolist()
data = data[data['code'].isin(active_stocks)]

print(f"  股票数: {len(active_stocks)}, 数据量: {len(data):,}")

# ============================================================
# 2. 计算反转因子
# ============================================================
print("\n[2/6] 计算反转因子...")

def calc_reversal_factors(df):
    """计算反转因子（A 股特化）"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    features = {}
    
    # ========== 反转因子（核心）==========
    # 过去收益的负值 = 反转信号
    for d in [5, 10, 20]:
        past_return = pd.Series(close).pct_change(d).values
        features[f'reversal_{d}d'] = -past_return  # 取负！过去跌 → 未来涨
    
    # 超跌程度：距离 N 日高点的跌幅
    for d in [10, 20, 60]:
        roll_high = pd.Series(high).rolling(d).max().values
        features[f'drawdown_{d}d'] = (close - roll_high) / (roll_high + 1e-10)  # 负值越大越超跌
    
    # 超涨程度：距离 N 日低点的涨幅（取负作为反转信号）
    for d in [10, 20, 60]:
        roll_low = pd.Series(low).rolling(d).min().values
        features[f'runup_{d}d'] = -((close - roll_low) / (roll_low + 1e-10))  # 涨太多 → 可能回调
    
    # ========== 波动率因子 ==========
    for d in [5, 10, 20]:
        features[f'volatility_{d}d'] = pd.Series(close).pct_change().rolling(d).std().values
    
    # 波动率变化（波动收敛可能预示突破）
    vol_5 = pd.Series(close).pct_change().rolling(5).std()
    vol_20 = pd.Series(close).pct_change().rolling(20).std()
    features['vol_contraction'] = (vol_5 / (vol_20 + 1e-10)).values
    
    # ========== 成交量因子 ==========
    for d in [5, 10, 20]:
        vol_ma = pd.Series(volume).rolling(d).mean().values
        features[f'vol_ratio_{d}d'] = volume / (vol_ma + 1e-10)
    
    # 量价背离：价格跌但成交量放大（可能是底部）
    price_change = pd.Series(close).pct_change(5).values
    vol_change = pd.Series(volume).pct_change(5).values
    features['vol_price_diverge'] = vol_change - price_change  # 量增价跌 → 正值
    
    # ========== 技术指标（偏反转）==========
    # RSI 超卖
    for d in [6, 14]:
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(d).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(d).mean()
        rsi = gain / (gain + loss + 1e-10)
        features[f'rsi_oversold_{d}'] = 0.5 - rsi.values  # RSI < 0.5 → 正值（超卖）
    
    # 布林带位置（偏下轨 = 超卖）
    ma20 = pd.Series(close).rolling(20).mean()
    std20 = pd.Series(close).rolling(20).std()
    features['bb_oversold'] = ((ma20 - close) / (2 * std20 + 1e-10)).values  # 低于均线 → 正值
    
    # MACD 背离
    ema12 = pd.Series(close).ewm(span=12).mean()
    ema26 = pd.Series(close).ewm(span=26).mean()
    macd = ema12 - ema26
    features['macd_reversal'] = -macd.values  # MACD 负值 → 可能反弹
    
    # ========== 截面排名因子 ==========
    # 这些在后面按日期计算
    
    return pd.DataFrame(features)

factor_dfs = []
for code in active_stocks:
    stock_data = data[data['code'] == code].copy()
    factors = calc_reversal_factors(stock_data)
    factors['code'] = code
    factors['date'] = stock_data['date'].values
    factors['close'] = stock_data['close'].values
    factor_dfs.append(factors)

factor_data = pd.concat(factor_dfs, ignore_index=True)
feature_cols = [c for c in factor_data.columns if c not in ['code', 'date', 'close']]
print(f"  因子数: {len(feature_cols)}")

# ============================================================
# 3. 准备标签和标准化
# ============================================================
print("\n[3/6] 准备标签...")

factor_data = factor_data.sort_values(['code', 'date'])
factor_data['future_return'] = factor_data.groupby('code')['close'].transform(
    lambda x: x.shift(-5) / x - 1
)
factor_data['label'] = factor_data.groupby('date')['future_return'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-10)
)

# 截面标准化特征
for col in feature_cols:
    factor_data[col] = factor_data.groupby('date')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )

factor_data = factor_data.dropna(subset=feature_cols + ['label'])

# 划分
train_data = factor_data[(factor_data['date'] >= '2023-01-01') & (factor_data['date'] < '2024-07-01')]
valid_data = factor_data[(factor_data['date'] >= '2024-07-01') & (factor_data['date'] < '2024-10-01')]
test_data = factor_data[factor_data['date'] >= '2024-10-01']

print(f"  训练集: {len(train_data):,}")
print(f"  验证集: {len(valid_data):,}")
print(f"  测试集: {len(test_data):,}")

# ============================================================
# 4. 训练模型
# ============================================================
print("\n[4/6] 训练模型...")

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

# 特征重要性
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)

print("\n  Top 10 重要因子:")
for _, row in importance.head(10).iterrows():
    print(f"    {row['feature']}: {row['importance']}")

# ============================================================
# 5. 评估
# ============================================================
print("\n[5/6] 评估...")

# IC
valid_data = valid_data.copy()
valid_data['pred'] = model.predict(X_valid)
valid_ic = valid_data.groupby('date').apply(lambda x: x['pred'].corr(x['label'])).mean()

test_data = test_data.copy()
X_test = test_data[feature_cols].values
test_data['pred'] = model.predict(X_test)
test_ic = test_data.groupby('date').apply(lambda x: x['pred'].corr(x['label'])).mean()

print(f"  验证集 IC: {valid_ic:.4f}")
print(f"  测试集 IC: {test_ic:.4f}")

# ============================================================
# 6. 滚动回测
# ============================================================
print("\n[6/6] 滚动回测...")

TOTAL_COST = 0.0036

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
    m = lgb.train(params, train_set, num_boost_round=200, callbacks=[lgb.log_evaluation(0)])
    
    m_data = factor_data.loc[test_mask, ['date', 'code', 'future_return', 'label']].copy()
    m_data['pred'] = m.predict(X_te)
    all_preds.append(m_data)

rolling_data = pd.concat(all_preds, ignore_index=True)

# 滚动 IC
rolling_ic = rolling_data.groupby('date').apply(lambda x: x['pred'].corr(x['label'])).mean()
print(f"  滚动 IC: {rolling_ic:.4f}")

# 分年 IC
for year in [2023, 2024]:
    year_data = rolling_data[rolling_data['date'].dt.year == year]
    if len(year_data) > 0:
        year_ic = year_data.groupby('date').apply(lambda x: x['pred'].corr(x['label'])).mean()
        print(f"  {year} IC: {year_ic:.4f}")

# 回测
def backtest(df, top_n):
    dates = sorted(df['date'].unique())
    daily_returns = []
    prev_holdings = set()
    
    for date in dates:
        day_data = df[df['date'] == date]
        if len(day_data) < top_n:
            continue
        
        top_stocks = day_data.nlargest(top_n, 'pred')['code'].tolist()
        current_holdings = set(top_stocks)
        
        if prev_holdings:
            turnover = len(current_holdings ^ prev_holdings) / (2 * top_n)
        else:
            turnover = 1.0
        
        day_return = day_data[day_data['code'].isin(top_stocks)]['future_return'].mean()
        if pd.isna(day_return):
            day_return = 0
        
        cost = turnover * TOTAL_COST
        net_return = day_return - cost
        
        daily_returns.append({
            'date': date,
            'net_return': net_return,
            'year': date.year
        })
        
        prev_holdings = current_holdings
    
    return pd.DataFrame(daily_returns)

print("\n  滚动回测结果:")
print()

for top_n in [10, 20, 30, 50]:
    result = backtest(rolling_data, top_n)
    
    if len(result) == 0:
        continue
    
    returns = result['net_return']
    cum_return = (1 + returns).prod() - 1
    sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
    
    cum_returns = (1 + returns).cumprod()
    max_dd = ((cum_returns - cum_returns.expanding().max()) / cum_returns.expanding().max()).min()
    
    # 分年
    y2023 = result[result['year'] == 2023]
    y2024 = result[result['year'] == 2024]
    
    ret_2023 = (1 + y2023['net_return']).prod() - 1 if len(y2023) > 0 else 0
    ret_2024 = (1 + y2024['net_return']).prod() - 1 if len(y2024) > 0 else 0
    
    sharpe_2023 = y2023['net_return'].mean() / (y2023['net_return'].std() + 1e-10) * np.sqrt(252) if len(y2023) > 0 else 0
    sharpe_2024 = y2024['net_return'].mean() / (y2024['net_return'].std() + 1e-10) * np.sqrt(252) if len(y2024) > 0 else 0
    
    print(f"  Top {top_n}:")
    print(f"    整体: 收益={cum_return*100:+.1f}%, 夏普={sharpe:.2f}, 回撤={max_dd*100:.1f}%")
    print(f"    2023: 收益={ret_2023*100:+.1f}%, 夏普={sharpe_2023:.2f}")
    print(f"    2024: 收益={ret_2024*100:+.1f}%, 夏普={sharpe_2024:.2f}")
    print()

# 基准
benchmark = rolling_data.groupby('date')['future_return'].mean()
bench_cum = (1 + benchmark).prod() - 1
print(f"  等权基准: 收益={bench_cum*100:+.1f}%")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
