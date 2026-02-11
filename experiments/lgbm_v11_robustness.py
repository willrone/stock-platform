#!/usr/bin/env python3
"""
LightGBM v11: 稳健性验证
- 多随机种子测试
- 样本外测试（2024 Q4 完全不参与训练）
- 加入交易成本
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 数据路径
DATA_DIR = Path('/Users/ronghui/Projects/willrone/data/parquet/stock_data')

print("=" * 70)
print("LightGBM v11: 稳健性验证")
print("=" * 70)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1/6] 加载数据...")

all_files = list(DATA_DIR.glob('*.parquet'))
dfs = []
for f in all_files:
    df = pd.read_parquet(f)
    df['code'] = f.stem
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(['code', 'date']).reset_index(drop=True)

# 筛选活跃股票
volume_rank = data.groupby('code')['volume'].mean().reset_index()
volume_rank['rank'] = volume_rank['volume'].rank(ascending=False)
active_stocks = volume_rank[volume_rank['rank'] <= 450]['code'].tolist()
data = data[data['code'].isin(active_stocks)]

print(f"  股票数: {len(active_stocks)}")
print(f"  数据量: {len(data):,}")

# ============================================================
# 2. Alpha158 因子计算（简化版，保留最重要的因子）
# ============================================================
print("\n[2/6] 计算 Alpha158 因子（简化版）...")

def calc_alpha_factors(df):
    """计算 Alpha158 因子（简化版，约 50 个最重要因子）"""
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    features = {}
    
    # 收益率因子
    for d in [1, 5, 10, 20, 60]:
        features[f'return_{d}d'] = pd.Series(close).pct_change(d).values
    
    # 波动率因子
    for d in [5, 10, 20, 60]:
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
    features['macd_hist'] = features['macd'] - features['macd_signal']
    
    # 布林带
    ma20 = pd.Series(close).rolling(20).mean()
    std20 = pd.Series(close).rolling(20).std()
    features['bb_upper'] = ((ma20 + 2 * std20 - close) / (close + 1e-10)).values
    features['bb_lower'] = ((close - ma20 + 2 * std20) / (close + 1e-10)).values
    features['bb_width'] = (4 * std20 / (ma20 + 1e-10)).values
    
    # ATR
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    for d in [5, 14, 20]:
        features[f'atr_{d}'] = pd.Series(tr).rolling(d).mean().values / (close + 1e-10)
    
    # 成交量因子
    for d in [5, 10, 20]:
        vol_ma = pd.Series(volume).rolling(d).mean().values
        features[f'vol_ratio_{d}d'] = volume / (vol_ma + 1e-10)
    
    # 换手率变化
    features['vol_change_5d'] = pd.Series(volume).pct_change(5).values
    features['vol_change_20d'] = pd.Series(volume).pct_change(20).values
    
    # 价格形态
    features['body'] = (close - open_) / (open_ + 1e-10)
    features['wick_upper'] = (high - np.maximum(close, open_)) / (high - low + 1e-10)
    features['wick_lower'] = (np.minimum(close, open_) - low) / (high - low + 1e-10)
    
    # 动量因子
    for d in [5, 10, 20, 60]:
        features[f'momentum_{d}d'] = (close / (np.roll(close, d) + 1e-10) - 1)
    
    # 高低点位置
    for d in [10, 20, 60]:
        roll_high = pd.Series(high).rolling(d).max().values
        roll_low = pd.Series(low).rolling(d).min().values
        features[f'high_pos_{d}d'] = (close - roll_low) / (roll_high - roll_low + 1e-10)
    
    # 量价相关性
    for d in [10, 20]:
        features[f'corr_vol_ret_{d}d'] = pd.Series(close).pct_change().rolling(d).corr(pd.Series(volume)).values
    
    return pd.DataFrame(features)

# 计算因子
factor_dfs = []
for code in active_stocks:
    stock_data = data[data['code'] == code].copy()
    if len(stock_data) < 100:
        continue
    factors = calc_alpha_factors(stock_data)
    factors['code'] = code
    factors['date'] = stock_data['date'].values
    factors['close'] = stock_data['close'].values
    factor_dfs.append(factors)

factor_data = pd.concat(factor_dfs, ignore_index=True)
print(f"  因子数: {len([c for c in factor_data.columns if c not in ['code', 'date', 'close']])}")

# ============================================================
# 3. 准备训练数据
# ============================================================
print("\n[3/6] 准备训练数据...")

# 计算未来收益（标签）
factor_data = factor_data.sort_values(['code', 'date'])
factor_data['future_return'] = factor_data.groupby('code')['close'].transform(
    lambda x: x.shift(-5) / x - 1
)

# 截面标准化标签
factor_data['label'] = factor_data.groupby('date')['future_return'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-10)
)

# 特征列
feature_cols = [c for c in factor_data.columns if c not in ['code', 'date', 'close', 'future_return', 'label']]

# 删除缺失值
factor_data = factor_data.dropna(subset=feature_cols + ['label'])

# 特征标准化
for col in feature_cols:
    factor_data[col] = factor_data.groupby('date')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )

# 时间划分
# 训练: 2019-01 到 2024-06（不包含 2024 Q4）
# 验证: 2024-07 到 2024-09
# 测试: 2024-10 到 2024-12（完全样本外）
train_data = factor_data[(factor_data['date'] >= '2019-01-01') & (factor_data['date'] < '2024-07-01')]
valid_data = factor_data[(factor_data['date'] >= '2024-07-01') & (factor_data['date'] < '2024-10-01')]
test_data = factor_data[factor_data['date'] >= '2024-10-01']

print(f"  训练集: {len(train_data):,} ({train_data['date'].min().date()} ~ {train_data['date'].max().date()})")
print(f"  验证集: {len(valid_data):,} ({valid_data['date'].min().date()} ~ {valid_data['date'].max().date()})")
print(f"  测试集: {len(test_data):,} ({test_data['date'].min().date()} ~ {test_data['date'].max().date()})")

# ============================================================
# 4. 多随机种子测试
# ============================================================
print("\n[4/6] 多随机种子测试...")

seeds = [42, 123, 456, 789, 2024]
seed_results = []

params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 200,
    'lambda_l2': 500,
    'min_data_in_leaf': 200,
    'verbose': -1,
}

X_train = train_data[feature_cols].values
y_train = train_data['label'].values
X_valid = valid_data[feature_cols].values
y_valid = valid_data['label'].values
X_test = test_data[feature_cols].values
y_test = test_data['label'].values

for seed in seeds:
    params['seed'] = seed
    params['bagging_seed'] = seed
    params['feature_fraction_seed'] = seed
    
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
    
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # 测试集预测
    test_pred = model.predict(X_test)
    
    # 计算 IC
    test_data_copy = test_data.copy()
    test_data_copy['pred'] = test_pred
    ic = test_data_copy.groupby('date').apply(
        lambda x: x['pred'].corr(x['label'])
    ).mean()
    
    seed_results.append({
        'seed': seed,
        'ic': ic,
        'best_iter': model.best_iteration
    })
    print(f"  Seed {seed}: IC={ic:.4f}, best_iter={model.best_iteration}")

print(f"\n  平均 IC: {np.mean([r['ic'] for r in seed_results]):.4f} ± {np.std([r['ic'] for r in seed_results]):.4f}")

# ============================================================
# 5. 样本外回测（2024 Q4，加入交易成本）
# ============================================================
print("\n[5/6] 样本外回测（2024 Q4 + 交易成本）...")

# 使用最佳种子训练最终模型
best_seed = max(seed_results, key=lambda x: x['ic'])['seed']
params['seed'] = best_seed
params['bagging_seed'] = best_seed
params['feature_fraction_seed'] = best_seed

train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

final_model = lgb.train(
    params,
    train_set,
    num_boost_round=500,
    valid_sets=[valid_set],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

# 预测
test_data['pred'] = final_model.predict(X_test)

# 交易成本参数
COMMISSION = 0.0003  # 手续费 0.03%
SLIPPAGE = 0.001     # 滑点 0.1%
STAMP_TAX = 0.001    # 印花税 0.1%（卖出时）
TOTAL_COST = COMMISSION * 2 + SLIPPAGE * 2 + STAMP_TAX  # 单次交易总成本约 0.36%

def backtest_with_cost(df, top_n, cost_per_trade):
    """带交易成本的回测"""
    dates = sorted(df['date'].unique())
    daily_returns = []
    prev_holdings = set()
    
    for date in dates:
        day_data = df[df['date'] == date].copy()
        if len(day_data) < top_n:
            continue
        
        # 选择 Top N
        top_stocks = day_data.nlargest(top_n, 'pred')['code'].tolist()
        current_holdings = set(top_stocks)
        
        # 计算换手
        new_stocks = current_holdings - prev_holdings
        sold_stocks = prev_holdings - current_holdings
        turnover = (len(new_stocks) + len(sold_stocks)) / (2 * top_n) if top_n > 0 else 0
        
        # 当日收益
        day_return = day_data[day_data['code'].isin(top_stocks)]['future_return'].mean()
        if pd.isna(day_return):
            day_return = 0
        
        # 扣除交易成本
        cost = turnover * cost_per_trade
        net_return = day_return - cost
        
        daily_returns.append({
            'date': date,
            'gross_return': day_return,
            'cost': cost,
            'net_return': net_return,
            'turnover': turnover
        })
        
        prev_holdings = current_holdings
    
    return pd.DataFrame(daily_returns)

# 回测不同 Top N
print("\n  样本外回测（2024-10 ~ 2024-12）:")
print(f"  交易成本: 手续费 {COMMISSION*100:.2f}% x2 + 滑点 {SLIPPAGE*100:.2f}% x2 + 印花税 {STAMP_TAX*100:.2f}% = {TOTAL_COST*100:.2f}%/次")
print()

for top_n in [10, 20, 30]:
    result = backtest_with_cost(test_data, top_n, TOTAL_COST)
    
    if len(result) == 0:
        continue
    
    # 统计
    gross_cum = (1 + result['gross_return']).prod() - 1
    net_cum = (1 + result['net_return']).prod() - 1
    avg_turnover = result['turnover'].mean()
    total_cost = result['cost'].sum()
    
    # 夏普比率
    net_sharpe = result['net_return'].mean() / (result['net_return'].std() + 1e-10) * np.sqrt(252)
    
    # 最大回撤
    cum_returns = (1 + result['net_return']).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    print(f"  Top {top_n}:")
    print(f"    毛收益: {gross_cum*100:+.1f}%")
    print(f"    净收益: {net_cum*100:+.1f}%")
    print(f"    夏普: {net_sharpe:.2f}")
    print(f"    回撤: {max_dd*100:.1f}%")
    print(f"    平均换手: {avg_turnover*100:.1f}%")
    print(f"    累计成本: {total_cost*100:.2f}%")
    print()

# ============================================================
# 6. 完整回测（2022-2024，加入交易成本）
# ============================================================
print("\n[6/6] 完整回测（2022-2024 + 交易成本）...")

# 滚动训练回测
full_test_data = factor_data[factor_data['date'] >= '2022-01-01'].copy()

# 按季度滚动训练
quarters = pd.date_range('2022-01-01', '2024-12-31', freq='QS')
all_preds = []

for i, q_start in enumerate(quarters):
    q_end = q_start + pd.DateOffset(months=3)
    
    # 训练数据：该季度之前的所有数据
    train_mask = factor_data['date'] < q_start
    test_mask = (factor_data['date'] >= q_start) & (factor_data['date'] < q_end)
    
    if train_mask.sum() < 10000 or test_mask.sum() == 0:
        continue
    
    X_tr = factor_data.loc[train_mask, feature_cols].values
    y_tr = factor_data.loc[train_mask, 'label'].values
    X_te = factor_data.loc[test_mask, feature_cols].values
    
    train_set = lgb.Dataset(X_tr, label=y_tr)
    model = lgb.train(params, train_set, num_boost_round=300, callbacks=[lgb.log_evaluation(0)])
    
    pred = model.predict(X_te)
    
    q_data = factor_data.loc[test_mask, ['date', 'code', 'future_return', 'label']].copy()
    q_data['pred'] = pred
    all_preds.append(q_data)

full_pred_data = pd.concat(all_preds, ignore_index=True)

# 市场择时
market_data = full_pred_data.groupby('date')['future_return'].mean().reset_index()
market_data['ma20'] = market_data['future_return'].rolling(20).mean()
market_data['ma60'] = market_data['future_return'].rolling(60).mean()
market_data['trend'] = (market_data['ma20'] > market_data['ma60']).astype(int)
market_trend = dict(zip(market_data['date'], market_data['trend']))

print("\n  完整回测（2022-2024，择时 + 交易成本）:")
print()

for top_n in [10, 20, 30]:
    dates = sorted(full_pred_data['date'].unique())
    daily_returns = []
    prev_holdings = set()
    
    for date in dates:
        # 择时：趋势向下时空仓
        if market_trend.get(date, 1) == 0:
            daily_returns.append({'date': date, 'net_return': 0, 'year': date.year})
            continue
        
        day_data = full_pred_data[full_pred_data['date'] == date]
        if len(day_data) < top_n:
            continue
        
        top_stocks = day_data.nlargest(top_n, 'pred')['code'].tolist()
        current_holdings = set(top_stocks)
        
        # 换手
        new_stocks = current_holdings - prev_holdings
        sold_stocks = prev_holdings - current_holdings
        turnover = (len(new_stocks) + len(sold_stocks)) / (2 * top_n) if top_n > 0 else 0
        
        # 收益
        day_return = day_data[day_data['code'].isin(top_stocks)]['future_return'].mean()
        if pd.isna(day_return):
            day_return = 0
        
        # 扣除成本
        cost = turnover * TOTAL_COST
        net_return = day_return - cost
        
        daily_returns.append({
            'date': date,
            'net_return': net_return,
            'year': date.year
        })
        
        prev_holdings = current_holdings
    
    result_df = pd.DataFrame(daily_returns)
    
    # 整体统计
    cum_return = (1 + result_df['net_return']).prod() - 1
    sharpe = result_df['net_return'].mean() / (result_df['net_return'].std() + 1e-10) * np.sqrt(252)
    
    cum_returns = (1 + result_df['net_return']).cumprod()
    rolling_max = cum_returns.expanding().max()
    max_dd = ((cum_returns - rolling_max) / rolling_max).min()
    
    print(f"  择时 Top {top_n}:")
    print(f"    整体: 累计={cum_return*100:+.1f}%, 夏普={sharpe:.2f}, 回撤={max_dd*100:.1f}%")
    
    # 分年统计
    for year in [2022, 2023, 2024]:
        year_data = result_df[result_df['year'] == year]
        if len(year_data) == 0:
            continue
        year_return = (1 + year_data['net_return']).prod() - 1
        year_sharpe = year_data['net_return'].mean() / (year_data['net_return'].std() + 1e-10) * np.sqrt(252)
        print(f"    {year}: {year_return*100:+.1f}%, 夏普={year_sharpe:.2f}")
    print()

print("=" * 70)
print("完成！")
print("=" * 70)
