#!/usr/bin/env python3
"""
LightGBM v15: 反转因子稳健性验证
- 多随机种子测试（5个种子）
- 验证 IC 稳定性
- 验证收益稳定性
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DATA_DIR = Path('/Users/ronghui/Projects/willrone/data/parquet/stock_data')
TRAIN_START = '2018-01-01'
TRAIN_END = '2022-12-31'
TEST_START = '2023-01-01'
TEST_END = '2024-12-31'

SEEDS = [42, 123, 456, 789, 2024]  # 5个随机种子

# ==================== 加载数据 ====================
print("=" * 70)
print("LightGBM v15: 反转因子稳健性验证")
print("=" * 70)
print("\n[1/5] 加载数据...")

all_data = []
for file in DATA_DIR.glob('*.parquet'):
    df = pd.read_parquet(file)
    df['ts_code'] = file.stem
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(['ts_code', 'date']).reset_index(drop=True)

print(f"  股票数: {data['ts_code'].nunique()}, 数据量: {len(data):,}")

# ==================== 计算反转因子 ====================
print("\n[2/5] 计算反转因子...")

def calculate_reversal_features(df):
    """计算反转因子"""
    df = df.copy()
    
    # 1. 短期反转（5/10/20日）
    df['reversal_5d'] = -df.groupby('ts_code')['close'].pct_change(5)
    df['reversal_10d'] = -df.groupby('ts_code')['close'].pct_change(10)
    df['reversal_20d'] = -df.groupby('ts_code')['close'].pct_change(20)
    
    # 2. 超买超卖（RSI反转）
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = df.groupby('ts_code')['close'].transform(lambda x: calc_rsi(x, 14))
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)  # 超卖信号
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)  # 超买信号
    
    # 3. 布林带反转
    df['ma_20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
    df['std_20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).std())
    df['bb_upper'] = df['ma_20'] + 2 * df['std_20']
    df['bb_lower'] = df['ma_20'] - 2 * df['std_20']
    df['bb_oversold'] = ((df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))).astype(int)
    df['bb_overbought'] = ((df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))).astype(int)
    
    # 4. 回撤深度（买入机会）
    df['high_60d'] = df.groupby('ts_code')['high'].transform(lambda x: x.rolling(60).max())
    df['drawdown_60d'] = (df['close'] - df['high_60d']) / df['high_60d']
    
    # 5. 涨幅过大（卖出信号）
    df['low_20d'] = df.groupby('ts_code')['low'].transform(lambda x: x.rolling(20).min())
    df['runup_20d'] = (df['close'] - df['low_20d']) / df['low_20d']
    df['low_10d'] = df.groupby('ts_code')['low'].transform(lambda x: x.rolling(10).min())
    df['runup_10d'] = (df['close'] - df['low_10d']) / df['low_10d']
    df['low_60d'] = df.groupby('ts_code')['low'].transform(lambda x: x.rolling(60).min())
    df['runup_60d'] = (df['close'] - df['low_60d']) / df['low_60d']
    
    # 6. MACD 反转
    ema_12 = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=12).mean())
    ema_26 = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=26).mean())
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df.groupby('ts_code')['macd'].transform(lambda x: x.ewm(span=9).mean())
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_reversal'] = ((df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0)).astype(int)
    
    # 7. 波动率（高波动 = 反转机会）
    df['volatility_5d'] = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change().rolling(5).std())
    df['volatility_20d'] = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change().rolling(20).std())
    
    # 8. 成交量反转
    df['volume_ma_20'] = df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(20).mean())
    df['vol_ratio_5d'] = df['volume'] / df['volume_ma_20']
    
    return df

data = calculate_reversal_features(data)

feature_cols = [
    'reversal_5d', 'reversal_10d', 'reversal_20d',
    'rsi_14', 'rsi_oversold', 'rsi_overbought',
    'bb_oversold', 'bb_overbought',
    'drawdown_60d', 'runup_20d', 'runup_10d', 'runup_60d',
    'macd', 'macd_signal', 'macd_hist', 'macd_reversal',
    'volatility_5d', 'volatility_20d',
    'vol_ratio_5d'
]

print(f"  因子数: {len(feature_cols)}")

# ==================== 准备标签 ====================
print("\n[3/5] 准备标签...")

data['return_5d'] = data.groupby('ts_code')['close'].pct_change(5).shift(-5)
data = data.dropna(subset=feature_cols + ['return_5d'])

train_data = data[(data['date'] >= TRAIN_START) & (data['date'] <= TRAIN_END)].copy()
test_data = data[(data['date'] >= TEST_START) & (data['date'] <= TEST_END)].copy()

print(f"  训练集: {len(train_data):,}")
print(f"  测试集: {len(test_data):,}")

# ==================== 多种子训练 ====================
print("\n[4/5] 多种子训练...")

results = []

for seed in SEEDS:
    print(f"\n  种子 {seed}:")
    
    # 训练模型
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': seed,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
    }
    
    train_set = lgb.Dataset(train_data[feature_cols], train_data['return_5d'])
    
    model = lgb.train(
        params,
        train_set,
        num_boost_round=100,
        valid_sets=[train_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # 预测
    test_data_copy = test_data.copy()
    test_data_copy['pred'] = model.predict(test_data_copy[feature_cols])
    
    # 计算 IC
    ic_by_date = test_data_copy.groupby('date').apply(
        lambda x: x['pred'].corr(x['return_5d'], method='spearman')
    )
    ic_mean = ic_by_date.mean()
    
    # 2023/2024 分别的 IC
    ic_2023 = test_data_copy[test_data_copy['date'].dt.year == 2023].groupby('date').apply(
        lambda x: x['pred'].corr(x['return_5d'], method='spearman')
    ).mean()
    
    ic_2024 = test_data_copy[test_data_copy['date'].dt.year == 2024].groupby('date').apply(
        lambda x: x['pred'].corr(x['return_5d'], method='spearman')
    ).mean()
    
    # 回测 Top 10
    backtest_results = []
    for date in sorted(test_data_copy['date'].unique()):
        day_data = test_data_copy[test_data_copy['date'] == date].copy()
        day_data = day_data.sort_values('pred', ascending=False)
        
        top10_return = day_data.head(10)['return_5d'].mean()
        backtest_results.append({'date': date, 'return': top10_return})
    
    backtest_df = pd.DataFrame(backtest_results)
    backtest_df['cum_return'] = (1 + backtest_df['return']).cumprod() - 1
    
    total_return = backtest_df['cum_return'].iloc[-1]
    sharpe = backtest_df['return'].mean() / backtest_df['return'].std() * np.sqrt(252)
    
    # 2023/2024 分别收益
    backtest_2023 = backtest_df[backtest_df['date'].dt.year == 2023]
    backtest_2024 = backtest_df[backtest_df['date'].dt.year == 2024]
    
    return_2023 = (1 + backtest_2023['return']).prod() - 1 if len(backtest_2023) > 0 else 0
    return_2024 = (1 + backtest_2024['return']).prod() - 1 if len(backtest_2024) > 0 else 0
    
    print(f"    IC: {ic_mean:.4f} (2023: {ic_2023:.4f}, 2024: {ic_2024:.4f})")
    print(f"    Top10: 收益={total_return*100:+.1f}%, 夏普={sharpe:.2f}")
    print(f"           2023={return_2023*100:+.1f}%, 2024={return_2024*100:+.1f}%")
    
    results.append({
        'seed': seed,
        'ic': ic_mean,
        'ic_2023': ic_2023,
        'ic_2024': ic_2024,
        'return': total_return,
        'return_2023': return_2023,
        'return_2024': return_2024,
        'sharpe': sharpe
    })

# ==================== 汇总统计 ====================
print("\n[5/5] 稳健性统计...")

results_df = pd.DataFrame(results)

print(f"\n  IC 稳定性:")
print(f"    均值: {results_df['ic'].mean():.4f}")
print(f"    标准差: {results_df['ic'].std():.4f}")
print(f"    范围: [{results_df['ic'].min():.4f}, {results_df['ic'].max():.4f}]")

print(f"\n  2023 IC:")
print(f"    均值: {results_df['ic_2023'].mean():.4f}")
print(f"    标准差: {results_df['ic_2023'].std():.4f}")

print(f"\n  2024 IC:")
print(f"    均值: {results_df['ic_2024'].mean():.4f}")
print(f"    标准差: {results_df['ic_2024'].std():.4f}")

print(f"\n  Top10 收益稳定性:")
print(f"    均值: {results_df['return'].mean()*100:+.1f}%")
print(f"    标准差: {results_df['return'].std()*100:.1f}%")
print(f"    范围: [{results_df['return'].min()*100:+.1f}%, {results_df['return'].max()*100:+.1f}%]")

print(f"\n  2023 收益:")
print(f"    均值: {results_df['return_2023'].mean()*100:+.1f}%")
print(f"    标准差: {results_df['return_2023'].std()*100:.1f}%")

print(f"\n  2024 收益:")
print(f"    均值: {results_df['return_2024'].mean()*100:+.1f}%")
print(f"    标准差: {results_df['return_2024'].std()*100:.1f}%")

print(f"\n  夏普比率:")
print(f"    均值: {results_df['sharpe'].mean():.2f}")
print(f"    标准差: {results_df['sharpe'].std():.2f}")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
