#!/usr/bin/env python3
"""
LightGBM v4 扩展测试
- 更多 Top N 选股（50/100/200）
- 更长时间回测（2020-2025）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/Users/ronghui/Projects/willrone/data/parquet/stock_data')
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
    
    # MA
    for window in [5, 10, 20, 60]:
        ma = close.rolling(window).mean()
        df[f'ma_ratio_{window}'] = close / ma - 1
    
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
    df['turnover_proxy'] = volume / (vol_ma20 + 1)
    
    price_change = close.pct_change()
    vol_change = volume.pct_change()
    df['vol_price_corr'] = price_change.rolling(10).corr(vol_change)
    
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
    df['bb_width'] = 4 * bb_std / (bb_mid + 1e-10)
    
    # 波动率
    df['volatility_5'] = close.pct_change().rolling(5).std()
    df['volatility_20'] = close.pct_change().rolling(20).std()
    df['vol_regime'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)
    
    returns = close.pct_change()
    df['vol_skew'] = returns.rolling(20).skew()
    
    # 形态
    df['body'] = (close - open_) / (open_ + 1e-10)
    df['wick_ratio'] = (high - low) / (close + 1e-10)
    df['up_streak'] = (close > close.shift(1)).rolling(5).sum()
    
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
    """截面特征"""
    rank_cols = ['return_1d', 'return_5d', 'momentum_1m', 'vol_ratio', 'volatility_20', 'rsi_14']
    
    for col in rank_cols:
        if col in data.columns:
            data[f'{col}_rank'] = data.groupby('date')[col].rank(pct=True)
    
    daily_mean = data.groupby('date')['future_return'].transform('mean')
    data['market_return'] = daily_mean
    data['excess_return'] = data['future_return'] - daily_mean
    data['label_relative'] = (data['excess_return'] > 0).astype(int)
    
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
        'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_60', 'ma_score',
        'vol_ratio', 'vol_trend', 'turnover_proxy', 'vol_price_corr',
        'rsi_6', 'rsi_14', 'rsi_deviation',
        'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'bb_width',
        'volatility_5', 'volatility_20', 'vol_regime', 'vol_skew',
        'body', 'wick_ratio', 'up_streak',
        'price_pos_20', 'price_pos_60',
        'atr_pct', 'di_diff',
    ]
    rank = ['return_1d_rank', 'return_5d_rank', 'momentum_1m_rank',
            'vol_ratio_rank', 'volatility_20_rank', 'rsi_14_rank']
    return base + rank

def run_extended_backtest(data):
    """扩展回测"""
    feature_cols = get_feature_cols()
    label_col = 'label_relative'
    
    df = data.replace([np.inf, -np.inf], np.nan)
    required = [c for c in feature_cols if c in df.columns] + [label_col, 'future_return', 'excess_return']
    df = df.dropna(subset=required)
    df = df.sort_values('date').reset_index(drop=True)
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"有效数据: {len(df)} 条")
    
    # 训练集: 2018-2022, 测试集: 2022-2025 (3年)
    train_end = '2022-01-01'
    
    train = df[df['date'] < train_end]
    test = df[df['date'] >= train_end]
    
    print(f"\n训练集: {len(train)} ({train['date'].min().date()} - {train['date'].max().date()})")
    print(f"测试集: {len(test)} ({test['date'].min().date()} - {test['date'].max().date()})")
    
    X_train, y_train = train[feature_cols], train[label_col]
    X_test, y_test = test[feature_cols], test[label_col]
    
    # 训练
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.01,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_child_samples': 100,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'verbose': -1,
        'seed': 42,
    }
    
    train_data = lgb.Dataset(X_train, y_train)
    test_data = lgb.Dataset(X_test, y_test, reference=train_data)
    
    print("\n训练中...")
    model = lgb.train(
        params, train_data,
        num_boost_round=1500,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(300)]
    )
    
    pred_test = model.predict(X_test)
    auc = roc_auc_score(y_test, pred_test)
    print(f"\n测试集 AUC: {auc:.4f}")
    
    # 回测
    result = pd.DataFrame({
        'date': test['date'].values,
        'ts_code': test['ts_code'].values,
        'prob': pred_test,
        'label': y_test.values,
        'return': test['future_return'].values,
        'excess_return': test['excess_return'].values
    })
    
    result['date_str'] = result['date'].astype(str)
    result['year'] = pd.to_datetime(result['date']).dt.year
    
    COST = 0.001  # 单边 0.1%
    
    print("\n" + "="*70)
    print("扩展回测结果（2022-2025，3年）")
    print("="*70)
    
    for top_n in [50, 100, 200]:
        print(f"\n{'='*70}")
        print(f"每日 Top {top_n} 策略")
        print('='*70)
        
        daily_top = result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'prob')
        ).reset_index(drop=True)
        
        daily_returns = daily_top.groupby('date_str')['return'].mean() - 2 * COST
        
        total_days = len(daily_returns)
        win_rate = (daily_returns > 0).mean()
        avg_daily = daily_returns.mean()
        cumsum = daily_returns.cumsum()
        total_return = (1 + daily_returns).prod() - 1
        sharpe = avg_daily / (daily_returns.std() + 1e-10) * np.sqrt(252)
        max_dd = (cumsum - cumsum.cummax()).min()
        
        print(f"\n整体表现:")
        print(f"  交易天数: {total_days}")
        print(f"  日胜率: {win_rate:.2%}")
        print(f"  日均收益: {avg_daily*100:.3f}%")
        print(f"  年化收益: {avg_daily*252*100:.1f}%")
        print(f"  累计收益: {total_return*100:.1f}%")
        print(f"  夏普比率: {sharpe:.2f}")
        print(f"  最大回撤: {max_dd*100:.1f}%")
        
        # 分年度表现
        print(f"\n分年度表现:")
        daily_top['date_dt'] = pd.to_datetime(daily_top['date'])
        daily_top['year'] = daily_top['date_dt'].dt.year
        
        for year in sorted(daily_top['year'].unique()):
            year_data = daily_top[daily_top['year'] == year]
            year_returns = year_data.groupby('date_str')['return'].mean() - 2 * COST
            
            if len(year_returns) > 0:
                y_win = (year_returns > 0).mean()
                y_avg = year_returns.mean()
                y_total = (1 + year_returns).prod() - 1
                y_sharpe = y_avg / (year_returns.std() + 1e-10) * np.sqrt(252)
                y_cumsum = year_returns.cumsum()
                y_dd = (y_cumsum - y_cumsum.cummax()).min()
                
                print(f"  {year}: 天数={len(year_returns):>3}, 胜率={y_win:.1%}, "
                      f"日均={y_avg*100:.2f}%, 年收益={y_total*100:.1f}%, "
                      f"夏普={y_sharpe:.2f}, 回撤={y_dd*100:.1f}%")
    
    # 十分位分析
    print("\n" + "="*70)
    print("十分位分析")
    print("="*70)
    
    result['decile'] = pd.qcut(result['prob'], 10, labels=False, duplicates='drop')
    
    print(f"\n{'分位':>6} {'样本数':>10} {'标签命中':>10} {'绝对收益':>12} {'超额收益':>12}")
    print("-" * 55)
    
    for d in sorted(result['decile'].unique()):
        g = result[result['decile'] == d]
        print(f"{d:>6} {len(g):>10} {g['label'].mean():>10.2%} "
              f"{g['return'].mean()*100:>11.3f}% {g['excess_return'].mean()*100:>11.3f}%")
    
    return model

if __name__ == '__main__':
    data = load_data(n_stocks=450, start_date='2018-01-01', end_date='2025-01-01')
    model = run_extended_backtest(data)
