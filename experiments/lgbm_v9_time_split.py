#!/usr/bin/env python3
"""
LightGBM v9 - 按时间分离（更接近实际场景）

分离方式：
- 训练：2017-2021 所有股票
- 回测：2022-2024 所有股票（完全未见过的时间段）

这比按股票分离更合理，因为实际使用时：
- 我们用历史数据训练模型
- 然后在未来时间段使用
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/Users/ronghui/Documents/GitHub/willrone/data/parquet/stock_data')
np.random.seed(42)

def compute_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算 Alpha 因子"""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_ = df['open']
    
    # KBAR 因子
    df['KMID'] = (close - open_) / (open_ + 1e-10)
    df['KLEN'] = (high - low) / (open_ + 1e-10)
    df['KMID2'] = (close - open_) / (high - low + 1e-10)
    df['KUP'] = (high - np.maximum(open_, close)) / (high - low + 1e-10)
    df['KLOW'] = (np.minimum(open_, close) - low) / (high - low + 1e-10)
    df['KSFT'] = (2 * close - high - low) / (open_ + 1e-10)
    
    # 收益率
    for d in [1, 2, 3, 5, 10, 20, 30, 60]:
        df[f'ROC_{d}'] = close.pct_change(d)
    
    # 价格位置
    for d in [5, 10, 20, 30, 60]:
        df[f'MAX_{d}'] = high.rolling(d).max()
        df[f'MIN_{d}'] = low.rolling(d).min()
        df[f'QTLU_{d}'] = (close - df[f'MIN_{d}']) / (df[f'MAX_{d}'] - df[f'MIN_{d}'] + 1e-10)
    
    # MA 因子
    for d in [5, 10, 20, 30, 60]:
        ma = close.rolling(d).mean()
        df[f'MA_{d}'] = ma
        df[f'MA_RATIO_{d}'] = close / (ma + 1e-10) - 1
    
    df['MA_CROSS_5_20'] = df['MA_5'] / (df['MA_20'] + 1e-10) - 1
    df['MA_CROSS_10_30'] = df['MA_10'] / (df['MA_30'] + 1e-10) - 1
    df['MA_CROSS_20_60'] = df['MA_20'] / (df['MA_60'] + 1e-10) - 1
    
    # 波动率
    returns = close.pct_change()
    for d in [5, 10, 20, 30, 60]:
        df[f'STD_{d}'] = returns.rolling(d).std()
    
    df['VSTD_5_20'] = df['STD_5'] / (df['STD_20'] + 1e-10)
    df['VSTD_10_60'] = df['STD_10'] / (df['STD_60'] + 1e-10)
    
    # 成交量
    for d in [5, 10, 20, 30, 60]:
        vol_ma = volume.rolling(d).mean()
        df[f'VOLUME_MA_{d}'] = vol_ma
        df[f'VOLUME_RATIO_{d}'] = volume / (vol_ma + 1e-10)
    
    # 量价相关
    df['CORR_CLOSE_VOL_5'] = close.rolling(5).corr(volume)
    df['CORR_CLOSE_VOL_10'] = close.rolling(10).corr(volume)
    df['CORR_CLOSE_VOL_20'] = close.rolling(20).corr(volume)
    
    # RSI
    for d in [6, 12, 24]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=d, adjust=False).mean()
        avg_loss = loss.ewm(span=d, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f'RSI_{d}'] = 100 - (100 / (1 + rs))
    
    # MACD
    for (fast, slow, signal) in [(12, 26, 9), (6, 12, 6)]:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        df[f'MACD_{fast}_{slow}'] = macd / (close + 1e-10)
        df[f'MACD_SIGNAL_{fast}_{slow}'] = macd_signal / (close + 1e-10)
        df[f'MACD_HIST_{fast}_{slow}'] = (macd - macd_signal) / (close + 1e-10)
    
    # 布林带
    for d in [10, 20]:
        bb_mid = close.rolling(d).mean()
        bb_std = close.rolling(d).std()
        df[f'BOLL_UP_{d}'] = (bb_mid + 2 * bb_std - close) / (close + 1e-10)
        df[f'BOLL_DOWN_{d}'] = (close - bb_mid + 2 * bb_std) / (close + 1e-10)
        df[f'BOLL_WIDTH_{d}'] = 4 * bb_std / (bb_mid + 1e-10)
    
    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    for d in [6, 12, 24]:
        df[f'ATR_{d}'] = tr.rolling(d).mean() / (close + 1e-10)
    
    # 标签
    df['future_return_1d'] = close.shift(-1) / close - 1
    
    return df

def add_cross_sectional_features(data: pd.DataFrame) -> pd.DataFrame:
    """添加截面特征"""
    rank_cols = ['ROC_5', 'ROC_20', 'STD_20', 'VOLUME_RATIO_20', 'RSI_12']
    for col in rank_cols:
        if col in data.columns:
            data[f'{col}_RANK'] = data.groupby('date')[col].rank(pct=True)
    
    data['RELATIVE_RETURN_5'] = data.groupby('date')['ROC_5'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )
    data['RELATIVE_RETURN_20'] = data.groupby('date')['ROC_20'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )
    
    # 市场状态
    daily_stats = data.groupby('date').agg({
        'ROC_1': 'mean',
        'STD_20': 'mean',
    }).reset_index()
    daily_stats.columns = ['date', 'market_return', 'market_vol']
    daily_stats = daily_stats.sort_values('date')
    
    daily_stats['market_ma10'] = daily_stats['market_return'].rolling(10).mean()
    daily_stats['market_ma30'] = daily_stats['market_return'].rolling(30).mean()
    daily_stats['market_trend'] = (daily_stats['market_ma10'] > daily_stats['market_ma30']).astype(int)
    daily_stats['market_vol_ma'] = daily_stats['market_vol'].rolling(20).mean()
    daily_stats['high_vol'] = (daily_stats['market_vol'] > daily_stats['market_vol_ma'] * 1.2).astype(int)
    
    data = data.merge(daily_stats[['date', 'market_trend', 'high_vol']], on='date', how='left')
    
    # 标签：截面标准化
    data['label'] = data.groupby('date')['future_return_1d'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )
    
    return data

def load_data(n_stocks=450, start_date='2017-01-01', end_date='2025-01-01'):
    """加载所有股票数据"""
    print(f"加载数据: {start_date} - {end_date}")
    print("="*60)
    
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
        df = compute_alpha_features(df)
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    data = add_cross_sectional_features(data)
    print(f"总数据: {len(data)} 条")
    return data

def get_feature_cols(data):
    """获取特征列"""
    exclude = ['date', 'ts_code', 'future_return_1d', 'label', 'market_trend', 
               'high_vol', 'open', 'high', 'low', 'close', 'volume', 'amount']
    exclude += [f'MA_{d}' for d in [5, 10, 20, 30, 60]]
    exclude += [f'MAX_{d}' for d in [5, 10, 20, 30, 60]]
    exclude += [f'MIN_{d}' for d in [5, 10, 20, 30, 60]]
    exclude += [f'VOLUME_MA_{d}' for d in [5, 10, 20, 30, 60]]
    
    feature_cols = [c for c in data.columns if c not in exclude and not c.startswith('_')]
    return feature_cols

def time_split_backtest(data):
    """
    按时间分离的回测
    - 滚动训练：每季度用过去 3 年数据训练
    - 回测：2022-2024（完全未见过的时间段）
    """
    feature_cols = get_feature_cols(data)
    label_col = 'label'
    
    data = data.replace([np.inf, -np.inf], np.nan)
    required = [c for c in feature_cols if c in data.columns] + [label_col, 'future_return_1d', 'market_trend', 'high_vol']
    data = data.dropna(subset=required)
    feature_cols = [c for c in feature_cols if c in data.columns]
    
    print(f"\n有效数据: {len(data)} 条")
    print(f"特征数: {len(feature_cols)}")
    
    data['date_dt'] = pd.to_datetime(data['date'])
    data['year_month'] = data['date_dt'].dt.to_period('M')
    
    all_months = sorted(data['year_month'].unique())
    
    train_months = 36  # 3 年训练窗口
    test_months = 3    # 每季度更新模型
    
    # 找到 2022-01 开始的位置
    start_test_idx = None
    for i, m in enumerate(all_months):
        if m >= pd.Period('2022-01'):
            start_test_idx = i
            break
    
    if start_test_idx is None or start_test_idx < train_months:
        print("数据不足")
        return None
    
    # Qlib 风格参数
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 128,
        'max_depth': 8,
        'min_data_in_leaf': 200,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 200.0,
        'lambda_l2': 500.0,
        'min_gain_to_split': 0.01,
        'verbose': -1,
        'seed': 42,
        'num_threads': 8,
    }
    
    print(f"\n模型参数:")
    print(f"  任务: 回归")
    print(f"  L1/L2 正则: {params['lambda_l1']}/{params['lambda_l2']}")
    print(f"  训练窗口: {train_months} 个月")
    
    all_results = []
    
    print(f"\n滚动训练（按时间分离）")
    print("="*70)
    
    test_idx = start_test_idx
    while test_idx + test_months <= len(all_months):
        train_start = all_months[test_idx - train_months]
        train_end = all_months[test_idx - 1]
        test_start = all_months[test_idx]
        test_end = all_months[min(test_idx + test_months - 1, len(all_months) - 1)]
        
        # 训练数据：过去 3 年
        train_mask = (data['year_month'] >= train_start) & (data['year_month'] <= train_end)
        train_df = data[train_mask]
        
        # 测试数据：未来 3 个月
        test_mask = (data['year_month'] >= test_start) & (data['year_month'] <= test_end)
        test_df = data[test_mask]
        
        if len(train_df) < 1000 or len(test_df) < 100:
            test_idx += test_months
            continue
        
        X_train = train_df[feature_cols]
        y_train = train_df[label_col]
        X_test = test_df[feature_cols]
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 分出验证集
        val_size = int(len(X_train_scaled) * 0.1)
        X_tr, X_val = X_train_scaled[:-val_size], X_train_scaled[-val_size:]
        y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]
        
        train_set = lgb.Dataset(X_tr, y_tr)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)
        
        model = lgb.train(
            params, train_set,
            num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        
        pred = model.predict(X_test_scaled)
        
        result = pd.DataFrame({
            'date': test_df['date'].values,
            'ts_code': test_df['ts_code'].values,
            'pred': pred,
            'return': test_df['future_return_1d'].values,
            'market_trend': test_df['market_trend'].values,
            'high_vol': test_df['high_vol'].values,
        })
        
        all_results.append(result)
        
        ic = spearmanr(pred, test_df['future_return_1d'].values)[0]
        print(f"  {test_start} - {test_end}: IC={ic:.4f} (训练{len(train_df)}条 → 测试{len(test_df)}条)")
        
        test_idx += test_months
    
    return pd.concat(all_results, ignore_index=True)

def run_strategies(result):
    """运行策略回测"""
    print("\n" + "="*70)
    print("策略回测（按时间分离）")
    print("="*70)
    
    COST = 0.001
    
    result['date_str'] = result['date'].astype(str)
    result['date_dt'] = pd.to_datetime(result['date'])
    result['year'] = result['date_dt'].dt.year
    
    # IC 分析
    overall_ic = spearmanr(result['pred'], result['return'])[0]
    print(f"\n整体 IC: {overall_ic:.4f}")
    
    daily_ic = result.groupby('date_str').apply(
        lambda x: spearmanr(x['pred'], x['return'])[0] if len(x) > 5 else np.nan
    )
    print(f"日均 IC: {daily_ic.mean():.4f} ± {daily_ic.std():.4f}")
    print(f"IC > 0 比例: {(daily_ic > 0).mean():.1%}")
    print(f"IC IR: {daily_ic.mean() / (daily_ic.std() + 1e-10):.4f}")
    
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
    
    # 策略1: 择时 + Top N
    print("\n【策略1】择时 + Top N")
    for top_n in [10, 20, 30, 50]:
        daily_top = result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'pred')
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
    
    # 策略2: 全风控
    print("\n【策略2】全风控（择时 + 波动调仓 + 止损）")
    for top_n in [10, 20, 30]:
        daily_top = result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'pred')
        ).reset_index(drop=True)
        
        daily_agg = daily_top.groupby('date_str').agg({
            'return': 'mean',
            'market_trend': 'first',
            'high_vol': 'first',
        }).reset_index()
        daily_agg = daily_agg.sort_values('date_str').reset_index(drop=True)
        
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
            
            position = 0.5 if row['high_vol'] == 1 else 1.0
            ret = row['return'] * position - 2 * COST * position
            strategy_returns.append(ret)
            
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
    
    # 策略3: 多空对冲
    print("\n【策略3】多空对冲")
    for top_n in [10, 20, 30]:
        daily_long = result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'pred')['return'].mean()
        )
        daily_short = result.groupby('date_str').apply(
            lambda x: x.nsmallest(top_n, 'pred')['return'].mean()
        )
        
        daily_ls = daily_long - daily_short - 4 * COST
        
        calc_metrics(daily_ls, f"多空 Top/Bottom {top_n}")
    
    # 策略4: 纯多头
    print("\n【策略4】纯多头（对照组）")
    for top_n in [10, 20, 30]:
        daily_top = result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'pred')
        ).reset_index(drop=True)
        
        daily_agg = daily_top.groupby('date_str').agg({
            'return': 'mean',
        }).reset_index()
        
        daily_agg['strategy_return'] = daily_agg['return'] - 2 * COST
        
        calc_metrics(daily_agg['strategy_return'], f"纯多头 Top {top_n}")
        print_yearly(daily_agg, f"纯多头 Top {top_n}")

def main():
    print("="*70)
    print("LightGBM v9 - 按时间分离")
    print("="*70)
    print("\n分离方式（更接近实际场景）:")
    print("  - 滚动训练：每季度用过去 3 年数据")
    print("  - 回测：2022-2024（完全未见过的时间段）")
    print("  - 同一批股票，不同时间段")
    print()
    
    # 加载数据
    data = load_data(n_stocks=450)
    
    # 按时间分离的回测
    result = time_split_backtest(data)
    
    if result is not None:
        run_strategies(result)
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)

if __name__ == '__main__':
    main()
