#!/usr/bin/env python3
"""
LightGBM v8 - 参考微软 Qlib 最佳实践

关键改进（基于网上研究）：
1. 回归任务（预测收益率）而非分类
2. 超强正则化（L1/L2 > 100）
3. 更多 Alpha 因子（参考 Alpha158）
4. 特征标准化（ZScore）
5. 标签截面标准化
6. 训练/回测股票分离
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/Users/ronghui/Projects/willrone/data/parquet/stock_data')
np.random.seed(42)

def compute_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 Alpha 因子（参考 Qlib Alpha158）
    增加更多经过验证的因子
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_ = df['open']
    amount = df.get('amount', volume * close)  # 成交额
    
    # ========== KBAR 因子 ==========
    # K线形态
    df['KMID'] = (close - open_) / (open_ + 1e-10)
    df['KLEN'] = (high - low) / (open_ + 1e-10)
    df['KMID2'] = (close - open_) / (high - low + 1e-10)
    df['KUP'] = (high - np.maximum(open_, close)) / (high - low + 1e-10)
    df['KUP2'] = (high - np.maximum(open_, close)) / (open_ + 1e-10)
    df['KLOW'] = (np.minimum(open_, close) - low) / (high - low + 1e-10)
    df['KLOW2'] = (np.minimum(open_, close) - low) / (open_ + 1e-10)
    df['KSFT'] = (2 * close - high - low) / (open_ + 1e-10)
    df['KSFT2'] = (2 * close - high - low) / (high - low + 1e-10)
    
    # ========== 价格因子 ==========
    # 收益率（多周期）
    for d in [1, 2, 3, 5, 10, 20, 30, 60]:
        df[f'ROC_{d}'] = close.pct_change(d)
    
    # 价格相对位置
    for d in [5, 10, 20, 30, 60]:
        df[f'MAX_{d}'] = high.rolling(d).max()
        df[f'MIN_{d}'] = low.rolling(d).min()
        df[f'QTLU_{d}'] = (close - df[f'MIN_{d}']) / (df[f'MAX_{d}'] - df[f'MIN_{d}'] + 1e-10)
    
    # MA 因子
    for d in [5, 10, 20, 30, 60]:
        ma = close.rolling(d).mean()
        df[f'MA_{d}'] = ma
        df[f'MA_RATIO_{d}'] = close / (ma + 1e-10) - 1
    
    # MA 交叉
    df['MA_CROSS_5_20'] = df['MA_5'] / (df['MA_20'] + 1e-10) - 1
    df['MA_CROSS_10_30'] = df['MA_10'] / (df['MA_30'] + 1e-10) - 1
    df['MA_CROSS_20_60'] = df['MA_20'] / (df['MA_60'] + 1e-10) - 1
    
    # ========== 波动率因子 ==========
    returns = close.pct_change()
    for d in [5, 10, 20, 30, 60]:
        df[f'STD_{d}'] = returns.rolling(d).std()
    
    # 波动率比率
    df['VSTD_5_20'] = df['STD_5'] / (df['STD_20'] + 1e-10)
    df['VSTD_10_60'] = df['STD_10'] / (df['STD_60'] + 1e-10)
    
    # ========== 成交量因子 ==========
    for d in [5, 10, 20, 30, 60]:
        vol_ma = volume.rolling(d).mean()
        df[f'VOLUME_MA_{d}'] = vol_ma
        df[f'VOLUME_RATIO_{d}'] = volume / (vol_ma + 1e-10)
    
    # 量价背离
    df['CORR_CLOSE_VOL_5'] = close.rolling(5).corr(volume)
    df['CORR_CLOSE_VOL_10'] = close.rolling(10).corr(volume)
    df['CORR_CLOSE_VOL_20'] = close.rolling(20).corr(volume)
    
    # ========== 动量因子 ==========
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
    
    # ========== ATR 因子 ==========
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    for d in [6, 12, 24]:
        df[f'ATR_{d}'] = tr.rolling(d).mean() / (close + 1e-10)
    
    # ========== 趋势因子 ==========
    # 线性回归斜率
    for d in [5, 10, 20]:
        df[f'SLOPE_{d}'] = close.rolling(d).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] / (x.mean() + 1e-10) if len(x) == d else np.nan,
            raw=False
        )
    
    # ========== 标签：未来收益率 ==========
    df['future_return_1d'] = close.shift(-1) / close - 1
    df['future_return_2d'] = close.shift(-2) / close - 1
    
    return df

def add_cross_sectional_features(data: pd.DataFrame) -> pd.DataFrame:
    """添加截面特征和标签标准化"""
    
    # 截面排名特征
    rank_cols = ['ROC_5', 'ROC_20', 'STD_20', 'VOLUME_RATIO_20', 'RSI_12']
    for col in rank_cols:
        if col in data.columns:
            data[f'{col}_RANK'] = data.groupby('date')[col].rank(pct=True)
    
    # 截面相对强度
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
    
    # 标签：截面标准化的未来收益率（关键！）
    data['label'] = data.groupby('date')['future_return_1d'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )
    
    return data

def load_split_data(start_date='2017-01-01', end_date='2025-01-01'):
    """加载数据，分离训练股票和回测股票"""
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
    
    n_train = 300
    n_test = 150
    
    train_files = valid[:n_train]
    test_files = valid[n_train:n_train + n_test]
    
    print(f"训练股票: {len(train_files)} 只")
    print(f"回测股票: {len(test_files)} 只")
    
    # 加载训练股票
    print("\n加载训练股票...")
    train_data = []
    for f, _ in train_files:
        df = pd.read_parquet(f)
        df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
        df = compute_alpha_features(df)
        train_data.append(df)
    train_data = pd.concat(train_data, ignore_index=True)
    train_data = add_cross_sectional_features(train_data)
    print(f"训练数据: {len(train_data)} 条")
    
    # 加载回测股票
    print("\n加载回测股票...")
    test_data = []
    for f, _ in test_files:
        df = pd.read_parquet(f)
        df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
        df = compute_alpha_features(df)
        test_data.append(df)
    test_data = pd.concat(test_data, ignore_index=True)
    test_data = add_cross_sectional_features(test_data)
    print(f"回测数据: {len(test_data)} 条")
    
    return train_data, test_data

def get_feature_cols(data):
    """获取特征列（排除标签和元数据）"""
    exclude = ['date', 'ts_code', 'future_return_1d', 'future_return_2d', 
               'label', 'market_trend', 'high_vol', 'open', 'high', 'low', 
               'close', 'volume', 'amount']
    
    # 排除 MA 绝对值（只保留比率）
    exclude += [f'MA_{d}' for d in [5, 10, 20, 30, 60]]
    exclude += [f'MAX_{d}' for d in [5, 10, 20, 30, 60]]
    exclude += [f'MIN_{d}' for d in [5, 10, 20, 30, 60]]
    exclude += [f'VOLUME_MA_{d}' for d in [5, 10, 20, 30, 60]]
    
    feature_cols = [c for c in data.columns if c not in exclude and not c.startswith('_')]
    return feature_cols

def rolling_train_and_backtest(train_data, test_data):
    """
    滚动训练 + 回测
    使用回归任务 + 超强正则化（参考 Qlib）
    """
    feature_cols = get_feature_cols(train_data)
    label_col = 'label'
    
    # 清洗数据
    for df in [train_data, test_data]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    required = [c for c in feature_cols if c in train_data.columns] + [label_col, 'future_return_1d', 'market_trend', 'high_vol']
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
    
    train_months = 36
    test_months = 3
    
    start_test_idx = None
    for i, m in enumerate(all_months):
        if m >= pd.Period('2022-01'):
            start_test_idx = i
            break
    
    if start_test_idx is None or start_test_idx < train_months:
        print("数据不足")
        return None
    
    # Qlib 风格的 LightGBM 参数（回归 + 超强正则化）
    params = {
        'objective': 'regression',  # 回归任务！
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 128,
        'max_depth': 8,
        'min_data_in_leaf': 200,      # 更大的叶子最小样本
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 200.0,           # 超强 L1 正则（参考 Qlib）
        'lambda_l2': 500.0,           # 超强 L2 正则（参考 Qlib）
        'min_gain_to_split': 0.01,
        'verbose': -1,
        'seed': 42,
        'num_threads': 8,
    }
    
    print(f"\n模型参数（Qlib 风格）:")
    print(f"  任务: 回归（预测标准化收益率）")
    print(f"  L1 正则: {params['lambda_l1']}")
    print(f"  L2 正则: {params['lambda_l2']}")
    print(f"  min_data_in_leaf: {params['min_data_in_leaf']}")
    
    all_results = []
    
    print(f"\n滚动训练 (训练{train_months}月, 测试{test_months}月)")
    print("="*70)
    
    test_idx = start_test_idx
    while test_idx + test_months <= len(all_months):
        train_start = all_months[test_idx - train_months]
        train_end = all_months[test_idx - 1]
        test_start = all_months[test_idx]
        test_end = all_months[min(test_idx + test_months - 1, len(all_months) - 1)]
        
        train_mask = (train_data['year_month'] >= train_start) & (train_data['year_month'] <= train_end)
        train_df = train_data[train_mask]
        
        backtest_mask = (test_data['year_month'] >= test_start) & (test_data['year_month'] <= test_end)
        backtest_df = test_data[backtest_mask]
        
        if len(train_df) < 1000 or len(backtest_df) < 100:
            test_idx += test_months
            continue
        
        X_train = train_df[feature_cols]
        y_train = train_df[label_col]
        X_backtest = backtest_df[feature_cols]
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_backtest_scaled = scaler.transform(X_backtest)
        
        # 分出验证集用于早停
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
        
        pred = model.predict(X_backtest_scaled)
        
        result = pd.DataFrame({
            'date': backtest_df['date'].values,
            'ts_code': backtest_df['ts_code'].values,
            'pred': pred,  # 预测的标准化收益率
            'return': backtest_df['future_return_1d'].values,
            'market_trend': backtest_df['market_trend'].values,
            'high_vol': backtest_df['high_vol'].values,
        })
        
        all_results.append(result)
        
        # 计算 IC（信息系数）
        ic = spearmanr(pred, backtest_df['future_return_1d'].values)[0]
        print(f"  {test_start} - {test_end}: IC={ic:.4f} (训练{len(train_df)}条 → 回测{len(backtest_df)}条)")
        
        test_idx += test_months
    
    return pd.concat(all_results, ignore_index=True)

def run_strategies(result):
    """运行策略回测"""
    print("\n" + "="*70)
    print("策略回测（Qlib 风格：回归预测 + 股票分离）")
    print("="*70)
    
    COST = 0.001
    
    result['date_str'] = result['date'].astype(str)
    result['date_dt'] = pd.to_datetime(result['date'])
    result['year'] = result['date_dt'].dt.year
    
    # 计算整体 IC
    overall_ic = spearmanr(result['pred'], result['return'])[0]
    print(f"\n整体 IC（信息系数）: {overall_ic:.4f}")
    
    # 按日计算 IC
    daily_ic = result.groupby('date_str').apply(
        lambda x: spearmanr(x['pred'], x['return'])[0] if len(x) > 5 else np.nan
    )
    print(f"日均 IC: {daily_ic.mean():.4f} ± {daily_ic.std():.4f}")
    print(f"IC > 0 比例: {(daily_ic > 0).mean():.1%}")
    
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
    
    # 策略1: 基础择时（按预测值排序选股）
    print("\n【策略1】择时 + Top N（按预测值排序）")
    for top_n in [10, 15, 20, 30]:
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
    
    # 策略2: 多空对冲（做多 Top N，做空 Bottom N）
    print("\n【策略2】多空对冲（Top N - Bottom N）")
    for top_n in [10, 15, 20]:
        daily_long = result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'pred')['return'].mean()
        )
        daily_short = result.groupby('date_str').apply(
            lambda x: x.nsmallest(top_n, 'pred')['return'].mean()
        )
        
        # 多空收益 = 做多收益 - 做空收益
        daily_ls = daily_long - daily_short - 4 * COST  # 双边成本
        
        calc_metrics(daily_ls, f"多空 Top/Bottom {top_n}")
    
    # 策略3: 纯多头（无择时）
    print("\n【策略3】纯多头（无择时，对照组）")
    for top_n in [10, 15, 20]:
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
    print("LightGBM v8 - Qlib 风格优化")
    print("="*70)
    print("\n关键改进（基于网上研究）:")
    print("  1. 回归任务（预测收益率）而非分类")
    print("  2. 超强正则化（L1=200, L2=500）")
    print("  3. 更多 Alpha 因子（~80个）")
    print("  4. 特征 ZScore 标准化")
    print("  5. 标签截面标准化")
    print("  6. 训练/回测股票完全分离")
    print("  7. 早停防止过拟合")
    print()
    
    # 加载分离的数据
    train_data, test_data = load_split_data()
    
    # 滚动训练 + 回测
    result = rolling_train_and_backtest(train_data, test_data)
    
    if result is not None:
        run_strategies(result)
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)

if __name__ == '__main__':
    main()
