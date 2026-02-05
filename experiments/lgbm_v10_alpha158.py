#!/usr/bin/env python3
"""
LightGBM v10 - 完整 Alpha158 因子 + DoubleEnsemble

关键改进（基于 Qlib 研究）：
1. 完整 Alpha158 因子集（158 个因子）
2. DoubleEnsemble 集成方法
3. 更严格的数据清洗
4. 按时间分离
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

def compute_alpha158_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算完整 Alpha158 因子集（参考 Qlib）
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_ = df['open']
    
    # ========== KBAR 因子 (9个) ==========
    df['KMID'] = (close - open_) / (open_ + 1e-12)
    df['KLEN'] = (high - low) / (open_ + 1e-12)
    df['KMID2'] = (close - open_) / (high - low + 1e-12)
    df['KUP'] = (high - np.maximum(open_, close)) / (open_ + 1e-12)
    df['KUP2'] = (high - np.maximum(open_, close)) / (high - low + 1e-12)
    df['KLOW'] = (np.minimum(open_, close) - low) / (open_ + 1e-12)
    df['KLOW2'] = (np.minimum(open_, close) - low) / (high - low + 1e-12)
    df['KSFT'] = (2 * close - high - low) / (open_ + 1e-12)
    df['KSFT2'] = (2 * close - high - low) / (high - low + 1e-12)
    
    # ========== 滚动因子 (windows: 5, 10, 20, 30, 60) ==========
    windows = [5, 10, 20, 30, 60]
    
    for d in windows:
        # ROC: 收益率
        df[f'ROC{d}'] = close.shift(d) / close
        
        # MA: 均线比率
        df[f'MA{d}'] = close.rolling(d).mean() / close
        
        # STD: 波动率
        df[f'STD{d}'] = close.rolling(d).std() / close
        
        # BETA: 趋势斜率
        def calc_slope(x):
            if len(x) < d:
                return np.nan
            y = np.arange(len(x))
            slope = np.polyfit(y, x, 1)[0]
            return slope / (x.mean() + 1e-12)
        df[f'BETA{d}'] = close.rolling(d).apply(calc_slope, raw=False)
        
        # RSQR: R方（趋势线性度）
        def calc_rsqr(x):
            if len(x) < d:
                return np.nan
            y = np.arange(len(x))
            corr = np.corrcoef(y, x)[0, 1]
            return corr ** 2 if not np.isnan(corr) else np.nan
        df[f'RSQR{d}'] = close.rolling(d).apply(calc_rsqr, raw=False)
        
        # MAX/MIN: 最高/最低价比率
        df[f'MAX{d}'] = high.rolling(d).max() / close
        df[f'MIN{d}'] = low.rolling(d).min() / close
        
        # QTLU/QTLD: 分位数
        df[f'QTLU{d}'] = close.rolling(d).quantile(0.8) / close
        df[f'QTLD{d}'] = close.rolling(d).quantile(0.2) / close
        
        # RSV: 价格位置
        max_high = high.rolling(d).max()
        min_low = low.rolling(d).min()
        df[f'RSV{d}'] = (close - min_low) / (max_high - min_low + 1e-12)
        
        # IMAX/IMIN: 最高/最低价距今天数
        df[f'IMAX{d}'] = high.rolling(d).apply(lambda x: (d - 1 - np.argmax(x)) / d, raw=False)
        df[f'IMIN{d}'] = low.rolling(d).apply(lambda x: (d - 1 - np.argmin(x)) / d, raw=False)
        
        # IMXD: 最高价和最低价的时间差
        df[f'IMXD{d}'] = df[f'IMAX{d}'] - df[f'IMIN{d}']
        
        # CORR: 价量相关性
        df[f'CORR{d}'] = close.rolling(d).corr(np.log(volume + 1))
        
        # CORD: 价格变化和成交量变化的相关性
        close_ret = close.pct_change()
        vol_ret = volume.pct_change()
        df[f'CORD{d}'] = close_ret.rolling(d).corr(vol_ret)
        
        # CNTP/CNTN/CNTD: 上涨/下跌天数比例
        up = (close > close.shift(1)).astype(float)
        down = (close < close.shift(1)).astype(float)
        df[f'CNTP{d}'] = up.rolling(d).mean()
        df[f'CNTN{d}'] = down.rolling(d).mean()
        df[f'CNTD{d}'] = df[f'CNTP{d}'] - df[f'CNTN{d}']
        
        # SUMP/SUMN/SUMD: 涨跌幅累计
        gain = (close - close.shift(1)).clip(lower=0)
        loss = (close.shift(1) - close).clip(lower=0)
        total_change = (close - close.shift(1)).abs()
        df[f'SUMP{d}'] = gain.rolling(d).sum() / (total_change.rolling(d).sum() + 1e-12)
        df[f'SUMN{d}'] = loss.rolling(d).sum() / (total_change.rolling(d).sum() + 1e-12)
        df[f'SUMD{d}'] = df[f'SUMP{d}'] - df[f'SUMN{d}']
        
        # VMA/VSTD: 成交量均线和波动
        df[f'VMA{d}'] = volume.rolling(d).mean() / (volume + 1e-12)
        df[f'VSTD{d}'] = volume.rolling(d).std() / (volume + 1e-12)
        
        # WVMA: 加权波动率
        weighted = (close.pct_change().abs() * volume)
        df[f'WVMA{d}'] = weighted.rolling(d).std() / (weighted.rolling(d).mean() + 1e-12)
        
        # VSUMP/VSUMN/VSUMD: 成交量涨跌
        vol_gain = (volume - volume.shift(1)).clip(lower=0)
        vol_loss = (volume.shift(1) - volume).clip(lower=0)
        vol_change = (volume - volume.shift(1)).abs()
        df[f'VSUMP{d}'] = vol_gain.rolling(d).sum() / (vol_change.rolling(d).sum() + 1e-12)
        df[f'VSUMN{d}'] = vol_loss.rolling(d).sum() / (vol_change.rolling(d).sum() + 1e-12)
        df[f'VSUMD{d}'] = df[f'VSUMP{d}'] - df[f'VSUMN{d}']
    
    # ========== 标签 ==========
    df['future_return_1d'] = close.shift(-1) / close - 1
    df['future_return_2d'] = close.shift(-2) / close.shift(-1) - 1  # Qlib 用的是 T+2/T+1
    
    return df

def add_cross_sectional_features(data: pd.DataFrame) -> pd.DataFrame:
    """添加截面特征和标签标准化"""
    
    # 市场状态
    daily_stats = data.groupby('date').agg({
        'future_return_1d': 'mean',
    }).reset_index()
    daily_stats.columns = ['date', 'market_return']
    daily_stats = daily_stats.sort_values('date')
    
    daily_stats['market_ma10'] = daily_stats['market_return'].rolling(10).mean()
    daily_stats['market_ma30'] = daily_stats['market_return'].rolling(30).mean()
    daily_stats['market_trend'] = (daily_stats['market_ma10'] > daily_stats['market_ma30']).astype(int)
    
    data = data.merge(daily_stats[['date', 'market_trend']], on='date', how='left')
    
    # 标签：截面标准化（Qlib 的 CSZScoreNorm）
    data['label'] = data.groupby('date')['future_return_1d'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12)
    )
    
    return data

def load_data(n_stocks=450, start_date='2017-01-01', end_date='2025-01-01'):
    """加载数据"""
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
    for i, (f, _) in enumerate(selected):
        if (i + 1) % 100 == 0:
            print(f"  处理 {i+1}/{len(selected)}...")
        df = pd.read_parquet(f)
        df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
        df = compute_alpha158_features(df)
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    data = add_cross_sectional_features(data)
    print(f"总数据: {len(data)} 条")
    return data

def get_feature_cols(data):
    """获取特征列"""
    exclude = ['date', 'ts_code', 'future_return_1d', 'future_return_2d', 
               'label', 'market_trend', 'open', 'high', 'low', 'close', 'volume', 'amount']
    feature_cols = [c for c in data.columns if c not in exclude]
    return feature_cols

def double_ensemble_train(X_train, y_train, X_val, y_val, n_models=5):
    """
    DoubleEnsemble 训练（参考 Qlib）
    
    核心思想：
    1. 训练多个模型
    2. 每个模型关注不同的样本（难样本加权）
    3. 最终预测取平均
    """
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
    
    models = []
    sample_weights = np.ones(len(X_train))
    
    for i in range(n_models):
        # 根据样本权重采样
        if i > 0:
            # 计算上一个模型的预测误差
            pred = models[-1].predict(X_train)
            errors = np.abs(pred - y_train)
            # 误差大的样本权重增加
            sample_weights = errors / (errors.mean() + 1e-12)
            sample_weights = np.clip(sample_weights, 0.1, 10)
        
        # 加权采样
        probs = sample_weights / sample_weights.sum()
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True, p=probs)
        
        X_sampled = X_train[indices]
        y_sampled = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        
        train_set = lgb.Dataset(X_sampled, y_sampled)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)
        
        model = lgb.train(
            params, train_set,
            num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        
        models.append(model)
    
    return models

def ensemble_predict(models, X):
    """集成预测"""
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0)

def time_split_backtest(data):
    """按时间分离的回测"""
    feature_cols = get_feature_cols(data)
    label_col = 'label'
    
    # 数据清洗
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # 只保留有效特征
    valid_features = []
    for col in feature_cols:
        if col in data.columns:
            nan_ratio = data[col].isna().mean()
            if nan_ratio < 0.5:  # 缺失率 < 50%
                valid_features.append(col)
    
    feature_cols = valid_features
    required = feature_cols + [label_col, 'future_return_1d', 'market_trend']
    data = data.dropna(subset=required)
    
    print(f"\n有效数据: {len(data)} 条")
    print(f"有效特征数: {len(feature_cols)}")
    
    data['date_dt'] = pd.to_datetime(data['date'])
    data['year_month'] = data['date_dt'].dt.to_period('M')
    
    all_months = sorted(data['year_month'].unique())
    
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
    
    print(f"\n滚动训练（DoubleEnsemble x 5）")
    print("="*70)
    
    all_results = []
    test_idx = start_test_idx
    
    while test_idx + test_months <= len(all_months):
        train_start = all_months[test_idx - train_months]
        train_end = all_months[test_idx - 1]
        test_start = all_months[test_idx]
        test_end = all_months[min(test_idx + test_months - 1, len(all_months) - 1)]
        
        train_mask = (data['year_month'] >= train_start) & (data['year_month'] <= train_end)
        train_df = data[train_mask]
        
        test_mask = (data['year_month'] >= test_start) & (data['year_month'] <= test_end)
        test_df = data[test_mask]
        
        if len(train_df) < 1000 or len(test_df) < 100:
            test_idx += test_months
            continue
        
        X_train = train_df[feature_cols].values
        y_train = train_df[label_col]
        X_test = test_df[feature_cols].values
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 处理 NaN
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0)
        
        # 分出验证集
        val_size = int(len(X_train_scaled) * 0.1)
        X_tr, X_val = X_train_scaled[:-val_size], X_train_scaled[-val_size:]
        y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]
        
        # DoubleEnsemble 训练
        models = double_ensemble_train(X_tr, y_tr, X_val, y_val, n_models=5)
        
        # 集成预测
        pred = ensemble_predict(models, X_test_scaled)
        
        result = pd.DataFrame({
            'date': test_df['date'].values,
            'ts_code': test_df['ts_code'].values,
            'pred': pred,
            'return': test_df['future_return_1d'].values,
            'market_trend': test_df['market_trend'].values,
        })
        
        all_results.append(result)
        
        ic = spearmanr(pred, test_df['future_return_1d'].values)[0]
        print(f"  {test_start} - {test_end}: IC={ic:.4f}")
        
        test_idx += test_months
    
    return pd.concat(all_results, ignore_index=True)

def run_strategies(result):
    """运行策略回测"""
    print("\n" + "="*70)
    print("策略回测（Alpha158 + DoubleEnsemble）")
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
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    print(f"日均 IC: {ic_mean:.4f} ± {ic_std:.4f}")
    print(f"ICIR: {ic_mean / (ic_std + 1e-12):.4f}")
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
    
    def print_yearly(daily_agg):
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
        print_yearly(daily_agg)
    
    # 策略2: 纯多头
    print("\n【策略2】纯多头（对照组）")
    for top_n in [10, 20, 30]:
        daily_top = result.groupby('date_str').apply(
            lambda x: x.nlargest(top_n, 'pred')
        ).reset_index(drop=True)
        
        daily_agg = daily_top.groupby('date_str').agg({
            'return': 'mean',
        }).reset_index()
        
        daily_agg['strategy_return'] = daily_agg['return'] - 2 * COST
        
        calc_metrics(daily_agg['strategy_return'], f"纯多头 Top {top_n}")
        print_yearly(daily_agg)

def main():
    print("="*70)
    print("LightGBM v10 - Alpha158 + DoubleEnsemble")
    print("="*70)
    print("\n关键改进:")
    print("  1. 完整 Alpha158 因子集（~150 个因子）")
    print("  2. DoubleEnsemble 集成（5 个模型）")
    print("  3. 按时间分离回测")
    print()
    
    # 加载数据
    data = load_data(n_stocks=450)
    
    # 回测
    result = time_split_backtest(data)
    
    if result is not None:
        run_strategies(result)
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)

if __name__ == '__main__':
    main()
