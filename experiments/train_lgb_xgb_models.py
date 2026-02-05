#!/usr/bin/env python3
"""
训练 LightGBM + XGBoost 集成模型并保存

基于 test_lgb_xgb_ensemble.py，训练模型并保存到 data/models/ 目录
供 ml_ensemble_strategy.py 加载使用
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path('/Users/ronghui/Documents/GitHub/willrone/data/parquet/stock_data')
MODEL_DIR = Path('/Users/ronghui/Documents/GitHub/willrone/data/models')

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标特征"""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_ = df['open']
    
    # 基础收益
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'return_{period}d'] = close.pct_change(period)
    
    # 动量因子
    df['momentum_short'] = df['return_5d'] - df['return_10d']
    df['momentum_long'] = df['return_10d'] - df['return_20d']
    df['momentum_reversal'] = -df['return_1d']
    
    returns = close.pct_change()
    for period in [5, 10, 20]:
        up_days = (returns > 0).rolling(period).sum()
        df[f'momentum_strength_{period}'] = up_days / period
    
    # 移动平均
    for window in [5, 10, 20, 60]:
        ma = close.rolling(window).mean()
        df[f'ma_ratio_{window}'] = close / ma - 1
        df[f'ma_slope_{window}'] = ma.pct_change(5)
    
    df['ma_5'] = close.rolling(5).mean()
    df['ma_10'] = close.rolling(10).mean()
    df['ma_20'] = close.rolling(20).mean()
    
    df['ma_alignment'] = (
        (df['ma_5'] > df['ma_10']).astype(int) +
        (df['ma_10'] > df['ma_20']).astype(int)
    )
    
    # 波动率因子
    for window in [5, 20, 60]:
        df[f'volatility_{window}'] = returns.rolling(window).std()
    
    df['vol_regime'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)
    df['volatility_skew'] = returns.rolling(20).skew()
    
    # 成交量特征
    vol_ma20 = volume.rolling(20).mean()
    vol_ma5 = volume.rolling(5).mean()
    
    df['vol_ratio'] = volume / (vol_ma20 + 1)
    df['vol_ma_ratio'] = vol_ma5 / (vol_ma20 + 1)
    df['vol_std'] = volume.rolling(20).std() / (vol_ma20 + 1)
    
    df['price_up'] = (close > close.shift(1)).astype(int)
    df['vol_up'] = (volume > volume.shift(1)).astype(int)
    df['vol_price_diverge'] = (df['price_up'] != df['vol_up']).astype(int)
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    for period in [6, 14]:
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    df['rsi_diff'] = df['rsi_6'] - df['rsi_14']
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_hist_slope'] = df['macd_hist'].diff(3)
    
    # 布林带
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_position'] = (close - bb_mid) / (2 * bb_std + 1e-10)
    df['bb_width'] = 4 * bb_std / (bb_mid + 1e-10)
    
    # 价格形态
    df['body'] = (close - open_) / (open_ + 1e-10)
    df['wick_upper'] = (high - np.maximum(close, open_)) / (high - low + 1e-10)
    df['wick_lower'] = (np.minimum(close, open_) - low) / (high - low + 1e-10)
    df['range_pct'] = (high - low) / (close + 1e-10)
    
    df['consecutive_up'] = (close > close.shift(1)).rolling(5).sum()
    df['consecutive_down'] = (close < close.shift(1)).rolling(5).sum()
    
    # 价格位置
    for window in [20, 60]:
        high_n = high.rolling(window).max()
        low_n = low.rolling(window).min()
        df[f'price_pos_{window}'] = (close - low_n) / (high_n - low_n + 1e-10)
        df[f'dist_high_{window}'] = (high_n - close) / (close + 1e-10)
        df[f'dist_low_{window}'] = (close - low_n) / (close + 1e-10)
    
    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / (close + 1e-10)
    
    # 趋势强度
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-10)
    df['di_diff'] = plus_di - minus_di
    df['adx'] = (plus_di - minus_di).abs().rolling(14).mean()
    
    # 未来收益
    df['future_return'] = close.shift(-1) / close - 1
    df['label'] = (df['future_return'] > 0.003).astype(int)
    
    df = df.dropna()
    return df

def compute_cross_sectional_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """计算截面特征（每日排名）"""
    df = df_all.copy()
    
    for col in ['return_1d', 'return_5d', 'return_20d', 'volume', 'volatility_20']:
        if col in df.columns:
            df[f'{col}_rank'] = df.groupby('date')[col].rank(pct=True)
    
    df['market_up_ratio'] = df.groupby('date')['return_1d'].transform(
        lambda x: (x > 0).sum() / len(x)
    )
    
    market_return_5d = df.groupby('date')['return_5d'].transform('mean')
    market_return_20d = df.groupby('date')['return_20d'].transform('mean')
    
    df['relative_strength_5d'] = df['return_5d'] - market_return_5d
    df['relative_strength_20d'] = df['return_20d'] - market_return_20d
    df['relative_momentum'] = df['relative_strength_5d'] - df['relative_strength_20d']
    
    return df

def load_data(n_stocks=350, start_date='2018-01-01', end_date='2025-01-01'):
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
    print(f"总数据: {len(data)} 条")
    
    print("计算截面特征...")
    data = compute_cross_sectional_features(data)
    
    print(f"标签分布: {data['label'].value_counts().to_dict()}")
    
    return data

def get_feature_cols():
    """特征列（与 ml_ensemble_strategy.py 保持一致）"""
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
        'atr_pct',
        'di_diff', 'adx',
    ]

def backtest(result_df, top_n=5, transaction_cost=0.001):
    """回测策略"""
    result_df = result_df.copy()
    result_df['date_str'] = result_df['date'].astype(str)
    
    daily_top_list = []
    for date_str, group in result_df.groupby('date_str'):
        top_stocks = group.nlargest(top_n, 'prob')
        daily_top_list.append(top_stocks)
    daily_top = pd.concat(daily_top_list, ignore_index=True)
    
    daily_returns = daily_top.groupby('date_str')['return'].mean() - transaction_cost
    
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
    
    return {
        'sharpe': sharpe,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_return': total_return,
    }

def main():
    print("="*60)
    print("训练 LightGBM + XGBoost 集成模型")
    print("="*60)
    
    # 确保模型目录存在
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    df = load_data(n_stocks=350, start_date='2018-01-01', end_date='2025-01-01')
    feature_cols = get_feature_cols()
    
    # 时间划分
    train_end = '2023-07-01'
    val_end = '2024-01-01'
    
    train = df[df['date'] < train_end]
    val = df[(df['date'] >= train_end) & (df['date'] < val_end)]
    test = df[df['date'] >= val_end]
    
    print(f"\n训练集: {len(train)} ({train['date'].min().date()} - {train['date'].max().date()})")
    print(f"验证集: {len(val)} ({val['date'].min().date()} - {val['date'].max().date()})")
    print(f"测试集: {len(test)} ({test['date'].min().date()} - {test['date'].max().date()})")
    
    X_train, y_train = train[feature_cols], train['label']
    X_val, y_val = val[feature_cols], val['label']
    X_test, y_test = test[feature_cols], test['label']
    
    # ========== 训练 LightGBM ==========
    print("\n" + "="*60)
    print("训练 LightGBM...")
    print("="*60)
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 45,
        'max_depth': 6,
        'learning_rate': 0.015,
        'feature_fraction': 0.65,
        'bagging_fraction': 0.65,
        'bagging_freq': 5,
        'min_child_samples': 100,
        'reg_alpha': 0.6,
        'reg_lambda': 0.6,
        'verbose': -1,
        'seed': SEED,
    }
    
    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)
    
    lgb_model = lgb.train(
        lgb_params, train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )
    
    pred_lgb_val = lgb_model.predict(X_val)
    pred_lgb_test = lgb_model.predict(X_test)
    lgb_val_auc = roc_auc_score(y_val, pred_lgb_val)
    lgb_test_auc = roc_auc_score(y_test, pred_lgb_test)
    print(f"LightGBM 验证 AUC: {lgb_val_auc:.4f}")
    print(f"LightGBM 测试 AUC: {lgb_test_auc:.4f}")
    
    # ========== 训练 XGBoost ==========
    print("\n" + "="*60)
    print("训练 XGBoost...")
    print("="*60)
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 5,
        'learning_rate': 0.015,
        'subsample': 0.65,
        'colsample_bytree': 0.65,
        'min_child_weight': 100,
        'reg_alpha': 0.6,
        'reg_lambda': 0.6,
        'seed': SEED,
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    xgb_model = xgb.train(
        xgb_params, dtrain,
        num_boost_round=2000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=200
    )
    
    pred_xgb_val = xgb_model.predict(dval)
    pred_xgb_test = xgb_model.predict(dtest)
    xgb_val_auc = roc_auc_score(y_val, pred_xgb_val)
    xgb_test_auc = roc_auc_score(y_test, pred_xgb_test)
    print(f"XGBoost 验证 AUC: {xgb_val_auc:.4f}")
    print(f"XGBoost 测试 AUC: {xgb_test_auc:.4f}")
    
    # ========== 集成评估 ==========
    print("\n" + "="*60)
    print("集成模型评估（等权重 50/50）")
    print("="*60)
    
    # 等权重集成
    pred_ensemble_val = 0.5 * pred_lgb_val + 0.5 * pred_xgb_val
    pred_ensemble_test = 0.5 * pred_lgb_test + 0.5 * pred_xgb_test
    
    ensemble_val_auc = roc_auc_score(y_val, pred_ensemble_val)
    ensemble_test_auc = roc_auc_score(y_test, pred_ensemble_test)
    print(f"集成 验证 AUC: {ensemble_val_auc:.4f}")
    print(f"集成 测试 AUC: {ensemble_test_auc:.4f}")
    
    # 回测评估
    test_df = pd.DataFrame({
        'date': test['date'].values,
        'ts_code': test['ts_code'].values,
        'label': y_test.values,
        'return': test['future_return'].values,
        'prob': pred_ensemble_test,
    })
    
    bt_results = backtest(test_df, top_n=5)
    print(f"\n回测结果（Top 5 选股）:")
    print(f"  夏普比率: {bt_results['sharpe']:.2f}")
    print(f"  年化收益: {bt_results['annual_return']*100:.1f}%")
    print(f"  最大回撤: {bt_results['max_drawdown']*100:.1f}%")
    print(f"  胜率: {bt_results['win_rate']*100:.1f}%")
    print(f"  总收益: {bt_results['total_return']*100:.1f}%")
    
    # ========== 保存模型 ==========
    print("\n" + "="*60)
    print("保存模型...")
    print("="*60)
    
    lgb_path = MODEL_DIR / "lgb_model.pkl"
    xgb_path = MODEL_DIR / "xgb_model.pkl"
    
    with open(lgb_path, 'wb') as f:
        pickle.dump(lgb_model, f)
    print(f"LightGBM 模型已保存: {lgb_path}")
    
    with open(xgb_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"XGBoost 模型已保存: {xgb_path}")
    
    # 保存模型元数据
    metadata = {
        'train_date_range': [str(train['date'].min().date()), str(train['date'].max().date())],
        'val_date_range': [str(val['date'].min().date()), str(val['date'].max().date())],
        'test_date_range': [str(test['date'].min().date()), str(test['date'].max().date())],
        'n_stocks': 350,
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'lgb_params': lgb_params,
        'xgb_params': xgb_params,
        'metrics': {
            'lgb_val_auc': lgb_val_auc,
            'lgb_test_auc': lgb_test_auc,
            'xgb_val_auc': xgb_val_auc,
            'xgb_test_auc': xgb_test_auc,
            'ensemble_val_auc': ensemble_val_auc,
            'ensemble_test_auc': ensemble_test_auc,
            'sharpe': bt_results['sharpe'],
            'annual_return': bt_results['annual_return'],
            'max_drawdown': bt_results['max_drawdown'],
            'win_rate': bt_results['win_rate'],
            'total_return': bt_results['total_return'],
        },
        'ensemble_weights': {'lgb': 0.5, 'xgb': 0.5},
    }
    
    metadata_path = MODEL_DIR / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"模型元数据已保存: {metadata_path}")
    
    # 打印文件大小
    print(f"\n模型文件大小:")
    print(f"  lgb_model.pkl: {lgb_path.stat().st_size / 1024:.1f} KB")
    print(f"  xgb_model.pkl: {xgb_path.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    
    return {
        'lgb_model': lgb_model,
        'xgb_model': xgb_model,
        'metrics': metadata['metrics'],
    }

if __name__ == '__main__':
    main()
