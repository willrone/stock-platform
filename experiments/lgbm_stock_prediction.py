#!/usr/bin/env python3
"""
LightGBM 股票预测模型
- 特征：技术指标（MA, RSI, MACD, Bollinger, Momentum等）
- 标签：未来5日收益率方向（涨/跌）
- 数据：2015-2025年，100只高流动性股票
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/Users/ronghui/Projects/willrone/data/parquet/stock_data')

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标特征"""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # 移动平均
    for window in [5, 10, 20, 60]:
        df[f'ma_{window}'] = close.rolling(window).mean()
        df[f'ma_ratio_{window}'] = close / df[f'ma_{window}']
    
    # 成交量移动平均
    df['vol_ma_5'] = volume.rolling(5).mean()
    df['vol_ma_20'] = volume.rolling(20).mean()
    df['vol_ratio'] = volume / df['vol_ma_20']
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 布林带
    df['bb_mid'] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # 动量指标
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = close.pct_change(period)
        df[f'volatility_{period}'] = close.pct_change().rolling(period).std()
    
    # 价格位置
    df['high_low_ratio'] = (close - low) / (high - low + 1e-10)
    df['price_range'] = (high - low) / close
    
    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr_14'] / close
    
    # 收益率
    df['return_1d'] = close.pct_change()
    df['return_5d'] = close.pct_change(5)
    
    # 标签：未来5日收益率方向
    df['future_return_5d'] = close.shift(-5) / close - 1
    df['label'] = (df['future_return_5d'] > 0).astype(int)
    
    return df

def load_and_prepare_data(n_stocks: int = 100, start_date: str = '2015-01-01', end_date: str = '2025-01-01'):
    """加载数据并准备特征"""
    print(f"加载数据: {n_stocks} 只股票, {start_date} - {end_date}")
    
    # 获取所有股票文件
    stock_files = list(DATA_DIR.glob('*.parquet'))
    print(f"找到 {len(stock_files)} 个股票文件")
    
    # 先读取所有股票，按平均成交量排序选择流动性最好的
    stock_volumes = []
    for f in stock_files:
        try:
            df = pd.read_parquet(f)
            df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
            if len(df) > 500:  # 至少500个交易日
                avg_vol = df['volume'].mean()
                stock_volumes.append((f, avg_vol, len(df)))
        except Exception as e:
            continue
    
    # 按成交量排序，选择前n_stocks只
    stock_volumes.sort(key=lambda x: x[1], reverse=True)
    selected_stocks = stock_volumes[:n_stocks]
    print(f"选择了 {len(selected_stocks)} 只高流动性股票")
    
    # 加载并计算特征
    all_data = []
    for f, avg_vol, n_days in selected_stocks:
        try:
            df = pd.read_parquet(f)
            df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
            df = compute_features(df)
            all_data.append(df)
        except Exception as e:
            print(f"处理 {f.name} 出错: {e}")
            continue
    
    # 合并所有数据
    data = pd.concat(all_data, ignore_index=True)
    print(f"总数据量: {len(data)} 条")
    
    return data

def train_model(data: pd.DataFrame):
    """训练 LightGBM 模型"""
    
    # 特征列
    feature_cols = [
        'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_60',
        'vol_ratio', 'rsi_14',
        'macd', 'macd_signal', 'macd_hist',
        'bb_width', 'bb_position',
        'momentum_5', 'momentum_10', 'momentum_20',
        'volatility_5', 'volatility_10', 'volatility_20',
        'high_low_ratio', 'price_range',
        'atr_ratio', 'return_1d', 'return_5d'
    ]
    
    # 删除缺失值
    df = data.dropna(subset=feature_cols + ['label'])
    print(f"有效数据: {len(df)} 条")
    
    X = df[feature_cols]
    y = df['label']
    
    # 按时间划分训练集和测试集（避免未来数据泄露）
    df_sorted = df.sort_values('date')
    split_idx = int(len(df_sorted) * 0.8)
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_test = test_df[feature_cols]
    y_test = test_df['label']
    
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    print(f"训练集时间范围: {train_df['date'].min()} - {train_df['date'].max()}")
    print(f"测试集时间范围: {test_df['date'].min()} - {test_df['date'].max()}")
    
    # LightGBM 参数
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # 训练
    print("\n开始训练...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # 预测
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 评估
    print("\n" + "="*50)
    print("模型评估结果")
    print("="*50)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['跌', '涨']))
    
    # 特征重要性
    print("\n特征重要性 (Top 10):")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.2f}")
    
    # 分析预测分布
    print("\n预测概率分布:")
    print(f"  均值: {y_pred_proba.mean():.4f}")
    print(f"  标准差: {y_pred_proba.std():.4f}")
    print(f"  预测为涨的比例: {y_pred.mean():.4f}")
    print(f"  实际涨的比例: {y_test.mean():.4f}")
    
    # 按概率分组看准确率
    print("\n按预测概率分组的准确率:")
    test_result = pd.DataFrame({
        'prob': y_pred_proba,
        'pred': y_pred,
        'actual': y_test.values
    })
    
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    test_result['prob_bin'] = pd.cut(test_result['prob'], bins=bins)
    
    for bin_range, group in test_result.groupby('prob_bin', observed=True):
        if len(group) > 0:
            acc = (group['pred'] == group['actual']).mean()
            print(f"  {bin_range}: 准确率 {acc:.4f}, 样本数 {len(group)}")
    
    return model, importance

if __name__ == '__main__':
    # 加载数据
    data = load_and_prepare_data(n_stocks=100, start_date='2015-01-01', end_date='2025-01-01')
    
    # 训练模型
    model, importance = train_model(data)
    
    # 保存模型
    model_path = Path('/Users/ronghui/Projects/willrone/experiments/lgbm_model.txt')
    model.save_model(str(model_path))
    print(f"\n模型已保存到: {model_path}")
