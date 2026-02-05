#!/usr/bin/env python3
"""
LightGBM 股���预测模型 v2
改进：
1. 放宽股票筛选，使用更多数据
2. 增加更多特征（趋势、交叉信号等）
3. 使用未来1日收益（更短期，信号更强）
4. 调整类别平衡
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/Users/ronghui/Documents/GitHub/willrone/data/parquet/stock_data')

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标特征 - 增强版"""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_ = df['open']
    
    # ========== 价格特征 ==========
    # 移动平均
    for window in [5, 10, 20, 60]:
        df[f'ma_{window}'] = close.rolling(window).mean()
        df[f'ma_ratio_{window}'] = close / df[f'ma_{window}'] - 1  # 偏离度
    
    # MA 交叉信号
    df['ma_cross_5_20'] = (df['ma_5'] > df['ma_20']).astype(int)
    df['ma_cross_10_60'] = (df['ma_10'] > df['ma_60']).astype(int)
    df['ma_trend'] = (df['ma_5'] > df['ma_10']).astype(int) + (df['ma_10'] > df['ma_20']).astype(int)
    
    # EMA
    for span in [12, 26]:
        df[f'ema_{span}'] = close.ewm(span=span, adjust=False).mean()
        df[f'ema_ratio_{span}'] = close / df[f'ema_{span}'] - 1
    
    # ========== 成交量特征 ==========
    df['vol_ma_5'] = volume.rolling(5).mean()
    df['vol_ma_20'] = volume.rolling(20).mean()
    df['vol_ratio_5'] = volume / (df['vol_ma_5'] + 1)
    df['vol_ratio_20'] = volume / (df['vol_ma_20'] + 1)
    df['vol_change'] = volume.pct_change()
    df['vol_trend'] = (df['vol_ma_5'] > df['vol_ma_20']).astype(int)
    
    # 量价配合
    df['price_vol_corr'] = close.pct_change().rolling(10).corr(volume.pct_change())
    
    # ========== RSI ==========
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    for period in [6, 14, 24]:
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # RSI 超买超卖
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    
    # ========== MACD ==========
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
    df['macd_hist_change'] = df['macd_hist'].diff()
    
    # ========== 布林带 ==========
    df['bb_mid'] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-10)
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
    
    # ========== 动量 ==========
    for period in [1, 3, 5, 10, 20]:
        df[f'return_{period}d'] = close.pct_change(period)
    
    # 动量变化
    df['momentum_accel'] = df['return_5d'] - df['return_5d'].shift(5)
    
    # ========== 波动率 ==========
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = close.pct_change().rolling(period).std()
    
    df['volatility_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)
    
    # ========== K线形态 ==========
    df['body'] = close - open_
    df['body_ratio'] = df['body'] / (high - low + 1e-10)
    df['upper_shadow'] = high - np.maximum(close, open_)
    df['lower_shadow'] = np.minimum(close, open_) - low
    df['shadow_ratio'] = (df['upper_shadow'] - df['lower_shadow']) / (high - low + 1e-10)
    
    # 连续涨跌
    df['up_day'] = (close > open_).astype(int)
    df['consecutive_up'] = df['up_day'].rolling(5).sum()
    
    # ========== ATR ==========
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr_14'] / (close + 1e-10)
    
    # ========== 价格位置 ==========
    df['high_20d'] = high.rolling(20).max()
    df['low_20d'] = low.rolling(20).min()
    df['price_position_20d'] = (close - df['low_20d']) / (df['high_20d'] - df['low_20d'] + 1e-10)
    
    df['high_60d'] = high.rolling(60).max()
    df['low_60d'] = low.rolling(60).min()
    df['price_position_60d'] = (close - df['low_60d']) / (df['high_60d'] - df['low_60d'] + 1e-10)
    
    # ========== 标签 ==========
    # 未来1日收益
    df['future_return_1d'] = close.shift(-1) / close - 1
    df['label'] = (df['future_return_1d'] > 0).astype(int)
    
    # 未来3日收益（备用）
    df['future_return_3d'] = close.shift(-3) / close - 1
    df['label_3d'] = (df['future_return_3d'] > 0).astype(int)
    
    return df

def load_and_prepare_data(n_stocks: int = 300, start_date: str = '2018-01-01', end_date: str = '2025-01-01'):
    """加载数据并准备特征"""
    print(f"加载数据: 最多 {n_stocks} 只股票, {start_date} - {end_date}")
    
    stock_files = list(DATA_DIR.glob('*.parquet'))
    print(f"找到 {len(stock_files)} 个股票文件")
    
    # 筛选有足够数据的股票
    valid_stocks = []
    for f in stock_files:
        try:
            df = pd.read_parquet(f)
            df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
            if len(df) >= 200:  # 至少200个交易日
                avg_vol = df['volume'].mean()
                valid_stocks.append((f, avg_vol, len(df)))
        except:
            continue
    
    # 按成交量排序
    valid_stocks.sort(key=lambda x: x[1], reverse=True)
    selected = valid_stocks[:n_stocks]
    print(f"选择了 {len(selected)} 只股票")
    
    # 加载并计算特征
    all_data = []
    for f, _, _ in selected:
        try:
            df = pd.read_parquet(f)
            df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
            df = compute_features(df)
            all_data.append(df)
        except Exception as e:
            continue
    
    data = pd.concat(all_data, ignore_index=True)
    print(f"总数据量: {len(data)} 条")
    
    return data

def train_model(data: pd.DataFrame):
    """训练 LightGBM 模型"""
    
    # 特征列
    feature_cols = [
        # MA 相关
        'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_60',
        'ma_cross_5_20', 'ma_cross_10_60', 'ma_trend',
        'ema_ratio_12', 'ema_ratio_26',
        # 成交量
        'vol_ratio_5', 'vol_ratio_20', 'vol_change', 'vol_trend',
        # RSI
        'rsi_6', 'rsi_14', 'rsi_24', 'rsi_oversold', 'rsi_overbought',
        # MACD
        'macd', 'macd_signal', 'macd_hist', 'macd_cross', 'macd_hist_change',
        # 布林带
        'bb_width', 'bb_position', 'bb_squeeze',
        # 动量
        'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
        'momentum_accel',
        # 波动率
        'volatility_5', 'volatility_10', 'volatility_20', 'volatility_ratio',
        # K线
        'body_ratio', 'shadow_ratio', 'consecutive_up',
        # ATR
        'atr_ratio',
        # 价格位置
        'price_position_20d', 'price_position_60d',
    ]
    
    # 删除缺失值和无穷值
    df = data.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ['label'])
    print(f"有效数据: {len(df)} 条")
    
    # 按时间排序并划分
    df = df.sort_values('date').reset_index(drop=True)
    split_date = '2024-01-01'
    
    train_df = df[df['date'] < split_date]
    test_df = df[df['date'] >= split_date]
    
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_test = test_df[feature_cols]
    y_test = test_df['label']
    
    print(f"\n训练集: {len(X_train)} 条 ({train_df['date'].min().date()} - {train_df['date'].max().date()})")
    print(f"测试集: {len(X_test)} 条 ({test_df['date'].min().date()} - {test_df['date'].max().date()})")
    print(f"训练集正样本比例: {y_train.mean():.4f}")
    print(f"测试集正样本比例: {y_test.mean():.4f}")
    
    # 计算���别权重
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"类别权重: {scale_pos_weight:.4f}")
    
    # LightGBM 参数
    params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.02,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'is_unbalance': True,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    print("\n开始训练...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # 预测
    y_pred_proba = model.predict(X_test)
    
    # 找最优阈值
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0.3, 0.7, 0.02):
        y_pred_temp = (y_pred_proba > threshold).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred_temp)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n最优阈值: {best_threshold:.2f}")
    y_pred = (y_pred_proba > best_threshold).astype(int)
    
    # 评估
    print("\n" + "="*60)
    print("模型评估结果")
    print("="*60)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['跌', '涨']))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:")
    print(f"           预测跌  预测涨")
    print(f"  实际跌    {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"  实际涨    {cm[1,0]:5d}   {cm[1,1]:5d}")
    
    # 特征重要性
    print("\n特征重要性 (Top 15):")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.2f}")
    
    # 预测分布
    print("\n预测概率分布:")
    print(f"  均值: {y_pred_proba.mean():.4f}")
    print(f"  标准差: {y_pred_proba.std():.4f}")
    print(f"  最小值: {y_pred_proba.min():.4f}")
    print(f"  最大值: {y_pred_proba.max():.4f}")
    
    # 按概率分组
    print("\n按预测概率分组的表现:")
    test_result = pd.DataFrame({
        'date': test_df['date'].values,
        'prob': y_pred_proba,
        'actual': y_test.values,
        'future_return': test_df['future_return_1d'].values
    })
    
    bins = [0, 0.35, 0.45, 0.55, 0.65, 1.0]
    labels = ['<0.35', '0.35-0.45', '0.45-0.55', '0.55-0.65', '>0.65']
    test_result['prob_bin'] = pd.cut(test_result['prob'], bins=bins, labels=labels)
    
    print(f"{'概率区间':^12} {'样本数':>8} {'实际涨比例':>12} {'平均收益':>12}")
    print("-" * 50)
    for bin_label in labels:
        group = test_result[test_result['prob_bin'] == bin_label]
        if len(group) > 0:
            actual_up = group['actual'].mean()
            avg_return = group['future_return'].mean() * 100
            print(f"{bin_label:^12} {len(group):>8} {actual_up:>12.2%} {avg_return:>11.2f}%")
    
    # 策略回测
    print("\n" + "="*60)
    print("简单策略回测 (预测概率 > 0.55 时买入)")
    print("="*60)
    
    high_prob = test_result[test_result['prob'] > 0.55]
    if len(high_prob) > 0:
        win_rate = (high_prob['future_return'] > 0).mean()
        avg_return = high_prob['future_return'].mean() * 100
        total_return = (1 + high_prob['future_return']).prod() - 1
        
        print(f"交易次数: {len(high_prob)}")
        print(f"胜率: {win_rate:.2%}")
        print(f"平均单次收益: {avg_return:.3f}%")
        print(f"累计收益: {total_return*100:.2f}%")
    else:
        print("没有高概率信号")
    
    return model, importance

if __name__ == '__main__':
    data = load_and_prepare_data(n_stocks=300, start_date='2018-01-01', end_date='2025-01-01')
    model, importance = train_model(data)
    
    # 保存模型
    model_path = Path('/Users/ronghui/Documents/GitHub/willrone/experiments/lgbm_model_v2.txt')
    model.save_model(str(model_path))
    print(f"\n模型已保存到: {model_path}")
