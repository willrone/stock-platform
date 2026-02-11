#!/usr/bin/env python3
"""
测试不同集成权重方案
"""
import sys
sys.path.insert(0, '/Users/ronghui/Projects/willrone/experiments')

from ensemble_v1 import EnsembleModel, load_and_prepare_data
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def test_weight_schemes():
    """测试不同权重方案"""
    print("="*60)
    print("加载数据...")
    print("="*60)
    
    df, feature_cols = load_and_prepare_data()
    
    # 时间划分
    train_end = '2023-07-01'
    val_end = '2024-01-01'
    
    train = df[df['date'] < train_end]
    val = df[(df['date'] >= train_end) & (df['date'] < val_end)]
    test = df[df['date'] >= val_end]
    
    X_train, y_train = train[feature_cols], train['label']
    X_val, y_val = val[feature_cols], val['label']
    X_test, y_test = test[feature_cols], test['label']
    
    # 训练模型
    print("\n训练三个基模型...")
    ensemble = EnsembleModel()
    ensemble.train_all(X_train, y_train, X_val, y_val)
    
    # 获取各模型预测
    _, pred_lgb, pred_xgb, pred_cat = ensemble.predict(X_test)
    
    # 测试不同权重方案
    weight_schemes = {
        'v1_optimized': [0.215, 0.200, 0.584],  # 原始优化结果
        'equal': [1/3, 1/3, 1/3],  # 等权重
        'lgb_xgb_only': [0.5, 0.5, 0.0],  # 只用 LGB + XGB
        'xgb_heavy': [0.25, 0.50, 0.25],  # XGB 为主（AUC 最高）
        'lgb_heavy': [0.50, 0.30, 0.20],  # LGB 为主
        'balanced': [0.35, 0.40, 0.25],  # 平衡方案
    }
    
    results = []
    
    print("\n" + "="*60)
    print("测试不同权重方案")
    print("="*60)
    
    test_df = pd.DataFrame({
        'date': test['date'].values,
        'ts_code': test['ts_code'].values,
        'label': y_test.values,
        'return': test['future_return'].values,
        'pred_lgb': pred_lgb,
        'pred_xgb': pred_xgb,
        'pred_cat': pred_cat,
    })
    
    TRANSACTION_COST = 0.001
    
    for name, weights in weight_schemes.items():
        # 计算集成预测
        pred = weights[0] * pred_lgb + weights[1] * pred_xgb + weights[2] * pred_cat
        auc = roc_auc_score(y_test, pred)
        
        # 策略回测 (Top 5)
        test_df['prob'] = pred
        test_df['date_str'] = test_df['date'].astype(str)
        
        daily_top_list = []
        for date_str, group in test_df.groupby('date_str'):
            top_stocks = group.nlargest(5, 'prob')
            daily_top_list.append(top_stocks)
        daily_top = pd.concat(daily_top_list, ignore_index=True)
        
        daily_returns = daily_top.groupby('date_str')['return'].mean() - TRANSACTION_COST
        
        total_days = len(daily_returns)
        win_days = (daily_returns > 0).sum()
        win_rate = win_days / total_days
        avg_daily = daily_returns.mean()
        
        cumulative = (1 + daily_returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / total_days) - 1
        
        # 计算回撤
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 夏普比率
        sharpe = avg_daily / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        results.append({
            'scheme': name,
            'weights': f"[{weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}]",
            'auc': auc,
            'sharpe': sharpe,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
        })
        
        print(f"\n{name}:")
        print(f"  权重: LGB={weights[0]:.2f}, XGB={weights[1]:.2f}, CAT={weights[2]:.2f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  夏普比率: {sharpe:.2f}")
        print(f"  年化收益: {annual_return*100:.1f}%")
        print(f"  最大回撤: {max_drawdown*100:.1f}%")
        print(f"  日胜率: {win_rate*100:.1f}%")
    
    # 汇总表格
    print("\n" + "="*60)
    print("汇总对比")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # 找最佳方案
    print("\n" + "="*60)
    print("最佳方案推荐")
    print("="*60)
    
    # 综合评分：AUC 权重 0.3，夏普 0.4，回撤 0.3
    results_df['score'] = (
        0.3 * (results_df['auc'] - 0.55) / 0.02 +  # AUC 归一化
        0.4 * results_df['sharpe'] / 10 +  # 夏普归一化
        0.3 * (1 + results_df['max_drawdown']) / 0.3  # 回撤归一化（越小越好）
    )
    
    best = results_df.loc[results_df['score'].idxmax()]
    print(f"\n综合最佳: {best['scheme']}")
    print(f"  权重: {best['weights']}")
    print(f"  AUC: {best['auc']:.4f}")
    print(f"  夏普: {best['sharpe']:.2f}")
    print(f"  回撤: {best['max_drawdown']*100:.1f}%")
    
    # 保存结果
    results_df.to_csv('/Users/ronghui/Projects/willrone/experiments/weight_comparison.csv', index=False)
    print("\n结果已保存到 weight_comparison.csv")

if __name__ == '__main__':
    test_weight_schemes()
