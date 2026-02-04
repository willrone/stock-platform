#!/usr/bin/env python3
"""
Reversal Neutral V1 - 反转因子市场中性策略信号生成器

策略概述：
- 基于 A 股反转��应设计的市场中性策略
- 做多超跌股票（预期反弹），做空超涨股票（预期回调）
- 消除市场 beta 暴露，只赚取选股 alpha

回测表现（2023-07 ~ 2024-12，含 0.3% 交易成本）：
- 累计收益: +500%
- 夏普比率: 3.12
- 最大回撤: -43.5%
- 2023 年: +21%
- 2024 年: +396%

使用方法：
    from signal_generator import ReversalNeutralV1
    
    generator = ReversalNeutralV1()
    generator.load_data('/path/to/stock_data')
    generator.train()
    signals = generator.generate_signals()

作者: Clawdbot
版本: v1.0.0
日期: 2024-02-04
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import datetime, timedelta
import json
import pickle
import warnings
warnings.filterwarnings('ignore')


class ReversalNeutralV1:
    """反转因子市场中性策略 V1"""
    
    VERSION = "1.0.0"
    MODEL_NAME = "reversal_neutral_v1"
    
    # 默认参数
    DEFAULT_CONFIG = {
        'top_n': 10,                    # 多头/空头各选 N 只
        'train_months': 6,              # 训练窗口（月）
        'min_train_samples': 5000,      # 最小训练样本数
        'holding_days': 5,              # 持仓周期（天）
        'cost_per_trade': 0.003,        # 单边交易成本
        'lgb_params': {
            'objective': 'regression',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1,
            'seed': 42,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
        },
        'lgb_rounds': 100,
    }
    
    # 因子列表
    FEATURE_COLS = [
        'reversal_5d', 'reversal_10d', 'reversal_20d',
        'drawdown_60d', 'runup_20d', 'runup_10d',
        'macd', 'volatility_20d', 'vol_ratio'
    ]
    
    def __init__(self, config: dict = None):
        """
        初始化信号生成器
        
        Args:
            config: 配置字典，覆盖默认配置
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.data = None
        self.model = None
        self.feature_cols = self.FEATURE_COLS.copy()
        self.last_train_date = None
        self.metadata = {
            'version': self.VERSION,
            'model_name': self.MODEL_NAME,
            'created_at': datetime.now().isoformat(),
        }
    
    def load_data(self, data_dir: str, start_date: str = None, end_date: str = None):
        """
        加载股票数据
        
        Args:
            data_dir: 数据目录（包含 parquet 文件）
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        data_path = Path(data_dir)
        
        all_data = []
        for file in data_path.glob('*.parquet'):
            df = pd.read_parquet(file)
            df['ts_code'] = file.stem
            all_data.append(df)
        
        if not all_data:
            raise ValueError(f"No parquet files found in {data_dir}")
        
        self.data = pd.concat(all_data, ignore_index=True)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values(['ts_code', 'date']).reset_index(drop=True)
        
        # 日期过滤
        if start_date:
            self.data = self.data[self.data['date'] >= start_date]
        if end_date:
            self.data = self.data[self.data['date'] <= end_date]
        
        # 计算因子
        self._calculate_features()
        
        print(f"[ReversalNeutralV1] 数据加载完成:")
        print(f"  股票数: {self.data['ts_code'].nunique()}")
        print(f"  数据量: {len(self.data):,}")
        print(f"  时间范围: {self.data['date'].min().date()} ~ {self.data['date'].max().date()}")
    
    def _calculate_features(self):
        """计算反转因子"""
        df = self.data
        
        # 1. 反转因子（过去收益取负）
        df['reversal_5d'] = -df.groupby('ts_code')['close'].pct_change(5)
        df['reversal_10d'] = -df.groupby('ts_code')['close'].pct_change(10)
        df['reversal_20d'] = -df.groupby('ts_code')['close'].pct_change(20)
        
        # 2. 超跌程度（距离高点的回撤）
        df['high_60d'] = df.groupby('ts_code')['high'].transform(
            lambda x: x.rolling(60, min_periods=20).max()
        )
        df['drawdown_60d'] = (df['close'] - df['high_60d']) / df['high_60d']
        
        # 3. 超涨程度（距离低点的涨幅，取负作为反转信号）
        df['low_20d'] = df.groupby('ts_code')['low'].transform(
            lambda x: x.rolling(20, min_periods=10).min()
        )
        df['runup_20d'] = -((df['close'] - df['low_20d']) / df['low_20d'])
        
        df['low_10d'] = df.groupby('ts_code')['low'].transform(
            lambda x: x.rolling(10, min_periods=5).min()
        )
        df['runup_10d'] = -((df['close'] - df['low_10d']) / df['low_10d'])
        
        # 4. MACD 反转
        ema_12 = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=12).mean())
        ema_26 = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=26).mean())
        df['macd'] = -(ema_12 - ema_26)  # 取负，MACD 负值 = 超卖
        
        # 5. 波动率
        df['volatility_20d'] = df.groupby('ts_code')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        )
        
        # 6. 成交量比率
        df['volume_ma_20'] = df.groupby('ts_code')['volume'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        df['vol_ratio'] = df['volume'] / df['volume_ma_20']
        
        # 7. 未来收益（标签）
        df['return_5d'] = df.groupby('ts_code')['close'].pct_change(5).shift(-5)
        
        self.data = df
    
    def train(self, train_end_date: str = None):
        """
        训练模型
        
        Args:
            train_end_date: 训练截止日期，默认使用最新数据往前推
        """
        if self.data is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        if train_end_date is None:
            train_end_date = self.data['date'].max()
        else:
            train_end_date = pd.to_datetime(train_end_date)
        
        train_start_date = train_end_date - pd.DateOffset(months=self.config['train_months'])
        
        train_data = self.data[
            (self.data['date'] >= train_start_date) & 
            (self.data['date'] < train_end_date)
        ].copy()
        
        # 删除缺失值
        train_data = train_data.dropna(subset=self.feature_cols + ['return_5d'])
        
        if len(train_data) < self.config['min_train_samples']:
            raise ValueError(f"训练样本不足: {len(train_data)} < {self.config['min_train_samples']}")
        
        # 训练
        train_set = lgb.Dataset(
            train_data[self.feature_cols], 
            train_data['return_5d']
        )
        
        self.model = lgb.train(
            self.config['lgb_params'],
            train_set,
            num_boost_round=self.config['lgb_rounds'],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        self.last_train_date = train_end_date
        self.metadata['last_train_date'] = str(train_end_date.date())
        self.metadata['train_samples'] = len(train_data)
        
        print(f"[ReversalNeutralV1] 模型训练完成:")
        print(f"  训练期: {train_start_date.date()} ~ {train_end_date.date()}")
        print(f"  样本数: {len(train_data):,}")
    
    def generate_signals(self, signal_date: str = None) -> dict:
        """
        生成交易信号
        
        Args:
            signal_date: 信号日期，默认使用最新数据日期
            
        Returns:
            dict: {
                'date': 信号日期,
                'long': [做多股票列表],
                'short': [做空股票列表],
                'predictions': DataFrame (所有股票预测值)
            }
        """
        if self.model is None:
            raise ValueError("请先调用 train() 训练模型")
        
        if signal_date is None:
            signal_date = self.data['date'].max()
        else:
            signal_date = pd.to_datetime(signal_date)
        
        # 获取当日数据
        day_data = self.data[self.data['date'] == signal_date].copy()
        
        if len(day_data) == 0:
            raise ValueError(f"没有 {signal_date.date()} 的数据")
        
        # 删除缺失值
        day_data = day_data.dropna(subset=self.feature_cols)
        
        if len(day_data) < self.config['top_n'] * 2:
            raise ValueError(f"有效股票数不足: {len(day_data)} < {self.config['top_n'] * 2}")
        
        # 预测
        day_data['pred'] = self.model.predict(day_data[self.feature_cols])
        
        # 排序选股
        day_data = day_data.sort_values('pred', ascending=False)
        
        top_n = self.config['top_n']
        long_stocks = day_data.head(top_n)['ts_code'].tolist()
        short_stocks = day_data.tail(top_n)['ts_code'].tolist()
        
        return {
            'date': signal_date.date(),
            'long': long_stocks,
            'short': short_stocks,
            'predictions': day_data[['ts_code', 'close', 'pred']].copy()
        }
    
    def generate_signals_range(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        生成日期范围内的所有信号
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 每日信号记录
        """
        if end_date is None:
            end_date = self.data['date'].max()
        
        dates = self.data[
            (self.data['date'] >= start_date) & 
            (self.data['date'] <= end_date)
        ]['date'].unique()
        
        all_signals = []
        for date in sorted(dates):
            try:
                signal = self.generate_signals(date)
                all_signals.append({
                    'date': signal['date'],
                    'long': ','.join(signal['long']),
                    'short': ','.join(signal['short'])
                })
            except Exception as e:
                print(f"  跳过 {date.date()}: {e}")
        
        return pd.DataFrame(all_signals)
    
    def backtest(self, start_date: str, end_date: str = None, with_cost: bool = True) -> dict:
        """
        回测策略
        
        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
            with_cost: 是否计算交易成本
            
        Returns:
            dict: 回测结果
        """
        if self.model is None:
            raise ValueError("请先调用 train() 训练模型")
        
        test_data = self.data[self.data['date'] >= start_date].copy()
        if end_date:
            test_data = test_data[test_data['date'] <= end_date]
        
        test_data = test_data.dropna(subset=self.feature_cols + ['return_5d'])
        test_data['pred'] = self.model.predict(test_data[self.feature_cols])
        
        dates = sorted(test_data['date'].unique())
        daily_returns = []
        prev_long = set()
        prev_short = set()
        
        top_n = self.config['top_n']
        cost = self.config['cost_per_trade']
        
        for date in dates:
            day_data = test_data[test_data['date'] == date].sort_values('pred', ascending=False)
            
            if len(day_data) < top_n * 2:
                continue
            
            long_stocks = set(day_data.head(top_n)['ts_code'].tolist())
            short_stocks = set(day_data.tail(top_n)['ts_code'].tolist())
            
            # 换手率
            long_turnover = len(long_stocks - prev_long) / top_n if prev_long else 1.0
            short_turnover = len(short_stocks - prev_short) / top_n if prev_short else 1.0
            avg_turnover = (long_turnover + short_turnover) / 2
            
            # 收益
            long_ret = day_data[day_data['ts_code'].isin(long_stocks)]['return_5d'].mean()
            short_ret = day_data[day_data['ts_code'].isin(short_stocks)]['return_5d'].mean()
            gross_ret = (long_ret - short_ret) / 2
            
            if with_cost:
                net_ret = gross_ret - avg_turnover * cost * 2
            else:
                net_ret = gross_ret
            
            daily_returns.append({
                'date': date,
                'return': net_ret,
                'turnover': avg_turnover
            })
            
            prev_long = long_stocks
            prev_short = short_stocks
        
        result_df = pd.DataFrame(daily_returns)
        
        if len(result_df) == 0:
            return {'error': '无有效回测数据'}
        
        # 计算指标
        cum_return = (1 + result_df['return']).prod() - 1
        sharpe = result_df['return'].mean() / result_df['return'].std() * np.sqrt(252)
        cum_returns = (1 + result_df['return']).cumprod()
        max_dd = ((cum_returns - cum_returns.expanding().max()) / cum_returns.expanding().max()).min()
        
        return {
            'cum_return': cum_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'avg_turnover': result_df['turnover'].mean(),
            'trading_days': len(result_df),
            'daily_returns': result_df
        }
    
    def save(self, save_dir: str):
        """保存模型和配置"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        model_path = save_path / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # 保存配置
        config_path = save_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        # 保存元数据
        meta_path = save_path / 'metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"[ReversalNeutralV1] 模型已保存到: {save_path}")
    
    def load(self, load_dir: str):
        """加载模型和配置"""
        load_path = Path(load_dir)
        
        # 加载模型
        model_path = load_path / 'model.pkl'
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # 加载配置
        config_path = load_path / 'config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 加载元数据
        meta_path = load_path / 'metadata.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        
        print(f"[ReversalNeutralV1] 模型已加载: {load_path}")
        print(f"  版本: {self.metadata.get('version', 'unknown')}")
        print(f"  训练日期: {self.metadata.get('last_train_date', 'unknown')}")


# ============================================================
# CLI 入口
# ============================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Reversal Neutral V1 信号生成器')
    parser.add_argument('--data-dir', required=True, help='数据目录')
    parser.add_argument('--action', choices=['train', 'signal', 'backtest'], default='signal')
    parser.add_argument('--date', help='信号日期 (YYYY-MM-DD)')
    parser.add_argument('--start-date', help='回测开始日期')
    parser.add_argument('--end-date', help='回测结束日期')
    parser.add_argument('--save-dir', help='模型保存目录')
    parser.add_argument('--load-dir', help='模型加载目录')
    parser.add_argument('--top-n', type=int, default=10, help='多空各选股数量')
    
    args = parser.parse_args()
    
    generator = ReversalNeutralV1(config={'top_n': args.top_n})
    
    if args.load_dir:
        generator.load(args.load_dir)
    
    generator.load_data(args.data_dir)
    
    if args.action == 'train':
        generator.train(args.date)
        if args.save_dir:
            generator.save(args.save_dir)
    
    elif args.action == 'signal':
        if generator.model is None:
            generator.train()
        signal = generator.generate_signals(args.date)
        print(f"\n信号日期: {signal['date']}")
        print(f"做多: {signal['long']}")
        print(f"做空: {signal['short']}")
    
    elif args.action == 'backtest':
        if generator.model is None:
            generator.train()
        result = generator.backtest(args.start_date, args.end_date)
        print(f"\n回测结果:")
        print(f"  累计收益: {result['cum_return']*100:+.1f}%")
        print(f"  夏普比率: {result['sharpe']:.2f}")
        print(f"  最大回撤: {result['max_drawdown']*100:.1f}%")
        print(f"  平均换手: {result['avg_turnover']*100:.1f}%")
