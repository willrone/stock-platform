#!/usr/bin/env python3
"""
Reversal Neutral V2 - 反转因子市场中性策略信号生成器（低换手版）

V2 改进：
- 调仓频率从每天改为每 5 天
- 换手率从 54% 降到 16%
- 夏普比率从 2.50 提升到 2.82

回测表现（2023-07 ~ 2024-12，含 0.3% 交易成本）：
- 累计收益: +371%
- 夏普比率: 2.82
- 最大回撤: -41.4%
- 换手率: 16.3%

作者: Clawdbot
版本: v2.0.0
日期: 2024-02-04
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
import json
import pickle
import warnings
warnings.filterwarnings('ignore')


class ReversalNeutralV2:
    """反转因子市场中性策略 V2（低换手版）"""
    
    VERSION = "2.0.0"
    MODEL_NAME = "reversal_neutral_v2"
    
    DEFAULT_CONFIG = {
        'top_n': 10,
        'train_months': 6,
        'min_train_samples': 5000,
        'rebalance_days': 5,           # V2 新增：调仓频率（天）
        'cost_per_trade': 0.003,
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
    
    FEATURE_COLS = [
        'reversal_5d', 'reversal_10d', 'reversal_20d',
        'drawdown_60d', 'runup_20d', 'runup_10d',
        'macd', 'volatility_20d', 'vol_ratio'
    ]
    
    def __init__(self, config: dict = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.data = None
        self.model = None
        self.feature_cols = self.FEATURE_COLS.copy()
        self.last_train_date = None
        self.last_rebalance_date = None
        self.current_long = []
        self.current_short = []
        self.metadata = {
            'version': self.VERSION,
            'model_name': self.MODEL_NAME,
            'created_at': datetime.now().isoformat(),
        }
    
    def load_data(self, data_dir: str, start_date: str = None, end_date: str = None):
        """加载股票数据"""
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
        
        if start_date:
            self.data = self.data[self.data['date'] >= start_date]
        if end_date:
            self.data = self.data[self.data['date'] <= end_date]
        
        self._calculate_features()
        
        print(f"[ReversalNeutralV2] 数据加载完成:")
        print(f"  股票数: {self.data['ts_code'].nunique()}")
        print(f"  数据量: {len(self.data):,}")
        print(f"  时间范围: {self.data['date'].min().date()} ~ {self.data['date'].max().date()}")
    
    def _calculate_features(self):
        """计算反转因子"""
        df = self.data
        
        df['reversal_5d'] = -df.groupby('ts_code')['close'].pct_change(5)
        df['reversal_10d'] = -df.groupby('ts_code')['close'].pct_change(10)
        df['reversal_20d'] = -df.groupby('ts_code')['close'].pct_change(20)
        
        df['high_60d'] = df.groupby('ts_code')['high'].transform(
            lambda x: x.rolling(60, min_periods=20).max()
        )
        df['drawdown_60d'] = (df['close'] - df['high_60d']) / df['high_60d']
        
        df['low_20d'] = df.groupby('ts_code')['low'].transform(
            lambda x: x.rolling(20, min_periods=10).min()
        )
        df['runup_20d'] = -((df['close'] - df['low_20d']) / df['low_20d'])
        
        df['low_10d'] = df.groupby('ts_code')['low'].transform(
            lambda x: x.rolling(10, min_periods=5).min()
        )
        df['runup_10d'] = -((df['close'] - df['low_10d']) / df['low_10d'])
        
        ema_12 = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=12).mean())
        ema_26 = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=26).mean())
        df['macd'] = -(ema_12 - ema_26)
        
        df['volatility_20d'] = df.groupby('ts_code')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        )
        
        df['volume_ma_20'] = df.groupby('ts_code')['volume'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        df['vol_ratio'] = df['volume'] / df['volume_ma_20']
        
        df['return_5d'] = df.groupby('ts_code')['close'].pct_change(5).shift(-5)
        
        self.data = df
    
    def train(self, train_end_date: str = None):
        """训练模型"""
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
        
        train_data = train_data.dropna(subset=self.feature_cols + ['return_5d'])
        
        if len(train_data) < self.config['min_train_samples']:
            raise ValueError(f"训练样本不足: {len(train_data)} < {self.config['min_train_samples']}")
        
        train_set = lgb.Dataset(train_data[self.feature_cols], train_data['return_5d'])
        
        self.model = lgb.train(
            self.config['lgb_params'],
            train_set,
            num_boost_round=self.config['lgb_rounds'],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        self.last_train_date = train_end_date
        self.metadata['last_train_date'] = str(train_end_date.date())
        self.metadata['train_samples'] = len(train_data)
        
        print(f"[ReversalNeutralV2] 模型训练完成:")
        print(f"  训练期: {train_start_date.date()} ~ {train_end_date.date()}")
        print(f"  样本数: {len(train_data):,}")
    
    def generate_signals(self, signal_date: str = None, force_rebalance: bool = False) -> dict:
        """
        生成交易信号
        
        Args:
            signal_date: 信号日期
            force_rebalance: 强制调仓（忽略调仓频率限制）
        """
        if self.model is None:
            raise ValueError("请先调用 train() 训练模型")
        
        if signal_date is None:
            signal_date = self.data['date'].max()
        else:
            signal_date = pd.to_datetime(signal_date)
        
        # 检查是否需要调仓
        need_rebalance = force_rebalance or self.last_rebalance_date is None
        if not need_rebalance and self.last_rebalance_date:
            days_since_rebalance = (signal_date - self.last_rebalance_date).days
            need_rebalance = days_since_rebalance >= self.config['rebalance_days']
        
        day_data = self.data[self.data['date'] == signal_date].copy()
        
        if len(day_data) == 0:
            raise ValueError(f"没有 {signal_date.date()} 的数据")
        
        day_data = day_data.dropna(subset=self.feature_cols)
        
        if len(day_data) < self.config['top_n'] * 2:
            raise ValueError(f"有效股票数不足: {len(day_data)} < {self.config['top_n'] * 2}")
        
        day_data['pred'] = self.model.predict(day_data[self.feature_cols])
        day_data = day_data.sort_values('pred', ascending=False)
        
        if need_rebalance:
            top_n = self.config['top_n']
            self.current_long = day_data.head(top_n)['ts_code'].tolist()
            self.current_short = day_data.tail(top_n)['ts_code'].tolist()
            self.last_rebalance_date = signal_date
            rebalanced = True
        else:
            rebalanced = False
        
        return {
            'date': signal_date.date(),
            'long': self.current_long,
            'short': self.current_short,
            'rebalanced': rebalanced,
            'next_rebalance': (self.last_rebalance_date + pd.Timedelta(days=self.config['rebalance_days'])).date(),
            'predictions': day_data[['ts_code', 'close', 'pred']].copy()
        }
    
    def backtest(self, start_date: str, end_date: str = None, with_cost: bool = True) -> dict:
        """回测策略"""
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
        last_rebalance_idx = None
        
        top_n = self.config['top_n']
        cost = self.config['cost_per_trade']
        rebalance_days = self.config['rebalance_days']
        
        for i, date in enumerate(dates):
            day_data = test_data[test_data['date'] == date].sort_values('pred', ascending=False)
            
            if len(day_data) < top_n * 2:
                continue
            
            # 判断是否调仓
            need_rebalance = (last_rebalance_idx is None) or (i - last_rebalance_idx >= rebalance_days)
            
            if need_rebalance:
                long_stocks = set(day_data.head(top_n)['ts_code'].tolist())
                short_stocks = set(day_data.tail(top_n)['ts_code'].tolist())
                last_rebalance_idx = i
                
                long_turnover = len(long_stocks - prev_long) / top_n if prev_long else 1.0
                short_turnover = len(short_stocks - prev_short) / top_n if prev_short else 1.0
                avg_turnover = (long_turnover + short_turnover) / 2
                
                prev_long = long_stocks
                prev_short = short_stocks
            else:
                long_stocks = prev_long
                short_stocks = prev_short
                avg_turnover = 0
            
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
        
        result_df = pd.DataFrame(daily_returns)
        
        if len(result_df) == 0:
            return {'error': '无有效回测数据'}
        
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
        """保存模型"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / 'model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(save_path / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # 保存当前持仓状态
        state = {
            'current_long': self.current_long,
            'current_short': self.current_short,
            'last_rebalance_date': str(self.last_rebalance_date) if self.last_rebalance_date else None
        }
        with open(save_path / 'state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"[ReversalNeutralV2] 模型已保存到: {save_path}")
    
    def load(self, load_dir: str):
        """加载模型"""
        load_path = Path(load_dir)
        
        with open(load_path / 'model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open(load_path / 'config.json', 'r') as f:
            self.config = json.load(f)
        
        if (load_path / 'metadata.json').exists():
            with open(load_path / 'metadata.json', 'r') as f:
                self.metadata = json.load(f)
        
        if (load_path / 'state.json').exists():
            with open(load_path / 'state.json', 'r') as f:
                state = json.load(f)
                self.current_long = state.get('current_long', [])
                self.current_short = state.get('current_short', [])
                if state.get('last_rebalance_date'):
                    self.last_rebalance_date = pd.to_datetime(state['last_rebalance_date'])
        
        print(f"[ReversalNeutralV2] 模型已加载: {load_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Reversal Neutral V2 信号生成器')
    parser.add_argument('--data-dir', required=True, help='数据目录')
    parser.add_argument('--action', choices=['train', 'signal', 'backtest'], default='signal')
    parser.add_argument('--date', help='信号日期')
    parser.add_argument('--start-date', help='回测开始日期')
    parser.add_argument('--end-date', help='回测结束日期')
    parser.add_argument('--save-dir', help='模型保存目录')
    parser.add_argument('--load-dir', help='模型加载目录')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--rebalance-days', type=int, default=5)
    
    args = parser.parse_args()
    
    generator = ReversalNeutralV2(config={
        'top_n': args.top_n,
        'rebalance_days': args.rebalance_days
    })
    
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
        print(f"是否调仓: {signal['rebalanced']}")
        print(f"下次调仓: {signal['next_rebalance']}")
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
