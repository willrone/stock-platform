"""
V5 策略回测脚本 - 市场状态自适应版
"""
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "strategies" / "reversal_neutral_v5"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from signal_generator import ReversalNeutralV5, MarketRegime


def load_stock_data(data_dir: Path) -> dict:
    """加载所有股票数据"""
    stock_data = {}
    parquet_dir = data_dir / "parquet" / "stock_data"
    
    for f in parquet_dir.glob("*.parquet"):
        # 文件名格式: 000001_SZ.parquet -> 000001.SZ
        ts_code = f.stem.replace('_', '.')
        try:
            df = pd.read_parquet(f)
            # 兼容不同列名
            date_col = 'trade_date' if 'trade_date' in df.columns else 'date'
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
                df = df.set_index(date_col)
                # 兼容 volume/vol 列名
                if 'volume' in df.columns and 'vol' not in df.columns:
                    df['vol'] = df['volume']
                stock_data[ts_code] = df
        except Exception as e:
            pass
    
    return stock_data


def prepare_features(stock_data: dict, date: pd.Timestamp, lookback: int = 20) -> pd.DataFrame:
    """准备特征数据"""
    features_list = []
    
    for ts_code, df in stock_data.items():
        try:
            # 获取历史数据
            hist = df[df.index < date].tail(lookback + 10)
            if len(hist) < lookback:
                continue
            
            # 计算特征
            close = hist['close']
            volume = hist['vol'] if 'vol' in hist.columns else hist.get('volume', pd.Series([1]*len(hist)))
            
            # 反转因子
            ret_5d = close.pct_change(5).iloc[-1]
            ret_10d = close.pct_change(10).iloc[-1]
            ret_20d = close.pct_change(20).iloc[-1]
            
            # 波动率
            volatility = close.pct_change().std()
            
            # 成交量变化
            vol_ratio = volume.iloc[-5:].mean() / (volume.iloc[-20:].mean() + 1e-8)
            
            features_list.append({
                'ts_code': ts_code,
                'ret_5d': ret_5d,
                'ret_10d': ret_10d,
                'ret_20d': ret_20d,
                'volatility': volatility,
                'vol_ratio': vol_ratio,
            })
        except:
            continue
    
    if not features_list:
        return pd.DataFrame()
    
    return pd.DataFrame(features_list).set_index('ts_code')


def simple_predict(features: pd.DataFrame) -> pd.Series:
    """
    简单预测模型：反转因子
    预测下期收益 = -过去收益（超跌反弹）
    """
    if features.empty:
        return pd.Series()
    
    # 综合反转得分
    score = -(features['ret_5d'] * 0.5 + features['ret_10d'] * 0.3 + features['ret_20d'] * 0.2)
    
    # 波动率惩罚（高波动股票降权）
    vol_penalty = features['volatility'].clip(upper=0.05) / 0.05
    score = score * (1 - vol_penalty * 0.3)
    
    return score


def get_market_index(stock_data: dict, end_date: pd.Timestamp, lookback: int = 60) -> pd.Series:
    """
    计算市场指数（等权平均）
    """
    all_returns = []
    
    for ts_code, df in stock_data.items():
        try:
            hist = df[df.index <= end_date].tail(lookback)
            if len(hist) >= lookback:
                # 归一化价格
                normalized = hist['close'] / hist['close'].iloc[0]
                all_returns.append(normalized)
        except:
            continue
    
    if not all_returns:
        return pd.Series()
    
    # 等权平均
    market_index = pd.concat(all_returns, axis=1).mean(axis=1)
    return market_index


def run_backtest(
    stock_data: dict,
    start_date: str,
    end_date: str,
    strategy: ReversalNeutralV5,
    initial_capital: float = 1000000,
) -> pd.DataFrame:
    """运行回测"""
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # 获取所有交易日
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index.tolist())
    
    trading_days = sorted([d for d in all_dates if start <= d <= end])
    
    print(f"回测期间: {start_date} ~ {end_date}")
    print(f"交易日数: {len(trading_days)}")
    
    # 回测记录
    records = []
    capital = initial_capital
    positions = {}  # ts_code -> shares
    last_signal = None
    
    for i, date in enumerate(trading_days):
        if i < 30:  # 需要足够历史数据
            continue
        
        # 1. 计算当日收益（基于昨日持仓）
        daily_pnl = 0
        if positions:
            for ts_code, shares in positions.items():
                if ts_code in stock_data:
                    df = stock_data[ts_code]
                    if date in df.index:
                        today_close = df.loc[date, 'close']
                        # 找前一个交易日
                        prev_dates = df.index[df.index < date]
                        if len(prev_dates) > 0:
                            prev_close = df.loc[prev_dates[-1], 'close']
                            pnl = shares * (today_close - prev_close)
                            daily_pnl += pnl
        
        daily_return = daily_pnl / capital if capital > 0 else 0
        capital += daily_pnl
        
        # 2. 准备特征和预测
        features = prepare_features(stock_data, date)
        if features.empty:
            continue
        
        predictions = simple_predict(features)
        if predictions.empty:
            continue
        
        # 3. 获取市场指数
        market_index = get_market_index(stock_data, date, lookback=60)
        if market_index.empty:
            continue
        
        # 4. 生成信号
        signal = strategy.generate_signals(
            date=date.strftime('%Y-%m-%d'),
            predictions=predictions,
            market_prices=market_index,
            portfolio_return=daily_return if i > 30 else None,
        )
        
        # 5. 更新持仓
        if signal.long_stocks != (last_signal.long_stocks if last_signal else []):
            # 调仓
            positions = {}
            if signal.position_scale > 0 and signal.long_stocks:
                position_value = capital * signal.position_scale / 2  # 多头一半
                per_stock = position_value / len(signal.long_stocks)
                
                for ts_code in signal.long_stocks:
                    if ts_code in stock_data:
                        df = stock_data[ts_code]
                        if date in df.index:
                            price = df.loc[date, 'close']
                            shares = per_stock / price
                            positions[ts_code] = shares
        
        last_signal = signal
        
        # 6. 记录
        records.append({
            'date': date,
            'capital': capital,
            'daily_return': daily_return,
            'regime': signal.market_regime.value,
            'position_scale': signal.position_scale,
            'stop_loss': signal.stop_loss_active,
            'long_count': len(signal.long_stocks),
        })
        
        # 进度
        if i % 50 == 0:
            print(f"  {date.strftime('%Y-%m-%d')}: 资金={capital:,.0f}, 状态={signal.market_regime.value}, 仓位={signal.position_scale:.1f}")
    
    return pd.DataFrame(records)


def analyze_results(results: pd.DataFrame, initial_capital: float = 1000000):
    """分析回测结果"""
    if results.empty:
        print("无回测结果")
        return
    
    results['date'] = pd.to_datetime(results['date'])
    results = results.set_index('date')
    
    # 基本指标
    total_return = (results['capital'].iloc[-1] / initial_capital - 1) * 100
    
    # 年化收益
    days = (results.index[-1] - results.index[0]).days
    annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100 if days > 0 else 0
    
    # 夏普比率
    daily_returns = results['daily_return'].dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    # 最大回撤
    cummax = results['capital'].cummax()
    drawdown = (results['capital'] - cummax) / cummax
    max_drawdown = drawdown.min() * 100
    
    # 市场状态统计
    regime_counts = results['regime'].value_counts()
    
    # 分年统计
    results['year'] = results.index.year
    yearly_returns = results.groupby('year').apply(
        lambda x: (x['capital'].iloc[-1] / x['capital'].iloc[0] - 1) * 100
    )
    
    print("\n" + "=" * 60)
    print("📊 V5 策略回测结果 - 市场状态自适应版")
    print("=" * 60)
    print(f"总收益率: {total_return:.1f}%")
    print(f"年化收益: {annual_return:.1f}%")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {max_drawdown:.1f}%")
    print(f"\n📈 市场状态分布:")
    for regime, count in regime_counts.items():
        pct = count / len(results) * 100
        print(f"  {regime}: {count} 天 ({pct:.1f}%)")
    
    print(f"\n📅 分年收益:")
    for year, ret in yearly_returns.items():
        print(f"  {year}: {ret:+.1f}%")
    
    # 止损统计
    stop_loss_days = results['stop_loss'].sum()
    print(f"\n🛑 止损天数: {stop_loss_days} 天 ({stop_loss_days/len(results)*100:.1f}%)")
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'yearly_returns': yearly_returns.to_dict(),
    }


def main():
    """主函数"""
    print("🚀 V5 策略回测 - 市场状态自适应版")
    print("=" * 60)
    
    # 加载数据
    data_dir = project_root / "data"
    print(f"加载数据目录: {data_dir}")
    
    stock_data = load_stock_data(data_dir)
    print(f"加载股票数: {len(stock_data)}")
    
    if len(stock_data) < 50:
        print("❌ 股票数据不足，请先获取数据")
        return
    
    # 创建策略
    strategy = ReversalNeutralV5(
        top_n=10,
        rebalance_days=5,
        # 熊市保护
        bear_position=0.3,  # 熊市只用 30% 仓位
        # 止损
        stop_loss_threshold=-0.08,  # 8% 止损
        stop_loss_lookback=10,
    )
    
    print(f"\n策略配置:")
    config = strategy.get_config()
    print(f"  Top N: {config['top_n']}")
    print(f"  调仓周期: {config['rebalance_days']} 天")
    print(f"  熊市仓位: {config['position_control']['bear']}")
    print(f"  止损阈值: {config['stop_loss']['threshold']}")
    
    # 运行回测
    results = run_backtest(
        stock_data=stock_data,
        start_date='2023-07-01',
        end_date='2026-02-01',
        strategy=strategy,
    )
    
    # 分析结果
    metrics = analyze_results(results)
    
    # 保存结果
    output_dir = project_root / "strategies" / "reversal_neutral_v5" / "results"
    output_dir.mkdir(exist_ok=True)
    results.to_csv(output_dir / "backtest_results.csv", index=False)
    print(f"\n结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
