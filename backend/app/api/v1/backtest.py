"""
回测服务路由
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
from datetime import datetime
import os
from app.core.config import settings

from app.api.v1.schemas import StandardResponse, BacktestRequest
from app.services.backtest import BacktestExecutor, BacktestConfig

router = APIRouter(prefix="/backtest", tags=["回测服务"])


def _parse_bool_env(var_name: str, default: bool = False) -> bool:
    """从环境变量解析布尔值（支持 1/true/yes/on）。"""
    val = os.getenv(var_name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


@router.get("/strategies", response_model=StandardResponse)
async def get_available_strategies():
    """获取可用策略列表"""
    try:
        from app.services.backtest import StrategyFactory, AdvancedStrategyFactory
        
        # 获取基础策略
        basic_strategies = StrategyFactory.get_available_strategies()
        
        # 获取高级策略分类
        advanced_categories = AdvancedStrategyFactory.get_available_strategies()
        
        # 合并所有策略
        all_strategies = set(basic_strategies)
        for category_strategies in advanced_categories.values():
            all_strategies.update(category_strategies)
        
        # 策略描述和参数
        strategy_descriptions = {
            # 基础技术分析策略
            'moving_average': {
                'name': '移动平均策略',
                'description': '基于短期和长期移动平均线的交叉信号，金叉买入，死叉卖出',
                'category': 'technical',
                'parameters': {
                    'short_window': {'type': 'int', 'default': 5, 'description': '短期均线周期', 'min': 2, 'max': 50},
                    'long_window': {'type': 'int', 'default': 20, 'description': '长期均线周期', 'min': 5, 'max': 200},
                    'signal_threshold': {'type': 'float', 'default': 0.02, 'description': '信号阈值', 'min': 0.001, 'max': 0.1}
                }
            },
            'rsi': {
                'name': 'RSI策略（优化版）',
                'description': '基于业界最佳实践的RSI策略，包含趋势对齐、背离检测和RSI穿越信号。在上升趋势中等待RSI回调买入，在下降趋势中等待RSI反弹卖出，避免逆势交易。',
                'category': 'technical',
                'parameters': {
                    'rsi_period': {'type': 'int', 'default': 14, 'description': 'RSI周期', 'min': 2, 'max': 50},
                    'oversold_threshold': {'type': 'int', 'default': 30, 'description': '超卖阈值', 'min': 10, 'max': 40},
                    'overbought_threshold': {'type': 'int', 'default': 70, 'description': '超买阈值', 'min': 60, 'max': 90},
                    'trend_ma_period': {'type': 'int', 'default': 50, 'description': '趋势判断均线周期（用于判断趋势方向）', 'min': 20, 'max': 200},
                    'enable_trend_alignment': {'type': 'boolean', 'default': True, 'description': '启用趋势对齐（在上升趋势中只在回调时买入，在下降趋势中只在反弹时卖出）'},
                    'enable_divergence': {'type': 'boolean', 'default': True, 'description': '启用背离检测（检测价格与RSI的背离作为反转信号）'},
                    'enable_crossover': {'type': 'boolean', 'default': True, 'description': '启用RSI穿越信号（等待RSI从超买超卖区域穿越回来，而不是仅仅在超买超卖区域就交易）'},
                    'uptrend_buy_threshold': {'type': 'int', 'default': 40, 'description': '上升趋势中的买入阈值（RSI回调到此值以上时买入）', 'min': 30, 'max': 60},
                    'downtrend_sell_threshold': {'type': 'int', 'default': 60, 'description': '下降趋势中的卖出阈值（RSI反弹到此值以下时卖出）', 'min': 40, 'max': 70}
                }
            },
            'macd': {
                'name': 'MACD策略',
                'description': '基于MACD指标的趋势跟踪策略，MACD柱状图由负转正买入，由正转负卖出',
                'category': 'technical',
                'parameters': {
                    'fast_period': {'type': 'int', 'default': 12, 'description': '快线周期', 'min': 5, 'max': 30},
                    'slow_period': {'type': 'int', 'default': 26, 'description': '慢线周期', 'min': 10, 'max': 50},
                    'signal_period': {'type': 'int', 'default': 9, 'description': '信号线周期', 'min': 3, 'max': 20}
                }
            },
            
            # 新增技术分析策略
            'bollinger': {
                'name': '布林带策略',
                'description': '基于布林带的突破策略，价格突破下轨买入，突破上轨卖出',
                'category': 'technical',
                'parameters': {
                    'period': {'type': 'int', 'default': 20, 'description': '布林带周期', 'min': 5, 'max': 50},
                    'std_dev': {'type': 'float', 'default': 2, 'description': '标准差倍数', 'min': 1, 'max': 3},
                    'entry_threshold': {'type': 'float', 'default': 0.02, 'description': '入场阈值', 'min': 0.01, 'max': 0.1}
                }
            },
            'stochastic': {
                'name': '随机指标策略',
                'description': '基于随机指标(K%D)的超买超卖策略，低位金叉买入，高位死叉卖出',
                'category': 'technical',
                'parameters': {
                    'k_period': {'type': 'int', 'default': 14, 'description': '%K周期', 'min': 5, 'max': 30},
                    'd_period': {'type': 'int', 'default': 3, 'description': '%D周期', 'min': 2, 'max': 10},
                    'oversold': {'type': 'int', 'default': 20, 'description': '超卖阈值', 'min': 10, 'max': 30},
                    'overbought': {'type': 'int', 'default': 80, 'description': '超买阈值', 'min': 70, 'max': 90}
                }
            },
            'cci': {
                'name': 'CCI策略',
                'description': '基于商品通道指数的趋势策略，CCI低于-100后回升买入，高于100后回落卖出',
                'category': 'technical',
                'parameters': {
                    'period': {'type': 'int', 'default': 20, 'description': 'CCI周期', 'min': 5, 'max': 50},
                    'oversold': {'type': 'int', 'default': -100, 'description': '超卖阈值', 'min': -200, 'max': -50},
                    'overbought': {'type': 'int', 'default': 100, 'description': '超买阈值', 'min': 50, 'max': 200}
                }
            },
            
            # 统计套利策略
            'pairs_trading': {
                'name': '配对交易策略',
                'description': '基于两只高度相关股票价差的统计套利策略，价差偏离均值时做多被低估的，做空被高估的',
                'category': 'statistical_arbitrage',
                'parameters': {
                    'correlation_threshold': {'type': 'float', 'default': 0.8, 'description': '相关性阈值', 'min': 0.5, 'max': 0.95},
                    'entry_threshold': {'type': 'float', 'default': 2.0, 'description': '入场Z-score阈值', 'min': 1.0, 'max': 3.0},
                    'exit_threshold': {'type': 'float', 'default': 0.5, 'description': '出场Z-score阈值', 'min': 0.1, 'max': 1.0},
                    'max_holding_period': {'type': 'int', 'default': 60, 'description': '最大持有期（天）', 'min': 20, 'max': 120}
                }
            },
            'mean_reversion': {
                'name': '均值回归策略',
                'description': '基于价格偏离均值的回归策略，价格偏离均值超过Z-score阈值时入场，回归时平仓',
                'category': 'statistical_arbitrage',
                'parameters': {
                    'lookback_period': {'type': 'int', 'default': 20, 'description': '回看周期', 'min': 10, 'max': 60},
                    'zscore_threshold': {'type': 'float', 'default': 2.0, 'description': 'Z-score阈值', 'min': 1.0, 'max': 3.0},
                    'position_size': {'type': 'float', 'default': 0.1, 'description': '仓位比例', 'min': 0.05, 'max': 0.3}
                }
            },
            'cointegration': {
                'name': '协整策略',
                'description': '基于协整关系的统计套利策略，寻找具有长期均衡关系的资产，偏离均衡时套利',
                'category': 'statistical_arbitrage',
                'parameters': {
                    'lookback_period': {'type': 'int', 'default': 60, 'description': '回看周期', 'min': 30, 'max': 120},
                    'entry_threshold': {'type': 'float', 'default': 2.0, 'description': '入场阈值', 'min': 1.0, 'max': 3.0},
                    'half_life': {'type': 'int', 'default': 20, 'description': '半衰期', 'min': 5, 'max': 60}
                }
            },
            
            # 因子投资策略
            'value_factor': {
                'name': '价值因子策略',
                'description': '基于估值因子的投资策略，选择PE、PB、PS等估值指标较低的股票',
                'category': 'factor_investment',
                'parameters': {
                    'pe_weight': {'type': 'float', 'default': 0.25, 'description': 'PE权重', 'min': 0, 'max': 1},
                    'pb_weight': {'type': 'float', 'default': 0.25, 'description': 'PB权重', 'min': 0, 'max': 1},
                    'ps_weight': {'type': 'float', 'default': 0.25, 'description': 'PS权重', 'min': 0, 'max': 1},
                    'ev_ebitda_weight': {'type': 'float', 'default': 0.25, 'description': 'EV/EBITDA权重', 'min': 0, 'max': 1}
                }
            },
            'momentum_factor': {
                'name': '动量因子策略',
                'description': '基于价格动量的投资策略，买入近期表现好的股票，卖出近期表现差的股票',
                'category': 'factor_investment',
                'parameters': {
                    'momentum_periods': {'type': 'json', 'default': [21, 63, 126], 'description': '动量计算周期（天）'},
                    'momentum_weights': {'type': 'json', 'default': [0.5, 0.3, 0.2], 'description': '各周期权重'}
                }
            },
            'low_volatility': {
                'name': '低波动因子策略',
                'description': '基于波动率因子的投资策略，选择历史波动率较低的股票构建组合',
                'category': 'factor_investment',
                'parameters': {
                    'volatility_period': {'type': 'int', 'default': 21, 'description': '波动率周期', 'min': 10, 'max': 60},
                    'volatility_window': {'type': 'int', 'default': 63, 'description': '波动率窗口', 'min': 20, 'max': 120}
                }
            },
            'multi_factor': {
                'name': '多因子组合策略',
                'description': '综合多个因子（价值、动量、低波动等）进行选股和加权，构建多元化投资组合',
                'category': 'factor_investment',
                'parameters': {
                    'factors': {'type': 'json', 'default': ['value', 'momentum', 'low_volatility'], 'description': '使用的因子列表'},
                    'factor_weights': {'type': 'json', 'default': [0.33, 0.33, 0.34], 'description': '因子权重'},
                    'weighting_method': {'type': 'string', 'default': 'equal', 'description': '权重方法', 'options': ['equal', 'ic', 'optimize']},
                    'market_cap_neutral': {'type': 'boolean', 'default': False, 'description': '是否市值中性化'},
                    'industry_neutral': {'type': 'boolean', 'default': False, 'description': '是否行业中性化'}
                }
            }
        }
        
        result = []
        for strategy_key in all_strategies:
            if strategy_key in strategy_descriptions:
                result.append({
                    'key': strategy_key,
                    **strategy_descriptions[strategy_key]
                })
            else:
                result.append({
                    'key': strategy_key,
                    'name': strategy_key.replace('_', ' ').title(),
                    'description': '自定义策略',
                    'category': 'other',
                    'parameters': {}
                })
        
        # 按分类排序
        category_order = {'technical': 0, 'statistical_arbitrage': 1, 'factor_investment': 2, 'other': 3}
        result.sort(key=lambda x: (category_order.get(x.get('category'), 3), x['key']))
        
        return StandardResponse(
            success=True,
            message="获取策略列表成功",
            data=result
        )
    except Exception as e:
        logger.error(f"获取策略列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取策略列表失败: {str(e)}")


@router.post("", response_model=StandardResponse)
async def run_backtest(request: BacktestRequest):
    """
    运行回测（支持单策略和组合策略）
    
    组合策略配置格式：
    {
        "strategy_name": "portfolio",
        "strategy_config": {
            "strategies": [
                {
                    "name": "rsi",
                    "weight": 0.4,
                    "config": {"rsi_period": 14}
                },
                {
                    "name": "macd",
                    "weight": 0.3,
                    "config": {"fast_period": 12}
                }
            ],
            "integration_method": "weighted_voting"
        }
    }
    """
    try:
        # 检测是否为组合策略
        is_portfolio = (
            request.strategy_name.lower() == "portfolio" or
            (request.strategy_config and "strategies" in request.strategy_config)
        )
        
        strategy_display = "组合策略" if is_portfolio else request.strategy_name
        logger.info(f"开始回测: 策略={strategy_display}, 股票={request.stock_codes}, 期间={request.start_date} - {request.end_date}")
        
        # 创建回测执行器
        executor = BacktestExecutor(
            data_dir=str(settings.DATA_ROOT_PATH),
            enable_performance_profiling=_parse_bool_env("ENABLE_BACKTEST_PERFORMANCE_PROFILING", default=False),
        )
        
        # 验证参数
        strategy_config = request.strategy_config or {}
        executor.validate_backtest_parameters(
            strategy_name=request.strategy_name,
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
            strategy_config=strategy_config
        )
        
        # 创建回测配置
        backtest_config = BacktestConfig(
            initial_cash=request.initial_cash,
            commission_rate=strategy_config.get("commission_rate", 0.0003),
            slippage_rate=strategy_config.get("slippage_rate", 0.0001)
        )
        
        # 执行回测（StrategyFactory会自动检测是否为组合策略）
        backtest_report = executor.run_backtest(
            strategy_name=request.strategy_name,
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
            strategy_config=strategy_config,
            backtest_config=backtest_config
        )
        
        # 转换数据格式以匹配前端期望
        portfolio_history = backtest_report.get("portfolio_history", [])
        dates = [snapshot["date"] for snapshot in portfolio_history]
        equity_curve = [snapshot["portfolio_value"] for snapshot in portfolio_history]
        
        # 计算回撤曲线
        drawdown_curve = []
        peak = backtest_config.initial_cash
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak * 100 if peak > 0 else 0
            drawdown_curve.append(drawdown)
        
        # 格式化交易记录
        trade_history = []
        for trade in backtest_report.get("trade_history", []):
            trade_history.append({
                "date": trade.get("timestamp", ""),
                "action": "buy" if trade.get("action") == "BUY" else "sell",
                "price": trade.get("price", 0),
                "quantity": trade.get("quantity", 0),
                "pnl": trade.get("pnl", 0)
            })
        
        # 构建返回结果（匹配前端期望的格式）
        # 如果是组合策略，获取组合信息
        portfolio_info = None
        if is_portfolio:
            # 尝试从回测报告中提取组合信息
            portfolio_info = {
                "is_portfolio": True,
                "strategies": strategy_config.get("strategies", []) if strategy_config else []
            }
        
        result = {
            "strategy_name": backtest_report.get("strategy_name", request.strategy_name),
            "is_portfolio": is_portfolio,
            "portfolio_info": portfolio_info,
            "period": {
                "start_date": backtest_report.get("start_date", request.start_date.isoformat()),
                "end_date": backtest_report.get("end_date", request.end_date.isoformat())
            },
            "portfolio": {
                "initial_cash": backtest_report.get("initial_cash", request.initial_cash),
                "final_value": backtest_report.get("final_value", request.initial_cash),
                "total_return": backtest_report.get("total_return", 0),
                "annualized_return": backtest_report.get("annualized_return", 0)
            },
            "risk_metrics": {
                "volatility": backtest_report.get("volatility", 0),
                "sharpe_ratio": backtest_report.get("sharpe_ratio", 0),
                "max_drawdown": backtest_report.get("max_drawdown", 0)
            },
            "trading_stats": {
                "total_trades": backtest_report.get("total_trades", 0),
                "win_rate": backtest_report.get("win_rate", 0),
                "profit_factor": backtest_report.get("profit_factor", 0)
            },
            "trade_history": trade_history,
            # 添加前端需要的图表数据
            "equity_curve": equity_curve,
            "drawdown_curve": drawdown_curve,
            "dates": dates
        }
        
        logger.info(f"回测完成: 总收益={backtest_report.get('total_return', 0):.2%}, 夏普比率={backtest_report.get('sharpe_ratio', 0):.2f}")
        
        return StandardResponse(
            success=True,
            message="回测执行成功",
            data=result
        )
        
    except Exception as e:
        logger.error(f"回测执行失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"回测执行失败: {str(e)}")


@router.get("/portfolio-templates", response_model=StandardResponse)
async def get_portfolio_templates():
    """获取预设的策略组合模板"""
    try:
        templates = [
            {
                "name": "技术指标组合",
                "description": "结合RSI、MACD和布林带的经典技术指标组合",
                "strategies": [
                    {
                        "name": "rsi",
                        "weight": 0.4,
                        "config": {
                            "rsi_period": 14,
                            "oversold_threshold": 30,
                            "overbought_threshold": 70
                        }
                    },
                    {
                        "name": "macd",
                        "weight": 0.3,
                        "config": {
                            "fast_period": 12,
                            "slow_period": 26,
                            "signal_period": 9
                        }
                    },
                    {
                        "name": "bollinger",
                        "weight": 0.3,
                        "config": {
                            "period": 20,
                            "std_dev": 2
                        }
                    }
                ],
                "integration_method": "weighted_voting"
            },
            {
                "name": "趋势跟踪组合",
                "description": "移动平均和MACD的组合，适合趋势市场",
                "strategies": [
                    {
                        "name": "moving_average",
                        "weight": 0.5,
                        "config": {
                            "short_window": 5,
                            "long_window": 20
                        }
                    },
                    {
                        "name": "macd",
                        "weight": 0.5,
                        "config": {
                            "fast_period": 12,
                            "slow_period": 26,
                            "signal_period": 9
                        }
                    }
                ],
                "integration_method": "weighted_voting"
            },
            {
                "name": "均值回归组合",
                "description": "RSI和随机指标的组合，适合震荡市场",
                "strategies": [
                    {
                        "name": "rsi",
                        "weight": 0.5,
                        "config": {
                            "rsi_period": 14,
                            "oversold_threshold": 30,
                            "overbought_threshold": 70
                        }
                    },
                    {
                        "name": "stochastic",
                        "weight": 0.5,
                        "config": {
                            "k_period": 14,
                            "d_period": 3,
                            "oversold": 20,
                            "overbought": 80
                        }
                    }
                ],
                "integration_method": "weighted_voting"
            }
        ]
        
        return StandardResponse(
            success=True,
            message="获取策略组合模板成功",
            data=templates
        )
    except Exception as e:
        logger.error(f"获取策略组合模板失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取策略组合模板失败: {str(e)}")

