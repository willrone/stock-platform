"""
回测服务路由
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
from datetime import datetime
from app.core.config import settings

from app.api.v1.schemas import StandardResponse, BacktestRequest
from app.services.backtest import BacktestExecutor, BacktestConfig

router = APIRouter(prefix="/backtest", tags=["回测服务"])


@router.get("/strategies", response_model=StandardResponse)
async def get_available_strategies():
    """获取可用策略列表"""
    try:
        from app.services.backtest import StrategyFactory
        
        strategies = StrategyFactory.get_available_strategies()
        
        # 策略描述
        strategy_descriptions = {
            'moving_average': {
                'name': '移动平均策略',
                'description': '基于短期和长期移动平均线的交叉信号',
                'parameters': {
                    'short_window': {'type': 'int', 'default': 5, 'description': '短期均线周期'},
                    'long_window': {'type': 'int', 'default': 20, 'description': '长期均线周期'},
                    'signal_threshold': {'type': 'float', 'default': 0.02, 'description': '信号阈值'}
                }
            },
            'rsi': {
                'name': 'RSI策略',
                'description': '基于相对强弱指数的超买超卖策略',
                'parameters': {
                    'rsi_period': {'type': 'int', 'default': 14, 'description': 'RSI周期'},
                    'oversold_threshold': {'type': 'int', 'default': 30, 'description': '超卖阈值'},
                    'overbought_threshold': {'type': 'int', 'default': 70, 'description': '超买阈值'}
                }
            },
            'macd': {
                'name': 'MACD策略',
                'description': '基于MACD指标的趋势跟踪策略',
                'parameters': {
                    'fast_period': {'type': 'int', 'default': 12, 'description': '快线周期'},
                    'slow_period': {'type': 'int', 'default': 26, 'description': '慢线周期'},
                    'signal_period': {'type': 'int', 'default': 9, 'description': '信号线周期'}
                }
            }
        }
        
        result = []
        for strategy_key in strategies:
            if strategy_key in strategy_descriptions:
                result.append({
                    'key': strategy_key,
                    **strategy_descriptions[strategy_key]
                })
            else:
                result.append({
                    'key': strategy_key,
                    'name': strategy_key,
                    'description': '自定义策略',
                    'parameters': {}
                })
        
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
    """运行回测"""
    try:
        logger.info(f"开始回测: 策略={request.strategy_name}, 股票={request.stock_codes}, 期间={request.start_date} - {request.end_date}")
        
        # 创建回测执行器
        executor = BacktestExecutor(data_dir=str(settings.DATA_ROOT_PATH))
        
        # 验证参数
        executor.validate_backtest_parameters(
            strategy_name=request.strategy_name,
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
            strategy_config=request.strategy_config or {}
        )
        
        # 创建回测配置
        backtest_config = BacktestConfig(
            initial_cash=request.initial_cash,
            commission_rate=request.strategy_config.get("commission_rate", 0.0003) if request.strategy_config else 0.0003,
            slippage_rate=request.strategy_config.get("slippage_rate", 0.0001) if request.strategy_config else 0.0001
        )
        
        # 执行回测
        backtest_report = executor.run_backtest(
            strategy_name=request.strategy_name,
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
            strategy_config=request.strategy_config or {},
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
        result = {
            "strategy_name": backtest_report.get("strategy_name", request.strategy_name),
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

