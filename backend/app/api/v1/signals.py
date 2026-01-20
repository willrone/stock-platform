"""
策略信号API

支持：
- 全市场（分页）获取“最新信号”（近N天窗口内最后一次BUY/SELL事件；若无则HOLD）
- 单只股票获取近N天“信号历史”（BUY/SELL事件列表）
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from app.api.v1.schemas import StandardResponse
from app.core.config import settings
from app.core.container import get_data_service
from app.services.backtest import AdvancedStrategyFactory, SignalType, StrategyFactory
from app.services.data import SimpleDataService


router = APIRouter(prefix="/signals", tags=["策略信号"])


SignalSource = Literal["local", "remote"]


def _infer_warmup_days(strategy_name: str, days: int) -> int:
    """
    不同策略对历史数据长度要求不同；为了在“近N天窗口”内也能计算指标，这里加热身窗口。
    """
    name = (strategy_name or "").lower()
    warmup_map = {
        # 基础技术策略
        "moving_average": 80,
        "rsi": 60,
        "macd": 120,
        # 高级技术策略
        "bollinger": 80,
        "stochastic": 80,
        "cci": 80,
        # 统计套利
        "pairs_trading": 120,
        "mean_reversion": 120,
        "cointegration": 200,
        # 因子类（通常需要更长历史）
        "value_factor": 320,
        "momentum_factor": 320,
        "low_volatility": 200,
        "multi_factor": 260,
    }
    return max(warmup_map.get(name, 120), days)


def _list_local_stock_codes() -> List[str]:
    """
    从本地 parquet 文件名快速获取股票列表：
    data_root/parquet/stock_data/{code}_{market}.parquet  ->  {code}.{MARKET}
    """
    from pathlib import Path

    data_root = Path(settings.DATA_ROOT_PATH)
    stock_data_dir = data_root / "parquet" / "stock_data"
    if not stock_data_dir.exists():
        # 兼容：有些环境 DATA_ROOT_PATH 可能是相对路径
        backend_dir = Path(__file__).parent.parent.parent.parent
        stock_data_dir = (backend_dir / data_root / "parquet" / "stock_data").resolve()

    if not stock_data_dir.exists():
        raise RuntimeError(f"未找到本地股票数据目录: {stock_data_dir}")

    codes: List[str] = []
    for fp in stock_data_dir.glob("*.parquet"):
        stem = fp.stem  # 000001_SZ
        if "_" not in stem:
            continue
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        code = parts[0].strip()
        market = parts[1].strip().upper()
        if not code or market not in {"SZ", "SH", "BJ"}:
            continue
        codes.append(f"{code}.{market}")

    codes = sorted(set(codes))
    return codes


async def _get_universe_stock_codes(
    source: SignalSource,
    data_service: SimpleDataService,
) -> List[str]:
    if source == "local":
        return _list_local_stock_codes()

    stocks = await data_service.get_remote_stock_list()
    if not stocks:
        raise RuntimeError("无法从远端数据服务获取股票列表")
    codes = sorted({s.get("ts_code") for s in stocks if s.get("ts_code")})
    return codes


def _create_strategy(strategy_name: str, strategy_config: Dict[str, Any]) -> Any:
    """
    统一创建基础/高级策略实例。
    """
    try:
        return StrategyFactory.create_strategy(strategy_name, strategy_config)
    except Exception:
        return AdvancedStrategyFactory.create_strategy(strategy_name, strategy_config)


def _stockdata_list_to_df(stock_code: str, rows: List[Any]) -> pd.DataFrame:
    """
    SimpleDataService.get_stock_data 返回的是 StockData 列表；这里转换为 DataFrame。
    """
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        data.append(
            {
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "adj_close": getattr(r, "adj_close", None),
            }
        )
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.attrs["stock_code"] = stock_code
    return df


def _compute_latest_signal_for_stock(
    strategy: Any,
    df: pd.DataFrame,
    window_days: int,
) -> Tuple[str, Optional[datetime], float, Optional[float], Optional[str]]:
    """
    计算近window_days窗口内最后一次 BUY/SELL 事件；若无则 HOLD。
    返回：signal_side, signal_date, strength, price, reason
    """
    if df.empty:
        return "HOLD", None, 0.0, None, None

    dates = list(df.index.unique())
    if not dates:
        return "HOLD", None, 0.0, None, None

    # 仅取最后window_days个交易日（不是自然日）
    recent_dates = dates[-window_days:]
    last_event: Optional[Tuple[str, datetime, float, float, str]] = None

    for dt in recent_dates:
        try:
            signals = strategy.generate_signals(df, dt.to_pydatetime())
        except Exception as e:
            logger.debug(f"生成信号失败: {df.attrs.get('stock_code')} @ {dt}: {e}")
            continue

        for s in signals or []:
            if s.signal_type == SignalType.BUY:
                last_event = ("BUY", s.timestamp, float(s.strength), float(s.price), s.reason)
            elif s.signal_type == SignalType.SELL:
                last_event = ("SELL", s.timestamp, float(s.strength), float(s.price), s.reason)

    if last_event is None:
        # 无事件，仍返回窗口末端日期作为“最后观察日”
        return "HOLD", recent_dates[-1].to_pydatetime(), 0.0, float(df["close"].iloc[-1]), "窗口内无买卖事件"

    side, ts, strength, price, reason = last_event
    return side, ts, strength, price, reason


@router.get("/latest", response_model=StandardResponse)
async def get_latest_signals(
    strategy_name: str = Query(..., description="策略名称（与 /backtest/strategies 的 key 对应）"),
    days: int = Query(60, ge=5, le=365, description="观察窗口：最近N个交易日"),
    source: SignalSource = Query("local", description="股票池来源：local=本地parquet，remote=远端数据服务"),
    limit: int = Query(200, ge=1, le=2000, description="分页大小（全市场很大，建议分页）"),
    offset: int = Query(0, ge=0, description="分页偏移"),
    data_service: SimpleDataService = Depends(get_data_service),
):
    """
    获取股票池内每只股票的“最新信号”（近N个交易日窗口内最后一次BUY/SELL；若无则HOLD）。
    """
    try:
        all_codes = await _get_universe_stock_codes(source=source, data_service=data_service)
        total = len(all_codes)
        page_codes = all_codes[offset : offset + limit]

        warmup = _infer_warmup_days(strategy_name, days)
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days + warmup + 30)  # 额外+30自然日，避免交易日不足

        strategy = _create_strategy(strategy_name, {})

        results: List[Dict[str, Any]] = []
        failures: List[str] = []

        for code in page_codes:
            try:
                rows = await data_service.get_stock_data(code, start_dt, end_dt)
                df = _stockdata_list_to_df(code, rows or [])
                side, ts, strength, price, reason = _compute_latest_signal_for_stock(strategy, df, days)
                results.append(
                    {
                        "stock_code": code,
                        "latest_signal": side,
                        "signal_date": ts.isoformat() if ts else None,
                        "strength": strength,
                        "price": price,
                        "reason": reason,
                    }
                )
            except Exception as e:
                failures.append(f"{code}: {type(e).__name__} {e}")
                continue

        return StandardResponse(
            success=True,
            message="获取最新信号成功",
            data={
                "strategy_name": strategy_name,
                "days": days,
                "source": source,
                "pagination": {"total": total, "limit": limit, "offset": offset},
                "signals": results,
                "failures": failures[:20],
            },
        )
    except Exception as e:
        logger.error(f"获取最新信号失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取最新信号失败: {str(e)}")


@router.get("/history", response_model=StandardResponse)
async def get_signal_history(
    stock_code: str = Query(..., description="股票代码，如 000001.SZ"),
    strategy_name: str = Query(..., description="策略名称（与 /backtest/strategies 的 key 对应）"),
    days: int = Query(60, ge=5, le=365, description="最近N个交易日"),
    data_service: SimpleDataService = Depends(get_data_service),
):
    """
    获取单只股票近N个交易日的BUY/SELL信号事件列表（不包含HOLD）。
    """
    try:
        warmup = _infer_warmup_days(strategy_name, days)
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days + warmup + 30)

        rows = await data_service.get_stock_data(stock_code, start_dt, end_dt)
        df = _stockdata_list_to_df(stock_code, rows or [])
        if df.empty:
            return StandardResponse(
                success=True,
                message="无数据",
                data={
                    "stock_code": stock_code,
                    "strategy_name": strategy_name,
                    "days": days,
                    "events": [],
                },
            )

        strategy = _create_strategy(strategy_name, {})
        dates = list(df.index.unique())[-days:]

        events: List[Dict[str, Any]] = []
        for dt in dates:
            try:
                signals = strategy.generate_signals(df, dt.to_pydatetime())
            except Exception:
                continue
            for s in signals or []:
                if s.signal_type not in (SignalType.BUY, SignalType.SELL):
                    continue
                events.append(
                    {
                        "timestamp": s.timestamp.isoformat(),
                        "signal": "BUY" if s.signal_type == SignalType.BUY else "SELL",
                        "strength": float(s.strength),
                        "price": float(s.price),
                        "reason": s.reason,
                        "metadata": s.metadata or {},
                    }
                )

        return StandardResponse(
            success=True,
            message="获取信号历史成功",
            data={
                "stock_code": stock_code,
                "strategy_name": strategy_name,
                "days": days,
                "events": events,
            },
        )
    except Exception as e:
        logger.error(f"获取信号历史失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取信号历史失败: {str(e)}")

