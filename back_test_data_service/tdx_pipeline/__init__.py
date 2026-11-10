"""通达信盘后包数据管线 — 免费、无 token 的 A 股全市场日线源。

替代已失效的 Tushare（会员过期）。实测对比:
- 通达信盘后包: 2.6MB/天，全市场含北交所，8s/天
- Baostock: 7.92s/只 × 5500 只 = 12 小时，且并行会互相阻塞、后期限流

模块:
- tdx_parser   盘后包下载与二进制解析（.cod/.md1 布局）
- pipeline     增量拉取 + 按日期分区存储 + 断点续传
- ashare       A 股筛选（盘后包含债券/基金/指数等约 5 万条）
- adjust       前复权 — 通达信加法模型（mootdx xdxr 事件）
- factor_store 除权除息事件批量存储

**存不复权价 + 事件表，不存前复权价** — 增量拉取不会让历史数据漂移。

用法:
    from back_test_data_service.tdx_pipeline import run_incremental, load_range

    # 增量拉取
    stats = run_incremental(start="2026-05-09", end="2026-10-16")

    # 加载 + 筛 A 股
    from back_test_data_service.tdx_pipeline import load_range, filter_ashare
    a = filter_ashare(load_range(start="2026-05-09"))

    # 前复权
    from back_test_data_service.tdx_pipeline import fetch_events, to_qfq
    evs = fetch_events("000001.SZ")
    qfq = to_qfq(raw_price=10.98, events=evs, date="2026-06-05")
"""

from .adjust import (
    AdjustmentEvent,
    compute_adjustment,
    fetch_events,
    factor_series,
    to_hfq,
    to_qfq,
)
from .ashare import (
    BJ_PREFIXES,
    SH_PREFIXES,
    SZ_PREFIXES,
    ashare_count_by_market,
    filter_ashare,
    is_ashare_code,
)
from .factor_store import (
    all_ts_codes,
    get_events,
    load_event_table,
    refresh_events,
    save_event_table,
)
from .pipeline import (
    PipelineState,
    bars_to_dataframe,
    date_partition_path,
    fetch_and_store,
    has_partition,
    load_range,
    load_state,
    loaded_dates,
    run_incremental,
    save_state,
    trading_days_between,
)
from .tdx_parser import (
    TDX_BJ_FIRST_DAY,
    TDX_MIN_PRICED,
    TDX_PACKAGE_URL,
    TdxBar,
    fetch_daily,
    is_trading_day,
    parse_package,
)

__all__ = [
    # adjust
    "AdjustmentEvent", "compute_adjustment", "fetch_events",
    "factor_series", "to_hfq", "to_qfq",
    # ashare
    "BJ_PREFIXES", "SH_PREFIXES", "SZ_PREFIXES",
    "ashare_count_by_market", "filter_ashare", "is_ashare_code",
    # factor_store
    "all_ts_codes", "get_events", "load_event_table",
    "refresh_events", "save_event_table",
    # pipeline
    "PipelineState", "bars_to_dataframe", "date_partition_path",
    "fetch_and_store", "has_partition", "load_range", "load_state",
    "loaded_dates", "run_incremental", "save_state", "trading_days_between",
    # tdx_parser
    "TDX_BJ_FIRST_DAY", "TDX_MIN_PRICED", "TDX_PACKAGE_URL", "TdxBar",
    "fetch_daily", "is_trading_day", "parse_package",
]
