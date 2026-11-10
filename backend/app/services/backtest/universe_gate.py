"""宇宙「可筛选性」闸门（2026-11-06，台账 §12.66-D）。

问题：本仓库三个数据库覆盖不一致 ——
  * 回测价格源：`data/qlib_data/features/day/{code}.parquet`（引擎实际读它，见
    `app/services/backtest/execution/data_loader.py`）—— 北交所(9xxxxx.BJ)有全历史；
  * 否决/风控源：`data/parquet/baostock_daily/{code}.parquet`（`is_st` / `trade_status` / `amount`）——
    **北交所只有 2026-05-11 起的 122 行**。
  → 结果：BJ 票「选得出来、筛不掉」，整窗 312 只从未被 ST/LOWLIQ/HALT 任何一条规则检查过。

铁律：**查不到 = 排除，不是放行**。所以任何带风控/否决层的宇宙，必须先过这道闸门。

口径（不用未来信息，只在回测窗内比较两个库的覆盖）：
    price_days(code)  = 窗内有价格行的交易日
    screen_days(code) = 窗内有风控行的交易日
    screen_ratio      = |price_days ∩ screen_days| / |price_days|
    screen_ratio < MIN_SCREEN_RATIO → 剔除
合法中途上市的票不会被误杀：它的 price_days 与 screen_days 同时开始，ratio≈1。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import pandas as pd

MIN_SCREEN_RATIO = 0.95


def _norm_day(x) -> int:
    """把各种日期表示统一成「当日 00:00 的 ns 整数」，便于跨库取交集。"""
    return int(pd.Timestamp(x).normalize().value)


def coverage_ratio(price_days: Iterable[int], screen_days: Iterable[int]) -> float:
    """纯函数核心：可筛选比例。price_days 为空 → 0（没有价格就不该进宇宙）。"""
    p = set(price_days)
    if not p:
        return 0.0
    return len(p & set(screen_days)) / len(p)


@dataclass
class GateReport:
    window: tuple[str, str]
    min_ratio: float
    universe_size: int
    kept: list[str]
    dropped: list[dict] = field(default_factory=list)

    @property
    def dropped_share(self) -> float:
        return len(self.dropped) / max(1, self.universe_size)

    def summary(self) -> str:
        by_reason: dict[str, int] = {}
        for d in self.dropped:
            by_reason[d["reason"]] = by_reason.get(d["reason"], 0) + 1
        return (
            f"宇宙 {self.universe_size} 只 → 保留 {len(self.kept)} 只，"
            f"剔除 {len(self.dropped)} 只（{self.dropped_share*100:.1f}%）；原因分布 {by_reason}"
        )

    def to_json(self) -> dict:
        return {
            "window": list(self.window),
            "min_screen_ratio": self.min_ratio,
            "universe_size": self.universe_size,
            "kept": len(self.kept),
            "dropped_count": len(self.dropped),
            "dropped_share_pct": round(self.dropped_share * 100, 2),
            "dropped": self.dropped,
            "kept_codes": self.kept,
        }


def _days_in_window(
    path: str, lo_ns: int, hi_ns: int, date_column: str | None
) -> set[int]:
    """读一个 parquet，返回窗内出现过的交易日（ns 整数）。date_column=None 表示日期在索引上。

    qlib 的 `features/day` 索引是 MultiIndex(stock_code, date)，且同一个文件里 stock_code 可能有
    两种拼写（`920819.BJ` 与 `920819_bj`）→ 这里只取日期层，不依赖代码层。
    """
    if not os.path.exists(path):
        return set()
    try:
        if date_column is None:
            df = pd.read_parquet(path, columns=["$close"])
            idx = df.index
            if isinstance(idx, pd.MultiIndex):
                picked = None
                for i in range(idx.nlevels):
                    vals = idx.get_level_values(i)
                    if isinstance(vals, pd.DatetimeIndex):
                        picked = vals
                        break
                if picked is None:  # 兜底：逐层试解析
                    for i in range(idx.nlevels):
                        try:
                            got = pd.to_datetime(idx.get_level_values(i))
                            if got.max() > pd.Timestamp("2000-01-01"):
                                picked = got
                                break
                        except Exception:
                            continue
                idx = picked if picked is not None else pd.DatetimeIndex([])
            else:
                idx = pd.to_datetime(idx)
            # ⚠️ qlib 的 features/day 里含「A股休市日」的行（端午/国庆/春节等，本窗 58 个），
            # 值全是 NaN。只数「有有效价格」的日期，否则分母被虚增 → 正常票会被闸门误杀
            # （2026-11-06 第一版就误杀了 5241/5513）。
            close = pd.to_numeric(df["$close"], errors="coerce").to_numpy()
            days = {_norm_day(d) for d, c in zip(idx, close) if pd.notna(c)}
        else:
            df = pd.read_parquet(path, columns=[date_column])
            days = {_norm_day(x) for x in pd.to_datetime(df[date_column])}
    except Exception:
        return set()
    return {d for d in days if lo_ns <= d <= hi_ns}


def build_gate_report(
    universe: list[str],
    start: datetime | str,
    end: datetime | str,
    data_root: str,
    min_ratio: float = MIN_SCREEN_RATIO,
) -> GateReport:
    qlib_dir = os.path.join(data_root, "qlib_data", "features", "day")
    bs_dir = os.path.join(data_root, "parquet", "baostock_daily")
    lo = int(pd.Timestamp(start).normalize().value)
    hi = int(pd.Timestamp(end).normalize().value)

    kept: list[str] = []
    dropped: list[dict] = []
    for code in universe:
        safe = code.replace(".", "_")
        p = _days_in_window(os.path.join(qlib_dir, f"{safe}.parquet"), lo, hi, None)
        s = _days_in_window(os.path.join(bs_dir, f"{safe}.parquet"), lo, hi, "date")
        ratio = coverage_ratio(p, s)
        if ratio >= min_ratio:
            kept.append(code)
            continue
        inter = sorted(p & s)
        blind = sorted(p - s)
        dropped.append(
            dict(
                code=code,
                screen_ratio=round(ratio, 4),
                price_days=len(p),
                screen_days=len(s),
                reason=(
                    "无价格数据"
                    if not p
                    else ("风控库完全无窗内数据" if not s else "风控覆盖不足")
                ),
                blind_span=(
                    [
                        pd.Timestamp(blind[0]).date().isoformat(),
                        pd.Timestamp(blind[-1]).date().isoformat(),
                    ]
                    if blind
                    else None
                ),
                screened_span=(
                    [
                        pd.Timestamp(inter[0]).date().isoformat(),
                        pd.Timestamp(inter[-1]).date().isoformat(),
                    ]
                    if inter
                    else None
                ),
            )
        )
    return GateReport(
        window=(
            pd.Timestamp(start).date().isoformat(),
            pd.Timestamp(end).date().isoformat(),
        ),
        min_ratio=min_ratio,
        universe_size=len(universe),
        kept=kept,
        dropped=dropped,
    )


def gate_report_path(data_root: str) -> str:
    return os.path.join(data_root, "strategy_baseline", "universe_gate_2026-11-06.json")


def raw_universe(data_root: str) -> list[str]:
    """引擎的价格源目录 = 宇宙来源（和 data_loader 读的是同一个库）。"""
    import glob

    files = glob.glob(
        os.path.join(data_root, "qlib_data", "features", "day", "*.parquet")
    )
    return sorted(
        os.path.basename(f)[: -len(".parquet")].replace("_", ".") for f in files
    )


def gated_universe(
    data_root: str,
    start,
    end,
    min_ratio: float = MIN_SCREEN_RATIO,
    enabled: bool = True,
    force: bool = False,
) -> tuple[list[str], "GateReport | None"]:
    """实验脚本统一入口：enabled=False 只做复现旧数字用，正常实验一律带闸门。"""
    if not enabled:
        return raw_universe(data_root), None
    rep = load_or_build_report(
        raw_universe(data_root),
        start,
        end,
        data_root,
        gate_report_path(data_root),
        min_ratio=min_ratio,
        force=force,
    )
    return rep.kept, rep


def load_or_build_report(
    universe: list[str],
    start,
    end,
    data_root: str,
    report_path: str,
    min_ratio: float = MIN_SCREEN_RATIO,
    force: bool = False,
) -> GateReport:
    """产物缓存：同窗口/同宇宙/同阈值直接读 JSON，避免每次跑实验都重扫 5513 个文件。"""
    if os.path.exists(report_path) and not force:
        raw = json.load(open(report_path))
        if (
            raw.get("window")
            == [
                pd.Timestamp(start).date().isoformat(),
                pd.Timestamp(end).date().isoformat(),
            ]
            and raw.get("min_screen_ratio") == min_ratio
            and raw.get("universe_size") == len(universe)
        ):
            return GateReport(
                window=tuple(raw["window"]),
                min_ratio=min_ratio,
                universe_size=raw["universe_size"],
                kept=list(raw["kept_codes"]),
                dropped=list(raw["dropped"]),
            )
    rep = build_gate_report(universe, start, end, data_root, min_ratio)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(rep.to_json(), fh, ensure_ascii=False, indent=1)
    return rep


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="宇宙可筛选性闸门（查不到 = 排除）")
    ap.add_argument("--start", default="2023-05-12")
    ap.add_argument("--end", default="2026-05-08")
    ap.add_argument("--min-ratio", type=float, default=MIN_SCREEN_RATIO)
    ap.add_argument(
        "--out", default="data/strategy_baseline/universe_gate_2026-11-06.json"
    )
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_root = args.data_root
    if data_root is None:
        from app.core.config import settings

        data_root = str(settings.DATA_ROOT_PATH)
    universe = raw_universe(data_root)
    rep = load_or_build_report(
        universe,
        args.start,
        args.end,
        data_root,
        args.out,
        min_ratio=args.min_ratio,
        force=args.force,
    )
    print(rep.summary(), flush=True)
    bj = [d for d in rep.dropped if d["code"].endswith(".BJ")]
    print(
        f"其中北交所 {len(bj)} 只；非北交所被剔除 {len(rep.dropped) - len(bj)} 只",
        flush=True,
    )
    for d in rep.dropped[:12]:
        print(
            f"  {d['code']:12s} ratio={d['screen_ratio']:.3f} 价格{d['price_days']:4d}天 "
            f"风控{d['screen_days']:4d}天 盲区{d['blind_span']} {d['reason']}",
            flush=True,
        )
    print("saved ->", args.out, flush=True)


if __name__ == "__main__":
    main()
