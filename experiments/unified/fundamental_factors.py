"""
基本面因子模块

从 Tushare 获取基本面数据（PE、PB、ROE、营收增长率等），
整合到现有技术因子中。

注意事项：
  - 基本面数据为季度频率，需要 forward fill 对齐到日频
  - 使用 point-in-time 原则，避免前瞻偏差
  - 缺失值用截面中位数填充
"""
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .constants import DATA_DIR, EPSILON

# === 基本面因子常量 ===
FUNDAMENTAL_CACHE_DIR = DATA_DIR.parent / "fundamental_cache"

# 需要获取的基本面指标
DAILY_BASIC_FIELDS = ["pe_ttm", "pb", "ps_ttm", "dv_ratio", "total_mv"]
FINA_INDICATOR_FIELDS = ["roe", "roa", "grossprofit_margin", "revenue_yoy"]

# 基本面因子衍生特征
FUNDAMENTAL_FEATURE_NAMES = [
    # 估值因子 (4)
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ratio",
    # 盈利因子 (3)
    "roe",
    "roa",
    "grossprofit_margin",
    # 成长因子 (1)
    "revenue_yoy",
    # 市值因子 (1)
    "log_market_cap",
    # 衍生因子 (4)
    "ep_ratio",
    "bp_ratio",
    "roe_stability",
    "value_growth_composite",
]


def get_fundamental_feature_names() -> List[str]:
    """获取基本面因子列名列表"""
    return FUNDAMENTAL_FEATURE_NAMES.copy()


def load_fundamental_data(
    ts_codes: List[str],
    start_date: str,
    end_date: str,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    加载基本面数据（优先从缓存读取）

    Args:
        ts_codes: 股票代码列表（如 ['000001.SZ', '600000.SH']）
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        cache_dir: 缓存目录

    Returns:
        包含 (ts_code, date, 基本面因子) 的 DataFrame
    """
    if cache_dir is None:
        cache_dir = FUNDAMENTAL_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / "fundamental_daily.parquet"

    if cache_path.exists():
        logger.info(f"从缓存加载基本面数据: {cache_path}")
        return _load_from_cache(cache_path, ts_codes, start_date, end_date)

    logger.info("缓存不存在，从 Tushare 获取基本面数据...")
    return _fetch_from_tushare(
        ts_codes, start_date, end_date, cache_path
    )


def _load_from_cache(
    cache_path: Path,
    ts_codes: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """从缓存文件加载并过滤"""
    df = pd.read_parquet(cache_path)

    mask = (
        df["ts_code"].isin(ts_codes)
        & (df["date"] >= start_date)
        & (df["date"] < end_date)
    )
    filtered = df[mask].copy()
    logger.info(f"缓存加载完成: {len(filtered)} 条记录")
    return filtered


def _fetch_from_tushare(
    ts_codes: List[str],
    start_date: str,
    end_date: str,
    cache_path: Path,
) -> pd.DataFrame:
    """从 Tushare 获取基本面数据"""
    try:
        import tushare as ts
    except ImportError:
        logger.warning("Tushare 未安装，返回空 DataFrame")
        return pd.DataFrame()

    import os

    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        logger.warning("TUSHARE_TOKEN 未设置，返回空 DataFrame")
        return pd.DataFrame()

    ts.set_token(token)
    pro = ts.pro_api()

    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")

    all_data = []
    total = len(ts_codes)

    for idx, code in enumerate(ts_codes):
        if idx % 50 == 0:
            logger.info(f"获取基本面数据: {idx}/{total}")

        try:
            daily_basic = _fetch_daily_basic(pro, code, start_fmt, end_fmt)
            fina_data = _fetch_fina_indicator(pro, code, start_fmt, end_fmt)

            if daily_basic is not None:
                if fina_data is not None:
                    daily_basic = _merge_fina_to_daily(daily_basic, fina_data)
                all_data.append(daily_basic)
        except Exception as e:
            logger.debug(f"获取 {code} 基本面数据失败: {e}")
            continue

    if not all_data:
        logger.warning("未获取到任何基本面数据")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)

    # 保存缓存
    result.to_parquet(cache_path, index=False)
    logger.info(f"基本面数据已缓存: {cache_path} ({len(result)} 条)")

    return result


def _fetch_daily_basic(
    pro, ts_code: str, start_date: str, end_date: str
) -> Optional[pd.DataFrame]:
    """获取每日基本面指标（PE、PB 等）"""
    try:
        df = pro.daily_basic(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,trade_date,pe_ttm,pb,ps_ttm,dv_ratio,total_mv",
        )
        if df is None or df.empty:
            return None

        df = df.rename(columns={"trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def _fetch_fina_indicator(
    pro, ts_code: str, start_date: str, end_date: str
) -> Optional[pd.DataFrame]:
    """获取财务指标（ROE、ROA、营收增长率等）"""
    try:
        df = pro.fina_indicator(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=(
                "ts_code,ann_date,roe,roa,"
                "grossprofit_margin,revenue_yoy"
            ),
        )
        if df is None or df.empty:
            return None

        df = df.rename(columns={"ann_date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        # 去重：同一公告日可能有多条记录，保留最新
        df = df.drop_duplicates(subset=["ts_code", "date"], keep="first")
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def _merge_fina_to_daily(
    daily: pd.DataFrame, fina: pd.DataFrame
) -> pd.DataFrame:
    """
    将季度财务数据合并到日频数据

    使用 merge_asof 实现 point-in-time 对齐：
    只使用公告日 <= 当前交易日的最新财务数据
    """
    daily = daily.sort_values("date")
    fina = fina.sort_values("date")

    fina_cols = [c for c in FINA_INDICATOR_FIELDS if c in fina.columns]
    if not fina_cols:
        return daily

    fina_subset = fina[["ts_code", "date"] + fina_cols].copy()

    merged = pd.merge_asof(
        daily,
        fina_subset,
        on="date",
        by="ts_code",
        direction="backward",
    )
    return merged


def compute_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算基本面衍生特征

    输入 DataFrame 需包含原始基本面字段（pe_ttm, pb, roe 等）

    新增特征：
      - log_market_cap: 对数市值（降低偏度）
      - ep_ratio: 1/PE（盈利收益率）
      - bp_ratio: 1/PB（账面价值收益率）
      - roe_stability: ROE 滚动标准差（盈利稳定性）
      - value_growth_composite: 价值成长复合因子
    """
    df = df.copy()

    # 对数市值
    if "total_mv" in df.columns:
        df["log_market_cap"] = np.log1p(
            df["total_mv"].clip(lower=0).fillna(0)
        )
    else:
        df["log_market_cap"] = 0.0

    # 盈利收益率 EP = 1/PE
    if "pe_ttm" in df.columns:
        pe = df["pe_ttm"].replace(0, np.nan)
        df["ep_ratio"] = 1.0 / (pe.abs() + EPSILON)
        # PE 为负时 EP 也为负
        df.loc[pe < 0, "ep_ratio"] = -df.loc[pe < 0, "ep_ratio"]
    else:
        df["ep_ratio"] = 0.0

    # 账面价值收益率 BP = 1/PB
    if "pb" in df.columns:
        pb = df["pb"].replace(0, np.nan)
        df["bp_ratio"] = 1.0 / (pb.abs() + EPSILON)
        df.loc[pb < 0, "bp_ratio"] = -df.loc[pb < 0, "bp_ratio"]
    else:
        df["bp_ratio"] = 0.0

    # ROE 稳定性（滚动 4 季度标准差）
    if "roe" in df.columns:
        df["roe_stability"] = (
            df.groupby("ts_code")["roe"]
            .transform(lambda x: x.rolling(4, min_periods=2).std())
        )
        df["roe_stability"] = df["roe_stability"].fillna(0)
    else:
        df["roe_stability"] = 0.0

    # 价值成长复合因子 = EP_rank + ROE_rank - Revenue_YoY_rank
    df["value_growth_composite"] = _compute_value_growth_composite(df)

    return df


def _compute_value_growth_composite(df: pd.DataFrame) -> pd.Series:
    """计算价值成长复合因子（截面排名组合）"""
    composite = pd.Series(0.0, index=df.index)

    if "ep_ratio" in df.columns and "date" in df.columns:
        composite += df.groupby("date")["ep_ratio"].rank(pct=True).fillna(0.5)

    if "roe" in df.columns and "date" in df.columns:
        composite += df.groupby("date")["roe"].rank(pct=True).fillna(0.5)

    if "revenue_yoy" in df.columns and "date" in df.columns:
        # 高增长排名靠前，但我们要 value+growth，所以加上
        composite += (
            df.groupby("date")["revenue_yoy"].rank(pct=True).fillna(0.5)
        )

    return composite


def merge_fundamental_to_technical(
    tech_df: pd.DataFrame,
    fund_df: pd.DataFrame,
    stock_col: str = "stock_code",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    将基本面因子合并到技术因子 DataFrame

    使用 forward fill 对齐频率差异：
      - 技术因子：日频
      - 基本面因子：日频（daily_basic）或季频（fina_indicator，已 merge_asof）

    Args:
        tech_df: 技术因子 DataFrame（含 stock_code, date 列）
        fund_df: 基本面因子 DataFrame（含 ts_code, date 列）
        stock_col: 技术 DF 中的股票代码列名
        date_col: 日期列名

    Returns:
        合并后的 DataFrame
    """
    if fund_df.empty:
        logger.warning("基本面数据为空，跳过合并")
        return tech_df

    # 统一股票代码格式
    fund_df = fund_df.copy()
    if "ts_code" in fund_df.columns and stock_col != "ts_code":
        fund_df = fund_df.rename(columns={"ts_code": stock_col})

    # 确保日期格式一致
    tech_df[date_col] = pd.to_datetime(tech_df[date_col])
    fund_df[date_col] = pd.to_datetime(fund_df[date_col])

    # 计算衍生特征
    fund_df = compute_fundamental_features(fund_df)

    # 选择要合并的列
    fund_feature_cols = [
        c for c in FUNDAMENTAL_FEATURE_NAMES if c in fund_df.columns
    ]
    merge_cols = [stock_col, date_col] + fund_feature_cols
    fund_subset = fund_df[
        [c for c in merge_cols if c in fund_df.columns]
    ].copy()

    # 去重
    fund_subset = fund_subset.drop_duplicates(
        subset=[stock_col, date_col], keep="last"
    )

    # 合并
    merged = tech_df.merge(
        fund_subset, on=[stock_col, date_col], how="left"
    )

    # Forward fill 基本面因子（按股票分组）
    for col in fund_feature_cols:
        if col in merged.columns:
            merged[col] = merged.groupby(stock_col)[col].ffill()

    # 剩余缺失值用截面中位数填充
    for col in fund_feature_cols:
        if col in merged.columns:
            median_fill = merged.groupby(date_col)[col].transform("median")
            merged[col] = merged[col].fillna(median_fill).fillna(0)

    n_fund = len(fund_feature_cols)
    n_tech = len(tech_df.columns) - 2  # 减去 stock_code 和 date
    logger.info(
        f"因子合并完成: {n_tech} 技术因子 + {n_fund} 基本面因子 "
        f"= {n_tech + n_fund} 总因子"
    )

    return merged
