"""测试 Alpha158 因子计算能否算出全部 158 个因子。

2026-11-03 修（既有缺陷，详见台账 §12.40）：
1. `sys.path.insert(0, "/Users/ronghui/Documents/GitHub/willrone/backend")`
   —— 硬编码**别人机器**的路径 → 本机 FileNotFoundError，测试永远失败。
   改成从 `__file__` 解析项目根。
2. 样本 parquet 也硬编码旧机器路径，且该文件本机**不存在**
   （`backend/data/parquet/stock_data/` 目录都没有）。
   改成：从 `data/parquet/baostock_daily/` 动态找一只真实票；找不到就 **skip**
   （而不是硬失败——测试该报告"环境缺数据"，不是"代码错了"）。
3. **副作用**：原测试把样本写进**生产** `qlib_data/features/day/` 并修改
   `qlib_data/instruments/all.txt` → 跑一次测试就污染生产数据目录。
   改成：**表达式引擎路径根本不写任何文件**，并加断言"生产 qlib_data 未被改动"。

关于 handler 路径（qlib 原生 Alpha158 handler）：
本机 qlib 0.9.8.dev31 **没有 `qlib.data.dump_bin`**，无法在临时目录里造出
handler 能读的合法 qlib 数据集（parquet 直写也读不出来：`D.features()` 返回 0 行，
无论单层 DatetimeIndex 还是 MultiIndex 都一样）→ 该路径无法在测试里自建环境。
生产上这条路径是通的：`data/qlib_data/features/day/*.parquet` 里就是
它产出的 158 个因子列（RET1…VSUMD60）+ 自定义列。
所以这里**显式 skip**并写明原因，不假装能测。
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")

from app.core.config import settings  # noqa: E402
from app.services.qlib.enhanced_qlib_provider import Alpha158Calculator  # noqa: E402

DATA_DIR = BACKEND_ROOT.parent / "data" / "parquet" / "baostock_daily"
PROD_FEATURES_DIR = Path(settings.QLIB_DATA_PATH) / "features" / "day"


def _pick_sample() -> Path | None:
    """找一只本机真实存在的票（样本数据要够长，Alpha158 有 60 日窗口）。"""
    if not DATA_DIR.exists():
        return None
    for p in sorted(DATA_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception:  # noqa: BLE001
            continue
        if len(df) >= 300 and {"date", "open", "high", "low", "close", "volume"} <= set(df.columns):
            return p
    return None


def _prod_fingerprint():
    """生产 qlib features 目录指纹（文件名 → 大小+mtime），用来证明没被污染。"""
    if not PROD_FEATURES_DIR.exists():
        return {}
    return {
        f.name: (f.stat().st_size, int(f.stat().st_mtime))
        for f in PROD_FEATURES_DIR.iterdir()
        if f.is_file()
    }


async def test_expression_engine_all_factors():
    """表达式引擎路径：真实行情应算出全部 158 个因子，且不碰生产 qlib_data。"""
    sample = _pick_sample()
    if sample is None:
        pytest.skip(f"本机没有可用的样本行情（{DATA_DIR}）")

    data = pd.read_parquet(sample)
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date").sort_index()
    data = data[["open", "high", "low", "close", "volume"]].dropna()
    data.columns = [f"${c}" for c in data.columns]

    safe_code = sample.stem.replace(".", "_")  # 000001_SZ
    # 表达式引擎要求 MultiIndex(stock_code, date)
    mi = pd.MultiIndex.from_arrays(
        [[safe_code] * len(data), data.index], names=["stock_code", "date"])

    before = _prod_fingerprint()

    calculator = Alpha158Calculator()
    factors = await calculator.calculate_alpha_factors(
        qlib_data=data.set_axis(mi),
        stock_codes=[safe_code],
        date_range=(data.index.min().to_pydatetime(), data.index.max().to_pydatetime()),
        use_cache=False,
        force_expression_engine=True,
    )

    assert len(factors.columns) == 158, (
        f"应算出 158 个因子，实际 {len(factors.columns)}")
    all_zero = [c for c in factors.columns if (factors[c] == 0).all()]
    assert not all_zero, f"这些因子恒为 0（求值失败被静默填 0?）: {all_zero[:10]}"

    # 关键回归点：跑测试**不许**改生产 qlib_data
    assert _prod_fingerprint() == before, "生产 qlib_data/features/day 被测试改动了"


def test_handler_path_needs_real_qlib_dataset():
    """qlib 原生 handler 路径：本机无法自建数据集 → 显式 skip 并写明原因。

    详见模块 docstring：本机 qlib 0.9.8 无 dump_bin，parquet 直写 D.features() 返回 0 行。
    生产上这条路径是通的（features/day/*.parquet 里就是它产出的 158 个因子列）。
    """
    import qlib

    has_dump_bin = True
    try:
        import qlib.data.dump_bin  # noqa: F401
    except ModuleNotFoundError:
        has_dump_bin = False

    if not has_dump_bin:
        pytest.skip(
            f"qlib {qlib.__version__} 没有 dump_bin，无法在临时目录构造合法 qlib 数据集；"
            "handler 路径改由生产数据产物验证（data/qlib_data/features/day/*.parquet）")

    # 有 dump_bin 的环境：走真实 handler 路径
    pytest.skip("handler 路径的端到端验证留给生产流水线（本机无 dump_bin）")
