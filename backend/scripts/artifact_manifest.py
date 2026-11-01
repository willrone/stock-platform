#!/usr/bin/env python3
"""回测产物指纹清单 —— 防"静默覆盖 / 产物丢失"。

## 为什么需要

- `data/` 被 .gitignore，**所有回测产物零版本控制**。
- 文档里的每个数字都靠 `audit_doc_numbers.py` 回读 JSON 产物核对。
- 一旦某个历史脚本（或迟到的后台进程）覆盖了 JSON，文档数字就不可验证，
  而且**不会有任何报错** —— 审计会"成功"核对出一份已经被改过的产物。

## 做法

`build` 模式：对关键产物算 SHA256，连同生成时间、窗口、脚本 git commit 写进
`docs/backtest/ARTIFACT_MANIFEST.json`（**这个文件纳入 git**，是唯一的版本化锚）。

`verify` 模式：重算 SHA256 与清单比对，任何一个产物被改动/丢失都报出来。

## 用法

    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/artifact_manifest.py build
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/artifact_manifest.py verify
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
DATA_DIR = os.path.join(ROOT, "data", "backtest")
MANIFEST = os.path.join(ROOT, "docs", "backtest", "ARTIFACT_MANIFEST.json")
DOCS_DIR = os.path.join(ROOT, "docs", "backtest")

# 关键产物：文档数字全部依赖这些文件
TRACKED = [
    "ALL_real_lag1_v2.json",              # A 股腿权威基线
    "multi_asset_l3.json",                # L3 类内动量
    "multi_asset_l3_oos.json",
    "multi_asset_p2_main.json",           # P2 商品类内 + 类间配权
    "multi_asset_p2_oos.json",
    "multi_asset_p2_long.json",
    "multi_asset_p2_attrib.json",
    "multi_asset_p3_main.json",           # P3 bond 类纠错
    "multi_asset_p3_oos.json",
    "multi_asset_p3_long.json",
    "p3_recon.json",
    "p3_bond_recon.json",
    "multi_asset_p4_main.json",           # P4 配权
    "multi_asset_p4_oos.json",
    "multi_asset_p4_long.json",
    "p5_class_audit.json",                # P5 分类审计
    "multi_asset_p6_main.json",           # P6 主腿权重敏感性
    "multi_asset_p6_oos.json",
    "p6_order_diag.json",                 # 顺序不变性回归测试
    "multi_asset_p7_main.json",           # P7 主腿因子消融
    "multi_asset_p7_oos.json",
    "multi_asset_p8_main.json",           # P8 暴露中性化复裁
    "multi_asset_p8_oos.json",
    "multi_asset_p9_main.json",           # P9 主腿调仓频率曲线
    "multi_asset_p9_oos.json",
    # ↓ 以下为 audit_doc_numbers.py 的 ART 字典里其余产物
    "multi_asset_final_robustness.json",  # FR 稳健性（main）
    "multi_asset_final_robustness_oos.json",
    "multi_asset_long_history.json",      # 7.7 年长历史
    "multi_asset_controls.json",          # 控制组
    "multi_asset_xclass.json",            # P1 跨类动量
    "multi_asset_xclass_oos.json",
    "multi_asset_xclass_long_history.json",
    "multi_asset_p10_gate.json",        # P10 验证门：features_full 逐位复现
    "multi_asset_p10_long.json",        # P10 A 股腿 5 窗口长历史
    "multi_asset_p10_stats.json",       # P10 跨窗口统计（派生）
    "multi_asset_p11_alpha.json",       # P11 A 股腿 alpha/beta 分解（4 基准口径）
    "multi_asset_p11_stats.json",       # P11 跨窗口统计（派生）
    "multi_asset_p12_main.json",        # P12 α 夏普口径重评 w（main）
    "multi_asset_p12_oos.json",         # P12 α 夏普口径重评 w（oos）
    "multi_asset_p12_main_degraded.json",  # P12 配置敏感性（降级袖套, main）
    "multi_asset_p12_oos_degraded.json",   # P12 配置敏感性（降级袖套, oos）
    "multi_asset_p12_long.json",        # P12 长历史 3 窗口组合分解
    "multi_asset_p12_longw.json",       # P12 长历史 3 窗口 w 曲线 + 5 窗口汇总
    "multi_asset_p12_diag.json",        # P12 E1~E6 根因诊断
    # ── P13 / P14（2026-10-31 补登：此前 P13 产物漏登记，等于没有指纹保护）──
    "multi_asset_p13_main.json",        # P13 袖套资产级权重优化（main）
    "multi_asset_p13_oos.json",         # P13 袖套资产级权重优化（oos）
    "multi_asset_p13_long.json",        # P13 长历史 3 窗口
    "multi_asset_p14_main.json",        # P14 Oracle 上界（main）
    "multi_asset_p14_oos.json",         # P14 Oracle 上界（oos）
    # ── P15：past_j 长历史独立审判（2026-11-01 补登）──
    # 这一条是**实查出来的洞**：P15 产物既不在 TRACKED 也不在 AUDITED，
    # 于是一份未提交的中间版脚本可以把它整个覆盖掉，而**没有任何机制发现**
    # （文档数字还指着它，却没人比对它）。补登记后指纹门开始守它。
    "multi_asset_p15_long_pastj.json",  # P15 长历史 3 窗口 past_j/oracle_s
    # ── P17：类内选标的 oracle 上界 + 袖套调仓频率 ──
    "multi_asset_p17_single_main.json",
    "multi_asset_p17_single_oos.json",
    "multi_asset_p17_verify_main.json",
    "multi_asset_p17_verify_oos.json",
    "multi_asset_p17_freq_main.json",
    "multi_asset_p17_freq_oos.json",
    # ── P18：42f 之上叠加 gbm 增量（分数级 z-score 融合）──
    # p18_oos 按预注册**故意不存在**（main 即被判据 (a) 否决 → §10.2/§11.6 不跑 oos）
    # → 放 EXPECTED_MISSING，而不是从 TRACKED 删掉（删掉会破坏 AUDITED ⊆ TRACKED 覆盖门）
    "multi_asset_p18_main.json",
    "multi_asset_p18_oos.json",
    # ── 试验预算账本（只增不减；register 后由 trial_budget.py 自动重建清单）──
    "trial_budget.json",
    # ── 文档引用件（2026-11-01 补）────────────────────────────────
    # 背景：实证发现**文档正文引用的 21 个产物此前一份都没有指纹保护**。
    # 篡改实验：改 pool_partition.json 的 baseline_sharpe/universe_size →
    #   指纹门 ✅ 通过（不归它管）＋ 文档审计 ✅ 也通过（它只查被引用的那几个数）
    #   → 引用得保护、清单不保护 = 静默漂移的合法通道。
    # 处置：凡是 docs/backtest/*.md 正文点名的产物，一律纳入指纹。
    #（`uncovered_doc_cited` 闸门保证以后新文档引用新产物时会自动报错）
    "_blend_nav_25f.json",
    "baseline_final.json",
    "deflated_sharpe.json",
    "ensemble_correlation.json",
    "full_history_features_progress.json",
    "full_history_reconcile.json",
    "fundamental_ic.json",
    "fundseg_seg3_fund.json",
    "gbm_lag1.json",
    "gbm_ref_baselines.json",
    "methods_5y_validation.json",
    "methods_engine_v2.json",
    "methods_weights_fixed.json",
    "multi_asset_leg.json",
    "multi_asset_leg_oos.json",
    "multi_asset_oos_validation.json",
    "multi_asset_p18_oos_smoke.json",
    "multi_asset_robustness.json",
    "neut_segments.json",
    "pool_partition.json",
    "turnover_truth.json",
    # ── 闸门自检件（2026-11-01 §12.23 D）────────────────────────────
    # 这份产物原本既不在 TRACKED、也不被任何文档引用 → 不受任何门保护。
    # 我在台账 §12.23 正文里**点了它的名字**（本想举例）→ 它当场变成
    # 「文档引用件」→ 新闸门立刻报「文档新引用了未登记产物」并让 verify 变红。
    # → 这就是闸门设计意图的直接实证：**差集藏不住，而且当场就藏不住**。
    "verify_10y_pit.json",
    # ── 袖套实现等价性验证结论（2026-11-01 归位）────────────────────
    # 原写在 backend/scripts/sleeve_verify.json → 落在指纹门覆盖范围之外，
    # 等于一份没人守护的结论。归位到 data/backtest/ 并纳入指纹。
    "sleeve_verify.json",
    # ── 产品口径 parity + 袖套漂移诊断（2026-11-01 归位）──────────
    # 三个产物原先都写在 backend/scripts/ → 在指纹门覆盖范围之外，
    # 而它们的结论都被台账正文引用（§12.12 交付清单第 5 项、§12.11 类权重偏离口径）。
    # 归位到 data/backtest/ 并纳入指纹。三个产物均无易变字段 → 指纹稳定。
    "parity_backtest_main.json",
    "diag_sleeve_drift.json",
    "diag_sleeve_class_order.json",
]

# ── 跨目录证据件（键 = 相对 ROOT 的路径）──────────────────────────
# 2026-11-01：台账之外的旧文档（BASELINE_CORRECTION / CONCLUSION /
# METHODS_RESEARCH 等）还引用了 26 个**不在 data/backtest/ 下**的产物
# （多在 data/strategy_baseline/）→ 同样"文档引用 ≠ 被保护"。
# 键用相对 ROOT 的路径，避免与 data/backtest 的 basename 键空间撞键。
TRACKED_EXTRA = [
    # ── data/strategy_baseline/：A 股腿基线/IC/DSR/随机对照等研究证据 ──
    "data/strategy_baseline/baseline_full_market_3y.json",
    "data/strategy_baseline/expanded_legs_10y.json",
    "data/strategy_baseline/factor_ranking_10y_fixed.json",
    "data/strategy_baseline/ic_composite_10y_execution_ab.json",
    "data/strategy_baseline/ic_composite_10y_hygiene.json",
    "data/strategy_baseline/ic_composite_10y_registered_topk.json",
    "data/strategy_baseline/ic_composite_10y_signfix.json",
    "data/strategy_baseline/ic_composite_rolling_walkforward.json",
    "data/strategy_baseline/ic_composite_walkforward.json",
    "data/strategy_baseline/moving_average_fix_3y.json",
    "data/strategy_baseline/portfolio_vol_experiment.json",
    "data/strategy_baseline/quality_weight_10y.json",
    "data/strategy_baseline/risk_parity_10y_dsr_final.json",
    "data/strategy_baseline/risk_parity_10y_experiment.json",
    "data/strategy_baseline/risk_parity_3y_control.json",
    "data/strategy_baseline/risk_parity_dsr_final.json",
    "data/strategy_baseline/sharpe_overlay_experiment.json",
    "data/strategy_baseline/sharpe_serial_verify.json",
    "data/strategy_baseline/sharpe_vol_sweep.json",
    "data/strategy_baseline/sharpe_vol_sweep_serial.json",
    "data/strategy_baseline/stochastic_signal_fullmarket.json",
    "data/strategy_baseline/stochastic_signal_fullmarket_yearly.json",
    "data/strategy_baseline/stochastic_signal_grid.json",
    # ── 静态定义件（应当永不静默变化）──
    "data/etf/etf_universe.json",
    "data/industry/industry_map.json",
    "data/stocks/st_names.json",
]

# ── 活状态：明确**不**做指纹化（会天天重写，指纹化等于让门天天红）──
# 它们的可复现性由各自管道的幂等性保证（纸面盘同成交日重跑不重复下单、
# 信号按日重生成），不是靠哈希锁。
LIVE_STATE_PREFIXES = (
    "data/paper/",       # 纸面盘持仓/账本/快照：每周一重写
    "data/signals/",     # 每日信号产出：每天重写
    "data/tdx_daily/",   # 行情管道状态：每天重写
)

# ── 预期缺失（故意不产出的产物）──────────────────────────────────
# 语义区分：**产物从不存在**（按预注册不跑）vs **产物本来存在但丢了**（真故障）。
# 两类混在一个 `missing` 列表里 → verify 每次都打同一条 ⚠️ 假警报，
# 真故障发生时会被淹没（告警疲劳 = 告警失效）。
# 规则：进入本表的产物，必须在台账里写明「按预注册不产出」的理由与章节号。
EXPECTED_MISSING = {
    # P18 在 main 窗口即被预注册判据 (a) 否决 → 按 §10.2/§11.6「main 通过后才跑 oos」，
    # oos 产物故意不存在（不是漏跑、不是丢失）。
    "multi_asset_p18_oos.json": "P18 main 被判据(a)否决，按预注册 §10.2/§11.6 不跑 oos",
}

# 审计器读取的产物文件 —— 必须与 TRACKED 一一对应（漏一个 = 该产物可被静默覆盖）
AUDITED = [
    "multi_asset_final_robustness.json",
    "multi_asset_final_robustness_oos.json",
    "multi_asset_l3.json",
    "multi_asset_l3_oos.json",
    "multi_asset_long_history.json",
    "multi_asset_controls.json",
    "multi_asset_xclass.json",
    "multi_asset_xclass_oos.json",
    "multi_asset_xclass_long_history.json",
    "multi_asset_p2_main.json",
    "multi_asset_p2_oos.json",
    "multi_asset_p2_long.json",
    "multi_asset_p2_attrib.json",
    "multi_asset_p4_main.json",
    "multi_asset_p4_oos.json",
    "multi_asset_p4_long.json",
    "p5_class_audit.json",
    "multi_asset_p6_main.json",
    "multi_asset_p6_oos.json",
    "p6_order_diag.json",
    "multi_asset_p3_main.json",
    "multi_asset_p3_oos.json",
    "multi_asset_p3_long.json",
    "p3_bond_recon.json",
    "ALL_real_lag1_v2.json",
    "multi_asset_p7_main.json",
    "multi_asset_p7_oos.json",
    "multi_asset_p8_main.json",
    "multi_asset_p8_oos.json",
    "multi_asset_p9_main.json",
    "multi_asset_p9_oos.json",
    "multi_asset_p10_gate.json",
    "multi_asset_p10_long.json",
    "multi_asset_p10_stats.json",
    "multi_asset_p11_alpha.json",
    "multi_asset_p11_stats.json",
    "multi_asset_p12_main.json",
    "multi_asset_p12_oos.json",
    "multi_asset_p12_main_degraded.json",
    "multi_asset_p12_oos_degraded.json",
    "multi_asset_p12_long.json",
    "multi_asset_p12_longw.json",
    "multi_asset_p12_diag.json",
    # ── P13 / P14 / P15（审计器实际读取的产物）──
    # 2026-11-01：P13/P14/P15 三份文档此前一条表格断言都没有。
    # 补断言后，P13 main/oos/长历史、P14 main/oos、P15 长历史成了审计器
    # **真实读取**的产物 → 必须登记进 AUDITED
    # （否则它们可被静默覆盖，而文档数字还指着它们）。
    "multi_asset_p13_main.json",
    "multi_asset_p13_oos.json",
    "multi_asset_p13_long.json",
    "multi_asset_p14_main.json",
    "multi_asset_p14_oos.json",
    "multi_asset_p15_long_pastj.json",
    "multi_asset_p17_single_main.json",
    "multi_asset_p17_single_oos.json",
    "multi_asset_p17_verify_main.json",
    "multi_asset_p17_verify_oos.json",
    "multi_asset_p17_freq_main.json",
    "multi_asset_p17_freq_oos.json",
    # ── P18：42f 之上叠加 gbm 增量（分数级 z-score 融合）──
    "multi_asset_p18_main.json",
    "multi_asset_p18_oos.json",
]


# ── 指纹规范化 ──────────────────────────────────────────────
# 产物 JSON 里含 wall_s（墙钟耗时）。同一实验重跑时**所有策略结果字段逐位相同**，
# 只有耗时不同 → 整文件 sha256 天然不稳定 → 指纹门会对任何重跑误报「产物被覆盖」。
#
# 实证（本项目）：P7 同一臂连续跑两次，114 个字段中**仅** wall_s 不同（92.9 vs 92.6），
# 其余全部逐位一致 → 引擎数值是确定性的，不稳定源只有耗时。
#
# 因此：JSON 产物先剔除易变字段再算规范化 sha256；非 JSON 走整文件字节。
# 易变字段白名单必须**最窄** —— 逐个产物扫过，只收"墙钟耗时/生成时刻"类字段，
# 绝不误剔业务字段（如交易日数、样本区间）。
# 2026-11-01 扩到 21 个文档引用件后补到 5 个：
#   wall_s / total_wall_s / mainloop_sub_progress_update_s /
#   precompute_sub_build_date_index_s → 墙钟耗时（同一实验重跑只差这些）
#   generated_at                        → 运行时刻戳
# 其余（如 candidate_main_best_rule_arm、candidate_selected_on）是**确定性选择元信息**，
# 不剥 —— 剥了就等于放弃对"选中了哪条规则"的保护。
VOLATILE_KEYS = (
    "wall_s",
    "total_wall_s",
    "mainloop_sub_progress_update_s",
    "precompute_sub_build_date_index_s",
    "generated_at",
)
# 指纹算法版本：**白名单变了 → 版本必须变**，否则旧清单（少剥几个字段）
# 算出的哈希与新算法不一致，会被误读成"产物被覆盖"。
SHA256_MODE = "normalized-json-v2"


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(x) for x in obj]
    return obj


def sha256(path: str) -> str:
    """产物指纹。JSON 走规范化（剔易变字段 + 统一序列化），其余走整文件字节。"""
    if path.endswith(".json"):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except (ValueError, OSError):
            obj = None
        if obj is not None:
            canon = json.dumps(_strip_volatile(obj), ensure_ascii=False,
                               sort_keys=True, indent=2)
            return hashlib.sha256(canon.encode("utf-8")).hexdigest()
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=20,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _key_for(path: str) -> str:
    """清单键：data/backtest/ **直属**文件用 basename，其余用相对 ROOT 的路径。

    为什么不用一种键：data/backtest 下有同名风险（子目录与顶层可能撞名），
    而跨目录证据件本来就得靠路径区分 → 混用两种键，但**键里带 "/" 的一定是路径**，
    不会与 basename 撞键。
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent == os.path.abspath(DATA_DIR):
        return os.path.basename(path)
    return os.path.relpath(os.path.abspath(path), ROOT).replace(os.sep, "/")


def _resolve(key: str) -> str:
    """键 → 绝对路径（basename 键在 data/backtest 下，其余相对 ROOT）。"""
    if "/" in key:
        return os.path.join(ROOT, key.replace("/", os.sep))
    return os.path.join(DATA_DIR, key)


def _doc_mentioned_json_names() -> set:
    """docs/ 下所有 markdown 正文里出现过的 json 文件名（只取名字，不判存在）。"""
    names = set()
    docs_root = os.path.join(ROOT, "docs")
    paths: List[str] = []
    if os.path.isdir(docs_root):
        paths += glob.glob(os.path.join(docs_root, "**", "*.md"), recursive=True)
    paths += glob.glob(os.path.join(ROOT, "*.md"))
    for md in sorted(paths):
        try:
            with open(md, "r", encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        names.update(re.findall(r"([A-Za-z0-9_\-]+\.json)", text))
    return names


def doc_cited_artifacts() -> set:
    """docs/ 下**所有** markdown 正文点名的 data/ 产物（键空间与清单一致）。

    为什么需要它：2026-11-01 的篡改实验证明，**文档引用 ≠ 被保护**。
    审计器只校验"文档里写的那几个数字"，清单只保护"自己登记的那几个文件"，
    两者的差集就是一个合法的静默漂移通道。
    → 把"文档点名"变成清单的**自动输入**，差集就再也藏不住。

    扫描范围：整个 data/ 树（实测只有 322 个 json，扫全树很便宜）
    —— **不用目录白名单**：白名单会重犯"没被列到的就没人管"的原罪。
    活状态目录（data/paper、data/signals、data/tdx_daily）由
    LIVE_STATE_PREFIXES 显式排除，理由写在那个常量旁边。
    """
    cited = set()
    names = _doc_mentioned_json_names()
    if not names:
        return cited
    for path in glob.glob(os.path.join(ROOT, "data", "**", "*.json"), recursive=True):
        if os.path.basename(path) in names:
            cited.add(_key_for(path))
    return cited


def _is_live_state(key: str) -> bool:
    """该键是否属于显式排除的活状态目录（不要求指纹保护）。"""
    return any(key.startswith(pref) for pref in LIVE_STATE_PREFIXES)


def build() -> int:
    entries: Dict[str, Any] = {}
    missing: List[str] = []
    for name in TRACKED:
        p = os.path.join(DATA_DIR, name)
        if not os.path.exists(p):
            # 预期缺失（按预注册不产出）与真缺失分开记，别让假警报淹掉真故障
            if name not in EXPECTED_MISSING:
                missing.append(name)
            continue
        st = os.stat(p)
        entries[name] = {
            "sha256": sha256(p),
            "bytes": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        }
    # 跨目录证据件（键 = 相对 ROOT 的路径）
    for rel in TRACKED_EXTRA:
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.exists(p):
            missing.append(rel)
            continue
        st = os.stat(p)
        entries[rel] = {
            "sha256": sha256(p),
            "bytes": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        }
    # 覆盖完整性：审计器读的每个产物都必须在 TRACKED 里（否则它可被静默覆盖）
    uncovered = [n for n in AUDITED if n not in TRACKED]
    # 覆盖完整性（2026-11-01 新增）：**文档正文点名的产物**也必须被保护。
    # 审计器只查"文档里写的那几个数"；清单只护"自己登记的文件"。
    # 两者差集 = 合法静默漂移通道（已用 pool_partition.json 篡改实验实证）。
    doc_cited = doc_cited_artifacts()
    tracked_keys = set(TRACKED) | set(TRACKED_EXTRA)
    uncovered_doc = sorted(n for n in doc_cited
                           if n not in tracked_keys and not _is_live_state(n))
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "data_dir": os.path.relpath(DATA_DIR, ROOT),
        "sha256_mode": SHA256_MODE,
        "sha256_excludes": list(VOLATILE_KEYS),
        "files": entries,
        "missing": missing,
        "expected_missing": {k: v for k, v in EXPECTED_MISSING.items()
                             if not os.path.exists(os.path.join(DATA_DIR, k))},
        "uncovered_audited": uncovered,
        "doc_cited": sorted(doc_cited),
        "uncovered_doc_cited": uncovered_doc,
        "live_state_excluded": list(LIVE_STATE_PREFIXES),
    }
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"✓ 清单已写入 {os.path.relpath(MANIFEST, ROOT)}")
    print(f"  指纹 {len(entries)} 个产物"
          + (f"；缺失 {len(missing)} 个: {missing}" if missing else ""))
    return 0


def verify() -> int:
    if not os.path.exists(MANIFEST):
        print("❌ 清单不存在，请先 build")
        return 1
    with open(MANIFEST, encoding="utf-8") as fh:
        manifest = json.load(fh)
    bad: List[str] = []
    # 算法版本闸门：旧清单（整文件 sha256）不能冒充新算法 → 防止"换了算法却用旧清单"
    mode = manifest.get("sha256_mode")
    if mode != SHA256_MODE:
        bad.append(f"清单指纹算法版本不符：清单 {mode!r} ≠ 当前 {SHA256_MODE!r}"
                   f"（需重新 build；旧算法会把含 wall_s 的产物一律误报为被覆盖）")
    # 覆盖完整性：清单必须覆盖审计器依赖的全部产物
    for n in manifest.get("uncovered_audited", []):
        bad.append(f"{n}: 审计依赖但未纳入指纹清单")
    # 覆盖完整性（2026-11-01 新增）：文档正文点名的产物必须都被保护。
    # 两个口径（清单自报 + 当场按文档重算）都查：
    #   自报 → build 当时就漏了；重算 → 清单生成后文档又引用了新产物。
    for n in manifest.get("uncovered_doc_cited", []):
        bad.append(f"{n}: 文档引用但未纳入指纹清单")
    tracked_keys = set(TRACKED) | set(TRACKED_EXTRA)
    for n in sorted(doc_cited_artifacts() - tracked_keys):
        if _is_live_state(n):
            continue  # 活状态：显式排除（理由见 LIVE_STATE_PREFIXES）
        bad.append(f"{n}: 文档新引用了未登记产物 → 需登记进 TRACKED/TRACKED_EXTRA 并重新 build")
    for name, meta in manifest.get("files", {}).items():
        p = _resolve(name)
        if not os.path.exists(p):
            bad.append(f"{name}: 文件丢失")
            continue
        cur = sha256(p)
        if cur != meta["sha256"]:
            bad.append(f"{name}: sha256 不符\n      清单 {meta['sha256'][:16]}…"
                       f"\n      当前 {cur[:16]}…")
    # 预期缺失的产物如果**突然出现** → 必须重新 build 登记指纹
    # （否则它是个没人守护的产物：可被静默覆盖，而文档数字还指着它）
    for name in manifest.get("expected_missing", {}):
        if os.path.exists(os.path.join(DATA_DIR, name)):
            bad.append(f"{name}: 标记为「预期缺失」但文件已出现 → 需重新 build 登记指纹")
    if bad:
        print(f"❌ 产物指纹校验失败（{len(bad)} 项）：")
        for b in bad:
            print(f"   - {b}")
        print("\n   → 产物被覆盖或丢失，文档数字不可信。先查清是谁改的再重跑。")
        return 1
    if manifest.get("missing"):
        print(f"⚠️  清单记录的缺失产物：{manifest['missing']}")
    exp = manifest.get("expected_missing", {})
    if exp:
        print(f"ℹ️  预期缺失（按预注册不产出，非故障）：{list(exp)}")
    print(f"✅ 产物指纹校验通过：{len(manifest.get('files', {}))} 个产物与清单一致"
          f"（清单生成于 {manifest.get('generated_at')}，commit {manifest.get('git_commit')}）")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if mode == "build":
        sys.exit(build())
    elif mode == "verify":
        sys.exit(verify())
    else:
        print(__doc__)
        sys.exit(2)
