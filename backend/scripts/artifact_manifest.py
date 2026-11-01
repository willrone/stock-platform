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
]

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


def doc_cited_artifacts() -> set:
    """docs/ 下**所有** markdown 正文里点名的 data/backtest 产物（只算真实存在的）。

    为什么需要它：2026-11-01 的篡改实验证明，**文档引用 ≠ 被保护**。
    审计器只校验"文档里写的那几个数字"，清单只保护"自己登记的那几个文件"，
    两者之间的差集（22 个文档引用件）就是一个合法的静默漂移通道。
    → 把"文档点名"变成清单的**自动输入**，差集就再也藏不住。

    扫描范围：docs/**/*.md（含 docs/plans、docs/reports 等）+ 仓库根 md。
    2026-11-01 扩到全 docs 时新暴露 0 个（55 个引用件已全部在清单），
    但**范围必须写死在前门里**——否则下次有人在 docs/reports 引用产物又是盲区。
    """
    cited = set()
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
        for name in re.findall(r"([A-Za-z0-9_\-]+\.json)", text):
            if os.path.exists(os.path.join(DATA_DIR, name)):
                cited.add(name)
    return cited


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
    # 覆盖完整性：审计器读的每个产物都必须在 TRACKED 里（否则它可被静默覆盖）
    uncovered = [n for n in AUDITED if n not in TRACKED]
    # 覆盖完整性（2026-11-01 新增）：**文档正文点名的产物**也必须在 TRACKED 里。
    # 审计器只查"文档里写的那几个数"；清单只护"自己登记的文件"。
    # 两者差集 = 合法静默漂移通道（已用 pool_partition.json 篡改实验实证）。
    doc_cited = doc_cited_artifacts()
    uncovered_doc = sorted(n for n in doc_cited if n not in TRACKED)
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
    # 覆盖完整性（2026-11-01 新增）：文档正文点名的产物必须都在指纹清单里。
    # 两个口径（清单自报 + 当场按文档重算）都查：
    #   自报 → build 当时就漏了；重算 → 清单生成后文档又引用了新产物。
    for n in manifest.get("uncovered_doc_cited", []):
        bad.append(f"{n}: 文档引用但未纳入指纹清单")
    for n in sorted(doc_cited_artifacts() - set(TRACKED)):
        bad.append(f"{n}: 文档新引用了未登记产物 → 需登记进 TRACKED 并重新 build")
    for name, meta in manifest.get("files", {}).items():
        p = os.path.join(DATA_DIR, name)
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
