"""P12-E：w 曲线不可识别性的根因诊断 + 跨窗口「统一 w」评估。

P12-B/P12-D 已证：逐窗口 argmax 的最优 w 在 5 个窗口散开（跨度 0.85），
判据 (f)=False。但「散开」本身只是现象，本脚本要回答三个问题：

E1 峰位置是不是「事后择时」的影子？
   假设：组合夏普在「A 股腿相对袖套走弱的窗口」自然更靠 w→0，
        在「A 股腿走强的窗口」更靠 w→1。
   检验：最优 w 与该窗口 A 股腿夏普/相对袖套超额收益的 Spearman 秩相关。
   若秩相关≈1 → w 曲线在捕捉「事后选边」，不是可持有的配置比例。

E2 逐窗口 argmax 是过拟合，正确的 w 应该跨窗口「同一个值」。
   检验：在同一配置组内，选使「各窗口平均夏普」最大的统一 w，
        再看这个 w 在另一组（不同配置/时间）上的表现。

E3 定稿 w=0.50 在各窗口各口径下损失多少、排第几？
   （风险披露，不是调参依据）

只读：消费 P12 的 JSON 产物，不重跑回测。
"""
from __future__ import annotations

import json
import os
import statistics
from typing import Any, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "backtest")
DOCS = os.path.join(ROOT, "docs", "backtest")

METRICS = {
    "abs_sharpe": "combined_sharpe",
    "alpha_gz": "alpha_sharpe_IDX_399317",
    "alpha_ew": "alpha_sharpe_equal_weight_buyhold",
    "alpha_hs300": "alpha_sharpe_IDX_000300",
}
W_GRID = [round(0.05 * i, 2) for i in range(21)]


def spearman(xs: List[float], ys: List[float]) -> float:
    def rank(v: List[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def load_window(name: str) -> Dict[str, Any]:
    fp = os.path.join(OUT, f"multi_asset_p12_{name}.json")
    with open(fp, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    main_d = load_window("main")
    oos_d = load_window("oos")
    longw_d = load_window("longw")

    # ── 汇总各窗口的曲线 ──
    windows: Dict[str, Dict[str, Any]] = {}
    for wname, dd in (("main", main_d), ("oos", oos_d)):
        windows[wname] = {
            "group": "A_定稿三类配置",
            "a_sharpe": dd["a_share_sharpe"],
            "sleeve_sharpe": dd["sleeve"]["sharpe"],
            "results": dd["results"],
            "curves": dd["curves"],
        }
    for wname, entry in (longw_d.get("results") or {}).items():
        windows[wname] = {
            "group": "B_降级gold+bond",
            "a_sharpe": entry["a_share_sharpe"],
            "sleeve_sharpe": entry["sleeve_sharpe"],
            "results": entry["results"],
            "curves": entry["curves"],
            "held": entry.get("held"),
        }

    out: Dict[str, Any] = {"windows": {}, "diagnostics": {}}

    # ══ E1：最优 w 与 A 股腿强弱 / 相对袖套超额的秩相关 ══
    print("=" * 78)
    print("E1  峰位置是否 = 事后择时的影子（Spearman 秩相关）")
    print("=" * 78)
    e1: Dict[str, Any] = {}
    for mname, field in METRICS.items():
        best_ws, a_shs, rel = [], [], []
        for wname, w in windows.items():
            c = (w["curves"] or {}).get(mname)
            if not c:
                continue
            best_ws.append(float(c["best_w"]))
            a_shs.append(float(w["a_sharpe"]))
            # A 股腿相对袖套的超额（简单差，仅作强弱排序用）
            rel.append(float(w["a_sharpe"]) - float(w["sleeve_sharpe"]))
        if len(best_ws) >= 3:
            rho_a = spearman(best_ws, a_shs)
            rho_r = spearman(best_ws, rel)
            e1[mname] = {
                "n_windows": len(best_ws),
                "best_w": best_ws,
                "a_leg_sharpe": a_shs,
                "spearman_bestw_vs_aleg_sharpe": round(rho_a, 4),
                "spearman_bestw_vs_aleg_minus_sleeve": round(rho_r, 4),
            }
            print(f"  {mname:12s} n={len(best_ws)}  "
                  f"ρ(最优w, A腿夏普)={rho_a:+.3f}  ρ(最优w, A腿-袖套)={rho_r:+.3f}")
            print(f"      最优w  ={best_ws}")
            print(f"      A腿夏普={a_shs}")
    out["diagnostics"]["E1_peak_vs_leg_strength"] = e1

    # ══ E2：跨窗口「统一 w」—— 逐窗口 argmax 是过拟合的正确反例 ══
    print()
    print("=" * 78)
    print("E2  跨窗口统一 w：平均夏普最高的 w（不是逐窗口 argmax）")
    print("=" * 78)
    e2: Dict[str, Any] = {}
    for gname, wnames in (("A_定稿三类配置", ["main", "oos"]),
                          ("B_降级gold+bond", ["w2013", "w2016", "w2019"])):
        avail = [x for x in wnames if x in windows]
        if len(avail) < 2:
            continue
        for mname, field in METRICS.items():
            curve = []
            for w in W_GRID:
                lab = f"w{int(round(w * 100)):03d}"
                vals = [windows[x]["results"].get(lab, {}).get(field) for x in avail]
                vals = [v for v in vals if v is not None]
                if len(vals) != len(avail):
                    curve = []
                    break
                curve.append((w, statistics.fmean(vals)))
            if not curve:
                continue
            bw, bv = max(curve, key=lambda t: t[1])
            # 该统一 w 在各窗口的单独值 + 组内逐窗口 argmax 对比
            lab = f"w{int(round(bw * 100)):03d}"
            per = {x: windows[x]["results"].get(lab, {}).get(field) for x in avail}
            argmax_each = {x: (windows[x]["curves"].get(mname) or {}).get("best_w") for x in avail}
            e2[f"{gname}|{mname}"] = {
                "windows": avail,
                "uniform_best_w": bw,
                "uniform_best_mean": round(bv, 4),
                "per_window_value_at_uniform_w": per,
                "per_window_argmax": argmax_each,
                "mean_curve": [(w, round(v, 4)) for w, v in curve],
            }
            print(f"  [{gname}] {mname:12s} 统一最优 w={bw:.2f} 平均={bv:.4f}  "
                  f"(逐窗口 argmax: {argmax_each})")
    out["diagnostics"]["E2_uniform_w_across_windows"] = e2

    # ══ E3：定稿 w=0.50 的排名与损失 ══
    print()
    print("=" * 78)
    print("E3  定稿 w=0.50 在各窗口各口径下：排名 / 相对最优的损失")
    print("=" * 78)
    e3: Dict[str, Any] = {}
    for mname, field in METRICS.items():
        rows = []
        for wname, w in windows.items():
            c = (w["curves"] or {}).get(mname)
            if not c:
                continue
            vals = [(float(w["results"][f"w{int(round(ww * 100)):03d}"].get(field)), ww)
                    for ww in W_GRID
                    if w["results"].get(f"w{int(round(ww * 100)):03d}", {}).get(field) is not None]
            if not vals:
                continue
            vals_sorted = sorted(vals, key=lambda t: -t[0])
            v050 = w["results"].get("w050", {}).get(field)
            # w=0.50 在 21 点网格中的名次（1 = 最优）
            rank = next((i + 1 for i, (v, ww) in enumerate(vals_sorted)
                         if abs(ww - 0.50) < 1e-9), None)
            # 最优 w 的名次恒为 1（自检）
            assert vals_sorted[0][1] == float(c["best_w"]), (wname, mname)
            rows.append({
                "window": wname,
                "group": w["group"],
                "best_w": c["best_w"],
                "best_value": c["best_value"],
                "value_at_w050": v050,
                "loss_vs_best": round(c["best_value"] - v050, 4) if v050 is not None else None,
                "rank_of_w050": rank,
                "n_grid": len(vals),
            })
            print(f"  {mname:12s} {wname:7s} 最优 w={c['best_w']:.2f} 值={c['best_value']:.4f} | "
                  f"w=0.50 值={v050} 损失={rows[-1]['loss_vs_best']} 排名 {rank}/{len(vals)}")
        e3[mname] = rows
    out["diagnostics"]["E3_w050_risk_disclosure"] = e3

    # ══ E4：随机 w 对照 —— 逐窗口 argmax 的优势是否超过随机 ══
    print()
    print("=" * 78)
    print("E4  逐窗口 argmax 相对随机 w 的优势（是否只是曲线噪声）")
    print("=" * 78)
    e4: Dict[str, Any] = {}
    for mname, field in METRICS.items():
        rows = []
        for wname, w in windows.items():
            c = (w["curves"] or {}).get(mname)
            if not c or c.get("random_max") is None:
                continue
            gap = float(c["best_value"]) - float(c["random_max"])
            rows.append({"window": wname, "best_value": c["best_value"],
                         "random_max": c["random_max"], "gap": round(gap, 4),
                         "gap_pct_of_range": None})
            print(f"  {mname:12s} {wname:7s} 最优={c['best_value']:.4f} 随机最大={c['random_max']:.4f} "
                  f"优势={gap:+.4f}")
        e4[mname] = rows
    out["diagnostics"]["E4_argmax_vs_random"] = e4



    # ══ E5：跨窗口迁移测试 —— 在 A 窗口选的 w，用到 B 窗口还成立吗？ ══
    print()
    print("=" * 78)
    print("E5  跨窗口迁移：源窗口的 argmax w 在目标窗口 vs w=0.50 谁更好？")
    print("=" * 78)
    e5: Dict[str, Any] = {}
    wnames_all = list(windows.keys())
    for mname, field in METRICS.items():
        rows = []
        wins = 0
        tot = 0
        for src in wnames_all:
            csrc = (windows[src]["curves"] or {}).get(mname)
            if not csrc:
                continue
            src_w = float(csrc["best_w"])
            for tgt in wnames_all:
                if tgt == src:
                    continue
                tt = windows[tgt]["results"]
                lab = f"w{int(round(src_w * 100)):03d}"
                v_src = tt.get(lab, {}).get(field)
                v_050 = tt.get("w050", {}).get(field)
                if v_src is None or v_050 is None:
                    continue
                tot += 1
                better = v_src > v_050
                wins += 1 if better else 0
                rows.append({
                    "src": src, "tgt": tgt, "src_best_w": src_w,
                    "value_at_src_w": v_src, "value_at_w050": v_050,
                    "src_w_better": better,
                    "delta": round(v_src - v_050, 4),
                    "same_group": windows[src]["group"] == windows[tgt]["group"],
                })
        e5[mname] = {
            "n_pairs": tot,
            "src_w_beats_w050": wins,
            "hit_rate": round(wins / tot, 4) if tot else None,
            "rows": rows,
        }
        print(f"  {mname:12s} 迁移命中率 {wins}/{tot} = {wins / tot:.1%} "
              f"（源窗口 argmax w 优于目标窗口 w=0.50 的比例）")
    out["diagnostics"]["E5_cross_window_transfer"] = e5

    # ══ E6：同组内迁移（A 组内部、B 组内部）—— 配置不同不可直接迁移 ══
    print()
    print("=" * 78)
    print("E6  同配置组内迁移命中率（跨组迁移受配置差异污染，不计）")
    print("=" * 78)
    e6: Dict[str, Any] = {}
    for mname in METRICS:
        rows = e5[mname]["rows"]
        same = [r for r in rows if r["same_group"]]
        cross = [r for r in rows if not r["same_group"]]
        sh = sum(1 for r in same if r["src_w_better"])
        ch = sum(1 for r in cross if r["src_w_better"])
        e6[mname] = {
            "same_group_n": len(same), "same_group_hits": sh,
            "same_group_hit_rate": round(sh / len(same), 4) if same else None,
            "cross_group_n": len(cross), "cross_group_hits": ch,
            "cross_group_hit_rate": round(ch / len(cross), 4) if cross else None,
        }
        print(f"  {mname:12s} 同组 {sh}/{len(same)} = "
              f"{sh / len(same):.1%}" if same else f"  {mname:12s} 同组无样本", end="")
        print(f"  | 跨组 {ch}/{len(cross)} = {ch / len(cross):.1%}" if cross else "  | 跨组无样本")
    out["diagnostics"]["E6_within_group_transfer"] = e6

    # ══ F：最终预注册判据（跨窗口合并）—— 供文档表格逐格核对 ══
    # runner 的 ab 阶段只能判单窗口判据，(c)（oos 不劣于基线 −0.02）与 (f)
    # （独立样本外证据）必须等两个窗口都跑完才能判。这里补齐。
    print()
    print("=" * 78)
    print("F   最终预注册判据（主口径 alpha_gz，跨窗口合并）")
    print("=" * 78)
    primary = "alpha_gz"
    field = METRICS[primary]
    raw = {"main": main_d["criteria"], "oos": oos_d["criteria"]}

    # (c)：把 main 选出的 argmax w 迁到 oos 窗口，看是否 ≥ oos 基线 − 0.02
    src_w = float(main_d["curves"][primary]["best_w"])
    lab = f"w{int(round(src_w * 100)):03d}"
    v_src = oos_d["results"][lab][field]
    v_base = oos_d["results"]["w050"][field]
    c_delta = round(v_src - v_base, 4)
    c_pass = bool(c_delta >= -0.02)
    c_transfer = {
        "src": "main", "tgt": "oos", "src_best_w": src_w,
        "value_at_src_w": v_src, "value_at_w050": v_base,
        "delta": c_delta, "threshold": -0.02, "pass": c_pass,
        "note": ("主窗口 argmax w 迁到 oos 的表现；"
                 "用 oos 自身 argmax 比基线属于样本内比较，不作判据"),
    }

    # (f)：5 窗口一致（长历史）
    gz_cw = longw_d["cross_window"][primary]
    f_pass = bool(longw_d["criteria"]["f_alpha_gz_best_w_consistent_5w"])
    f_consistency = {
        "spread": gz_cw["spread"],
        "n_in_0.40_0.60": gz_cw["n_in_0.40_0.60"],
        "mode_best_w": gz_cw["mode_best_w"],
        "mode_count": gz_cw["mode_count"],
        "n_windows": gz_cw["n_windows"],
        "pass": f_pass,
    }

    table = {
        "main": dict(raw["main"]),
        "oos": dict(raw["oos"]),
    }
    table["oos"]["c_oos_alpha_sharpe_ge_base_minus_0.02"] = c_pass
    table["main"]["f_independent_oos_evidence"] = f_pass
    table["oos"]["f_independent_oos_evidence"] = f_pass
    out["diagnostics"]["F_final_criteria"] = {
        "primary_metric": primary,
        "criteria_raw": raw,
        "table": table,
        "cross_window": {"c_transfer": c_transfer, "f_consistency": f_consistency},
        "note": ("(a)(b)(d)(e) 由单窗口 runner 判定（主口径 alpha_gz）；"
                 "(c) 由 main→oos 迁移判定；(f) 由 5 窗口一致性判定。"
                 "全部判据必须同时通过才允许改 w —— 未全过 → 定稿不变。"),
    }
    for wname in ("main", "oos"):
        marks = {k: ("✅" if v is True else ("—" if v is None else "❌"))
                 for k, v in table[wname].items()}
        print(f"  {wname}: " + "  ".join(f"{k}={m}" for k, m in marks.items()))
    print(f"  (c) main→oos 迁移 Δ={c_delta:+.4f}（阈值 −0.02）→ "
          f"{'✅' if c_pass else '❌'}")
    print(f"  (f) 5 窗口跨度 {f_consistency['spread']:.2f} → "
          f"{'✅' if f_pass else '❌'}")

    fp = os.path.join(OUT, "multi_asset_p12_diag.json")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n✓ 写出 {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
