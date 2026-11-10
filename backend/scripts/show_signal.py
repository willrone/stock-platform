#!/usr/bin/env python3
"""读取最新纸面信号 JSON, 打印汇总并生成 data/signals/latest.html.

用法:
  env -u PYTHONPATH backend/.venv-py313/bin/python3 backend/scripts/show_signal.py
  open data/signals/latest.html
"""
from __future__ import annotations

import glob
import html
import json
import os
import webbrowser

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIGNAL_DIR = os.path.join(ROOT, "data", "signals")


def latest_signal() -> str:
    files = [f for f in glob.glob(os.path.join(SIGNAL_DIR, "signal_*.json"))
             if "_42f" not in os.path.basename(f)]
    if not files:
        raise SystemExit("没有 signal_*.json，先跑 weekly_signal.py")
    return max(files, key=os.path.getmtime)


def fmt_money(x) -> str:
    try:
        return f"¥{float(x):,.0f}"
    except Exception:
        return "-"


def table(rows, kind: str) -> str:
    if not rows:
        return '<p class="empty">无</p>'
    if kind == "hold":
        cells = "".join(f"<span class='tag'>{html.escape(str(c))}</span>" for c in rows)
        return f"<div class='tags'>{cells}</div>"
    head = ("代码", "名称", "股数", "价格", "金额", "成本/净得", "分数")
    trs = []
    for r in rows:
        cost = r.get("total_cost", r.get("net", ""))
        trs.append(
            "<tr>"
            f"<td class='mono'>{html.escape(str(r.get('code','')))}</td>"
            f"<td>{html.escape(str(r.get('name','')))}</td>"
            f"<td class='num'>{int(r.get('shares',0)):,}</td>"
            f"<td class='num'>{float(r.get('price',0)):.2f}</td>"
            f"<td class='num'>{fmt_money(r.get('amount'))}</td>"
            f"<td class='num'>{fmt_money(cost)}</td>"
            f"<td class='num'>{float(r.get('score',0)):+.3f}</td>"
            "</tr>"
        )
    ths = "".join(f"<th>{h}</th>" for h in head)
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def main() -> None:
    path = latest_signal()
    with open(path) as fh:
        d = json.load(fh)
    meta, port = d.get("meta", {}), d.get("portfolio", {})
    buys, sells, holds = d.get("buys", []), d.get("sells", []), d.get("holds", [])

    print(f"\n{'='*60}")
    print(f"纸面信号 {meta.get('date')} | 组合 {fmt_money(port.get('value'))} | "
          f"现金 {fmt_money(port.get('cash'))} | 持仓 {port.get('n_positions')}")
    print(f"买入 {len(buys)} | 卖出 {len(sells)} | 持有 {len(holds)}")
    for b in buys:
        print(f"  BUY  {b.get('code')} {b.get('name','')} {b.get('shares')}股 "
              f"@{b.get('price'):.2f} 成本 {fmt_money(b.get('total_cost'))} score={b.get('score'):+.3f}")
    for s in sells:
        print(f"  SELL {s.get('code')} {s.get('name','')} {s.get('shares')}股 "
              f"@{s.get('price'):.2f} 净得 {fmt_money(s.get('net'))}")
    print(f"\n[latest] {path}")

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>股票平台 · 纸面信号 {html.escape(str(meta.get('date','')))}</title>
<style>
:root{{--bg:#0e1117;--card:#161b22;--line:#30363d;--fg:#e6edf3;--mut:#8b949e;
--green:#3fb950;--red:#f85149;--gold:#d29922;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:24px 0 8px;color:var(--mut)}}
.meta{{display:flex;gap:18px;flex-wrap:wrap;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:14px 16px;margin:16px 0}}
.meta b{{display:block;font-size:18px}} .meta span{{color:var(--mut);font-size:12px}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);
border-radius:10px;overflow:hidden}} th,td{{padding:8px 10px;border-bottom:1px solid var(--line);
text-align:left}} th{{color:var(--mut);font-size:12px;font-weight:500}}
tr:last-child td{{border-bottom:none}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.tag{{display:inline-block;background:#1f2630;border:1px solid var(--line);border-radius:6px;
padding:3px 7px;margin:0 6px 6px 0;font-family:ui-monospace,Menlo,monospace;font-size:12px}}
.tags{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}}
.empty{{color:var(--mut)}} .note{{color:var(--mut);font-size:12px;margin-top:24px;line-height:1.7}}
.warn{{color:var(--gold)}}
</style></head><body>
<h1>股票平台 · 周频纸面信号</h1>
<p class="note">22f 基础版 · topk={meta.get('topk')} · n_drop={meta.get('n_drop')} ·
成交假设：{html.escape(str(meta.get('fill_assumption','')))}</p>
<div class="meta">
<div><b>{html.escape(str(meta.get('date','')))}</b><span>信号日期</span></div>
<div><b>{fmt_money(port.get('value'))}</b><span>组合总值</span></div>
<div><b>{fmt_money(port.get('cash'))}</b><span>现金</span></div>
<div><b>{port.get('n_positions',0)}</b><span>持仓数</span></div>
<div><b>{len(buys)} / {len(sells)} / {len(holds)}</b><span>买 / 卖 / 持</span></div>
</div>
<h2>买入 ({len(buys)})</h2>{table(buys,'trade')}
<h2>卖出 ({len(sells)})</h2>{table(sells,'trade')}
<h2>持有 ({len(holds)})</h2>{table(holds,'hold')}
<p class="note warn">这是纸面虚拟盘，不是实盘建议。回测口径：22f 周频 / 实盘成本；
2021-2026 OOS 9.38% 年化 / -27.1% 回撤 / 0.685 夏普。先纸面 4-8 周，跑赢基准再谈真钱。</p>
</body></html>"""
    out = os.path.join(SIGNAL_DIR, "latest.html")
    with open(out, "w") as fh:
        fh.write(page)
    print(f"[html] {out}")
    try:
        webbrowser.open("file://" + out)
    except Exception:
        pass


if __name__ == "__main__":
    main()
