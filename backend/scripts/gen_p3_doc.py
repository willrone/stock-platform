#!/usr/bin/env python3
"""生成 docs/backtest/P3_BOND_CLASS_2026-10-31.md（表格全部由 JSON 产物生成）。

为什么用生成器而不是手写文档：
- 本项目的硬规矩是「文档数字必须由产物生成或经自动审计」，手抄必然出错；
- 本轮 P3 表格有 4 张、60+ 个数据格，人工转录必错。

用法：
    cd ~/Projects/stock-platform/backend
    env -u PYTHONPATH .venv-py313/bin/python3 scripts/gen_p3_doc.py
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BACK = os.path.join(ROOT, 'data', 'backtest')
DOCS = os.path.join(ROOT, 'docs', 'backtest')
OUT = os.path.join(DOCS, 'P3_BOND_CLASS_2026-10-31.md')

SEEDS = (20261030, 7, 99, 1234, 555, 8888)


def J(name):
    with open(os.path.join(BACK, name), encoding='utf-8') as fh:
        return json.load(fh)


PM = J('multi_asset_p3_main.json')
PO = J('multi_asset_p3_oos.json')
PL = J('multi_asset_p3_long.json')
BR = J('p3_bond_recon.json')
RC = J('p3_recon.json')


def n(x, d=4):
    """格式化数字：去掉 -0.0，固定小数位。"""
    if x is None:
        return '—'
    v = round(float(x), d)
    if v == 0:
        v = 0.0
    return f'{v:.{d}f}'


def pct(x, d=2):
    if x is None:
        return '—'
    return f'{float(x) * 100:.{d}f}%'


def wan(x):
    """成交额（万元），不带千分位逗号 —— 逗号会让审计脚本的 parse_num 只取到首位。"""
    if x is None:
        return '—'
    return f'{float(x) / 1e4:.0f}'


def yi(x):
    if x is None:
        return '—'
    return f'{float(x) / 1e8:.3f}'


def tbl(header, rows):
    out = ['| ' + ' | '.join(header) + ' |',
           '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)


# ─────────────────────────── 表格 ───────────────────────────
def t0_mechanism():
    """§一：P2 逼出来的机制归因（来自 P2 归因产物）。"""
    pa = J('multi_asset_p2_attrib.json')
    rows = []
    for c in ('gold', 'bond', 'commodity'):
        w = pa['main']['classes'][c]
        o = pa['oos']['classes'][c]
        rows.append([c, pct(w['sleeve_vol']), n(w['sleeve_sharpe']),
                     f"`{n(w['corr_with_a_share'])}` / `{n(o['corr_with_a_share'])}`"])
    return tbl(['类', '年化波动', '夏普', 'ρ(类, A 股腿)'], rows)


def t1_t2_bond_pool():
    """§二：债券池逐只特征（来自 p3_bond_recon.json）。"""
    names = {
        '511260': '上证10年期国债ETF', '511010': '国泰上证5年期国债ETF',
        '511020': '平安5-10年期国债活跃券ETF', '511270': '海富通上证10年期地方政府债ETF',
        '511360': '海富通中证短融ETF', '511060': '海富通上证5年期地方政府债ETF',
        '511220': '海富通上证城投债ETF', '511380': '**博时可转债ETF**',
        '511180': '**投资级可转债ETF**',
    }
    kinds = {
        '511260': '长久期利率债', '511010': '中久期利率债', '511020': '中长久期利率债',
        '511270': '长久期地方债', '511360': '短融（纯债）', '511060': '地方债',
        '511220': '城投（信用）', '511380': '**可转债（股票 beta）**',
        '511180': '**可转债（股票 beta）**',
    }
    rows_main = []
    for r in BR['windows']['main']:
        c = r['code']
        rows_main.append([
            f"`{c}`", names.get(c, ''), f"`{n(r['rho'])}`", pct(r['vol']),
            wan(r['turnover']), kinds.get(c, ''),
        ])
    rows_oos = []
    for r in BR['windows']['oos']:
        c = r['code']
        rows_oos.append([f"`{c}`", f"`{n(r['rho'])}`", pct(r['vol']),
                         wan(r['turnover']), kinds.get(c, '')])
    h1 = ['代码', '名称', 'ρ(与 510300)', '年化波动', '窗口前 120 日均成交额(万)', '类型']
    h2 = ['代码', 'ρ(与 510300)', '年化波动', '窗口前 120 日均成交额(万)', '类型']
    return tbl(h1, rows_main), tbl(h2, rows_oos)


def t4_t5_results(win, store, with504):
    """§4.1：P3a 主表。"""
    order = [
        ('base', 'base（旧定稿）'),
        ('bond_no_cb', 'bond_no_cb'),
        ('bond_cb_to_lowrho', 'bond_cb_to_lowrho'),
        ('bond_lowrho3', '**bond_lowrho3**'),
        ('bond_all_rate', 'bond_all_rate'),
    ]
    for s in SEEDS:
        order.append((f'swap_rand_s{s}', f'swap_rand_s{s}'))
    order.append(('bond_cb_sep', 'bond_cb_sep'))
    order.append(('extra_lowrho3', 'extra_lowrho3'))
    for s in SEEDS:
        order.append((f'extra_rand_s{s}', f'extra_rand_s{s}'))

    rows = []
    for key, label in order:
        r = store['results'].get(key)
        if not r:
            continue
        held = r.get('held', {}).get('bond')
        held_s = ' / '.join(held) if held else '—'
        if key == 'base':
            held_s = ' / '.join(held) + ' ← 含可转债' if held else '—'
        cells = [
            label, f"`{held_s}`" if held else '—',
            n(r['combined_sharpe']), n(r.get('sharpe_ex_gold')),
            pct(r['combined_maxdd']), pct(r['combined_return']),
            n(r['corr_sleeve_ashare']),
            pct(r.get('rolling_sharpe_252d_win_rate'), 2),
        ]
        if with504:
            cells.append(pct(r.get('rolling_sharpe_504d_win_rate'), 2))
        rows.append(cells)

    hdr = ['配置', 'bond 类持仓', '组合夏普', '剔gold夏普', '组合回撤', '组合收益',
           'ρ(袖套,A)', '滚动252日胜率']
    if with504:
        hdr.append('滚动504日胜率')
    return tbl(hdr, rows)


def t6_p3b():
    rows = []
    r = PM['results']['extra_lowrho3']
    rows.append(['extra_lowrho3（全池ρ最低top-3作第4类）',
                 n(r['combined_sharpe']), n(PO['results']['extra_lowrho3']['combined_sharpe']),
                 '**双窗口均劣于基线**'])
    for s in SEEDS:
        k = f'extra_rand_s{s}'
        if k not in PM['results']:
            continue
        rows.append([f'extra_rand_s{s}',
                     n(PM['results'][k]['combined_sharpe']),
                     n(PO['results'][k]['combined_sharpe']),
                     '**劣于基线**'])
    return tbl(['臂', 'main 夏普', 'oos 夏普', '判定'], rows)


def t7_cb_sep():
    rows = []
    for w, st in (('main', PM), ('oos', PO)):
        r = st['results']['bond_cb_sep']
        rows.append([w, n(r['combined_sharpe']), n(r.get('sharpe_ex_gold')),
                     n(r['corr_sleeve_ashare']),
                     pct(r.get('rolling_sharpe_252d_win_rate'), 2),
                     pct(r.get('rolling_sharpe_504d_win_rate'), 2)])
    return tbl(['窗口', '组合夏普', '剔gold夏普', 'ρ(袖套,A)', '滚动252日胜率',
                '滚动504日胜率'], rows)


def t8_class_rho():
    rows = []
    for w, st, keys in (('main', PM, ('base', 'bond_lowrho3')),
                        ('oos', PO, ('base', 'bond_lowrho3'))):
        for k in keys:
            r = st['results'][k]
            cr = r['class_rho']['bond']
            rows.append([f'{w} · {k}', f"`{n(cr['rho_a'])}`", n(cr['sleeve_sharpe'])])
    return tbl(['窗口 · 配置', 'ρ(bond 类, A 股腿)', 'bond 类自身夏普'], rows)


def t9_yearly():
    rows = []
    for w, st in (('main', PM), ('oos', PO)):
        b = st['results']['base']['combined_yearly']
        l = st['results']['bond_lowrho3']['combined_yearly']
        for y in sorted(set(b) | set(l)):
            rows.append([f'{w} · {y}', pct(b.get(y)), pct(l.get(y))])
    return tbl(['窗口 · 年份', 'base', 'bond_lowrho3'], rows)


def t10_long():
    rows = []
    for k, label in (('base_long', 'base_long'),
                     ('lowrho3_long', 'lowrho3_long'),
                     ('plus_cb_long', '**plus_cb_long**')):
        r = PL['results'][k]
        held = ' / '.join(r.get('held', {}).get('bond') or [])
        rows.append([label, f"`{held}`", n(r['sleeve_sharpe']), pct(r['sleeve_vol']),
                     pct(r['sleeve_return']), pct(r['sleeve_maxdd']),
                     pct(r.get('rolling_sharpe_252d_win_rate'), 2),
                     pct(r.get('rolling_sharpe_504d_win_rate'), 2)])
    return tbl(['臂', 'bond 类持仓', '袖套夏普', '年化波动', '累计收益', '最大回撤',
                '滚动252日胜率', '滚动504日胜率'], rows)


def t11_drift():
    rows = [
        ['2019 起（长历史）', ' / '.join(PL['results']['base_long']['held']['bond']), '否 ✅'],
        ['2021-05 起（OOS）', ' / '.join(PO['results']['base']['held']['bond']), '**是** ❌'],
        ['2023-05 起（main）', ' / '.join(PM['results']['base']['held']['bond']), '**是** ❌'],
    ]
    return tbl(['窗口', 'bond 类"流动性 top-3"选出的结果', '是否含可转债'], rows)


def t12_cur_liq():
    cl = PM['current_liquidity']
    rows = []
    for c in ('511260', '511010', '511020'):
        v = cl.get(c, {})
        rows.append([f"`{c}`", yi(v.get('avg_amount_120d'))])
    rows.append(['（数据截止）', cl.get('511260', {}).get('last_date', '—')])
    return tbl(['代码', '日均成交额(亿)'], rows)


def t13_final():
    return tbl(['组件', '权重', '持仓'], [
        ['A 股腿（`honest_composite`）', '50%', '—'],
        ['gold', '16.67%', '`518880` / `159934` / `159937`'],
        ['**bond**', '**16.67%**', '**`511260` / `511010` / `511020`**（长久期利率债）'],
        ['commodity', '16.67%', '`159985` / `159981` / `159980`'],
    ])


def t14_before_after():
    mb = PM['results']['base']
    ml = PM['results']['bond_lowrho3']
    ob = PO['results']['base']
    ol = PO['results']['bond_lowrho3_noliq']
    rows = [
        ['组合夏普 · 推导窗口', n(mb['combined_sharpe']), f"**{n(ml['combined_sharpe'])}**"],
        ['组合夏普 · 样本外', n(ob['combined_sharpe']), f"**{n(ol['combined_sharpe'])}**"],
        ['组合回撤 · 推导窗口', pct(mb['combined_maxdd']), f"**{pct(ml['combined_maxdd'])}**"],
        ['组合回撤 · 样本外', pct(ob['combined_maxdd']), f"**{pct(ol['combined_maxdd'])}**"],
        ['ρ(袖套, A 股腿) · 推导', n(mb['corr_sleeve_ashare']),
         f"**{n(ml['corr_sleeve_ashare'])}**"],
        ['ρ(袖套, A 股腿) · 样本外', n(ob['corr_sleeve_ashare']),
         f"**{n(ol['corr_sleeve_ashare'])}**"],
        ['ρ(bond 类, A 股腿) · 推导', n(mb['class_rho']['bond']['rho_a']),
         f"**{n(ml['class_rho']['bond']['rho_a'])}**"],
        ['ρ(bond 类, A 股腿) · 样本外', n(ob['class_rho']['bond']['rho_a']),
         f"**{n(ol['class_rho']['bond']['rho_a'])}**"],
        ['剔gold夏普 · 推导', n(mb.get('sharpe_ex_gold')),
         f"**{n(ml.get('sharpe_ex_gold'))}**"],
        ['剔gold夏普 · 样本外', n(ob.get('sharpe_ex_gold')),
         f"**{n(ol.get('sharpe_ex_gold'))}**"],
        ['滚动504日胜率 · 推导', '—',
         f"**{pct(ml.get('rolling_sharpe_504d_win_rate'), 2)}**"],
    ]
    return tbl(['指标', '旧定稿', '新定稿'], rows)


def t15_rejected():
    return tbl(['方向', '结论'], [
        ['可转债独立成第 4 类',
         f"组合夏普 `{n(PM['results']['bond_cb_sep']['combined_sharpe'])}` / "
         f"`{n(PO['results']['bond_cb_sep']['combined_sharpe'])}`，"
         f"滚动 504 日胜率 "
         f"`{pct(PM['results']['bond_cb_sep'].get('rolling_sharpe_504d_win_rate'), 2)}`"],
        ['新增低相关第 4 类',
         f"组合夏普 `{n(PM['results']['extra_lowrho3']['combined_sharpe'])}` / "
         f"`{n(PO['results']['extra_lowrho3']['combined_sharpe'])}`"],
        ['第 4 类随机抽取',
         f"`{n(min(PM['results'][f'extra_rand_s{s}']['combined_sharpe'] for s in SEEDS))}` ~ "
         f"`{n(max(PM['results'][f'extra_rand_s{s}']['combined_sharpe'] for s in SEEDS))}` / "
         f"`{n(min(PO['results'][f'extra_rand_s{s}']['combined_sharpe'] for s in SEEDS))}` ~ "
         f"`{n(max(PO['results'][f'extra_rand_s{s}']['combined_sharpe'] for s in SEEDS))}`"],
        ['「按 ρ 最低选标」本身当 alpha',
         f"OOS 随机组最高 `{n(max(PO['results'][f'swap_rand_s{s}']['combined_sharpe'] for s in SEEDS))}` "
         f"反而高于 `bond_lowrho3` 的 "
         f"`{n(PO['results']['bond_lowrho3']['combined_sharpe'])}`"],
    ])


def t16_rand_detail(win, store):
    rows = []
    for s in SEEDS:
        k = f'swap_rand_s{s}'
        r = store['results'][k]
        rows.append([f'swap_rand_s{s}', ' / '.join(r['held']['bond']),
                     n(r['combined_sharpe']),
                     pct(r.get('rolling_sharpe_252d_win_rate'), 2)])
    return tbl(['配置', 'bond 类持仓', '组合夏普', '滚动252日胜率'], rows)


def main():
    t0 = t0_mechanism()
    t1, t2 = t1_t2_bond_pool()
    t4 = t4_t5_results('main', PM, True)
    t5 = t4_t5_results('oos', PO, False)
    t6 = t6_p3b()
    t7 = t7_cb_sep()
    t8 = t8_class_rho()
    t9 = t9_yearly()
    t10 = t10_long()
    t11 = t11_drift()
    t12 = t12_cur_liq()
    t13 = t13_final()
    t14 = t14_before_after()
    t15 = t15_rejected()
    r16m = t16_rand_detail('main', PM)
    r16o = t16_rand_detail('oos', PO)

    mb = PM['results']['base']
    ml = PM['results']['bond_lowrho3']
    ob = PO['results']['base']
    ol = PO['results']['bond_lowrho3_noliq']
    rnd_m = [PM['results'][f'swap_rand_s{s}']['combined_sharpe'] for s in SEEDS]
    rnd_o = [PO['results'][f'swap_rand_s{s}']['combined_sharpe'] for s in SEEDS]

    doc = f"""# P3 低相关资产替换：bond 类里的可转债是分类错误（2026-10-31）

> **一句话结论**：定稿的 bond 类里混进了一只可转债 ETF（`511380`），它与 A 股代理的相关性高达
> `+0.76 / +0.80`，本质是"带债底的股票"。把它换成长久期利率债后，
> bond 类与 A 股腿的相关性从 `{n(mb['class_rho']['bond']['rho_a'])} / {n(ob['class_rho']['bond']['rho_a'])}`
> 变成 `{n(ml['class_rho']['bond']['rho_a'])} / {n(ol['class_rho']['bond']['rho_a'])}`，
> 组合夏普 **main `{n(mb['combined_sharpe'])}` → `{n(ml['combined_sharpe'])}`**、
> **oos `{n(ob['combined_sharpe'])}` → `{n(ol['combined_sharpe'])}`**，
> 回撤 `{pct(mb['combined_maxdd'])}` → `{pct(ml['combined_maxdd'])}`。
> **这是本项目唯一被采用的资产侧改进。**
>
> 同批证伪两个方向：可转债独立成第 4 类
> （`{n(PM['results']['bond_cb_sep']['combined_sharpe'])}` / {n(PO['results']['bond_cb_sep']['combined_sharpe'])}`）、
> 新增低相关第 4 类
> （`{n(PM['results']['extra_lowrho3']['combined_sharpe'])}` / {n(PO['results']['extra_lowrho3']['combined_sharpe'])}`）。

---

## 一、这个方向是从哪被逼出来的

P2（商品类内取舍 + 类间逆波动率配权）的结果是：**逆波动率配权在"袖套单独"口径上大幅赢**
（长历史 Δ夏普 `+0.21 ~ +1.00`，12 个独立子窗口 `8–11/12` 胜出），
**但在"组合"口径上两个窗口都输**（`1.3767` → `1.2612`）。

P2 的机制归因（`P2_CLASS_WEIGHT_2026-10-31.md`）给出了原因：

{t0}

**低波 ≠ 低相关。** bond 腿波动最低，却与 A 股腿最同向；逆波动率配权于是把仓位系统性地推向
最同向的资产（L120 无 cap 时 bond 平均权重 `68.56%`），袖套波动从 `8.91%` 砍到 `4.73%`、
袖套自身夏普升到 `1.8620`，但 `ρ(袖套, A 股腿)` 从 `0.3086` 升到 `0.4328`，组合夏普反而崩。

那问题就变成一个非常具体的问题：**bond 类里到底装了什么？**

查 P2 定稿的持仓：

```
main bond 类 top-3 = {PM['results']['base']['held']['bond']}
oos  bond 类 top-3 = {PO['results']['base']['held']['bond']}
```

`511380` 是**博时可转债 ETF**。它不是纯债，它是"债底 + 转股溢价 + 正股期权"，
本质上是**带债底的股票**。`511180`（投资级可转债）同理。

**这不是参数问题，是分类错误。** 本文档就是把这个分类错误修掉。

---

## 二、勘察：ETF 全池里与 A 股最低相关的是什么

### 2.1 为什么用 `510300` 做代理

选腿需要"和 A 股腿的相关性"，但 `honest_composite` 面板数据从 **2021-05-11** 起，
而 OOS 窗口的起点恰好就是 **2021-05-11** —— 窗口前**一天都没有**。
用推导窗口的 A 股腿去选 OOS 的资产就是前视偏差。

`510300`（沪深300ETF）的 parquet 可回溯到 2012 年，能提供真正的 point-in-time 代理。
选腿一律用 `510300`，事后组合验证仍用真实的 `honest_composite`，
**两者必须分开报告，代理 ρ 不得冒充真实 A 股腿 ρ**。

- ρ 计算：窗口**开始前** 250 个交易日的日收益
- 流动性门槛：窗口**开始前** 120 个交易日日均成交额 ≥ 5000 万

### 2.2 债券池逐只特征（产物：`p3_bond_recon.json`）

**推导窗口（起点 2023-05-08）**

{t1}

**样本外窗口（起点 2021-05-11）**

{t2}

**两个窗口独立收敛到同一批最低相关资产：`511260 / 511010 / 511020`，全部是长久期利率债。**
两只可转债的 ρ 稳定在 `+0.76 ~ +0.81`，与其他所有债券相差近 1.0 —— 是干净的分界线。

**额外勘察发现（`p3_recon.json`，567 只 ETF 全池）**：全池 ρ 最低的资产全部是
长久期利率债，没有出现"某个冷门行业 ETF 更低相关"的情况。这说明低相关来自
**资产性质**（利率 vs 股票），不是标的挑选技巧。

---

## 三、预注册假设

在跑任何回测之前先写死三条假设（见 `multi_asset_p3_bond.py` 文件头）：

| 编号 | 假设 | 对照 |
|---|---|---|
| **H1** | bond 类剔除可转债后，ρ(类, A 股腿) 显著下降，组合夏普提升 | `bond_no_cb` |
| **H2** | 在纯利率债内按 ρ 最低 top-3 选，相对"只剔除可转债"仍有增量 | `bond_cb_to_lowrho` / `bond_lowrho3` |
| **H3** | 新增一个低相关资产类（第 4 类）能进一步提升 | `extra_lowrho3` / `extra_rand_*` |

每个假设都配了**同池随机对照**（`swap_rand_s{{seed}}`、`extra_rand_s{{seed}}`，6 个固定种子），
用来排除"碰巧选中赢家"。基线复现门不通过（`|base − 定稿| > 5e-4`）直接退出脚本。

---

## 四、结果

### 4.1 P3a · bond 类内替换（gold / commodity 固定 top-3，仅动 bond 类）

**推导窗口（{PM['calendar']['start']} → {PM['calendar']['end']}，{PM['calendar']['n']} 个交易日）**

{t4}

**样本外窗口（{PO['calendar']['start']} → {PO['calendar']['end']}，{PO['calendar']['n']} 个交易日）**

{t5}

### 4.2 三个关键读数

**① 6/6 随机替换全部优于基线（两个窗口都是）。**
main 随机组 `{n(min(rnd_m))} ~ {n(max(rnd_m))}` 全部 > `base {n(mb['combined_sharpe'])}`；
oos 随机组 `{n(min(rnd_o))} ~ {n(max(rnd_o))}` 全部 > `base {n(ob['combined_sharpe'])}`。
**所以增量不来自"我挑中了 511260/511010"这个运气，来自"可转债这个成分本身是拖累"这个分类事实。**

main 随机对照逐项：

{r16m}

oos 随机对照逐项：

{r16o}

**② 反过来，"ρ 最低 top-3"这个选标规则本身没有 alpha。**
oos 窗口随机组最好的 `swap_rand_s7 = {n(PO['results']['swap_rand_s7']['combined_sharpe'])}`
**高于** `bond_lowrho3 = {n(PO['results']['bond_lowrho3']['combined_sharpe'])}`；
main 窗口随机组最好的 `{n(max(rnd_m))}` 也**低于** `bond_lowrho3` 的 `{n(ml['combined_sharpe'])}`
（差 `{n(ml['combined_sharpe'] - max(rnd_m))}`）。
所以定稿选 `511260/511010/511020` 的理由**不是回测择优**，而是
**资产性质**（纯利率债、负股票 beta）—— 这跟 P2 的教训完全一致：
袖套自身指标不能替代组合指标，选标技巧不能替代机制。

**③ `bond_lowrho3` 与 `bond_all_rate` 在两个窗口都收敛到同一组持仓**
（main: {PM['results']['bond_lowrho3']['held']['bond']}；
oos 流动性达标时 {PO['results']['bond_lowrho3']['held']['bond']}，
不限流动性时 {PO['results']['bond_lowrho3_noliq']['held']['bond']}）。
两条独立规则（"按 ρ 最低选" vs "按分类修正+流动性选"）指向同一答案 → 不是规则碰巧。

### 4.3 P3b · 新增低相关第 4 类 —— 证伪

{t6}

原因很清楚：**类间等权意味着新增一类会同时稀释 gold 和 commodity，
而新增类本身的夏普远不如 gold**。main 里 ρ 最低 top-3 是
`511010 / 518800(黄金) / 513050(中概互联)` —— 把黄金 ETF 塞进"第 4 类"
等于重复暴露 gold、把中概互联的高波动引进来。

**结论：低相关资产不能靠"多加一类"来用，只能靠"换掉错的类成分"来用。**

### 4.4 P3a-2 · 可转债独立成第 4 类 —— 证伪

可转债 ρ 高，但有独立的收益来源（债底 + 转股溢价 + 正股期权）。
"因为 ρ 高就丢掉"和"放在 bond 类里污染 ρ"都不对，单独成类才是公平处理 —— 结果：

{t7}

**组合夏普双窗口大幅劣于基线，滚动 504 日胜率
`{pct(PM['results']['bond_cb_sep'].get('rolling_sharpe_504d_win_rate'), 2)}`。**
可转债作为一个 1/4 权重的类，贡献的股票 beta 直接推高整个组合对 A 股的暴露
（`ρ(袖套,A)` 从 `{n(mb['corr_sleeve_ashare'])}` / `{n(ob['corr_sleeve_ashare'])}`
升到 `{n(PM['results']['bond_cb_sep']['corr_sleeve_ashare'])}` /
`{n(PO['results']['bond_cb_sep']['corr_sleeve_ashare'])}`）。

**注意剔gold口径的读数**：`bond_cb_sep` 的剔gold夏普
`{n(PM['results']['bond_cb_sep'].get('sharpe_ex_gold'))}`（main）反而**高于**
`bond_lowrho3` 的 `{n(ml.get('sharpe_ex_gold'))}`。这不矛盾 —— 剔掉 gold 后，
剩下 bond+commodity 更纯，可转债提供了额外的非 gold 收益来源。
**但定稿目标是组合夏普与回撤，不是剔gold夏普。**
这是本项目第一次明确遇到"辅助指标变好但主指标变差"的情况，按主指标判定。

---

## 五、机制解释

`bond` 类的 ρ 与自身夏普（产物：`multi_asset_p3_main.json` / `multi_asset_p3_oos.json`）：

{t8}

链条：

1. `511380`（可转债）与 A 股代理 ρ = `+0.7643 / +0.7976` → 混进 bond 类后，bond 类整体 ρ 被拉到
   `{n(mb['class_rho']['bond']['rho_a'])} / {n(ob['class_rho']['bond']['rho_a'])}`
2. bond 类是三类里权重最大的分散腿之一，它高相关 → `ρ(袖套, A 股腿)` 达
   `{n(mb['corr_sleeve_ashare'])} / {n(ob['corr_sleeve_ashare'])}`
3. 换成纯长久期利率债 → bond 类 ρ 变成 **`{n(ml['class_rho']['bond']['rho_a'])} /
   {n(ol['class_rho']['bond']['rho_a'])}`（负相关）**，袖套整体 ρ 降到
   `{n(ml['corr_sleeve_ashare'])} / {n(ol['corr_sleeve_ashare'])}`
4. 负相关的分散腿 + 每月再平衡 → 组合夏普
   `+{n(ml['combined_sharpe'] - mb['combined_sharpe'])} / +{n(ol['combined_sharpe'] - ob['combined_sharpe'])}`，
   回撤 `{pct(mb['combined_maxdd'])}` → `{pct(ml['combined_maxdd'])}`

**关键点：负相关的分散腿比"低波但同向"的腿有用得多。**
这与 P2 的结论完全一致，只是这次是从反面确认。

### 分年收益（组合口径）

{t9}

**不是靠某一年拉起来的**：4 个 main 年份里 2 年赢 2 年输，3 个 oos 年份里 1 赢 2 输。
夏普的提升主要来自**波动率下降**（`ρ` 下降 → 组合波动下降），
收益端基本持平（main `{pct(ml['combined_return'])}` vs `{pct(mb['combined_return'])}`，
oos `{pct(ol['combined_return'])}` vs `{pct(ob['combined_return'])}`）。
**这是降方差，不是提收益** —— 和此前所有被采纳的机制一样。

---

## 六、长历史交叉验证

产物：`multi_asset_p3_long.json`（{PL['calendar']['start']} → {PL['calendar']['end']}，**袖套单独口径**）

{t10}

**长历史能说什么、不能说什么（必须写清楚）**

- ✅ **能说的**：`plus_cb_long` 在 7.7 年长历史上，把一只可转债塞进长债 bond 类，
  袖套夏普从 `{n(PL['results']['base_long']['sleeve_sharpe'])}` 掉到
  `{n(PL['results']['plus_cb_long']['sleeve_sharpe'])}`，回撤从
  `{pct(PL['results']['base_long']['sleeve_maxdd'])}` 恶化到
  `{pct(PL['results']['plus_cb_long']['sleeve_maxdd'])}`，
  滚动 504 日夏普胜率 **`{pct(PL['results']['plus_cb_long'].get('rolling_sharpe_504d_win_rate'), 2)}`**。
  **"含可转债更差"在长历史上被独立确认。**
- ✅ **能说的**：长历史窗口（2019 起）的"流动性 top-3"自动选到了
  `{PL['results']['base_long']['held']['bond']}`（全是长债）—— 因为 `511380` 是 2020-04 才上市、
  `511360` 是 2020-09、`511520` 是 2022-10。
  **同一条"流动性 top-3"规则，在 2019 年给出正确答案，在 2023 年给出错误答案。规则会随市场规模漂移。**
- ❌ **不能说的**：长历史**不能**验证"选腿规则"本身。原因是
  `510300` 代理与所有 ETF 的 parquet **都从 2019-01-02 起**，
  长历史窗口**没有"窗口前"数据**，无法做 point-in-time 的 ρ 选腿。
- ❌ **不能说的**：长历史**没有 A 股腿**（`honest_composite` 面板从 2021-05-11 起），
  测不到决定组合夏普的 `ρ(袖套, A 股腿)`。P2 已经吃过这个亏：
  袖套单独赢 ≠ 组合赢。**所以长历史在本节只作为"含可转债更差"的交叉验证，不作为选腿规则的验证。**

### 6.1 长历史给出的额外设计原则：定稿必须用**固定持仓集**

{t11}

**"流动性 top-3"这条规则本身是不稳的** —— 短债 ETF 的规模在 2020–2023 年爆发式增长，
把可转债和短融顶进了流动性榜单，而可转债带着 `+0.8` 的股票 beta。

**因此定稿的 bond 类改为固定持仓集 `{{511260, 511010, 511020}}`**，
选入的依据是**资产性质**（纯长久期利率债、负股票 beta），
而不是"当前流动性排名"。配套纪律：**每季度复核一次**，
若某只成交额跌破 5000 万或规模逼近清盘线则替换。

**当前流动性**

{t12}

三只都是当前流动性最好的品种，**实盘完全可执行**。

---

## 七、定稿变更

### 7.1 新定稿

{t13}

类内等权，**每月（21 交易日）再平衡回类间等权**，其余口径全部不变
（信号日后下一交易日 `open` 成交、佣金 0.03% + 滑点 0.01%、ETF 免印花税）。

### 7.2 前后对比

{t14}

> 样本外一列取 `bond_lowrho3_noliq`（含 `511020`）：`511020` 在 2021-05 的成交额只有
> 12 万/日，**严格按 point-in-time 流动性门槛它在当时不可实盘**；
> 门槛内版本（`bond_lowrho3`，511260/511010）为 `{n(PO['results']['bond_lowrho3']['combined_sharpe'])}`。
> 结论在两个口径下都成立。

### 7.3 被证伪、不采用的

{t15}

---

## 八、诚实的局限

1. **长历史无法验证选腿规则**。`510300` 与全部 ETF 的数据都从 2019-01-02 起，
   长历史窗口没有"窗口前"数据；且长历史没有 A 股腿，测不到 `ρ(袖套, A 股腿)`。
   P3 的选腿规则只在 2 个短窗口上验证过。
2. **样本外窗口的债券流动性本身就很差**。2021-05 时流动性达标的债券只有 2 只
   （`511260`、`511010`），其余全部低于 5000 万门槛。这意味着 OOS 窗口的
   `bond_lowrho3`（严格门槛）实际只持 2 只。
3. **两个短窗口 = 两样本**。滚动 252/504 日胜率缓解了这个问题（main 上
   `bond_lowrho3` 的 504 日胜率
   `{pct(ml.get('rolling_sharpe_504d_win_rate'), 2)}`），但不能替代独立长历史审判。
4. **定稿的 bond 持仓集是固定值**，不是每期重选。它规避了"流动性排名漂移"，
   但也意味着如果 `511260` 之类的产品结构变化，需要人工复核。这是自觉的取舍。
5. **`extra_lowrho3` 抽到的第 4 类包含黄金 ETF（`518800`）**，
   说明"全池 ρ 最低"这个规则本身会重复暴露已有类 —— 这也是 P3b 失败的部分原因。
   P3b 的失败不能推广成"任何第 4 类都无用"，只能说"用 ρ 最低 + 类间等权的方式加第 4 类无用"。

---

## 九、复现

```bash
cd ~/Projects/stock-platform/backend
# 勘察
env -u PYTHONPATH .venv-py313/bin/python3 scripts/p3_recon.py       # → data/backtest/p3_recon.json
env -u PYTHONPATH .venv-py313/bin/python3 scripts/p3_bond_recon.py  # → data/backtest/p3_bond_recon.json
# 主实验
env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 scripts/multi_asset_p3_bond.py --window main
env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 scripts/multi_asset_p3_bond.py --window oos
env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 scripts/multi_asset_p3_bond.py --window long
# 文档生成（表格全部由产物渲染）
env -u PYTHONPATH .venv-py313/bin/python3 scripts/gen_p3_doc.py
# 数字审计
env -u PYTHONPATH .venv-py313/bin/python3 scripts/audit_doc_numbers.py
```

产物：`data/backtest/multi_asset_p3_main.json`、`multi_asset_p3_oos.json`、`multi_asset_p3_long.json`、
`p3_recon.json`、`p3_bond_recon.json`

### 本轮踩到的坑（已修，写下来防止复发）

1. **`| grep -v ... > log` 会把 Python 的退出码换成 grep 的退出码**。
   一轮实验实际以 `NameError` 崩溃，但 `EXIT=0`，我差点读了旧 JSON 产物当新结果。
   **正确写法**：先重定向到 `.raw`，取 `$?`，再 `grep` 生成 `.log`。
2. **随机对照的候选池若加流动性门槛，OOS 窗口会退化成单一配置**
   （流动性达标池只有 2 只 → 6 个种子抽到同一只），随机对照形同虚设。
   改为从"非可转债全池"抽样，并加断言：≥3 个种子时必须产生 ≥2 个不同持仓集。
3. **同一类内持仓代码重复会被 `dict` 静默覆盖** → 实际只剩 2 只有效持仓、权重之和 < 1。
   本轮 OOS 的 `bond_cb_to_lowrho` 就撞上了（`['511260','511360','511260']`）。
   已在 `run_cfg` 加类内唯一性断言。
4. **基线日历漂移**：价格索引扩展到全池后，若交易日历也用全池并集，
   基线夏普会从 `1.3767` 漂到 `1.4103`。**价格索引覆盖全部候选，交易日历只用基线候选并集**，
   并保留基线复现门。
5. **patch 工具是子串匹配**：`store = {{"window": ...}}` 这种写法会被
   `store: Dict[str, Any] = {{...}}` 的前半段误命中，把字面量拆开 → SyntaxError。
   改脚本后**必须先 `ast.parse` 验语法再跑**。

---

## 十、后续

- **P4 候选**：P2 的逆波动率配权在**修正后的 bond 类**上重做一次。
  机制上现在 bond 腿既低波（`1.4% ~ 2.0%`）**又**负相关（`{n(ml['class_rho']['bond']['rho_a'])} / {n(ol['class_rho']['bond']['rho_a'])}`），
  逆波动率配权不再会"把仓位推向最同向的资产" —— P2 的失败原因被抽掉了。
  但必须重跑全部对照，不能因为机制讲得通就默认有效。
- **P5 候选**：gold 类是否也有同样的分类问题？（`518880/159934/159937` 都是黄金 ETF，
  看起来干净，但值得用同样的流程过一遍。）
"""
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(doc)
    print(f'✓ 已生成 {OUT}（{len(doc)} 字符）')


if __name__ == '__main__':
    sys.exit(main())
