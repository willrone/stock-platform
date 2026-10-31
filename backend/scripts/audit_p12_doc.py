#!/usr/bin/env python3
"""P12 文档 ↔ 产物 逐格核对（一次性排查脚本）。

用途：找出 P12_ALPHA_WEIGHT_2026-10-31.md 里所有与产物不一致的数字。
排查完的锚点应迁进 audit_doc_numbers.py（长期守卫），本脚本仅作证据留存。
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BACK = os.path.join(ROOT, 'data', 'backtest')
DOC = os.path.join(ROOT, 'docs', 'backtest', 'P12_ALPHA_WEIGHT_2026-10-31.md')


def J(name):
    with open(os.path.join(BACK, name), encoding='utf-8') as fh:
        return json.load(fh)


def dig(obj, path):
    cur = obj
    for part in re.findall(r'[^.\[\]]+|\[\d+\]', path):
        cur = cur[int(part[1:-1])] if part.startswith('[') else cur[part]
    return cur


def norm(s):
    return re.sub(r'[\s`*]', '', s or '')


def extract_tables(text):
    tables = []
    cur = []
    for line in text.splitlines():
        if line.strip().startswith('|'):
            cur.append(line)
        else:
            if len(cur) >= 2:
                tables.append(cur)
            cur = []
    if len(cur) >= 2:
        tables.append(cur)
    out = []
    for t in tables:
        rows = [r.split('|')[1:-1] for r in t]
        rows = [r for r in rows
                if not all(re.fullmatch(r'\s*:?-{2,}:?\s*', c or ' ') for c in r)]
        if rows:
            out.append((rows[0], rows[1:]))
    return out


def cell_num(cell):
    txt = cell.replace('−', '-').replace('**', '')
    m = re.search(r'[-+]?\d+(?:\.\d+)?', txt)
    return float(m.group()) if m else None


def all_nums(cell):
    txt = cell.replace('−', '-').replace('**', '')
    return [float(m.group()) for m in re.finditer(r'[-+]?\d+(?:\.\d+)?', txt)]


A = {
    'p12m': J('multi_asset_p12_main.json'),
    'p12o': J('multi_asset_p12_oos.json'),
    'p12md': J('multi_asset_p12_main_degraded.json'),
    'p12od': J('multi_asset_p12_oos_degraded.json'),
    'p12l': J('multi_asset_p12_long.json'),
    'p12lw': J('multi_asset_p12_longw.json'),
    'p12d': J('multi_asset_p12_diag.json'),
}

METRIC_NAME = {
    'abs_sharpe': '绝对夏普',
    'alpha_gz': 'α 夏普（国证A指）',
    'alpha_ew': 'α 夏普（等权持有）',
    'alpha_hs300': 'α 夏普（沪深300）',
}
BENCH_NAME = {
    'equal_weight_buyhold': '等权买入持有',
    'equal_weight_rebalance': '等权日频再平衡',
    'IDX_000300': '沪深300',
    'IDX_399317': '国证A指',
}
WIN_NAME = {'main': 'main', 'oos': 'oos', 'w2013': 'w2013',
            'w2016': 'w2016', 'w2019': 'w2019'}

problems = []
checks = 0


def eq(where, got, want, tol=5e-4):
    global checks
    checks += 1
    if got is None:
        problems.append(f'{where}: 单元格非数字')
        return
    if abs(got - want) > tol:
        problems.append(f'{where}: 文档 {got} vs 产物 {want}')


def tables():
    with open(DOC, encoding='utf-8') as fh:
        return extract_tables(fh.read())


TS = tables()


def get(tidx, row, col):
    hdr, data = TS[tidx]
    hc = [norm(c) for c in hdr]
    ci = hc.index(norm(col))
    for r in data:
        if r and norm(r[0]) == norm(row):
            return r[ci]
    raise KeyError(f'T{tidx} 无行 {row}')


def num_at(tidx, row, col, want, tol=5e-4, label=None):
    """单元格首个数字 == want"""
    cell = get(tidx, row, col)
    n = cell_num(cell)
    eq(f'T{tidx}「{row}」×「{col}」{label or ""}', n, want, tol)


# ══ T1/T2 组合层分解 ══
for tidx, art in ((1, 'p12m'), (2, 'p12o')):
    for bkey, bname in BENCH_NAME.items():
        dec = dig(A[art], f'combo_decomposition_w050.{bkey}.decomp')
        # 行标签含 n= 或（反例口径）
        row = None
        for r in TS[tidx][1]:
            if norm(r[0]).startswith(norm(bname)):
                row = r[0]
                break
        if row is None:
            problems.append(f'T{tidx}: 找不到行 {bname}')
            continue
        num_at(tidx, row, '组合夏普', dec['leg_sharpe'])
        num_at(tidx, row, '基准夏普', dec['bench_sharpe'])
        num_at(tidx, row, '超额夏普', dec['excess_sharpe'])
        num_at(tidx, row, 'α 夏普', dec['alpha_sharpe'])
        num_at(tidx, row, 'β', dec['beta'])
        num_at(tidx, row, 'ρ', dec['corr'])

# ══ T3 长历史 3 窗口 ══
for w in ('w2013', 'w2016', 'w2019'):
    r = dig(A['p12l'], f'results.{w}')
    row = None
    for rr in TS[3][1]:
        if norm(rr[0]).startswith(w):
            row = rr[0]
            break
    if row is None:
        problems.append(f'T3: 找不到行 {w}')
        continue
    num_at(3, row, 'A 股腿夏普', r['a_share_sharpe'])
    num_at(3, row, '袖套夏普', r['sleeve_sharpe'])
    num_at(3, row, '组合夏普', r['combo_sharpe'])
    num_at(3, row, 'α 夏普（等权持有）', dig(A['p12l'], f'results.{w}.decomp_ew_buyhold.alpha_sharpe'))
    num_at(3, row, 'β', dig(A['p12l'], f'results.{w}.decomp_ew_buyhold.beta'))
    num_at(3, row, 'ρ', dig(A['p12l'], f'results.{w}.decomp_ew_buyhold.corr'))

# ══ T4/T5 w 曲线（main/oos）══
for tidx, art in ((4, 'p12m'), (5, 'p12o')):
    for m, mname in METRIC_NAME.items():
        c = dig(A[art], f'curves.{m}')
        num_at(tidx, mname, '最优 w', c['best_w'])
        num_at(tidx, mname, '最优值', c['best_value'])
        num_at(tidx, mname, '随机 w 最大', c['random_max'])
        # 子窗口众数格："0.15 (1/4)" → 首个数字 = 众数 w
        cell = get(tidx, mname, '子窗口众数')
        nums = all_nums(cell)
        eq(f'T{tidx}「{mname}」子窗口众数 w', nums[0], c['subwindow_argmax_mode'])
        eq(f'T{tidx}「{mname}」子窗口众数计数', nums[1], c['subwindow_mode_count'])
        eq(f'T{tidx}「{mname}」子窗口总数', nums[2], c['subwindow_n'])

# ══ T6 5 窗口汇总 ══
for m, mname in METRIC_NAME.items():
    cw = dig(A['p12lw'], f'cross_window.{m}')
    for w in ('main', 'oos', 'w2013', 'w2016', 'w2019'):
        num_at(6, mname, WIN_NAME[w], dig(A['p12lw'], f'cross_window.{m}.per_window.{w}.best_w'))
    num_at(6, mname, '落 [0.40,0.60]', cw['n_in_0.40_0.60'])
    cell = get(6, mname, '众数')
    nums = all_nums(cell)
    eq(f'T6「{mname}」众数 w', nums[0], cw['mode_best_w'])
    eq(f'T6「{mname}」众数计数', nums[1], cw['mode_count'])
    num_at(6, mname, '跨度', cw['spread'])

# ══ T7 配置敏感性 ══
for m, mname in METRIC_NAME.items():
    for col, art in (('main（定稿）', 'p12m'), ('main（降级）', 'p12md'),
                      ('oos（定稿）', 'p12o'), ('oos（降级）', 'p12od')):
        num_at(7, mname, col, dig(A[art], f'curves.{m}.best_w'))

# ══ T8 E1 Spearman ══
e1 = dig(A['p12d'], 'diagnostics.E1_peak_vs_leg_strength')
for m, mname in METRIC_NAME.items():
    num_at(8, mname, 'ρ(最优w,A腿夏普)', e1[m]['spearman_bestw_vs_aleg_sharpe'])
    num_at(8, mname, 'ρ(最优w,A腿−袖套)', e1[m]['spearman_bestw_vs_aleg_minus_sleeve'])

# ══ T9 E2 统一 w ══
e2 = dig(A['p12d'], 'diagnostics.E2_uniform_w_across_windows')
GRP_PRE = {'A': 'A_定稿三类配置', 'B': 'B_降级gold+bond'}
for grp in ('A', 'B'):
    for m, mname in METRIC_NAME.items():
        row = None
        for r in TS[9][1]:
            if norm(r[0]).startswith(norm(grp)) and norm(r[1]) == norm(mname):
                row = r
                break
        if row is None:
            problems.append(f'T9: 找不到行 {grp}/{mname}')
            continue
        e = e2[f'{GRP_PRE[grp]}|{m}']
        eq(f'T9「{grp}/{mname}」统一最优 w', cell_num(row[2]), e['uniform_best_w'])
        eq(f'T9「{grp}/{mname}」平均值', cell_num(row[3]), e['uniform_best_mean'])
        # 逐窗口 argmax
        got = [n for n in all_nums(row[4])]
        want = list(e['per_window_argmax'].values())
        if len(got) != len(want) or any(abs(a - b) > 1e-9 for a, b in zip(got, want)):
            problems.append(f'T9「{grp}/{mname}」逐窗口 argmax: 文档 {got} vs 产物 {want}')

# ══ T10 E5 迁移命中率 ══
e5 = dig(A['p12d'], 'diagnostics.E5_cross_window_transfer')
for m, mname in METRIC_NAME.items():
    cell = get(10, mname, '迁移命中率')
    nums = all_nums(cell)
    eq(f'T10「{mname}」命中数', nums[0], e5[m]['src_w_beats_w050'])
    eq(f'T10「{mname}」总对数', nums[1], e5[m]['n_pairs'])
    eq(f'T10「{mname}」命中率%', nums[2], e5[m]['hit_rate'] * 100, 0.05)

# ══ T11 E3 定稿 w 排名 ══
e3 = dig(A['p12d'], 'diagnostics.E3_w050_risk_disclosure')
for m, mname in METRIC_NAME.items():
    for w in ('main', 'oos', 'w2013', 'w2016', 'w2019'):
        row = next((r for r in e3[m] if r['window'] == w), None)
        if row is None:
            problems.append(f'T11: E3 缺 {m}/{w}')
            continue
        cell = get(11, mname, WIN_NAME[w])
        nums = all_nums(cell)
        eq(f'T11「{mname}」{w} 排名', nums[0], row['rank_of_w050'])
        eq(f'T11「{mname}」{w} 网格数', nums[1], row['n_grid'])

# ══ T12 E4 相对随机优势 ══
e4 = dig(A['p12d'], 'diagnostics.E4_argmax_vs_random')
for m, mname in METRIC_NAME.items():
    for w in ('main', 'oos', 'w2013', 'w2016', 'w2019'):
        row = next((r for r in e4[m] if r['window'] == w), None)
        if row is None:
            problems.append(f'T12: E4 缺 {m}/{w}')
            continue
        num_at(12, mname, WIN_NAME[w], row['gap'])

# ══ T13 判据汇总 ══
FIN = dig(A['p12d'], 'diagnostics.F_final_criteria')
CRIT_LABEL = {
    'a0_curve_has_interior_max_not_edge': '(a0)',
    'a_best_w_outside_0.40_0.60': '(a)',
    'b_alpha_sharpe_ge_base_plus_0.05': '(b)',
    'c_oos_alpha_sharpe_ge_base_minus_0.02': '(c)',
    'd_subwindow_majority_same_w': '(d)',
    'e_beats_all_random_w': '(e)',
    'f_independent_oos_evidence': '(f)',
}
hdr13 = [norm(c) for c in TS[13][0]]
for key, lab in CRIT_LABEL.items():
    row = None
    for r in TS[13][1]:
        if norm(r[0]).startswith(lab):
            row = r
            break
    if row is None:
        problems.append(f'T13: 找不到判据行 {lab}')
        continue
    for ci, wname in ((1, 'main'), (2, 'oos')):
        v = FIN['table'][wname][key]
        want = '✅' if v is True else ('—' if v is None else '❌')
        got = norm(row[ci])
        checks += 1
        if got != want:
            problems.append(
                f'T13「{lab}」×「{wname}」: 文档 {got} vs 产物 {want} '
                f'（原始值 {v}）')

print(f'共 {checks} 项核对')
if problems:
    print(f'\n❌ {len(problems)} 处不一致：')
    for p in problems:
        print('  -', p)
    sys.exit(1)
print('\n✅ 全部一致')
