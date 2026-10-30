"""文档数字 vs 产物 的一致性自动核对（表格语义级）。

v1 只问「文档里有没有出现过这个数」→ 同一文档别处的**矛盾旧值**抓不到。
v2 改成行级锚定，但同名行标签会撞（例：L3 文档里 'top3 每月再平衡' 出现在 5 张表）。
v3（本文件）用**表格语义三元组**定位，唯一且可读：

    (doc, table_index, row_label, column_header) → 期望值

    table_index   该文档内第几张 markdown 表（0 起）
    row_label     行首列的文字（去掉 ** 和空格后精确匹配）
    column_header 列名（去掉 ** 和空格后精确匹配）

期望值的取法：
    ('j', 'fr', 'F1_final.combined_sharpe')  从产物 JSON 取
    ('lit', 0.8661)                          直接给常量

比较时容差 5e-4（文档写 4 位小数，产物是全精度）；
比例类产物值额外接受 ×100 形式（-0.087 ↔ -8.70%）。

用法：
    python3 scripts/audit_doc_numbers.py          # 核对
    python3 scripts/audit_doc_numbers.py --neg    # 自检：注入错误看能否抓到
退出码：断言失败 → 1（可挂 CI）
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BACK = os.path.join(ROOT, 'data', 'backtest')
DOCS = os.path.join(ROOT, 'docs', 'backtest')


def J(name):
    with open(os.path.join(BACK, name), encoding='utf-8') as fh:
        return json.load(fh)


def dig(obj, path):
    cur = obj
    for part in re.findall(r'[^.\[\]]+|\[\d+\]', path):
        cur = cur[int(part[1:-1])] if part.startswith('[') else cur[part]
    return cur


ART = {
    'fr': J('multi_asset_final_robustness.json'),
    'fro': J('multi_asset_final_robustness_oos.json'),
    'l3': J('multi_asset_l3.json'),
    'l3o': J('multi_asset_l3_oos.json'),
    'lh': J('multi_asset_long_history.json'),
    'ctrl': J('multi_asset_controls.json'),
    'xc': J('multi_asset_xclass.json'),
    'xco': J('multi_asset_xclass_oos.json'),
    'xclh': J('multi_asset_xclass_long_history.json'),
}


def norm(s):
    """归一化单元格文字：去掉 markdown 强调、空白、反引号。"""
    return re.sub(r'[\s`*]', '', s or '')


# ─────────────────────────────────────────────────────────────
# 断言清单
#   (doc, table_idx, row_label, col_header, expect_spec)
# ─────────────────────────────────────────────────────────────
def build_anchors():
    """断言清单：(doc, table_idx, row_label, col_header, expect_spec)

    table_idx 是该文档内 markdown 表的序号（0 起）。
    行标签与列名都要与文档**归一化后**精确一致（去空白/**/反引号）。
    """
    F1M = ('j', 'fr', 'F1_final.combined_sharpe')
    F1O = ('j', 'fro', 'F1_final.combined_sharpe')
    A = []

    # ── FINAL_ROBUSTNESS 表1: F1 定稿配置全景 ──
    A += [
        ('FINAL_ROBUSTNESS', 1, 'A 股基线夏普', '推导窗口 (2023-05~2026-05)', ('j', 'fr', 'baseline.sharpe')),
        ('FINAL_ROBUSTNESS', 1, 'A 股基线夏普', '样本外 (2021-05~2023-05)', ('j', 'fro', 'baseline.sharpe')),
        ('FINAL_ROBUSTNESS', 1, '组合夏普', '推导窗口 (2023-05~2026-05)', F1M),
        ('FINAL_ROBUSTNESS', 1, '组合夏普', '样本外 (2021-05~2023-05)', F1O),
        ('FINAL_ROBUSTNESS', 1, '组合收益', '推导窗口 (2023-05~2026-05)', ('j', 'fr', 'F1_final.combined_return')),
        ('FINAL_ROBUSTNESS', 1, '组合收益', '样本外 (2021-05~2023-05)', ('j', 'fro', 'F1_final.combined_return')),
        ('FINAL_ROBUSTNESS', 1, '组合回撤', '推导窗口 (2023-05~2026-05)', ('j', 'fr', 'F1_final.combined_maxdd')),
        ('FINAL_ROBUSTNESS', 1, '组合回撤', '样本外 (2021-05~2023-05)', ('j', 'fro', 'F1_final.combined_maxdd')),
        ('FINAL_ROBUSTNESS', 1, '组合波动', '推导窗口 (2023-05~2026-05)', ('j', 'fr', 'F1_final.combined_vol')),
        ('FINAL_ROBUSTNESS', 1, '组合波动', '样本外 (2021-05~2023-05)', ('j', 'fro', 'F1_final.combined_vol')),
        ('FINAL_ROBUSTNESS', 1, '与 A 股相关', '推导窗口 (2023-05~2026-05)', ('j', 'fr', 'F1_final.corr_with_a_share')),
        ('FINAL_ROBUSTNESS', 1, '与 A 股相关', '样本外 (2021-05~2023-05)', ('j', 'fro', 'F1_final.corr_with_a_share')),
        ('FINAL_ROBUSTNESS', 1, '与 gold 相关', '推导窗口 (2023-05~2026-05)', ('j', 'fr', 'F1_final.corr_with_gold')),
        ('FINAL_ROBUSTNESS', 1, '与 gold 相关', '样本外 (2021-05~2023-05)', ('j', 'fro', 'F1_final.corr_with_gold')),
    ]
    # ── FINAL_ROBUSTNESS 表7: F7 再平衡贡献 ──
    A += [
        ('FINAL_ROBUSTNESS', 8, '买入持有夏普', '推导窗口', ('j', 'fr', 'F7_rebalance_contribution.买入持有.combined_sharpe')),
        ('FINAL_ROBUSTNESS', 8, '买入持有夏普', '样本外', ('j', 'fro', 'F7_rebalance_contribution.买入持有.combined_sharpe')),
        ('FINAL_ROBUSTNESS', 8, '每月再平衡夏普', '推导窗口', F1M),
        ('FINAL_ROBUSTNESS', 8, '每月再平衡夏普', '样本外', F1O),
        ('FINAL_ROBUSTNESS', 8, '再平衡 Δ夏普', '推导窗口', ('j', 'fr', 'F7_rebalance_contribution.rebal_sharpe_gain')),
        ('FINAL_ROBUSTNESS', 8, '再平衡 Δ夏普', '样本外', ('j', 'fro', 'F7_rebalance_contribution.rebal_sharpe_gain')),
        ('FINAL_ROBUSTNESS', 8, '再平衡的收益代价', '推导窗口', ('j', 'fr', 'F7_rebalance_contribution.rebal_return_cost')),
        ('FINAL_ROBUSTNESS', 8, '再平衡的收益代价', '样本外', ('j', 'fro', 'F7_rebalance_contribution.rebal_return_cost')),
    ]

    # ── L3_MOMENTUM 表0: 顶部摘要 ──
    A += [
        ('L3_MOMENTUM', 0, '推导窗口夏普', '新推荐（top3 每月再平衡）', F1M),
        ('L3_MOMENTUM', 0, '样本外夏普', '新推荐（top3 每月再平衡）', F1O),
    ]
    # ── L3_MOMENTUM 表5: §2.2 再平衡变体 ──
    A += [
        ('L3_MOMENTUM', 5, '旧定稿 top3 买入持有', 'main 夏普', ('j', 'l3', 'results.static_top3_buyhold.combined_sharpe')),
        ('L3_MOMENTUM', 5, '旧定稿 top3 买入持有', 'oos 夏普', ('j', 'l3o', 'results.static_top3_buyhold.combined_sharpe')),
        ('L3_MOMENTUM', 5, 'top3 每月再平衡', 'main 夏普', ('j', 'l3', 'results.static_top3_rebal_monthly.combined_sharpe')),
        ('L3_MOMENTUM', 5, 'top3 每月再平衡', 'oos 夏普', ('j', 'l3o', 'results.static_top3_rebal_monthly.combined_sharpe')),
        ('L3_MOMENTUM', 5, 'top3 双周再平衡', 'main 夏普', ('j', 'l3', 'results.static_top3_rebal_biweekly.combined_sharpe')),
        ('L3_MOMENTUM', 5, 'top3 双周再平衡', 'oos 夏普', ('j', 'l3o', 'results.static_top3_rebal_biweekly.combined_sharpe')),
        ('L3_MOMENTUM', 5, '全候选 每月再平衡', 'main 夏普', ('j', 'l3', 'results.static_all_rebal_monthly.combined_sharpe')),
        ('L3_MOMENTUM', 5, '全候选 每月再平衡', 'oos 夏普', ('j', 'l3o', 'results.static_all_rebal_monthly.combined_sharpe')),
        ('L3_MOMENTUM', 5, '全候选 双周再平衡', 'main 夏普', ('j', 'l3', 'results.static_all_rebal_biweekly.combined_sharpe')),
        ('L3_MOMENTUM', 5, '全候选 双周再平衡', 'oos 夏普', ('j', 'l3o', 'results.static_all_rebal_biweekly.combined_sharpe')),
    ]
    # ── L3_MOMENTUM 表6: §2.3 top1 vs top3 效应分解 ──
    A += [
        ('L3_MOMENTUM', 6, 'top1 买入持有', 'main', ('j', 'l3', 'results.static_top1_buyhold.combined_sharpe')),
        ('L3_MOMENTUM', 6, 'top1 买入持有', 'oos', ('j', 'l3o', 'results.static_top1_buyhold.combined_sharpe')),
        ('L3_MOMENTUM', 6, 'top1 每月再平衡', 'main', ('j', 'l3', 'results.static_top1_rebal_monthly.combined_sharpe')),
        ('L3_MOMENTUM', 6, 'top1 每月再平衡', 'oos', ('j', 'l3o', 'results.static_top1_rebal_monthly.combined_sharpe')),
        ('L3_MOMENTUM', 6, 'top3 买入持有', 'main', ('j', 'l3', 'results.static_top3_buyhold.combined_sharpe')),
        ('L3_MOMENTUM', 6, 'top3 买入持有', 'oos', ('j', 'l3o', 'results.static_top3_buyhold.combined_sharpe')),
        ('L3_MOMENTUM', 6, 'top3 每月再平衡', 'main', ('j', 'l3', 'results.static_top3_rebal_monthly.combined_sharpe')),
        ('L3_MOMENTUM', 6, 'top3 每月再平衡', 'oos', ('j', 'l3o', 'results.static_top3_rebal_monthly.combined_sharpe')),
    ]
    # ── L3_MOMENTUM 表8: §2.4 alt2/alt3 ──
    A += [
        ('L3_MOMENTUM', 8, 'top1 每月再平衡（流动性第 1 名）', 'main', ('j', 'l3', 'results.static_top1_rebal_monthly.combined_sharpe')),
        ('L3_MOMENTUM', 8, 'top1 每月再平衡（流动性第 1 名）', 'oos', ('j', 'l3o', 'results.static_top1_rebal_monthly.combined_sharpe')),
        ('L3_MOMENTUM', 8, 'top1 alt2 每月再平衡（第 2 名）', 'main', ('j', 'l3', 'results.static_top1_alt2_rebal_monthly.combined_sharpe')),
        ('L3_MOMENTUM', 8, 'top1 alt2 每月再平衡（第 2 名）', 'oos', ('j', 'l3o', 'results.static_top1_alt2_rebal_monthly.combined_sharpe')),
        ('L3_MOMENTUM', 8, 'top1 alt3 每月再平衡（第 3 名）', 'main', ('j', 'l3', 'results.static_top1_alt3_rebal_monthly.combined_sharpe')),
        ('L3_MOMENTUM', 8, 'top1 alt3 每月再平衡（第 3 名）', 'oos', ('j', 'l3o', 'results.static_top1_alt3_rebal_monthly.combined_sharpe')),
    ]
    # ── L3_MOMENTUM 表10: §2.5 gold 相关性 ──
    A += [
        ('L3_MOMENTUM', 10, 'top3 买入持有', 'main 与 gold 相关', ('j', 'l3', 'results.static_top3_buyhold.corr_with_gold')),
        ('L3_MOMENTUM', 10, 'top3 买入持有', 'oos 与 gold 相关', ('j', 'l3o', 'results.static_top3_buyhold.corr_with_gold')),
        ('L3_MOMENTUM', 10, 'top3 买入持有', 'main 剔gold夏普', ('j', 'l3', 'results.static_top3_buyhold.sharpe_ex_gold')),
        ('L3_MOMENTUM', 10, 'top3 买入持有', 'oos 剔gold夏普', ('j', 'l3o', 'results.static_top3_buyhold.sharpe_ex_gold')),
        ('L3_MOMENTUM', 10, 'top3 每月再平衡', 'main 与 gold 相关', ('j', 'l3', 'results.static_top3_rebal_monthly.corr_with_gold')),
        ('L3_MOMENTUM', 10, 'top3 每月再平衡', 'oos 与 gold 相关', ('j', 'l3o', 'results.static_top3_rebal_monthly.corr_with_gold')),
        ('L3_MOMENTUM', 10, 'top3 每月再平衡', 'main 剔gold夏普', ('j', 'l3', 'results.static_top3_rebal_monthly.sharpe_ex_gold')),
        ('L3_MOMENTUM', 10, 'top3 每月再平衡', 'oos 剔gold夏普', ('j', 'l3o', 'results.static_top3_rebal_monthly.sharpe_ex_gold')),
    ]
    # ── L3_MOMENTUM 表13: §三 最终推荐表 ──
    A += [
        ('L3_MOMENTUM', 13, 'top3 每月再平衡', '两窗口下界', ('j', 'l3o', 'results.static_top3_rebal_monthly.combined_sharpe')),
        ('L3_MOMENTUM', 13, '全候选 每月再平衡', '两窗口下界', ('j', 'l3o', 'results.static_all_rebal_monthly.combined_sharpe')),
        ('L3_MOMENTUM', 13, 'top3 双周再平衡', '两窗口下界', ('j', 'l3o', 'results.static_top3_rebal_biweekly.combined_sharpe')),
        ('L3_MOMENTUM', 13, 'top1 每月再平衡', '两窗口下界', ('j', 'l3o', 'results.static_top1_rebal_monthly.combined_sharpe')),
    ]
    # ── XCLASS_MOMENTUM 表0/1/2: P1 跨类动量 ──
    A += [
        ('XCLASS_MOMENTUM', 0, 'momentum 族均值（24 配置）', 'main', ('l3fam_mean', 'l3', 'mom')),
        ('XCLASS_MOMENTUM', 0, 'momentum 族均值（24 配置）', 'oos', ('l3fam_mean', 'l3o', 'mom')),
        ('XCLASS_MOMENTUM', 0, 'random 族均值（18 次运行）', 'main', ('l3fam_mean', 'l3', 'rand')),
        ('XCLASS_MOMENTUM', 0, 'random 族均值（18 次运行）', 'oos', ('l3fam_mean', 'l3o', 'rand')),
        ('XCLASS_MOMENTUM', 0, 'random L60 K1 月度均值 vs static top1 再平衡', 'main', ('l3_delta', 'l3')),
        ('XCLASS_MOMENTUM', 0, 'random L60 K1 月度均值 vs static top1 再平衡', 'oos', ('l3_delta', 'l3o')),
        ('XCLASS_MOMENTUM', 2, 'mom_L60_top1', 'main', ('j', 'xc', 'results.xclass_mom_L60_top1.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L60_top1', 'oos', ('j', 'xco', 'results.xclass_mom_L60_top1.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L60_top1', '随机均值(main)', ('mean_rand', 'xc', 'top1')),
        ('XCLASS_MOMENTUM', 2, 'mom_L60_top1', '随机均值(oos)', ('mean_rand', 'xco', 'top1')),
        ('XCLASS_MOMENTUM', 2, 'mom_L60_top2', 'main', ('j', 'xc', 'results.xclass_mom_L60_top2.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L60_top2', 'oos', ('j', 'xco', 'results.xclass_mom_L60_top2.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L60_top2', '随机均值(main)', ('mean_rand', 'xc', 'top2')),
        ('XCLASS_MOMENTUM', 2, 'mom_L60_top2', '随机均值(oos)', ('mean_rand', 'xco', 'top2')),
        ('XCLASS_MOMENTUM', 2, 'mom_L60_weighted', 'main', ('j', 'xc', 'results.xclass_mom_L60_weighted.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L60_weighted', 'oos', ('j', 'xco', 'results.xclass_mom_L60_weighted.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L60_weighted', '随机均值(main)', ('mean_rand', 'xc', 'weighted')),
        ('XCLASS_MOMENTUM', 2, 'mom_L60_weighted', '随机均值(oos)', ('mean_rand', 'xco', 'weighted')),
        ('XCLASS_MOMENTUM', 2, 'mom_L120_top1', 'main', ('j', 'xc', 'results.xclass_mom_L120_top1.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L120_top1', 'oos', ('j', 'xco', 'results.xclass_mom_L120_top1.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L120_top2', 'main', ('j', 'xc', 'results.xclass_mom_L120_top2.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L120_top2', 'oos', ('j', 'xco', 'results.xclass_mom_L120_top2.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L120_weighted', 'main', ('j', 'xc', 'results.xclass_mom_L120_weighted.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L120_weighted', 'oos', ('j', 'xco', 'results.xclass_mom_L120_weighted.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L250_top1', 'main', ('j', 'xc', 'results.xclass_mom_L250_top1.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L250_top1', 'oos', ('j', 'xco', 'results.xclass_mom_L250_top1.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L250_top2', 'main', ('j', 'xc', 'results.xclass_mom_L250_top2.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L250_top2', 'oos', ('j', 'xco', 'results.xclass_mom_L250_top2.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L250_weighted', 'main', ('j', 'xc', 'results.xclass_mom_L250_weighted.combined_sharpe')),
        ('XCLASS_MOMENTUM', 2, 'mom_L250_weighted', 'oos', ('j', 'xco', 'results.xclass_mom_L250_weighted.combined_sharpe')),
    ]
    # ── XCLASS_MOMENTUM 表3: 随机对照离散度 ──
    A += [
        ('XCLASS_MOMENTUM', 3, 'rand_top1', 'main 均值', ('mean_rand', 'xc', 'top1')),
        ('XCLASS_MOMENTUM', 3, 'rand_top1', 'oos 均值', ('mean_rand', 'xco', 'top1')),
        ('XCLASS_MOMENTUM', 3, 'rand_top2', 'main 均值', ('mean_rand', 'xc', 'top2')),
        ('XCLASS_MOMENTUM', 3, 'rand_top2', 'oos 均值', ('mean_rand', 'xco', 'top2')),
        ('XCLASS_MOMENTUM', 3, 'rand_weighted', 'main 均值', ('mean_rand', 'xc', 'weighted')),
        ('XCLASS_MOMENTUM', 3, 'rand_weighted', 'oos 均值', ('mean_rand', 'xco', 'weighted')),
        ('XCLASS_MOMENTUM', 3, '基线', 'main 均值', ('j', 'xc', 'results.base_top3_rebal_monthly.combined_sharpe')),
        ('XCLASS_MOMENTUM', 3, '基线', 'oos 均值', ('j', 'xco', 'results.base_top3_rebal_monthly.combined_sharpe')),
    ]
    # ── XCLASS_MOMENTUM 表4: 长历史独立审判 ──
    A += [
        ('XCLASS_MOMENTUM', 4, 'baseline_fixed_equal', '袖套夏普', ('j', 'xclh', 'results.baseline_fixed_equal.sleeve_sharpe')),
        ('XCLASS_MOMENTUM', 4, 'mom_L60_top2', '袖套夏普', ('j', 'xclh', 'results.mom_L60_top2.sleeve_sharpe')),
        ('XCLASS_MOMENTUM', 4, 'mom_L60_weighted', '袖套夏普', ('j', 'xclh', 'results.mom_L60_weighted.sleeve_sharpe')),
        ('XCLASS_MOMENTUM', 4, 'mom_L250_top2', '袖套夏普', ('j', 'xclh', 'results.mom_L250_top2.sleeve_sharpe')),
        ('XCLASS_MOMENTUM', 4, 'mom_L250_weighted', '袖套夏普', ('j', 'xclh', 'results.mom_L250_weighted.sleeve_sharpe')),
        ('XCLASS_MOMENTUM', 4, 'rand_top2（6 种子均值）', '袖套夏普', ('j', 'xclh', 'random_summary.top2.mean')),
        ('XCLASS_MOMENTUM', 4, 'rand_weighted（6 种子均值）', '袖套夏普', ('j', 'xclh', 'random_summary.weighted.mean')),
        ('XCLASS_MOMENTUM', 4, 'mom_L250_top2', '滚动504日夏普胜率', ('j', 'xclh', 'results.mom_L250_top2.rolling_sharpe_504d.sharpe_win_rate')),
        ('XCLASS_MOMENTUM', 4, 'mom_L60_top2', '滚动504日夏普胜率', ('j', 'xclh', 'results.mom_L60_top2.rolling_sharpe_504d.sharpe_win_rate')),
    ]
    # ── L3_MOMENTUM 表14: §三 推荐配置表 ──
    A += [
        ('L3_MOMENTUM', 14, '夏普', 'main（推导）', F1M),
        ('L3_MOMENTUM', 14, '夏普', 'oos（样本外）', F1O),
        ('L3_MOMENTUM', 14, '总收益', 'main（推导）', ('j', 'fr', 'F1_final.combined_return')),
        ('L3_MOMENTUM', 14, '总收益', 'oos（样本外）', ('j', 'fro', 'F1_final.combined_return')),
        ('L3_MOMENTUM', 14, '最大回撤', 'main（推导）', ('j', 'fr', 'F1_final.combined_maxdd')),
        ('L3_MOMENTUM', 14, '最大回撤', 'oos（样本外）', ('j', 'fro', 'F1_final.combined_maxdd')),
    ]
    return A


def parse_num(cell):
    txt = cell.replace('−', '-').replace('**', '').strip()
    m = re.search(r'[-+]?\d+(?:\.\d+)?', txt)
    return float(m.group()) if m else None


def extract_tables(text):
    """返回 [(表头行cells, [数据行cells...]), ...]"""
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
        # 去掉分隔行
        rows = [r for r in rows if not all(re.fullmatch(r'\s*:?-{2,}:?\s*', c or ' ') for c in r)]
        if rows:
            out.append((rows[0], rows[1:]))
    return out


def find_cell(header_cells, data_cells, row_label, col_header):
    """在表里定位 (行标签, 列名) 的单元格值，返回 (值文本, 错误信息)"""
    hdr = [norm(c) for c in header_cells]
    want_col = norm(col_header)
    if want_col not in hdr:
        return None, f'表头里没有列「{col_header}」，现有列：{hdr}'
    ci = hdr.index(want_col)
    want_row = norm(row_label)
    hit = None
    for r in data_cells:
        if r and norm(r[0]) == want_row:
            if hit is not None:
                return None, f'行标签「{row_label}」出现多次'
            hit = r
    if hit is None:
        rows = [norm(r[0]) for r in data_cells if r]
        return None, f'表里没有行「{row_label}」，现有行：{rows}'
    if ci >= len(hit):
        return None, f'行「{row_label}」缺少第 {ci} 列'
    return hit[ci], None


SEEDS = (20261030, 7, 99, 1234, 555, 8888)


def resolve(spec):
    if spec[0] == 'lit':
        return spec[1]
    if spec[0] == 'l3fam_mean':
        # L3 产物里某族（mom_/random_）所有 combined_sharpe 的均值
        art, fam = spec[1], spec[2]
        pre = 'mom_' if fam == 'mom' else 'random_'
        vals = [dig(ART[art], f'results.{k}.combined_sharpe')
                for k in ART[art]['results'] if k.startswith(pre)]
        return sum(vals) / len(vals)
    if spec[0] == 'l3_delta':
        # random L60 K1 月度均值 - static_top1_rebal_monthly
        art = spec[1]
        vals = [dig(ART[art], f'results.random_L60_K1_monthly_s{s}.combined_sharpe')
                for s in SEEDS]
        return sum(vals) / len(vals) - dig(ART[art], 'results.static_top1_rebal_monthly.combined_sharpe')
    if spec[0] == 'mean_rand':
        # 随机对照某个分组的夏普均值（产物路径由 art 决定）
        art, grp = spec[1], spec[2]
        vals = [dig(ART[art], f'results.xclass_rand_{grp}_s{s}.combined_sharpe') for s in SEEDS]
        return sum(vals) / len(vals)
    return dig(ART[spec[1]], spec[2])


TOL = 5e-4


def check(texts, anchors):
    problems = []
    for a in anchors:
        doc_pref, tidx, row_label, col_hdr, spec = a[0], a[1], a[2], a[3], a[4]
        doc = next((k for k in texts if k.startswith(doc_pref)), None)
        if not doc:
            problems.append(f'{doc_pref}: 文档不存在')
            continue
        tables = extract_tables(texts[doc])
        if isinstance(tidx, str):
            # 按表头关键字定位表
            cands = [i for i, (h, _) in enumerate(tables)
                     if any(norm(col_hdr) == norm(c) for c in h)]
            if not cands:
                problems.append(f'{doc}: 找不到含列「{col_hdr}」的表')
                continue
            tidx = cands[0]
        if tidx >= len(tables):
            problems.append(f'{doc}: 表序号 {tidx} 越界（共 {len(tables)} 张表）')
            continue
        header, data = tables[tidx]
        cell, err = find_cell(header, data, row_label, col_hdr)
        if err:
            problems.append(f'{doc}[表{tidx}] {err}')
            continue
        got = parse_num(cell)
        if got is None:
            problems.append(f'{doc}[表{tidx}] 「{row_label}」×「{col_hdr}」不是数字：{cell!r}')
            continue
        want = resolve(spec)
        cands = [want, want * 100]
        # 单元格写成百分比（0.1848 → "18.5%"）时，用 0.05 的绝对容差（四舍五入到 0.1%）
        if '%' in cell and abs(got - want * 100) <= 0.05:
            continue
        if not any(abs(got - c) <= TOL for c in cands):
            problems.append(
                f'{doc}[表{tidx}] 「{row_label}」×「{col_hdr}」= {got}，'
                f'产物 = {want}（×100={want*100:.4f}）')
    return problems


def load_docs():
    return {os.path.basename(f): open(f, encoding='utf-8').read()
            for f in sorted(glob.glob(os.path.join(DOCS, '*.md')))}


STALE = {
    '1.3285': '旧 top3 买入持有 main',
    '1.3581': '旧 top3 每月再平衡 main',
    '0.9323': '旧 top3 买入持有 oos',
    '1.0435': '旧 top3 每月再平衡 oos',
    '1.1628': '旧 top1 oos',
    '1.1240': '旧 top1 买入持有 oos',
    '0.7831': '旧剔 gold oos',
    '0.9557': '旧剔 gold main',
    '8/12': '旧子窗口回撤胜率',
    '0.0762': '旧平均 Δ夏普',
    '−0.192': '旧类内分散 oos',
    '−9.11%': '旧 oos 回撤',
}


def scan_stale(texts):
    hits = []
    for doc, body in texts.items():
        for old, why in STALE.items():
            for m in re.finditer(re.escape(old), body):
                ctx = body[max(0, m.start() - 70):m.start() + 50].replace('\n', '⏎')
                hits.append((doc, old, why, ctx))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--neg', action='store_true', help='自检：注入错误验证审计能抓到')
    args = ap.parse_args()

    texts = load_docs()
    anchors = build_anchors()

    print('=' * 78)
    print(f'表格语义核对：{len(anchors)} 条断言 × 逐格比对产物')
    print('=' * 78)
    problems = check(texts, anchors)
    for p in problems:
        print(f'  ❌ {p}')
    if not problems:
        print('  ✅ 全部断言通过')

    print()
    print('=' * 78)
    print('反向扫描：已知旧值（合法命中 = 『旧→新』对照说明）')
    print('=' * 78)
    hits = scan_stale(texts)
    for doc, old, why, ctx in hits:
        print(f'  ⚠️  [{doc}] {old!r} ({why})\n       ...{ctx}...')
    if not hits:
        print('  ✅ 没有发现旧值残留')

    if args.neg:
        print()
        print('=' * 78)
        print('自检：注入错误 → 期望审计失败')
        print('=' * 78)
        target = os.path.join(DOCS, 'L3_MOMENTUM_2026-10-30.md')
        original = open(target, encoding='utf-8').read()
        broken = original.replace(
            '| **top3 每月再平衡** | **1.3767** | **1.0382** | **1.0382** |',
            '| **top3 每月再平衡** | **1.9999** | **1.0382** | **1.0382** |')
        if broken == original:
            print('  ❌ 注入失败：目标字符串未找到')
            return 1
        try:
            open(target, 'w', encoding='utf-8').write(broken)
            t2 = load_docs()
            p2 = check(t2, anchors)
            print(f'  注入后断言失败数：{len(p2)}（期望 > 0）')
            for p in p2[:5]:
                print(f'      {p}')
            neg_ok = len(p2) > 0
        finally:
            open(target, 'w', encoding='utf-8').write(original)
        print(f'  自检结论：{"✅ 审计能抓到注入的错误" if neg_ok else "❌ 审计有漏洞"}')
        print('  已还原文档。')

    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
