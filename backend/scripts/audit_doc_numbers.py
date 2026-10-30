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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # artifact_manifest 同目录


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
    'p2m': J('multi_asset_p2_main.json'),
    'p2o': J('multi_asset_p2_oos.json'),
    'p2l': J('multi_asset_p2_long.json'),
    'p2a': J('multi_asset_p2_attrib.json'),
    'p4m': J('multi_asset_p4_main.json'),
    'p4o': J('multi_asset_p4_oos.json'),
    'p4l': J('multi_asset_p4_long.json'),
    'p5': J('p5_class_audit.json'),
    'p6m': J('multi_asset_p6_main.json'),
    'p6o': J('multi_asset_p6_oos.json'),
    'p3m': J('multi_asset_p3_main.json'),
    'p3o': J('multi_asset_p3_oos.json'),
    'p3l': J('multi_asset_p3_long.json'),
    'p3br': J('p3_bond_recon.json'),
    'p7m': J('multi_asset_p7_main.json'),
    'p7o': J('multi_asset_p7_oos.json'),
    'p8m': J('multi_asset_p8_main.json'),
    'p8o': J('multi_asset_p8_oos.json'),
}


def safe_dig(art, path):
    """取不到就返回 None（而不是抛 KeyError）——用于「该字段可能不存在」的锚点。"""
    try:
        return dig(ART[art], path)
    except (KeyError, TypeError, IndexError):
        return None


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
    # ── P2_CLASS_WEIGHT 表1: 每个类与 A 股腿的 ρ（机制核心）──
    for _w, _wn in (('main', '推导'), ('oos', '样本外')):
        for _c in ('gold', 'bond', 'commodity'):
            A += [
                ('P2_CLASS_WEIGHT', 1, f'{_wn} {_c}', '年化波动',
                 ('j', 'p2a', f'{_w}.classes.{_c}.sleeve_vol')),
                ('P2_CLASS_WEIGHT', 1, f'{_wn} {_c}', '夏普',
                 ('j', 'p2a', f'{_w}.classes.{_c}.sleeve_sharpe')),
                ('P2_CLASS_WEIGHT', 1, f'{_wn} {_c}', 'ρ(类, A 股腿)',
                 ('j', 'p2a', f'{_w}.classes.{_c}.corr_with_a_share')),
            ]
    # ── P2_CLASS_WEIGHT 表2: P2a 商品类内 ──
    for _disp, _k in (
        ('top1（流动性第 1）', 'com_top1'),
        ('top2', 'com_top2'),
        ('top3（定稿现状）', 'com_top3'),
        ('全部 4 只', 'com_all'),
        ('单只 159985（豆粕）', 'com_single_159985'),
        ('单只 159981（能源化工）', 'com_single_159981'),
        ('单只 159980（有色）', 'com_single_159980'),
        ('单只 510170', 'com_single_510170'),
        ('top2 + 510170', 'com_top2_plus_last'),
        ('top3 去掉第 1 + 510170', 'com_top3_minus_first'),
    ):
        A += [
            ('P2_CLASS_WEIGHT', 2, _disp, '推导', ('j', 'p2m', f'results.{_k}.combined_sharpe')),
            ('P2_CLASS_WEIGHT', 2, _disp, '样本外', ('j', 'p2o', f'results.{_k}.combined_sharpe')),
            ('P2_CLASS_WEIGHT', 2, _disp, '较差窗口',
             ('jmin', 'p2m', f'results.{_k}.combined_sharpe', 'p2o', f'results.{_k}.combined_sharpe')),
        ]
    # ── P2_CLASS_WEIGHT 表3: P2b 组合口径 ──
    for _disp, _k in (
        ('等权（定稿）', 'cls_equal'),
        ('逆波动率 L60（无 cap）', 'cls_rvol60'),
        ('逆波动率 L120（无 cap）', 'cls_rvol120'),
        ('逆波动率 L250（无 cap）', 'cls_rvol250'),
        ('逆波动率 L60 + cap', 'cls_rvol60_cap'),
        ('逆波动率 L120 + cap', 'cls_rvol120_cap'),
        ('逆波动率 L250 + cap', 'cls_rvol250_cap'),
    ):
        A += [
            ('P2_CLASS_WEIGHT', 4, _disp, '推导', ('j', 'p2m', f'results.{_k}.combined_sharpe')),
            ('P2_CLASS_WEIGHT', 4, _disp, '样本外', ('j', 'p2o', f'results.{_k}.combined_sharpe')),
            ('P2_CLASS_WEIGHT', 4, _disp, '袖套波动(推导)', ('j', 'p2m', f'results.{_k}.sleeve_vol')),
            ('P2_CLASS_WEIGHT', 4, _disp, 'ρ(袖套,A)(推导)', ('j', 'p2m', f'results.{_k}.corr_sleeve_ashare')),
        ]
    # ── P2_CLASS_WEIGHT 表4: P2b 长历史 ──
    for _disp, _k in (
        ('等权', 'cls_equal'),
        ('逆波动率 L60（无 cap）', 'cls_rvol60'),
        ('逆波动率 L120（无 cap）', 'cls_rvol120'),
        ('逆波动率 L250（无 cap）', 'cls_rvol250'),
        ('逆波动率 L60 + cap', 'cls_rvol60_cap'),
        ('逆波动率 L120 + cap', 'cls_rvol120_cap'),
        ('逆波动率 L250 + cap', 'cls_rvol250_cap'),
    ):
        A += [
            ('P2_CLASS_WEIGHT', 5, _disp, '长历史袖套夏普', ('j', 'p2l', f'results.{_k}.sleeve_sharpe')),
        ]
        if _k != 'cls_equal':
            A += [
                ('P2_CLASS_WEIGHT', 5, _disp, 'Δ vs 等权',
                 ('j', 'p2l', f'results.{_k}.delta_sharpe_vs_base')),
                ('P2_CLASS_WEIGHT', 5, _disp, '滚动 504 日夏普胜率',
                 ('j', 'p2l', f'results.{_k}.rolling_sharpe_504d.sharpe_win_rate')),
            ]
    # ── P2_CLASS_WEIGHT 表6: 随机对照均值 ──
    for _disp, _pre in (('cls_rand（每期重抽）', 'cls_rand_s'),
                        ('cls_randstatic（固定配置）', 'cls_randstatic_s')):
        A += [
            ('P2_CLASS_WEIGHT', 6, _disp, '推导均值',
             ('globmean', 'p2m', _pre, 'combined_sharpe')),
            ('P2_CLASS_WEIGHT', 6, _disp, '样本外均值',
             ('globmean', 'p2o', _pre, 'combined_sharpe')),
            ('P2_CLASS_WEIGHT', 6, _disp, '长历史均值',
             ('globmean', 'p2l', _pre, 'sleeve_sharpe')),
        ]
    # ── P2_CLASS_WEIGHT 表7: 随机固定配置逐个结果 ──
    for _s in (20261030, 7, 99, 1234, 555, 8888):
        A += [
            ('P2_CLASS_WEIGHT', 7, f'cls_randstatic_s{_s}', '推导',
             ('j', 'p2m', f'results.cls_randstatic_s{_s}.combined_sharpe')),
            ('P2_CLASS_WEIGHT', 7, f'cls_randstatic_s{_s}', '样本外',
             ('j', 'p2o', f'results.cls_randstatic_s{_s}.combined_sharpe')),
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

    # ═══════════════════════ P3_BOND_CLASS ═══════════════════════
    # P3: bond 类剔除可转债 → 换长久期利率债
    P3DOC = 'P3_BOND_CLASS'
    _br_main = [x['code'] for x in ART['p3br']['windows']['main']]
    _br_oos = [x['code'] for x in ART['p3br']['windows']['oos']]
    # ── 表1/表2: 债券池逐只特征（勘察产物）──
    for _w, _codes in (('main', _br_main), ('oos', _br_oos)):
        _tidx = 1 if _w == 'main' else 2
        _cols = (['ρ(与 510300)', '年化波动', '窗口前 120 日均成交额(万)']
                 if _w == 'main' else
                 ['ρ(与 510300)', '年化波动', '窗口前 120 日均成交额(万)'])
        for _i, _c in enumerate(_codes):
            for _col, _fld, _sp in (
                    (_cols[0], 'rho', 'j'),
                    (_cols[1], 'vol', 'j'),
                    (_cols[2], 'turnover', 'jwan')):
                if _sp == 'j':
                    A.append((P3DOC, _tidx, _c, _col,
                              ('j', 'p3br', f'windows.{_w}.[{_i}].{_fld}')))
                else:
                    A.append((P3DOC, _tidx, _c, _col,
                              (_sp, 'p3br', f'windows.{_w}.[{_i}].{_fld}')))
    # ── 表4/表5: P3a 主结果表 ──
    _p3_keys = (['base', 'bond_no_cb', 'bond_cb_to_lowrho', 'bond_lowrho3',
                 'bond_all_rate']
                + [f'swap_rand_s{s}' for s in SEEDS]
                + ['bond_cb_sep', 'extra_lowrho3']
                + [f'extra_rand_s{s}' for s in SEEDS])
    for _w, _art, _tidx, _with504 in (('main', 'p3m', 4, True),
                                      ('oos', 'p3o', 5, False)):
        for _k in _p3_keys:
            _lbl = 'base（旧定稿）' if _k == 'base' else _k
            A.append((P3DOC, _tidx, _lbl, '组合夏普',
                      ('j', _art, f'results.{_k}.combined_sharpe')))
            A.append((P3DOC, _tidx, _lbl, '剔gold夏普',
                      ('j', _art, f'results.{_k}.sharpe_ex_gold')))
            A.append((P3DOC, _tidx, _lbl, '组合回撤',
                      ('j', _art, f'results.{_k}.combined_maxdd')))
            A.append((P3DOC, _tidx, _lbl, '组合收益',
                      ('j', _art, f'results.{_k}.combined_return')))
            A.append((P3DOC, _tidx, _lbl, 'ρ(袖套,A)',
                      ('j', _art, f'results.{_k}.corr_sleeve_ashare')))
            # base 是参照系，本身没有"胜率"（文档写的是 —）
            if _k != 'base':
                A.append((P3DOC, _tidx, _lbl, '滚动252日胜率',
                          ('j', _art, f'results.{_k}.rolling_sharpe_252d_win_rate')))
                if _with504:
                    A.append((P3DOC, _tidx, _lbl, '滚动504日胜率',
                              ('j', _art, f'results.{_k}.rolling_sharpe_504d_win_rate')))
    # ── 表6/表7: 随机对照逐项 ──
    for _w, _art, _tidx in (('main', 'p3m', 6), ('oos', 'p3o', 7)):
        for _s in SEEDS:
            _k = f'swap_rand_s{_s}'
            A.append((P3DOC, _tidx, _k, '组合夏普',
                      ('j', _art, f'results.{_k}.combined_sharpe')))
            A.append((P3DOC, _tidx, _k, '滚动252日胜率',
                      ('j', _art, f'results.{_k}.rolling_sharpe_252d_win_rate')))
    # ── 表8: P3b 新增第 4 类 ──
    A.append((P3DOC, 8, 'extra_lowrho3（全池ρ最低top-3作第4类）', 'main 夏普',
              ('j', 'p3m', 'results.extra_lowrho3.combined_sharpe')))
    A.append((P3DOC, 8, 'extra_lowrho3（全池ρ最低top-3作第4类）', 'oos 夏普',
              ('j', 'p3o', 'results.extra_lowrho3.combined_sharpe')))
    for _s in SEEDS:
        _k = f'extra_rand_s{_s}'
        A.append((P3DOC, 8, _k, 'main 夏普',
                  ('j', 'p3m', f'results.{_k}.combined_sharpe')))
        A.append((P3DOC, 8, _k, 'oos 夏普',
                  ('j', 'p3o', f'results.{_k}.combined_sharpe')))
    # ── 表9: 可转债独立成第 4 类 ──
    for _w, _art in (('main', 'p3m'), ('oos', 'p3o')):
        for _col, _fld in (('组合夏普', 'combined_sharpe'),
                           ('剔gold夏普', 'sharpe_ex_gold'),
                           ('ρ(袖套,A)', 'corr_sleeve_ashare'),
                           ('滚动252日胜率', 'rolling_sharpe_252d_win_rate'),
                           ('滚动504日胜率', 'rolling_sharpe_504d_win_rate')):
            # oos 窗口只有 484 个交易日 < 504 → 504 日滚动胜率不存在
            if _col == '滚动504日胜率' and _w == 'oos':
                continue
            A.append((P3DOC, 9, _w, _col,
                      ('j', _art, f'results.bond_cb_sep.{_fld}')))
    # ── 表10: bond 类 ρ 与自身夏普 ──
    for _w, _art, _cfg in (('main', 'p3m', 'base'),
                            ('main', 'p3m', 'bond_lowrho3'),
                            ('oos', 'p3o', 'base'),
                            ('oos', 'p3o', 'bond_lowrho3')):
        _lbl = f'{_w} · {_cfg}'
        A.append((P3DOC, 10, _lbl, 'ρ(bond 类, A 股腿)',
                  ('j', _art, f'results.{_cfg}.class_rho.bond.rho_a')))
        A.append((P3DOC, 10, _lbl, 'bond 类自身夏普',
                  ('j', _art, f'results.{_cfg}.class_rho.bond.sleeve_sharpe')))
    # ── 表11: 分年收益 ──
    for _w, _art in (('main', 'p3m'), ('oos', 'p3o')):
        for _y in ART[_art]['results']['base']['combined_yearly']:
            A.append((P3DOC, 11, f'{_w} · {_y}', 'base',
                      ('j', _art, f'results.base.combined_yearly.{_y}')))
            A.append((P3DOC, 11, f'{_w} · {_y}', 'bond_lowrho3',
                      ('j', _art, f'results.bond_lowrho3.combined_yearly.{_y}')))
    # ── 表12: 长历史 ──
    for _k in ('base_long', 'lowrho3_long', 'plus_cb_long'):
        for _col, _fld in (('袖套夏普', 'sleeve_sharpe'),
                           ('年化波动', 'sleeve_vol'),
                           ('累计收益', 'sleeve_return'),
                           ('最大回撤', 'sleeve_maxdd'),
                           ('滚动252日胜率', 'rolling_sharpe_252d_win_rate'),
                           ('滚动504日胜率', 'rolling_sharpe_504d_win_rate')):
            # base_long 是参照系，本身没有胜率
            if _col.startswith('滚动') and _k == 'base_long':
                continue
            A.append((P3DOC, 12, _k, _col, ('j', 'p3l', f'results.{_k}.{_fld}')))
    # ── 表14: 当前流动性 ──
    for _c in ('511260', '511010', '511020'):
        A.append((P3DOC, 14, _c, '日均成交额(亿)',
                  ('jyi', 'p3m', f'current_liquidity.{_c}.avg_amount_120d')))
    # ── 表16: 前后对比 ──
    _p3_cmp = [
        ('组合夏普·推导窗口', 'main', 'base', 'bond_lowrho3', 'combined_sharpe'),
        ('组合夏普·样本外', 'oos', 'base', 'bond_lowrho3_noliq', 'combined_sharpe'),
        ('组合回撤·推导窗口', 'main', 'base', 'bond_lowrho3', 'combined_maxdd'),
        ('组合回撤·样本外', 'oos', 'base', 'bond_lowrho3_noliq', 'combined_maxdd'),
        ('ρ(袖套, A 股腿)·推导', 'main', 'base', 'bond_lowrho3', 'corr_sleeve_ashare'),
        ('ρ(袖套, A 股腿)·样本外', 'oos', 'base', 'bond_lowrho3_noliq', 'corr_sleeve_ashare'),
        ('ρ(bond 类, A 股腿)·推导', 'main', 'base', 'bond_lowrho3', 'class_rho.bond.rho_a'),
        ('ρ(bond 类, A 股腿)·样本外', 'oos', 'base', 'bond_lowrho3_noliq', 'class_rho.bond.rho_a'),
        ('剔gold夏普·推导', 'main', 'base', 'bond_lowrho3', 'sharpe_ex_gold'),
        ('剔gold夏普·样本外', 'oos', 'base', 'bond_lowrho3_noliq', 'sharpe_ex_gold'),
        ('滚动504日胜率·推导', 'main', None, 'bond_lowrho3', 'rolling_sharpe_504d_win_rate'),
    ]
    for _lbl, _w, _old, _new, _fld in _p3_cmp:
        _art = 'p3m' if _w == 'main' else 'p3o'
        if _old:
            A.append((P3DOC, 16, _lbl, '旧定稿', ('j', _art, f'results.{_old}.{_fld}')))
        A.append((P3DOC, 16, _lbl, '新定稿', ('j', _art, f'results.{_new}.{_fld}')))

    # ── P4 · 类间配权（表格 2/3/4/5/6/7/9）──
    P4DOC = 'P4_CLASS_WEIGHT'
    P4_CANDS = ['base_equal', 'invvol', 'invvol_nocap', 'minvol',
                 'riskparity', 'lowcorr', 'sharpew']
    P4_NAMES = {
        'base_equal': '类间等权（定稿基线）',
        'invvol': '逆波动率 cap40/floor8',
        'invvol_nocap': '逆波动率 无 cap',
        'minvol': '100% 给最低波动类',
        'riskparity': '风险平价',
        'lowcorr': '∝(1−ρ(类,A股腿))',
        'sharpew': '∝max(夏普,0)',
    }
    _p4_cols = [
        ('组合夏普', 'combined_sharpe', None),
        ('Δ基线', None, 'p4delta'),
        ('剔 gold 夏普', 'sharpe_ex_gold', None),
        ('ρ(袖套,A股腿)', 'corr_sleeve_ashare', None),
        ('最大回撤', 'combined_maxdd', None),
        ('收益', 'combined_return', None),
        ('换手', 'cost.turnover_ratio', 'j2'),
    ]
    for _tidx, _art in ((2, 'p4m'), (4, 'p4o')):
        for _k in P4_CANDS:
            _lbl = f"`{_k}` {P4_NAMES[_k]}"
            for _col, _fld, _spec in _p4_cols:
                if _spec == 'j2':
                    A.append((P4DOC, _tidx, _lbl, _col,
                              ('j2', _art, f'results.{_k}.{_fld}')))
                elif _spec:
                    A.append((P4DOC, _tidx, _lbl, _col, (_spec, _art, _k)))
                else:
                    A.append((P4DOC, _tidx, _lbl, _col,
                              ('j', _art, f'results.{_k}.{_fld}')))
            # 504 日胜率：base_equal 是参照系(None)；oos 窗口不足 504 天(None)
            if safe_dig(_art, f'results.{_k}.roll_win_504') is not None:
                A.append((P4DOC, _tidx, _lbl, '504 日胜率',
                          ('j', _art, f'results.{_k}.roll_win_504')))
    for _tidx, _art in ((3, 'p4m'), (5, 'p4o')):
        for _lbl, _pre in (('固定权重 `rand_s*`', 'rand_s'),
                           ('动态权重 `randdyn_s*`', 'randdyn_s')):
            A.append((P4DOC, _tidx, _lbl, 'n', ('p4fam', _art, _pre, 'n')))
            A.append((P4DOC, _tidx, _lbl, '均值', ('p4fam', _art, _pre, 'mean')))
            A.append((P4DOC, _tidx, _lbl, '范围', ('p4fam', _art, _pre, 'min')))
    _p4l_cols = [
        ('袖套夏普', 'sleeve_sharpe', None),
        ('Δ基线', None, 'p4ldelta'),
        ('袖套波动', 'sleeve_vol', None),
        ('ρ(袖套,510300)', 'corr_sleeve_proxy', None),
    ]
    for _k in P4_CANDS:
        _lbl = f"`{_k}` {P4_NAMES[_k]}"
        for _col, _fld, _spec in _p4l_cols:
            if _spec:
                A.append((P4DOC, 6, _lbl, _col, (_spec, 'p4l', _k)))
            else:
                A.append((P4DOC, 6, _lbl, _col, ('j', 'p4l', f'results.{_k}.{_fld}')))
        if safe_dig('p4l', f'results.{_k}.roll_win_504') is not None:
            A.append((P4DOC, 6, _lbl, '504 日胜率',
                      ('j', 'p4l', f'results.{_k}.roll_win_504')))
    for _lbl, _pre in (('固定权重 `rand_s*`', 'rand_s'),
                       ('动态权重 `randdyn_s*`', 'randdyn_s')):
        A.append((P4DOC, 7, _lbl, 'n', ('p4fam', 'p4l', _pre, 'n')))
        A.append((P4DOC, 7, _lbl, '袖套夏普均值', ('p4fam', 'p4l', _pre, 'mean')))
        A.append((P4DOC, 7, _lbl, '范围', ('p4fam', 'p4l', _pre, 'min')))
    for _lbl, _k in (('类间等权', 'base_equal'), ('逆波动率', 'invvol'),
                     ('逆波动率 无 cap', 'invvol_nocap'),
                     ('100% 最低波动类', 'minvol')):
        A.append((P4DOC, 9, _lbl, 'bond 平均权重',
                  ('j', 'p4m', f'results.{_k}.avg_class_weight.bond')))
        A.append((P4DOC, 9, _lbl, '袖套波动', ('j', 'p4m', f'results.{_k}.sleeve_vol')))
        A.append((P4DOC, 9, _lbl, 'ρ(袖套,A股腿)',
                  ('j', 'p4m', f'results.{_k}.corr_sleeve_ashare')))
        A.append((P4DOC, 9, _lbl, '组合收益', ('j', 'p4m', f'results.{_k}.combined_return')))
        A.append((P4DOC, 9, _lbl, '组合夏普', ('j', 'p4m', f'results.{_k}.combined_sharpe')))

    # ── P5 · 分类审计（gold 表 0/1/2，commodity 表 3/4/5）──
    P5DOC = 'P5_CLASS_AUDIT'
    for _cls, _off in (('gold', 0), ('commodity', 3)):
        _ref_col = 'ρ(纯金基准)' if _cls == 'gold' else 'ρ(有色基准)'
        for _w, _wi in (('w2020', 0), ('oos_2021', 1), ('main_2023', 2)):
            _tidx = _off + _wi
            for _row in ART['p5']['windows'][_w]['classes'][_cls]['rows']:
                _c = _row['code']
                _base = f"windows.{_w}.classes.{_cls}.by_code.{_c}"
                if _row.get('rho_ref') is not None:
                    A.append((P5DOC, _tidx, _c, _ref_col, ('j', 'p5', f'{_base}.rho_ref')))
                if _row.get('rho_proxy') is not None:
                    A.append((P5DOC, _tidx, _c, 'ρ(A股代理)',
                              ('j', 'p5', f'{_base}.rho_proxy')))
                if _row.get('vol') is not None:
                    A.append((P5DOC, _tidx, _c, '年化波动', ('j', 'p5', f'{_base}.vol')))
                if _row.get('turnover'):
                    A.append((P5DOC, _tidx, _c, '窗口前 120 日日均成交额',
                              ('jwan', 'p5', f'{_base}.turnover')))

    # ── P6 · 主腿权重敏感性（表格 0/1/2/3/4）──
    P6DOC = 'P6_WEIGHT_SENS'
    _p6_cols = [
        ('main 夏普', 'p6m', 'combined_sharpe', None),
        ('oos 夏普', 'p6o', 'combined_sharpe', None),
        ('oos Δ基线', 'p6o', None, 'p6delta'),
        ('main 回撤', 'p6m', 'combined_maxdd', None),
        ('oos 回撤', 'p6o', 'combined_maxdd', None),
        ('main 波动', 'p6m', 'combined_vol', None),
        ('oos 波动', 'p6o', 'combined_vol', None),
    ]
    for _w in ART['p6m']['w_grid']:
        _k = f"w{int(round(_w*100)):03d}"
        for _col, _art, _fld, _spec in _p6_cols:
            if _spec:
                A.append((P6DOC, 0, _k, _col, (_spec, _art, _k)))
            else:
                A.append((P6DOC, 0, _k, _col, ('j', _art, f'results.{_k}.{_fld}')))
    for _s in ART['p6m']['random_w']:
        _k = f"randw_s{_s}"
        A.append((P6DOC, 1, _k, '随机 w', ('j', 'p6m', f'random_w.{_s}')))
        A.append((P6DOC, 1, _k, 'main 夏普', ('j', 'p6m', f'results.{_k}.combined_sharpe')))
        A.append((P6DOC, 1, _k, 'oos 夏普', ('j', 'p6o', f'results.{_k}.combined_sharpe')))
        A.append((P6DOC, 1, _k, 'main Δ基线', ('p6delta', 'p6m', _k)))
    _p6_shape = [('最优 w', 'best_w'), ('最优夏普', 'best_sharpe'),
                 ('[0.40,0.60] 内夏普极差', 'mid_range_sharpe_span_040_060'),
                 ('相邻网格最大跳变', 'max_adjacent_jump'),
                 ('随机 w 夏普最大', 'random_w_sharpe_max'),
                 ('随机 w 夏普最小', 'random_w_sharpe_min')]
    for _wn, _art in (('main', 'p6m'), ('oos', 'p6o')):
        for _col, _fld in _p6_shape:
            A.append((P6DOC, 2, _wn, _col, ('j', _art, f'shape.{_fld}')))
    for _tidx, _art in ((3, 'p6m'), (4, 'p6o')):
        for _sub in sorted(ART[_art]['subwindows']):
            for _col, _k in (('w=0.50', 'w050'), ('w=0.40', 'w040'), ('w=0.60', 'w060')):
                _v = safe_dig(_art, f'subwindows.{_sub}.{_k}')
                if _v is not None:
                    A.append((P6DOC, _tidx, _sub, _col, ('j', _art, f'subwindows.{_sub}.{_k}')))
            A.append((P6DOC, _tidx, _sub, '天数',
                      ('j', _art, f'subwindows.{_sub}.n')))

    # ── P7 · 因子消融（表 2=main 排名, 3=随机对照, 5=oos, 6=子窗口）──
    P7DOC = 'P7_FACTOR_ABLATION'
    _p7_main_cols = [
        ('因子数', 'n_factors'), ('A腿夏普', 'a_share_sharpe'),
        ('组合夏普', 'combined_sharpe'),
        ('组合收益', 'combined_return'), ('组合回撤', 'combined_maxdd'),
        ('交易笔数', 'a_share_trades'),
    ]
    for _l in ART['p7m']['ranking']:
        for _col, _fld in _p7_main_cols:
            A.append((P7DOC, 2, _l, _col, ('j', 'p7m', f'results.{_l}.{_fld}')))
        A.append((P7DOC, 2, _l, 'Δ组合', ('p7delta', 'p7m', _l)))
    # 随机对照分组（表 3）：行标签 = keep 数
    for _keep, _g in ART['p7m']['random_groups'].items():
        _labs = _g['labels']
        A.append((P7DOC, 3, _keep, '组合夏普最大',
                  ('p7rand_max', 'p7m', _keep)))
        A.append((P7DOC, 3, _keep, '组合夏普最小',
                  ('p7rand_min', 'p7m', _keep)))
        for _i, _lab in enumerate(_labs[:3]):
            A.append((P7DOC, 3, _keep, f'随机臂 {_i + 1}',
                      ('j', 'p7m', f'results.{_lab}.combined_sharpe')))
    # oos 表（表 5）
    _p7_oos_cols = [
        ('因子数', 'n_factors'), ('A腿夏普', 'a_share_sharpe'),
        ('组合夏普', 'combined_sharpe'),
        ('组合收益', 'combined_return'), ('组合回撤', 'combined_maxdd'),
    ]
    for _l in ART['p7o']['results']:
        for _col, _fld in _p7_oos_cols:
            A.append((P7DOC, 5, _l, _col, ('j', 'p7o', f'results.{_l}.{_fld}')))
        A.append((P7DOC, 5, _l, 'Δ组合', ('p7delta', 'p7o', _l)))
    # 子窗口（表 6）：main base25 + oos base25
    for _sub in sorted(ART['p7m']['subwindows']):
        A.append((P7DOC, 6, _sub, 'base25',
                  ('j', 'p7m', f'subwindows.{_sub}.base25')))
        A.append((P7DOC, 6, _sub, '天数',
                  ('j', 'p7m', f'subwindows.{_sub}.n')))

    # ── P8 · 暴露中性化（表 1=主结果, 3=子窗口）──
    P8DOC = 'P8_EXPOSURE_NEUTRAL'
    _p8_cols = [
        ('A腿夏普(关)', 'base25', 'a_share_sharpe'),
        ('A腿夏普(开)', 'neut25', 'a_share_sharpe'),
        ('组合夏普(关)', 'base25', 'combined_sharpe'),
        ('组合夏普(开)', 'neut25', 'combined_sharpe'),
        ('组合回撤(关)', 'base25', 'combined_maxdd'),
        ('组合回撤(开)', 'neut25', 'combined_maxdd'),
    ]
    for _w, _art in (('main', 'p8m'), ('oos', 'p8o')):
        for _col, _arm, _fld in _p8_cols:
            A.append((P8DOC, 1, _w, _col, ('j', _art, f'results.{_arm}.{_fld}')))
        A.append((P8DOC, 1, _w, 'ΔA腿', ('p8delta', _art, 'a_share_sharpe')))
        A.append((P8DOC, 1, _w, 'Δ组合', ('p8delta', _art, 'combined_sharpe')))
    for _w, _art in (('main', 'p8m'), ('oos', 'p8o')):
        for _i, _sub in enumerate(sorted(ART[_art]['subwindows'])):
            A.append((P8DOC, 3, f"{_w}/{_sub}", 'base25(关)',
                      ('j', _art, f'subwindows.{_sub}.base25')))
            A.append((P8DOC, 3, f"{_w}/{_sub}", 'neut25(开)',
                      ('j', _art, f'subwindows.{_sub}.neut25')))
    return A


def parse_num(cell):
    txt = cell.replace('−', '-').replace('**', '').strip()
    m = re.search(r'[-+]?\d+(?:\.\d+)?', txt)
    return float(m.group()) if m else None


def cell_num_positions(cell):
    """单元格里所有数字及其位置：[(num, start, end), ...]

    注意：`1.3767 (+59.0%)` 这类**复合单元格**里，锚点对应的是**第一个**数字
    （主值），括号里的百分比是补充说明、不参与锚点。
    """
    txt = cell.replace('−', '-')
    return [(float(m.group()), m.start(), m.end())
            for m in re.finditer(r'[-+]?\d+(?:\.\d+)?', txt)]


def target_number(cell, want):
    """定位单元格中对应 `want` 的数字。

    规则（审计器与回填器必须完全一致，否则「审计通过」不代表「数字对」）：
      - 单元格**只有一个数字且含 `%`** → 这是百分比格，目标值 = want × 100；
      - 否则（纯数值格 / 复合格）→ 目标值 = want，取**第一个**数字。
    返回 (kind, num, start, end) 或 None。
    """
    nums = cell_num_positions(cell)
    if not nums:
        return None
    n, s, e = nums[0]
    if len(nums) == 1 and '%' in cell:
        if abs(n - want * 100) <= 0.05:
            return ('pct', n, s, e)
        if abs(n - want) <= TOL:
            return ('direct', n, s, e)
        return None
    if abs(n - want) <= TOL:
        return ('direct', n, s, e)
    return None


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
    if spec[0] == 'globmean':
        # 产物 results 里所有以 prefix 开头的键，某字段的均值
        art, pre, field = spec[1], spec[2], spec[3]
        vals = [dig(ART[art], f'results.{k}.{field}')
                for k in ART[art]['results'] if k.startswith(pre)]
        return sum(vals) / len(vals)
    if spec[0] == 'jwan':
        # 成交额（万元），四舍五入到整数 —— 文档展示就是取整的，期望值同样取整
        return round(dig(ART[spec[1]], spec[2]) / 1e4)
    if spec[0] == 'jyi':
        # 成交额（亿元），保留 3 位小数
        return round(dig(ART[spec[1]], spec[2]) / 1e8, 3)
    if spec[0] == 'j2':
        # 文档按 2 位小数展示的量（换手等）—— 期望值同样取到 2 位再比
        return round(dig(ART[spec[1]], spec[2]), 2)
    if spec[0] == 'p6delta':
        return (dig(ART[spec[1]], f'results.{spec[2]}.combined_sharpe')
                - dig(ART[spec[1]], 'results.w050.combined_sharpe'))
    if spec[0] == 'p7delta':
        # 该臂组合夏普 - 同窗口 base25 组合夏普
        return (dig(ART[spec[1]], f'results.{spec[2]}.combined_sharpe')
                - dig(ART[spec[1]], 'results.base25.combined_sharpe'))
    if spec[0] == 'p7rand_max':
        return dig(ART[spec[1]], f'random_groups.{spec[2]}.max')
    if spec[0] == 'p7rand_min':
        return dig(ART[spec[1]], f'random_groups.{spec[2]}.min')
    if spec[0] == 'p8delta':
        # neut25 - base25（指定字段）
        return (dig(ART[spec[1]], f'results.neut25.{spec[2]}')
                - dig(ART[spec[1]], f'results.base25.{spec[2]}'))
    if spec[0] == 'p4delta':
        # P4 组合夏普相对 base_equal 的差
        art, key = spec[1], spec[2]
        return (dig(ART[art], f'results.{key}.combined_sharpe')
                - dig(ART[art], 'results.base_equal.combined_sharpe'))
    if spec[0] == 'p4ldelta':
        # P4 袖套夏普相对 base_equal 的差
        art, key = spec[1], spec[2]
        return (dig(ART[art], f'results.{key}.sleeve_sharpe')
                - dig(ART[art], 'results.base_equal.sleeve_sharpe'))
    if spec[0] == 'p4fam':
        # P4 random 族统计：art, prefix('rand_s'|'randdyn_s'), stat('mean'|'max'|'min'|'n')
        art, pre, stat = spec[1], spec[2], spec[3]
        field = 'sleeve_sharpe' if art == 'p4l' else 'combined_sharpe'
        vals = [dig(ART[art], f'results.{k}.{field}')
                for k in ART[art]['results'] if k.startswith(pre)]
        if stat == 'n':
            return len(vals)
        if stat == 'mean':
            return sum(vals) / len(vals)
        if stat == 'max':
            return max(vals)
        if stat == 'min':
            return min(vals)
        raise SystemExit(f'[audit] 未知 p4fam stat {stat}')
    if spec[0] == 'jmin':
        # 两个产物值取小者
        return min(dig(ART[spec[1]], spec[2]), dig(ART[spec[3]], spec[4]))
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
        nums = cell_num_positions(cell)
        if not nums:
            problems.append(f'{doc}[表{tidx}] 「{row_label}」×「{col_hdr}」不是数字：{cell!r}')
            continue
        want = resolve(spec)
        hit = target_number(cell, want)
        if hit is None:
            got = nums[0][0]
            kind = '百分比格' if (len(nums) == 1 and '%' in cell) else '数值/复合格'
            problems.append(
                f'{doc}[表{tidx}] 「{row_label}」×「{col_hdr}」= {got}（{kind}），'
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
    ap.add_argument('--skip-manifest', action='store_true',
                    help='跳过产物指纹校验（仅在刚重跑完实验、清单待刷新时用）')
    args = ap.parse_args()

    # ── 产物指纹校验（必须在数字核对之前）──
    # data/ 被 gitignore → 产物没有版本控制。若某个迟到的后台进程覆盖了 JSON，
    # 下面的数字核对会"成功"核对出一份已被改过的产物，静默失效。
    # 所以先验 sha256，产物被动过就直接退出。
    if not args.skip_manifest:
        import artifact_manifest
        print('=' * 78)
        print('产物指纹校验（防静默覆盖）')
        print('=' * 78)
        if artifact_manifest.verify() != 0:
            print('  → 产物已被改动/丢失，后续数字核对无意义，退出。')
            return 2
        print()

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
        print('自检：动态注入错误 → 期望审计失败')
        print('=' * 78)
        # **动态注入**：从锚点实时定位真实单元格并改坏其中一个数字。
        # 旧实现把注入目标写死成字面量（如 `**1.3767**`），
        # 文档数字一更新注入点就失效 —— 自检形同虚设（而且是「静默失效」）。
        t2 = dict(texts)
        injected = []
        seen_docs = set()
        for a in anchors:
            if len(injected) >= 12:
                break
            doc_pref, tidx, row_label, col_hdr, spec = a[0], a[1], a[2], a[3], a[4]
            if doc_pref in seen_docs:
                continue
            doc = next((k for k in texts if k.startswith(doc_pref)), None)
            if not doc:
                continue
            try:
                tables = extract_tables(texts[doc])
                if isinstance(tidx, str):
                    cands = [i for i, (h, _) in enumerate(tables)
                             if any(norm(col_hdr) == norm(c) for c in h)]
                    if not cands:
                        continue
                    tidx = cands[0]
                if tidx >= len(tables):
                    continue
                header, data = tables[tidx]
                cell, err = find_cell(header, data, row_label, col_hdr)
                if err or cell is None:
                    continue
                nums = cell_num_positions(cell)
                if not nums:
                    continue
                n, s, e = nums[0]
                bad_cell = cell[:s] + f"{n + 0.5:.4f}" + cell[e:]
                if bad_cell == cell or texts[doc].count(cell) != 1:
                    continue
                t2[doc] = t2[doc].replace(cell, bad_cell, 1)
                injected.append((doc, row_label, col_hdr, cell, bad_cell))
                seen_docs.add(doc_pref)
            except Exception:
                continue
        if not injected:
            print('  ❌ 未能构造任何注入点（锚点结构变了？）')
            return 1
        for doc, rl, ch, oc, bc in injected:
            print(f'  注入 [{doc}] 「{rl[:18]}」×「{ch[:14]}」: {oc} → {bc}')
        p2 = check(t2, anchors)
        print(f'  注入后断言失败数：{len(p2)}（期望 > 0）')
        for p in p2[:8]:
            print(f'      {p}')
        neg_ok = len(p2) > 0
        print(f'  自检结论：{"✅ 审计能抓到注入的错误" if neg_ok else "❌ 审计有漏洞"}')
        print('  （动态注入不落盘，文档未被修改）')
        if not neg_ok:
            return 1
        return 0

    if False:
        target = os.path.join(DOCS, 'L3_MOMENTUM_2026-10-30.md')
        original = open(target, encoding='utf-8').read()
        broken = original.replace(
            '| **top3 每月再平衡** | **1.3767** | **1.0382** | **1.0382** |',
            '| **top3 每月再平衡** | **1.9999** | **1.0382** | **1.0382** |')
        if broken == original:
            print('  ❌ 注入失败：目标字符串未找到')
            return 1
        # 第二个注入点：P4 新增锚点（防止只对老文档有效）
        p4target = os.path.join(DOCS, 'P4_CLASS_WEIGHT_2026-10-31.md')
        p4orig = open(p4target, encoding='utf-8').read()
        p4broken = p4orig.replace(
            '| `lowcorr` ∝(1−ρ(类,A股腿)) | 1.4472 |',
            '| `lowcorr` ∝(1−ρ(类,A股腿)) | 9.9999 |')
        p4_ok = p4broken != p4orig
        try:
            open(target, 'w', encoding='utf-8').write(broken)
            if p4_ok:
                open(p4target, 'w', encoding='utf-8').write(p4broken)
            t2 = load_docs()
            p2 = check(t2, anchors)
            print(f'  注入后断言失败数：{len(p2)}（期望 > 0）')
            for p in p2[:6]:
                print(f'      {p}')
            neg_ok = len(p2) > 0 and p4_ok
            if not p4_ok:
                print('  ❌ P4 注入点未找到（文档结构变了？）')
        finally:
            open(target, 'w', encoding='utf-8').write(original)
            if p4_ok:
                open(p4target, 'w', encoding='utf-8').write(p4orig)
        print(f'  自检结论：{"✅ 审计能抓到注入的错误" if neg_ok else "❌ 审计有漏洞"}')
        print('  已还原文档。')

    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
