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
    """按路径取值。

    路径用 '.' 分隔；但键名里本身含小数点（如 'n_in_0.40_0.60'）会被误切，
    所以额外支持 '/' 分隔（'/' 优先，不做正则拆分，键名可原样带入）。
    """
    if isinstance(path, (list, tuple)):
        parts = list(path)
    elif '/' in path:
        parts = path.split('/')
    else:
        parts = re.findall(r'[^.\[\]]+|\[\d+\]', path)
    cur = obj
    for part in parts:
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
    'p12m': J('multi_asset_p12_main.json'),
    'p12o': J('multi_asset_p12_oos.json'),
    'p12md': J('multi_asset_p12_main_degraded.json'),
    'p12od': J('multi_asset_p12_oos_degraded.json'),
    'p12l': J('multi_asset_p12_long.json'),
    'p12lw': J('multi_asset_p12_longw.json'),
    'p12d': J('multi_asset_p12_diag.json'),
    'p3m': J('multi_asset_p3_main.json'),
    'p3o': J('multi_asset_p3_oos.json'),
    'p3l': J('multi_asset_p3_long.json'),
    'p3br': J('p3_bond_recon.json'),
    'p7m': J('multi_asset_p7_main.json'),
    'p7o': J('multi_asset_p7_oos.json'),
    'p8m': J('multi_asset_p8_main.json'),
    'p8o': J('multi_asset_p8_oos.json'),
    'p9m': J('multi_asset_p9_main.json'),
    'p9o': J('multi_asset_p9_oos.json'),
    'p10g': J('multi_asset_p10_gate.json'),
    'p10l': J('multi_asset_p10_long.json'),
    'p10s': J('multi_asset_p10_stats.json'),
    'p11': J('multi_asset_p11_alpha.json'),
    'p11s': J('multi_asset_p11_stats.json'),
    'p17sm': J('multi_asset_p17_single_main.json'),
    'p17so': J('multi_asset_p17_single_oos.json'),
    'p17vm': J('multi_asset_p17_verify_main.json'),
    'p17vo': J('multi_asset_p17_verify_oos.json'),
    'p17fm': J('multi_asset_p17_freq_main.json'),
    'p17fo': J('multi_asset_p17_freq_oos.json'),
}


def Jopt(name):
    """产物可能还不存在（如实验尚未跑完）→ 返回空 dict，不让整个审计器崩。"""
    p = os.path.join(BACK, name)
    if not os.path.exists(p):
        return {}
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


ART['p18m'] = Jopt('multi_asset_p18_main.json')
ART['p18o'] = Jopt('multi_asset_p18_oos.json')
# ⚠️ 2026-11-01 修正：原门控是 `p18m and p18o`（**and**）→ oos 产物按预注册
# 故意不产出时，**整个 P18 断言段永远被跳过**，P18 文档的数字一条都没被审过。
# 正确语义：P18 的判决**只依赖 main 阶段**（判据 (a) 在 main 上就能定），
# oos 产物缺席时那些 oos 专属断言应「不存在」，而不是「整段不审」。
P18_MAIN_READY = bool(ART['p18m'])
P18DOC = 'P18_FUSE_GBM'
P18_OOS_READY = bool(ART['p18m']) and bool(ART['p18o'])
# 兼容旧名（外部脚本可能引用）
P18_READY = P18_MAIN_READY
# P18 的融合权重网格（来自产物 prereg，不硬编码）
W18_LIST = ART['p18m'].get('prereg', {}).get('w_gbms', [0.10, 0.20, 0.30, 0.50]) if P18_MAIN_READY else []


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
def _p9_rule_labels(art):
    """P9 规则臂标签（固定顺序，排除随机对照臂）。"""
    res = art.get('results', {})
    order = ['daily', 'wk_fri', 'wk_mon', 'monthly']
    labs = [l for l in order if l in res]
    labs += sorted(l for l in res
                   if l not in order and not l.startswith('rand_'))
    return labs


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
        A.append((P4DOC, 10, _lbl, 'bond 平均权重',
                  ('j', 'p4m', f'results.{_k}.avg_class_weight.bond')))
        A.append((P4DOC, 10, _lbl, '袖套波动', ('j', 'p4m', f'results.{_k}.sleeve_vol')))
        A.append((P4DOC, 10, _lbl, 'ρ(袖套,A股腿)',
                  ('j', 'p4m', f'results.{_k}.corr_sleeve_ashare')))
        A.append((P4DOC, 10, _lbl, '组合收益', ('j', 'p4m', f'results.{_k}.combined_return')))
        A.append((P4DOC, 10, _lbl, '组合夏普', ('j', 'p4m', f'results.{_k}.combined_sharpe')))

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

    # ── P9 · 主腿调仓频率曲线 ──
    #   表 0=main规则臂, 1=main随机对照, 2=oos规则臂, 3=oos随机对照,
    #   4=main子窗口, 5=oos子窗口, 6=main判据, 7=oos判据
    P9DOC = 'P9_REBALANCE_FREQ'
    _p9_cols = [
        ('执行日数', 'rebalance_days'), ('交易笔数', 'total_trades'),
        ('A腿夏普', 'a_share_sharpe'), ('组合夏普', 'combined_sharpe'),
        ('组合收益', 'combined_return'), ('组合回撤', 'combined_maxdd'),
        ('组合波动', 'combined_vol'),
    ]
    for _art, _ti in (('p9m', 0), ('p9o', 2)):
        for _l in _p9_rule_labels(ART[_art]):
            for _col, _fld in _p9_cols:
                A.append((P9DOC, _ti, _l, _col,
                          ('j', _art, f'results.{_l}.{_fld}')))
            A.append((P9DOC, _ti, _l, 'Δ组合', ('p9delta', _art, _l)))
    for _art, _ti in (('p9m', 1), ('p9o', 3)):
        for _l in sorted(ART[_art].get('random_controls', {}) or {}):
            A.append((P9DOC, _ti, _l, '执行日数',
                      ('j', _art, f'results.{_l}.rebalance_days')))
            A.append((P9DOC, _ti, _l, '组合夏普',
                      ('j', _art, f'results.{_l}.combined_sharpe')))
            A.append((P9DOC, _ti, _l, 'Δ规则臂', ('p9randdelta', _art, _l)))
    for _art, _ti in (('p9m', 4), ('p9o', 5)):
        for _sub in sorted(ART[_art]['subwindows']):
            for _l in _p9_rule_labels(ART[_art]):
                if safe_dig(_art, f'subwindows.{_sub}.{_l}') is not None:
                    A.append((P9DOC, _ti, _sub, _l,
                              ('j', _art, f'subwindows.{_sub}.{_l}')))
            A.append((P9DOC, _ti, _sub, '天数',
                      ('j', _art, f'subwindows.{_sub}.n')))
    for _art, _ti in (('p9m', 6), ('p9o', 7)):
        for _l in _p9_rule_labels(ART[_art]):
            for _col, _path in (('Δ组合', 'delta_vs_base_combined'),
                                ('ΔA腿', 'delta_vs_base_ashare'),
                                ('子窗口胜', 'subwindow_wins'),
                                ('随机组最大', 'random_group_max'),
                                ('Δ随机组最大', 'delta_vs_random_max')):
                if safe_dig(_art, f'criteria.{_l}.{_path}') is not None:
                    A.append((P9DOC, _ti, _l, _col,
                              ('j', _art, f'criteria.{_l}.{_path}')))
            if safe_dig(_art, f'criteria.{_l}.criteria') is not None:
                A.append((P9DOC, _ti, _l, '通过判据数', ('p9npass', _art, _l)))

    # ── P10 · A 股腿长历史重建 ──
    #   表定位用**列名关键字**（check() 支持字符串 tidx），
    #   避免表格顺序变化导致锚点错位。
    P10DOC = 'P10_LONG_ASHARE'
    _p10_order = ['w2013', 'w2016', 'w2019', 'oos', 'main']
    _p10_name = {
        'w2013': 'w2013（2013-05~2016-05）',
        'w2016': 'w2016（2016-05~2019-05）',
        'w2019': 'w2019（2019-05~2021-05）',
        'oos': 'oos（2021-05~2023-05）',
        'main': 'main（2023-05~2026-05）',
    }
    # 表 A：验证门（features_full vs features/day）
    for _f in ('total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate',
               'total_trades'):
        for _col, _path in (('期望（features/day）', f'expected.{_f}'),
                            ('实得（features/full）', f'got.{_f}'),
                            ('diff', f'diff.{_f}')):
            A.append((P10DOC, 'diff', _f, _col, ('j', 'p10g', _path)))
    # 表 B：长历史曲线
    for _l in _p10_order:
        if _l not in (ART['p10l'].get('results') or {}):
            continue
        A.append((P10DOC, 'A腿夏普', _p10_name[_l], '起点年',
                  ('p10yr', 'p10l', _l, 'start')))
        A.append((P10DOC, 'A腿夏普', _p10_name[_l], '终点年',
                  ('p10yr', 'p10l', _l, 'end')))
        for _col, _fld in (('A腿夏普', 'a_share_sharpe'),
                           ('A腿收益', 'total_return'),
                           ('A腿回撤', 'max_drawdown'),
                           ('交易笔数', 'total_trades')):
            A.append((P10DOC, 'A腿夏普', _p10_name[_l], _col,
                      ('j', 'p10l', f'results.{_l}.{_fld}')))
    # 表 C：跨窗口统计
    for _row, _fld in (('窗口数', 'n_windows'), ('均值', 'mean_sharpe'),
                       ('中位数', 'median_sharpe'), ('最小值', 'min_sharpe'),
                       ('最大值', 'max_sharpe'), ('极差', 'range_sharpe'),
                       ('标准差', 'stdev_sharpe')):
        A.append((P10DOC, '值', _row, '值', ('j', 'p10s', _fld)))

    # ── P12 · α 夏普口径重评 w 曲线 ──
    # 文档：P12_ALPHA_WEIGHT_2026-10-31.md
    # 表号：0 问题表 / 1 main 分解 / 2 oos 分解 / 3 长历史 / 4 main w 曲线
    #       5 oos w 曲线 / 6 5 窗口 / 7 配置敏感性 / 8 E1 / 9 E2 / 10 E5
    #       11 E3 / 12 E4 / 13 判据汇总 / 14 产物清单
    P12 = 'P12_ALPHA_WEIGHT'
    _MET = [('绝对夏普', 'abs_sharpe'),
            ('α 夏普（国证A指）', 'alpha_gz'),
            ('α 夏普（等权持有）', 'alpha_ew'),
            ('α 夏普（沪深300）', 'alpha_hs300')]
    _WIN = ['main', 'oos', 'w2013', 'w2016', 'w2019']

    # T1/T2 组合层分解（main / oos）
    # 行标签必须与文档原文逐字一致（normalize 只去空白/**/反引号，
    # 括号里的 n=5031 是文档的一部分，不能省）。
    for _t, _art, _rows in (
            (1, 'p12m', (('equal_weight_buyhold', '等权**买入持有**（n=5031）'),
                         ('equal_weight_rebalance', '等权日频再平衡（反例口径）'),
                         ('IDX_000300', '沪深300'),
                         ('IDX_399317', '国证A指'))),
            (2, 'p12o', (('equal_weight_buyhold', '等权买入持有（n=4184）'),
                         ('equal_weight_rebalance', '等权日频再平衡'),
                         ('IDX_000300', '沪深300'),
                         ('IDX_399317', '国证A指')))):
        for _bk, _bn in _rows:
            for _col, _fld in (('组合夏普', 'leg_sharpe'),
                               ('基准夏普', 'bench_sharpe'),
                               ('超额夏普', 'excess_sharpe'),
                               ('α 夏普', 'alpha_sharpe'),
                               ('β', 'beta'),
                               ('ρ', 'corr')):
                A.append((P12, _t, _bn, _col, ('p12decomp', _art, _bk, _fld)))

    # T3 长历史 3 窗口分解（行标签含窗口区间）
    for _w, _wl in (('w2013', 'w2013 (2013-05~2016-05)'),
                    ('w2016', 'w2016 (2016-05~2019-05)'),
                    ('w2019', 'w2019 (2019-05~2021-05)')):
        for _col, _fld in (('A 股腿夏普', 'a_share_sharpe'),
                           ('袖套夏普', 'sleeve_sharpe'),
                           ('组合夏普', 'combo_sharpe')):
            A.append((P12, 3, _wl, _col, ('p12long', 'p12l', _w, _fld)))
        for _col, _fld in (('α 夏普（等权持有）', 'alpha_sharpe'),
                           ('β', 'beta'), ('ρ', 'corr')):
            A.append((P12, 3, _wl, _col,
                      ('p12long', 'p12l', _w, f'decomp_ew_buyhold.{_fld}')))

    # T4/T5 w 曲线（main / oos）
    for _t, _art in ((4, 'p12m'), (5, 'p12o')):
        for _mn, _mk in _MET:
            for _col, _fld in (('最优 w', 'best_w'),
                               ('最优值', 'best_value'),
                               ('随机 w 最大', 'random_max')):
                A.append((P12, _t, _mn, _col, ('p12curve', _art, _mk, _fld)))
            # 子窗口众数格「0.15 (1/4)」：第 1 位=众数 w，第 2 位=计数，第 3 位=总数
            A.append((P12, _t, _mn, '子窗口众数',
                      ('p12curve', _art, _mk, 'subwindow_argmax_mode')))
            A.append((P12, _t, _mn, '子窗口众数',
                      ('p12curve', _art, _mk, 'subwindow_mode_count'), 1))
            A.append((P12, _t, _mn, '子窗口众数',
                      ('p12curve', _art, _mk, 'subwindow_n'), 2))

    # T6 5 窗口汇总
    for _mn, _mk in _MET:
        for _w in _WIN:
            A.append((P12, 6, _mn, _w, ('p12cw', _mk, ('per_window', _w, 'best_w'))))
        A.append((P12, 6, _mn, '落 [0.40,0.60]', ('p12cw', _mk, ('n_in_0.40_0.60',))))
        A.append((P12, 6, _mn, '众数', ('p12cw', _mk, ('mode_best_w',))))
        A.append((P12, 6, _mn, '众数', ('p12cw', _mk, ('mode_count',)), 1))
        A.append((P12, 6, _mn, '跨度', ('p12cw', _mk, ('spread',))))

    # T7 配置敏感性（定稿 vs 降级袖套）
    for _mn, _mk in _MET:
        for _col, _art in (('main（定稿）', 'p12m'), ('main（降级）', 'p12md'),
                           ('oos（定稿）', 'p12o'), ('oos（降级）', 'p12od')):
            A.append((P12, 7, _mn, _col, ('p12curve', _art, _mk, 'best_w')))

    # T8 E1 Spearman
    for _mn, _mk in _MET:
        A.append((P12, 8, _mn, 'ρ(最优w,A腿夏普)',
                  ('p12diag', f'E1_peak_vs_leg_strength.{_mk}.spearman_bestw_vs_aleg_sharpe')))
        A.append((P12, 8, _mn, 'ρ(最优w,A腿−袖套)',
                  ('p12diag', f'E1_peak_vs_leg_strength.{_mk}.spearman_bestw_vs_aleg_minus_sleeve')))

    # T9 E2 统一 w（组列有重复值 → 用两列行键）
    for _grp, _pre in (('A', 'A_定稿三类配置'), ('B', 'B_降级gold+bond')):
        for _mn, _mk in _MET:
            A.append((P12, 9, (_grp, _mn), '统一最优 w',
                      ('p12e2', f'{_pre}|{_mk}', 'uniform_best_w')))
            A.append((P12, 9, (_grp, _mn), '平均值',
                      ('p12e2', f'{_pre}|{_mk}', 'uniform_best_mean')))
            # 逐窗口 argmax：「0.20 / 0.25」第 0/1 位
            _wins = ['main', 'oos'] if _grp == 'A' else ['w2013', 'w2016', 'w2019']
            for _i, _w in enumerate(_wins):
                A.append((P12, 9, (_grp, _mn), '逐窗口 argmax',
                          ('p12e2', f'{_pre}|{_mk}', f'per_window_argmax.{_w}'), _i))

    # T10 E5 跨窗口迁移命中率「10/20 = 50.0%」：第 0/1/2 位
    for _mn, _mk in _MET:
        A.append((P12, 10, _mn, '迁移命中率',
                  ('p12diag', f'E5_cross_window_transfer.{_mk}.src_w_beats_w050')))
        A.append((P12, 10, _mn, '迁移命中率',
                  ('p12diag', f'E5_cross_window_transfer.{_mk}.n_pairs'), 1))
        A.append((P12, 10, _mn, '迁移命中率',
                  ('p12e5pct', _mk), 2))

    # T11 E3 定稿 w=0.50 排名「11/21」
    for _mn, _mk in _MET:
        for _w in _WIN:
            A.append((P12, 11, _mn, _w, ('p12e3rank', _mk, _w)))
            A.append((P12, 11, _mn, _w, ('p12e3grid', _mk, _w), 1))

    # T12 E4 相对随机 w 的优势
    for _mn, _mk in _MET:
        for _w in _WIN:
            A.append((P12, 12, _mn, _w, ('p12e4gap', _mk, _w)))

    # ── P11 · A 股腿 alpha/beta 分解 ──
    P11DOC = 'P11_ALPHA_BETA'
    _p11_order = ['w2013', 'w2016', 'w2019', 'oos', 'main']
    _p11_name = {
        'w2013': 'w2013（2013-05~2016-05）',
        'w2016': 'w2016（2016-05~2019-05）',
        'w2019': 'w2019（2019-05~2021-05）',
        'oos': 'oos（2021-05~2023-05）',
        'main': 'main（2023-05~2026-05）',
    }
    # 表 A（§3 主分解，表头含「超额夏普」）：5 窗口 × 10 列
    for _l in _p11_order:
        if _l not in (ART['p11'].get('results') or {}):
            continue
        for _col, _fld in (('腿夏普', 'leg_sharpe'), ('基准夏普', 'bench_sharpe'),
                           ('腿年化', 'leg_return_ann'), ('基准年化', 'bench_return_ann'),
                           ('基准回撤', 'bench_maxdd'), ('超额夏普', 'excess_sharpe'),
                           ('α夏普', 'alpha_sharpe'), ('β', 'beta'), ('ρ', 'corr'),
                           ('IR', 'ir')):
            A.append((P11DOC, '超额夏普', _p11_name[_l], _col,
                      ('j', 'p11', f'results.{_l}.decomposition.{_fld}')))
    # 表 B（§4 基准敏感性，表头含「国证A指」）
    for _l in _p11_order:
        if _l not in (ART['p11'].get('results') or {}):
            continue
        A.append((P11DOC, '国证A指', _p11_name[_l], '等权买入持有',
                  ('j', 'p11', f'results.{_l}.decomposition.excess_sharpe')))
        A.append((P11DOC, '国证A指', _p11_name[_l], '日频再平衡',
                  ('j', 'p11', f'results.{_l}.alternatives.rebalance_benchmark.excess_sharpe')))
        A.append((P11DOC, '国证A指', _p11_name[_l], '国证A指',
                  ('j', 'p11', f'results.{_l}.alternatives.market_proxies.IDX_399317.excess_sharpe')))
        A.append((P11DOC, '国证A指', _p11_name[_l], '沪深300',
                  ('j', 'p11', f'results.{_l}.alternatives.market_proxies.IDX_000300.excess_sharpe')))
    # 表 C（§5.2 幸存者偏差情景，表头含「窗口内退市」）
    for _l in _p11_order:
        if safe_dig('p11', f'results.{_l}.survivorship.ok') is not True:
            continue
        for _col, _fld in (('窗口内退市', 'n_delisted_in_window'),
                           ('起点成分', 'n_members_start'), ('占比', 'share'),
                           ('基准年化（观测）', 'bench_ann_observed'),
                           ('基准年化（修正）', 'bench_ann_adjusted'),
                           ('修正量', 'bench_ann_shift')):
            A.append((P11DOC, '窗口内退市', _p11_name[_l], _col,
                      ('j', 'p11', f'results.{_l}.survivorship.{_fld}')))
    # 表 D（§6 随机子池，表头含「随机均值」）
    for _l in _p11_order:
        if _l not in (ART['p11'].get('results') or {}):
            continue
        A.append((P11DOC, '全市场超额夏普', _p11_name[_l], '全市场超额夏普',
                  ('j', 'p11', f'results.{_l}.decomposition.excess_sharpe')))
        for _sd in ('s7', 's99', 's1234'):
            A.append((P11DOC, '全市场超额夏普', _p11_name[_l], f'随机500 {_sd}',
                      ('j', 'p11', f'results.{_l}.alternatives.random500.{_sd}.excess_sharpe')))
        A.append((P11DOC, '全市场超额夏普', _p11_name[_l], '随机均值',
                  ('p11randmean', 'p11', _l)))
    # 表 E（§3 跨窗口统计，表头含「统计量」）
    for _row, _fld in (('窗口数', 'n_windows'), ('均值', 'mean_excess_sharpe'),
                       ('中位数', 'median_excess_sharpe'), ('最小值', 'min_excess_sharpe'),
                       ('最大值', 'max_excess_sharpe'), ('极差', 'range_excess_sharpe'),
                       ('标准差', 'stdev_excess_sharpe')):
        A.append((P11DOC, '统计量', _row, '超额夏普(等权买入持有)', ('j', 'p11s', _fld)))
    for _row, _fld in (('窗口数', 'n_windows'), ('均值', 'mean_alpha_sharpe'),
                       ('中位数', 'median_alpha_sharpe'), ('最小值', 'min_alpha_sharpe'),
                       ('最大值', 'max_alpha_sharpe'), ('标准差', 'stdev_alpha_sharpe')):
        A.append((P11DOC, '统计量', _row, 'α夏普(等权买入持有)', ('j', 'p11s', _fld)))

    # ── P18：42f+gbm 分数级融合 ──
    # 表定位用**行标签**（('row', 行标签)）而非表序号：P18 文档未来插表不会让锚点失效
    # （P17 教训：插表 → 表序号位移 → 锚点必须手工同步改）。
    # 同一个行标签可能出现在多张表 → 定位时再用**列名**筛一次（见 check()）。
    if P18_MAIN_READY:
        _DIAGROW = "('2023-05-08 00:00:00', '2026-05-08 00:00:00')"
        # §二 诊断表：正交性（否决前置）
        A.append((P18DOC, ('row', _DIAGROW), _DIAGROW, '可算天数',
                  ('j', 'p18m', ['diagnostics', _DIAGROW, 'n_days'])))
        A.append((P18DOC, ('row', _DIAGROW), _DIAGROW, 'Pearson 中位',
                  ('j', 'p18m', ['diagnostics', _DIAGROW, 'pearson_median'])))
        A.append((P18DOC, ('row', _DIAGROW), _DIAGROW, 'Spearman 中位',
                  ('j', 'p18m', ['diagnostics', _DIAGROW, 'spearman_median'])))
        A.append((P18DOC, ('row', _DIAGROW), _DIAGROW, 'gbm 覆盖率中位',
                  ('j', 'p18m', ['diagnostics', _DIAGROW, 'gbm_coverage_median'])))
        # §三 复现门表：orig42f 与 fuse_w0.00 必须逐位一致
        for _arm, _lab in (('orig42f', '`orig42f`（原始 42f 分数）'),
                           ('fuse_w0.00', '`fuse_w0.00`（z 化后，w=0）')):
            A.append((P18DOC, ('row', _lab), _lab, '组合夏普',
                      ('j', 'p18m', ['results', _arm, 'combined_sharpe'])))
            A.append((P18DOC, ('row', _lab), _lab, 'A腿夏普',
                      ('j', 'p18m', ['results', _arm, 'a_share_sharpe'])))
            A.append((P18DOC, ('row', _lab), _lab, '交易笔数',
                      ('j', 'p18m', ['results', _arm, 'total_trades'])))
        # §四 规则臂结果表
        for _w in W18_LIST:
            _arm = f'fuse_w{_w:.2f}'
            _lab = f'`{_arm}`'
            A.append((P18DOC, ('row', _lab), _lab, '组合夏普',
                      ('j', 'p18m', ['results', _arm, 'combined_sharpe'])))
            A.append((P18DOC, ('row', _lab), _lab, 'A腿夏普',
                      ('j', 'p18m', ['results', _arm, 'a_share_sharpe'])))
            A.append((P18DOC, ('row', _lab), _lab, '组合波动',
                      ('j', 'p18m', ['results', _arm, 'combined_vol'])))
            A.append((P18DOC, ('row', _lab), _lab, '笔数',
                      ('j', 'p18m', ['results', _arm, 'total_trades'])))
            A.append((P18DOC, ('row', _lab), _lab, 'Δ组合',
                      ('p18deltabase', 'p18m', _w)))
            A.append((P18DOC, ('row', _lab), _lab, 'placebo 最高',
                      ('p18plcmax', 'p18m', _w)))
            A.append((P18DOC, ('row', _lab), _lab, '滚动胜率',
                      ('j', 'p18m', ['criteria', _arm, 'rolling', 'win_rate'])))
        # §五 placebo 对照表
        for _w in W18_LIST:
            _rowlab = f'{_w:.2f}'
            A.append((P18DOC, ('row', _rowlab), _rowlab, '规则臂组合夏普',
                      ('j', 'p18m', ['results', f'fuse_w{_w:.2f}', 'combined_sharpe'])))
            A.append((P18DOC, ('row', _rowlab), _rowlab, 'placebo 最低',
                      ('p18plcmin', 'p18m', _w)))
            A.append((P18DOC, ('row', _rowlab), _rowlab, 'placebo 最高',
                      ('p18plcmax', 'p18m', _w)))
            A.append((P18DOC, ('row', _rowlab), _rowlab, 'placebo 均值',
                      ('p18plcmean', 'p18m', _w)))
            A.append((P18DOC, ('row', _rowlab), _rowlab, '规则 − placebo 最高',
                      ('p18delta', 'p18m', _w)))
        # §六 机制分解表
        for _w in W18_LIST:
            _rowlab = f'{_w:.2f}'
            A.append((P18DOC, ('row', _rowlab), _rowlab, '真实夏普',
                      ('j', 'p18m', ['results', f'fuse_w{_w:.2f}', 'combined_sharpe'])))
            A.append((P18DOC, ('row', _rowlab), _rowlab, '真实波动',
                      ('j', 'p18m', ['results', f'fuse_w{_w:.2f}', 'combined_vol'])))
            A.append((P18DOC, ('row', _rowlab), _rowlab, 'placebo 波动',
                      ('p18plcvolmean', 'p18m', _w)))
            A.append((P18DOC, ('row', _rowlab), _rowlab, 'Δ波动',
                      ('p18delta_vol', 'p18m', _w)))
            A.append((P18DOC, ('row', _rowlab), _rowlab, 'Δ夏普',
                      ('p18delta_real_vs_plcmean', 'p18m', _w)))
        # §六 年度分解表（基线 + 4 个真实臂的逐年收益）
        for _yr in ('2023', '2024', '2025', '2026'):
            A.append((P18DOC, ('row', '**基线 `fuse_w0.00`**'), '**基线 `fuse_w0.00`**', _yr,
                      ('p18year', 'p18m', 'fuse_w0.00', _yr)))
            for _w in W18_LIST:
                _arm = f'fuse_w{_w:.2f}'
                _lab = f'真实 `{_arm}`'
                A.append((P18DOC, ('row', _lab), _lab, _yr,
                          ('p18year', 'p18m', _arm, _yr)))
    return A


def build_text_anchors():
    """文本格断言：(doc, table_idx, row_label, col_header, expect_spec)

    数字格断言覆盖不到「持仓集」这类**字符串**单元格
    （P4 §6.1 长历史持仓集披露），也覆盖不到 ✅/❌/— 判据符号。
    """
    A = []
    P4DOC = 'P4_CLASS_WEIGHT'
    for _cls in ('gold', 'bond', 'commodity'):
        A.append((P4DOC, 8, f'`{_cls}`', 'main 窗口', ('p4held', 'p4m', _cls)))
        A.append((P4DOC, 8, f'`{_cls}`', 'oos 窗口', ('p4held', 'p4o', _cls)))
        A.append((P4DOC, 8, f'`{_cls}`', '长历史窗口', ('p4held', 'p4l', _cls)))
    return A


# ── 判据符号断言（✅ / ❌ / —）──────────────────────────────────
# P12 §七 判据汇总表：每一格是符号而不是数字。P6 的同款表也有。
# spec = ('p12crit', window, 判据键)
P12_CRIT = [
    ('a0_curve_has_interior_max_not_edge', '(a0) 曲线有内部极大（非端点）'),
    ('a_best_w_outside_0.40_0.60', '(a) 最优 w 落在 [0.40,0.60] **之外**'),
    ('b_alpha_sharpe_ge_base_plus_0.05', '(b) 最优值 ≥ w=0.50 基线 + 0.05'),
    ('c_oos_alpha_sharpe_ge_base_minus_0.02', '(c) oos 不劣于基线 − 0.02'),
    ('d_subwindow_majority_same_w', '(d) 子窗口众数 ≥ 3/4'),
    ('e_beats_all_random_w', '(e) 优于所有随机 w'),
    ('f_independent_oos_evidence', '(f) 独立样本外证据'),
]


def build_crit_anchors():
    """判据符号断言：(doc, table_idx, row_label, col_header, expect_spec)

    产物：P12 的 `diagnostics.F_final_criteria.table[window][crit_key]`
          （True→✅ / False→❌ / None→—）。
    """
    A = []
    for _key, _lab in P12_CRIT:
        A.append(('P12_ALPHA_WEIGHT', 13, _lab, 'main', ('p12crit', 'main', _key)))
        A.append(('P12_ALPHA_WEIGHT', 13, _lab, 'oos', ('p12crit', 'oos', _key)))
    # P18 §七 判据汇总表：main/oos 两列都从产物回读
    # （P18 的 oos 按预注册不跑 → oos 列必须是「不跑（按预注册）」，
    #   不是空白、也不是 ❌ —— 空着会让人以为是漏填）
    if P18_MAIN_READY:
        for _lab, _key in (('(a) main 组合夏普', 'a_main_combined_ge_base_plus_0.05'),
                           ('(c) 胜 placebo', 'c_beats_placebo'),
                           ('(d) 滚动 504 日夏普胜率', 'd_rolling_win_rate_ge_0.60')):
            A.append((P18DOC, ('row', _lab), _lab, 'main 结果',
                      ('p18crit', 'main', _key)))
            A.append((P18DOC, ('row', _lab), _lab, 'oos 结果',
                      ('p18crit_oos_na', _key)))
        # (e) DSR 是**数值**报告项（不是符号）→ 走数值锚点，见 build_anchors
    return A


def resolve_text(spec):
    if spec[0] == 'p4held':
        art, cls = spec[1], spec[2]
        return '/'.join(sorted(dig(ART[art], f'results.base_equal.held.{cls}')))
    raise KeyError(spec)


def resolve_crit(spec):
    """判据符号：产物 True→✅ / False→❌ / None→—"""
    if spec[0] == 'p18crit':
        # P18：候选臂（main 上组合夏普最高的规则臂）的判据布尔
        # 文档格格式是 '✅ True' / '❌ False' / '—'（与 P12 的纯符号不同）
        cand = dig(ART['p18m'], ['verdict', 'candidate_main_best_rule_arm'])
        v = dig(ART['p18m'], ['criteria', cand, spec[2]])
        return '✅ True' if v is True else ('—' if v is None else '❌ False')
    if spec[0] == 'p18crit_oos_na':
        # P18：oos 未跑时该列必须是明确的占位符，不能是空白
        if not P18_OOS_READY:
            return '不跑（按预注册）'
        cand = dig(ART['p18m'], ['verdict', 'candidate_main_best_rule_arm'])
        v = dig(ART['p18o'], ['criteria', cand, spec[1]])
        return '✅ True' if v is True else ('—' if v is None else '❌ False')
    v = dig(ART['p12d'], ['diagnostics', 'F_final_criteria', 'table', spec[1], spec[2]])
    return '✅' if v is True else ('—' if v is None else '❌')


def check_crit_cells(texts, anchors):
    """判据符号核对：单元格必须**恰好**等于产物给出的符号。

    用「包含」语义会误判（'❌' 与 '❌❌'、'✅' 被 '—' 包含等），
    所以这里要求归一化后**完全相等**。
    """
    problems = []
    for a in anchors:
        doc_pref, tidx, row_label, col_hdr, spec = a
        doc = next((k for k in texts if k.startswith(doc_pref)), None)
        if not doc:
            problems.append(f'{doc_pref}: 文档不存在')
            continue
        tables = extract_tables(texts[doc])
        if isinstance(tidx, tuple) and tidx and tidx[0] == 'row':
            cands = [i for i, (h, rows) in enumerate(tables)
                     if any(r and norm(r[0]) == norm(tidx[1]) for r in rows)]
            cands2 = [i for i in cands
                      if any(norm(c) == norm(col_hdr) for c in tables[i][0])]
            if cands2:
                cands = cands2
            if not cands:
                problems.append(f'{doc}: 找不到含行「{tidx[1]}」的表')
                continue
            tidx = cands[0]
        if tidx >= len(tables):
            problems.append(f'{doc}: 表序号 {tidx} 越界（共 {len(tables)} 张表）')
            continue
        header, data = tables[tidx]
        cell, err = find_cell(header, data, row_label, col_hdr)
        if err or cell is None:
            problems.append(f'{doc}[表{tidx}] {err or "单元格为空"}')
            continue
        want = resolve_crit(spec)
        got = norm(cell)
        want = norm(want)
        if got != want:
            problems.append(
                f'{doc}[表{tidx}] 「{row_label}」×「{col_hdr}」= {got!r}，'
                f'产物 = {want!r}')
    return problems


def check_text_cells(texts, anchors):
    """字符串单元格核对：单元格必须**包含**产物给出的代码串。"""
    problems = []
    for a in anchors:
        doc_pref, tidx, row_label, col_hdr, spec = a
        doc = next((k for k in texts if k.startswith(doc_pref)), None)
        if not doc:
            problems.append(f'{doc_pref}: 文档不存在')
            continue
        tables = extract_tables(texts[doc])
        if tidx >= len(tables):
            problems.append(f'{doc}: 表序号 {tidx} 越界（共 {len(tables)} 张表）')
            continue
        header, data = tables[tidx]
        cell, err = find_cell(header, data, row_label, col_hdr)
        if err or cell is None:
            problems.append(f'{doc}[表{tidx}] {err or "单元格为空"}')
            continue
        want = resolve_text(spec)
        norm_cell = cell.replace('`', '').replace(' ', '').replace('**', '')
        norm_want = want.replace('`', '').replace(' ', '')
        if norm_want not in norm_cell:
            problems.append(
                f'{doc}[表{tidx}] 「{row_label}」×「{col_hdr}」= {cell!r}，'
                f'产物 = {want!r}')
    return problems


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


def target_number(cell, want, idx=0):
    """定位单元格中对应 `want` 的数字。

    规则（审计器与回填器必须完全一致，否则「审计通过」不代表「数字对」）：
      - 单元格**只有一个数字且含 `%`** → 这是百分比格，目标值 = want × 100；
      - 否则（纯数值格 / 复合格）→ 目标值 = want，取第 `idx` 个数字（默认第一个）。
    返回 (kind, num, start, end) 或 None。
    """
    nums = cell_num_positions(cell)
    if not nums:
        return None
    if idx >= len(nums):
        return None
    n, s, e = nums[idx]
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
    # 行键可以是字符串（匹配第 0 列），也可以是元组（依次匹配前 N 列）——
    # 有些表（如 P12 §六 E2）第一列有重复值，单列定位会歧义。
    is_key = isinstance(row_label, (tuple, list))
    if is_key:
        want_cols = [norm(x) for x in row_label]
        n_key = len(want_cols)
    else:
        want_cols = [norm(row_label)]
        n_key = 1
    hit = None
    for r in data_cells:
        if not r or len(r) < n_key:
            continue
        if is_key:
            # 元组行键：首元素按**前缀**匹配 —— 文档里分组列的首行常写全称
            # （如 'A（定稿三类，main+oos）'），后续行只写 'A'。
            if not norm(r[0]).startswith(want_cols[0]):
                continue
            if all(norm(r[i]) == want_cols[i] for i in range(1, n_key)):
                if hit is not None:
                    return None, f'行键「{row_label}」出现多次'
                hit = r
        elif norm(r[0]) == want_cols[0]:
            # 普通行键：精确匹配（前缀匹配会误伤 'top2' vs 'top20'）
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
        vals = [dig(ART[art], ['results', k, 'combined_sharpe'])
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
    if spec[0] == 'p9delta':
        # P9：某臂组合夏普相对基线臂（base_label）的差
        art, lab = spec[1], spec[2]
        base = ART[art]['base_label']
        return (dig(ART[art], f'results.{lab}.combined_sharpe')
                - dig(ART[art], f'results.{base}.combined_sharpe'))
    if spec[0] == 'p9randdelta':
        # P9：随机臂组合夏普相对它对标的规则臂的差
        art, lab = spec[1], spec[2]
        rule = ART[art]['random_controls'][lab]['matched_rule']
        return (dig(ART[art], f'results.{lab}.combined_sharpe')
                - dig(ART[art], f'results.{rule}.combined_sharpe'))
    if spec[0] == 'p9npass':
        # P9：通过的预注册判据数（(f) 结构性缺失恒 False，不计入）
        art, lab = spec[1], spec[2]
        k = dig(ART[art], f'criteria.{lab}.criteria')
        keys = ('a_ashare_sharpe_gt_base',
                'b_main_combined_ge_base_plus_0.05',
                'c_oos_combined_ge_base_minus_0.02',
                'd_subwindow_majority_ge_3',
                'e_beats_same_freq_random_max')
        return sum(1 for key in keys if k.get(key) is True)
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
    if spec[0] == 'p10yr':
        # P10：窗口起止年（JSON 里是完整日期字符串）
        art, lab, which = spec[1], spec[2], spec[3]
        w = dig(ART[art], f'results.{lab}.window')
        return int(w[0 if which == 'start' else 1][:4])
    if spec[0] == 'p11randmean':
        # P11：某窗口 3 个随机 500 子池基准的超额夏普均值
        art, lab = spec[1], spec[2]
        rb = dig(ART[art], f'results.{lab}.alternatives.random500')
        vals = [dig(rb, k)['excess_sharpe'] for k in sorted(rb)]
        return sum(vals) / len(vals)
    if spec[0] == 'jmin':
        # 两个产物值取小者
        return min(dig(ART[spec[1]], spec[2]), dig(ART[spec[3]], spec[4]))
    if spec[0] == 'p12curve':
        # P12 w 曲线：art, metric, field（best_w/best_value/random_max/...）
        return dig(ART[spec[1]], f'curves.{spec[2]}.{spec[3]}')
    if spec[0] == 'p12decomp':
        # P12 组合层分解：art, 基准键, 字段（leg_sharpe/bench_sharpe/...）
        return dig(ART[spec[1]],
                   ['combo_decomposition_w050', spec[2], 'decomp', spec[3]])
    if spec[0] == 'p12long':
        # P12 长历史：art, window, path（'.' 分隔）
        return dig(ART[spec[1]], ['results', spec[2]] + spec[3].split('.'))
    if spec[0] == 'p12cw':
        # P12 5 窗口汇总：metric, 键名（键名可能含小数点 → 逐段显式传）
        return dig(ART['p12lw'], ['cross_window', spec[1]] + list(spec[2]))
    if spec[0] == 'p12e2':
        # P12 E2 统一 w：组前缀|口径 → uniform_best_w / uniform_best_mean
        return dig(ART['p12d'],
                   ['diagnostics', 'E2_uniform_w_across_windows', spec[1]]
                   + spec[2].split('.'))
    if spec[0] == 'p12diag':
        # P12 诊断：diagnostics 下的任意路径（'.' 分隔）
        return dig(ART['p12d'], ['diagnostics'] + spec[1].split('.'))
    if spec[0] == 'p12e5pct':
        # E5 迁移命中率（百分数形式）
        return dig(ART['p12d'],
                   f'diagnostics.E5_cross_window_transfer.{spec[1]}.hit_rate') * 100
    if spec[0] == 'p12e3rank':
        # E3：某口径某窗口下 w=0.50 的排名（1 起）
        rows = dig(ART['p12d'], f'diagnostics.E3_w050_risk_disclosure.{spec[1]}')
        return next(r['rank_of_w050'] for r in rows if r['window'] == spec[2])
    if spec[0] == 'p12e3grid':
        # E3：网格点数
        rows = dig(ART['p12d'], f'diagnostics.E3_w050_risk_disclosure.{spec[1]}')
        return next(r['n_grid'] for r in rows if r['window'] == spec[2])
    if spec[0] == 'p12e4gap':
        # E4：某口径某窗口下 argmax 相对随机 w 的优势
        rows = dig(ART['p12d'], f'diagnostics.E4_argmax_vs_random.{spec[1]}')
        return next(r['gap'] for r in rows if r['window'] == spec[2])
    if spec[0] == 'p12bestgap':
        # 最优点与 w=0.50 基线的差（该窗口主口径）
        art = 'p12m' if spec[1] == 'main' else 'p12o'
        return (dig(ART[art], 'curves.alpha_gz.best_value')
                - dig(ART[art], 'results.w050.alpha_sharpe_IDX_399317'))
    if spec[0] == 'p12e5range':
        # E5 迁移命中率区间：4 个口径的最小/最大（文档写 25%~50%）
        vals = []
        for _m in ('abs_sharpe', 'alpha_gz', 'alpha_ew', 'alpha_hs300'):
            vals.append(dig(ART['p12d'],
                            ['diagnostics', 'E5_cross_window_transfer', _m, 'hit_rate']) * 100)
        return min(vals)
    if spec[0] == 'p12e6same':
        # E6 同组迁移命中率
        return dig(ART['p12d'],
                   ['diagnostics', 'E6_within_group_transfer', spec[1],
                    'same_group_hit_rate']) * 100
    if spec[0] == 'p12longsum':
        # 长历史 α 夏普汇总统计
        return dig(ART['p12l'], ['summary', 'decomp_ew_buyhold', spec[1]])
    if spec[0] == 'p12lossrange':
        # E3 某窗口跨 4 口径的 w=0.50 损失区间（min/max）
        vals = []
        for _m in ('abs_sharpe', 'alpha_gz', 'alpha_ew', 'alpha_hs300'):
            rows = dig(ART['p12d'],
                       ['diagnostics', 'E3_w050_risk_disclosure', _m])
            vals.append(next(r['loss_vs_best'] for r in rows
                             if r['window'] == spec[1]))
        return min(vals) if spec[2] == 'min' else max(vals)
    if spec[0] == 'p12spreadrange':
        # 5 窗口跨 4 口径的最优 w 跨度区间（min/max）
        vals = [dig(ART['p12lw'], ['cross_window', _m, 'spread'])
                for _m in ('abs_sharpe', 'alpha_gz', 'alpha_ew', 'alpha_hs300')]
        return min(vals)
    if spec[0] == 'p12spreadmax':
        vals = [dig(ART['p12lw'], ['cross_window', _m, 'spread'])
                for _m in ('abs_sharpe', 'alpha_gz', 'alpha_ew', 'alpha_hs300')]
        return max(vals)
    if spec[0] == 'p12spread':
        # 某口径 5 窗口最优 w 的最小值
        cw = dig(ART['p12lw'], ['cross_window', spec[1]])
        return min(v['best_w'] for v in cw['per_window'].values())
    if spec[0] == 'p12betarange':
        # 组合 β 区间：main/oos × (国证A指/等权持有)
        vals = []
        for _art in ('p12m', 'p12o'):
            for _b in ('IDX_399317', 'equal_weight_buyhold'):
                vals.append(dig(ART[_art],
                                ['combo_decomposition_w050', _b, 'decomp', 'beta']))
        return min(vals)
    if spec[0] == 'p12corrhange':
        # 组合 ρ 区间（国证A指/等权持有，main/oos）
        vals = []
        for _art in ('p12m', 'p12o'):
            for _b in ('IDX_399317', 'equal_weight_buyhold'):
                vals.append(dig(ART[_art],
                                ['combo_decomposition_w050', _b, 'decomp', 'corr']))
        return min(vals)
    if spec[0] == 'p12ann':
        # 年化收益（% 展示）
        return dig(ART[spec[1]], ['combo_decomposition_w050',
                                  'equal_weight_buyhold', 'decomp', spec[2]]) * 100
    if spec[0] == 'p12configshift':
        # 配置敏感性：最优 w 的移动幅度（定稿 vs 降级）
        vals = []
        for _m in ('abs_sharpe', 'alpha_gz', 'alpha_ew', 'alpha_hs300'):
            for _a1, _a2 in (('p12m', 'p12md'), ('p12o', 'p12od')):
                vals.append(abs(dig(ART[_a1], ['curves', _m, 'best_w'])
                                - dig(ART[_a2], ['curves', _m, 'best_w'])))
        return min(vals)
    if spec[0] == 'p12alpharange':
        # 定稿三类下 α 夏普（国证A指）main/oos 的值
        art = 'p12m' if spec[1] == 'main' else 'p12o'
        return dig(ART[art], ['results', 'w050', 'alpha_sharpe_IDX_399317'])
    if spec[0] == 'p18deltabase':
        # P18：某臂组合夏普 − 基线（fuse_w0.00）组合夏普
        art, w = spec[1], spec[2]
        return dig(ART[art], ['criteria', f'fuse_w{w:.2f}', 'delta_combined'])
    if spec[0] == 'p18plcmax':
        # P18：某 w 下 placebo 臂组合夏普的最大值
        art, w = spec[1], spec[2]
        vals = [dig(ART[art], ['results', k, 'combined_sharpe'])
                for k in ART[art]['results'] if k.startswith(f'plc_w{w:.2f}_s')]
        return max(vals)
    if spec[0] == 'p18plcmin':
        # P18：某 w 下 placebo 臂组合夏普的最小值
        art, w = spec[1], spec[2]
        vals = [dig(ART[art], ['results', k, 'combined_sharpe'])
                for k in ART[art]['results'] if k.startswith(f'plc_w{w:.2f}_s')]
        return min(vals)
    if spec[0] == 'p18plcmean':
        # P18：某 w 下 placebo 臂组合夏普的均值
        art, w = spec[1], spec[2]
        vals = [dig(ART[art], ['results', k, 'combined_sharpe'])
                for k in ART[art]['results'] if k.startswith(f'plc_w{w:.2f}_s')]
        return sum(vals) / len(vals)
    if spec[0] == 'p18plcvolmean':
        # P18：某 w 下 placebo 臂组合波动的均值
        art, w = spec[1], spec[2]
        vals = [dig(ART[art], ['results', k, 'combined_vol'])
                for k in ART[art]['results'] if k.startswith(f'plc_w{w:.2f}_s')]
        return sum(vals) / len(vals)
    if spec[0] == 'p18plcretmean':
        # P18：某 w 下 placebo 臂组合收益的均值
        art, w = spec[1], spec[2]
        vals = [dig(ART[art], ['results', k, 'combined_return'])
                for k in ART[art]['results'] if k.startswith(f'plc_w{w:.2f}_s')]
        return sum(vals) / len(vals)
    if spec[0] == 'p18delta':
        # P18：某 w 下「规则臂夏普 − 同参数 placebo 最高夏普」
        art, w = spec[1], spec[2]
        rule = dig(ART[art], ['results', f'fuse_w{w:.2f}', 'combined_sharpe'])
        vals = [dig(ART[art], ['results', k, 'combined_sharpe'])
                for k in ART[art]['results'] if k.startswith(f'plc_w{w:.2f}_s')]
        return rule - max(vals)
    if spec[0] == 'p18delta_real_vs_plcmean':
        # P18：真实臂夏普 − 同参数 placebo 平均夏普
        art, w = spec[1], spec[2]
        rule = dig(ART[art], ['results', f'fuse_w{w:.2f}', 'combined_sharpe'])
        vals = [dig(ART[art], ['results', k, 'combined_sharpe'])
                for k in ART[art]['results'] if k.startswith(f'plc_w{w:.2f}_s')]
        return rule - sum(vals) / len(vals)
    if spec[0] == 'p18delta_vol':
        # P18：真实臂波动 − 同参数 placebo 平均波动
        art, w = spec[1], spec[2]
        rule = dig(ART[art], ['results', f'fuse_w{w:.2f}', 'combined_vol'])
        vals = [dig(ART[art], ['results', k, 'combined_vol'])
                for k in ART[art]['results'] if k.startswith(f'plc_w{w:.2f}_s')]
        return rule - sum(vals) / len(vals)
    if spec[0] == 'p18delta_ret':
        # P18：真实臂收益 − 同参数 placebo 平均收益（文档按 pp 展示）
        art, w = spec[1], spec[2]
        rule = dig(ART[art], ['results', f'fuse_w{w:.2f}', 'combined_return'])
        vals = [dig(ART[art], ['results', k, 'combined_return'])
                for k in ART[art]['results'] if k.startswith(f'plc_w{w:.2f}_s')]
        return (rule - sum(vals) / len(vals)) * 100
    if spec[0] == 'p18year':
        # P18：某臂某年组合收益（文档按百分比展示）
        art, lab, yr = spec[1], spec[2], spec[3]
        return dig(ART[art], ['results', lab, 'combined_yearly', yr])
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
        # 用局部变量 ti 承载解析后的表序号（避免与锚点里的原始形态混用类型）
        ti = a[1]
        if isinstance(ti, tuple) and ti and ti[0] == 'row':
            # 按**行标签**定位表：插表导致表序号位移时仍然稳（P17 教训：
            # 「往文档插新表会让后面表序号位移」→ 表序号锚点必须同步改，
            #  行标签锚点不用）。ti = ('row', 行标签, 列名)
            cands = [i for i, (h, rows) in enumerate(tables)
                     if any(r and norm(r[0]) == norm(ti[1]) for r in rows)]
            # 同一个行标签可能出现在多张表（如 P18 的 '0.10' 同时在 placebo 表和机制表）：
            # 用**列名**再筛一次，选出真正含该列的表。
            cands2 = [i for i in cands
                      if any(norm(c) == norm(col_hdr) for c in tables[i][0])]
            if cands2:
                cands = cands2
            if not cands:
                problems.append(f'{doc}: 找不到含行「{ti[1]}」的表')
                continue
            ti = cands[0]
        elif isinstance(ti, str):
            # 按表头关键字定位表
            cands = [i for i, (h, _) in enumerate(tables)
                     if any(norm(col_hdr) == norm(c) for c in h)]
            if not cands:
                problems.append(f'{doc}: 找不到含列「{col_hdr}」的表')
                continue
            ti = cands[0]
        else:
            ti = int(ti)
        if ti >= len(tables):
            problems.append(f'{doc}: 表序号 {ti} 越界（共 {len(tables)} 张表）')
            continue
        header, data = tables[ti]
        cell, err = find_cell(header, data, row_label, col_hdr)
        if err:
            problems.append(f'{doc}[表{ti}] {err}')
            continue
        nums = cell_num_positions(cell)
        if not nums:
            problems.append(f'{doc}[表{ti}] 「{row_label}」×「{col_hdr}」不是数字：{cell!r}')
            continue
        want = resolve(spec)
        # 锚点可带第 6 位：复合格里要核对**第 N 个**数字（默认第 0 个）
        idx = a[5] if len(a) > 5 else 0
        hit = target_number(cell, want, idx)
        if hit is None:
            got = nums[0][0]
            kind = '百分比格' if (len(nums) == 1 and '%' in cell) else '数值/复合格'
            problems.append(
                f'{doc}[表{ti}] 「{row_label}」×「{col_hdr}」= {got}（{kind}），'
                f'产物 = {want}（×100={want*100:.4f}）')
    return problems


# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# 正文数字断言：表格断言只管表格里的数字；正文散落的
# 区间 / 比例 / 结论里的具体值同样要机器核对。
#
# 机制：正则里用 (?P<gN>...) 捕获**待核对的数字**，捕获文本自身
# 决定比对精度（文档写 0.21 → 拿产物值 round 到 2 位再比）。
# 这样既不会因为文档做了合理舍入而误报，也不会漏掉「值写错了」
# 这类真错误（旧实现取「匹配文本里第一个数字」，会把
# 'main 0.20' 里的 20 当成被核对的值）。
# ─────────────────────────────────────────────────────────────

_BENCH3 = ('equal_weight_buyhold', 'IDX_000300', 'IDX_399317')
_METRICS4 = ('abs_sharpe', 'alpha_gz', 'alpha_ew', 'alpha_hs300')
_WINDOWS5 = ('main', 'oos', 'w2013', 'w2016', 'w2019')


def _decomp(art, bench, field):
    return ART[art]['combo_decomposition_w050'][bench]['decomp'][field]


def _beta_corr(field):
    """组合层 β / ρ 的区间：3 个基准（剔除「等权日频再平衡」反例口径）
    × 2 个窗口（main/oos）。"""
    vals = []
    for _art in ('p12m', 'p12o'):
        for _b in _BENCH3:
            vals.append(_decomp(_art, _b, field))
    return vals


def _e3_rows(metric):
    return dig(ART['p12d'], ['diagnostics', 'E3_w050_risk_disclosure', metric])


def _e4_rows(metric):
    return dig(ART['p12d'], ['diagnostics', 'E4_argmax_vs_random', metric])


def _cw(metric):
    return dig(ART['p12lw'], ['cross_window', metric])


def _curve(art, metric, field):
    return dig(ART[art], ['curves', metric, field])


def _w050(art, field):
    return dig(ART[art], ['results', 'w050', field])


def _config_shifts():
    """配置敏感性：定稿三类 → 降级两类，最优 w 的移动幅度。"""
    out = []
    for _m in _METRICS4:
        for _a1, _a2 in (('p12m', 'p12md'), ('p12o', 'p12od')):
            out.append(abs(_curve(_a1, _m, 'best_w') - _curve(_a2, _m, 'best_w')))
    return out


def _e2b(metric):
    return dig(ART['p12d'], ['diagnostics', 'E2_uniform_w_across_windows',
                             f'B_降级gold+bond|{metric}'])


def _e5_hits():
    return [dig(ART['p12d'], ['diagnostics', 'E5_cross_window_transfer', m,
                              'hit_rate']) for m in _METRICS4]


def _ew_alpha_vals():
    """定稿三类下、相对「等权买入持有」的 α 夏普（main + oos）。"""
    return [_w050('p12m', 'alpha_sharpe_equal_weight_buyhold'),
            _w050('p12o', 'alpha_sharpe_equal_weight_buyhold')]


def prose_value(spec):
    """把断言 spec 解析成期望值（float）。"""
    kind = spec[0]
    if kind == 'j':
        return dig(ART[spec[1]], spec[2])
    if kind == 'gate':
        return dig(ART[spec[1]], ['baseline_gate', 'w050_abs_sharpe'])
    if kind == 'decomp':
        return _decomp(spec[1], spec[2], spec[3])
    if kind == 'w050':
        return _w050(spec[1], spec[2])
    if kind == 'curve':
        return _curve(spec[1], spec[2], spec[3])
    if kind == 'randmax':
        return _curve(spec[1], spec[2], 'random_max')
    if kind == 'bestgap':
        # 最优点 − w=0.50 基线（主口径 α 夏普 / 国证A指）
        return (_curve(spec[1], 'alpha_gz', 'best_value')
                - _w050(spec[1], 'alpha_sharpe_IDX_399317'))
    if kind == 'beta_min':
        return min(_beta_corr('beta'))
    if kind == 'beta_max':
        return max(_beta_corr('beta'))
    if kind == 'corr_min':
        return min(_beta_corr('corr'))
    if kind == 'corr_max':
        return max(_beta_corr('corr'))
    if kind == 'long_ash':
        return dig(ART['p12l'], ['results', spec[1], 'a_share_sharpe'])
    if kind == 'long_sum':
        return dig(ART['p12l'], ['summary', 'decomp_ew_buyhold', spec[1]])
    if kind == 'cw':
        # 某口径某窗口的最优 w
        return dig(ART['p12lw'], ['cross_window', spec[1], 'per_window',
                                 spec[2], 'best_w'])
    if kind == 'cw_spread':
        return _cw(spec[1])['spread']
    if kind == 'cw_bestw_min':
        return min(v['best_w'] for v in _cw(spec[1])['per_window'].values())
    if kind == 'cw_bestw_max':
        return max(v['best_w'] for v in _cw(spec[1])['per_window'].values())
    if kind == 'e4gap':
        return next(r['gap'] for r in _e4_rows(spec[1])
                    if r['window'] == spec[2])
    if kind == 'e5_hit':
        return dig(ART['p12d'], ['diagnostics', 'E5_cross_window_transfer',
                                 spec[1], 'hit_rate'])
    if kind == 'e5_min':
        return min(_e5_hits())
    if kind == 'e5_max':
        return max(_e5_hits())
    if kind == 'e6_same':
        return dig(ART['p12d'], ['diagnostics', 'E6_within_group_transfer',
                                 spec[1], 'same_group_hit_rate'])
    if kind == 'e3_rank':
        return next(r['rank_of_w050'] for r in _e3_rows(spec[1])
                    if r['window'] == spec[2])
    if kind == 'e3_loss_min':
        # 某窗口跨 4 口径的 w=0.50 损失区间下界
        vals = [r['loss_vs_best'] for r in _e3_rows(spec[1])
                if r['window'] == spec[2]]
        return min(vals)
    if kind == 'e3_loss_max':
        vals = [r['loss_vs_best'] for r in _e3_rows(spec[1])
                if r['window'] == spec[2]]
        return max(vals)
    if kind == 'e3_loss_all_min':
        vals = []
        for _m in _METRICS4:
            vals += [r['loss_vs_best'] for r in _e3_rows(_m)
                     if r['window'] == spec[1]]
        return min(vals)
    if kind == 'e3_loss_all_max':
        vals = []
        for _m in _METRICS4:
            vals += [r['loss_vs_best'] for r in _e3_rows(_m)
                     if r['window'] == spec[1]]
        return max(vals)
    if kind == 'cfg_min':
        vals = [v for v in _config_shifts() if v > 0]
        return min(vals)
    if kind == 'cfg_max':
        return max(_config_shifts())
    if kind == 'e2b_uniform':
        return _e2b(spec[1])['uniform_best_w']
    if kind == 'e2b_span':
        vals = [_e2b(m)['uniform_best_w'] for m in _METRICS4]
        return max(vals) - min(vals)
    if kind == 'ew_alpha_min':
        return min(_ew_alpha_vals())
    if kind == 'ew_alpha_max':
        return max(_ew_alpha_vals())
    if kind == 'p3_final':
        # P3 定稿（剔除可转债后）某窗口的组合夏普
        d = dig(ART[f'p3{"m" if spec[1] == "main" else "o"}'],
                ['results', 'bond_lowrho3_noliq'
                 if 'bond_lowrho3_noliq' in dig(
                     ART[f'p3{"m" if spec[1] == "main" else "o"}'], ['results'])
                 else 'bond_lowrho3'])
        return d['combined_sharpe']
    if kind == 'p3_base':
        # P3 base（含可转债污染的旧现状）某窗口的组合夏普
        d = dig(ART[f'p3{"m" if spec[1] == "main" else "o"}'],
                ['results', 'base'])
        return d['combined_sharpe']
    if kind == 'p3_final_dd':
        d = dig(ART[f'p3{"m" if spec[1] == "main" else "o"}'],
                ['results', 'bond_lowrho3_noliq'
                 if 'bond_lowrho3_noliq' in dig(
                     ART[f'p3{"m" if spec[1] == "main" else "o"}'], ['results'])
                 else 'bond_lowrho3'])
        return abs(d['combined_maxdd']) * 100
    if kind == 'p3_base_dd':
        d = dig(ART[f'p3{"m" if spec[1] == "main" else "o"}'],
                ['results', 'base'])
        return abs(d['combined_maxdd']) * 100
    # ── P18：42f 之上叠加 gbm 增量（分数级融合）──
    if kind.startswith('p18_'):
        art = ART['p18m' if spec[1] == 'main' else 'p18o']
        cand = art.get('verdict', {}).get('candidate_main_best_rule_arm')
        if kind == 'p18_base':
            return dig(art, ['results', 'fuse_w0.00', 'combined_sharpe'])
        if kind == 'p18_cand':
            return dig(art, ['results', cand, 'combined_sharpe'])
        if kind == 'p18_delta':
            return dig(art, ['criteria', cand, 'delta_combined'])
        if kind == 'p18_corr':
            return dig(art, ['corr_median'])
        if kind == 'p18_dsr':
            return dig(art, ['verdict', 'dsr_report'])
        if kind == 'p18_dsr_thr':
            return dig(art, ['dsr95_threshold_sharpe'])
        if kind == 'p18_arm':
            return dig(art, ['results', spec[2], 'combined_sharpe'])
        if kind == 'p18_placebo_max':
            return dig(art, ['criteria', cand, 'placebo_max'])
    if kind == 'p17_oracle':
        # P17 三类同时取优的 oracle 组合夏普
        art = ART[f'p17v{"m" if spec[1] == "main" else "o"}']
        return next(r['combined_sharpe'] for r in art['results']
                    if r['label'] == 'all_three_oracles')
    if kind == 'p17_base':
        return ART[f'p17s{"m" if spec[1] == "main" else "o"}']['baseline']['combined_sharpe']
    if kind == 'p17_gain':
        art = ART[f'p17v{"m" if spec[1] == "main" else "o"}']
        base = ART[f'p17s{"m" if spec[1] == "main" else "o"}']['baseline']['combined_sharpe']
        orc = next(r['combined_sharpe'] for r in art['results']
                   if r['label'] == 'all_three_oracles')
        return orc - base
    if kind == 'p17_rmax':
        # 随机族最大 − 定稿基线
        art = ART[f'p17v{"m" if spec[1] == "main" else "o"}']
        base = ART[f'p17s{"m" if spec[1] == "main" else "o"}']['baseline']['combined_sharpe']
        rnd = [r['combined_sharpe'] for r in art['results']
               if r['label'].startswith('rand_')]
        return max(rnd) - base
    if kind == 'p17_freq':
        art = ART[f'p17f{"m" if spec[1] == "main" else "o"}']
        return next(r['combined_sharpe'] for r in art['results']
                    if r['label'] == spec[2])
    if kind == 'comb_min':
        return min(_w050('p12m', 'combined_sharpe'),
                   _w050('p12o', 'combined_sharpe'))
    if kind == 'comb_max':
        return max(_w050('p12m', 'combined_sharpe'),
                   _w050('p12o', 'combined_sharpe'))
    raise ValueError(f'未知 prose spec: {spec!r}')


def build_prose_anchors():
    """正文数字断言：(doc, 正则, spec)

    正则必须带 (?P<gN>...) 捕获组，捕获顺序与 spec 返回的期望值顺序
    一一对应。正则在文档中必须**恰好匹配一次**。
    """
    A = []
    P12 = 'P12_ALPHA_WEIGHT'

    def add(pattern, spec):
        A.append((P12, pattern, spec))

    # ── 摘要 ──
    add(r'打架（(?P<g1>-?\d+(?:\.\d+)?) vs (?P<g2>-?\d+(?:\.\d+)?)）',
        ('multi', ('curve', 'p12m', 'alpha_gz', 'best_w'),
         ('curve', 'p12o', 'alpha_gz', 'best_w')))
    add(r'5 窗口最优 w 跨度 (?P<g1>-?\d+(?:\.\d+)?)', ('cw_spread', 'alpha_gz'))
    add(r'迁移命中率只有 (?P<g1>-?\d+(?:\.\d+)?)%~(?P<g2>-?\d+(?:\.\d+)?)%',
        ('multi', ('e5_min', 'pct'), ('e5_max', 'pct')))

    # ── §二 组合层分解 ──
    add(r'基线复现门 ✅：w=0\.50 绝对夏普 \*\*(?P<g1>-?\d+(?:\.\d+)?)\*\*',
        ('gate', 'p12m'))
    add(r'基线复现门 ✅：\*\*(?P<g1>-?\d+(?:\.\d+)?)\*\*', ('gate', 'p12o'))
    add(r'组合 β 只有 (?P<g1>-?\d+(?:\.\d+)?)~(?P<g2>-?\d+(?:\.\d+)?)',
        ('multi', ('beta_min',), ('beta_max',)))
    add(r'（等权买入持有 (?P<g1>-?\d+(?:\.\d+)?)、国证A指 (?P<g2>-?\d+(?:\.\d+)?)）',
        ('multi', ('decomp', 'p12m', 'equal_weight_buyhold', 'excess_sharpe'),
         ('decomp', 'p12m', 'IDX_399317', 'excess_sharpe')))
    add(r'组合绝对收益 (?P<g1>-?\d+(?:\.\d+)?)%/年 < 等权买入持有 (?P<g2>-?\d+(?:\.\d+)?)%/年',
        ('multi', ('decomp', 'p12m', 'equal_weight_buyhold', 'leg_return_ann', 'pct'),
         ('decomp', 'p12m', 'equal_weight_buyhold', 'bench_return_ann', 'pct')))
    add(r'ρ (?P<g1>-?\d+(?:\.\d+)?)~(?P<g2>-?\d+(?:\.\d+)?)\*\*（不是 0）',
        ('multi', ('corr_min',), ('corr_max',)))

    # ── §三 长历史 ──
    add(r'（min (?P<g1>-?\d+(?:\.\d+)?) / max (?P<g2>-?\d+(?:\.\d+)?) / mean (?P<g3>-?\d+(?:\.\d+)?)）',
        ('multi', ('long_sum', 'min'), ('long_sum', 'max'),
         ('long_sum', 'mean')))

    # ── §四 w 曲线 ──
    add(r'绝对夏普口径\*\*（P6 原口径）：main (?P<g1>-?\d+(?:\.\d+)?) / oos (?P<g2>-?\d+(?:\.\d+)?)',
        ('multi', ('curve', 'p12m', 'abs_sharpe', 'best_w'),
         ('curve', 'p12o', 'abs_sharpe', 'best_w')))
    add(r'α 夏普口径\*\*（P11 §8\.2 指定口径）：main \*\*(?P<g1>-?\d+(?:\.\d+)?)\*\* / oos \*\*(?P<g2>-?\d+(?:\.\d+)?)\*\*',
        ('multi', ('curve', 'p12m', 'alpha_gz', 'best_w'),
         ('curve', 'p12o', 'alpha_gz', 'best_w')))
    add(r'0\.50 只差 (?P<g1>-?\d+(?:\.\d+)?)', ('bestgap', 'p12o'))
    add(r'差 (?P<g1>-?\d+(?:\.\d+)?)，比随机 w 最大值（(?P<g2>-?\d+(?:\.\d+)?)）只高 (?P<g3>-?\d+(?:\.\d+)?)',
        ('multi', ('bestgap', 'p12o'), ('randmax', 'p12o', 'alpha_gz'),
         ('e4gap', 'alpha_gz', 'oos')))
    add(r'最优值 \*\*(?P<g1>-?\d+(?:\.\d+)?) < 随机最大 (?P<g2>-?\d+(?:\.\d+)?)\*\*',
        ('multi', ('curve', 'p12o', 'alpha_ew', 'best_value'),
         ('randmax', 'p12o', 'alpha_ew')))

    # ── §五 5 窗口 + 配置敏感性 ──
    add(r'跨度 (?P<g1>-?\d+(?:\.\d+)?)~(?P<g2>-?\d+(?:\.\d+)?) 意味着',
        ('multi', ('cw_spread', 'abs_sharpe'), ('cw_spread', 'alpha_hs300')))
    add(r'最优 w 就移 (?P<g1>-?\d+(?:\.\d+)?)~(?P<g2>-?\d+(?:\.\d+)?)',
        ('multi', ('cfg_min',), ('cfg_max',)))

    # ── §六 诊断 ──
    add(r'A 股腿夏普 (?P<g1>-?\d+(?:\.\d+)?) 的熊市', ('long_ash', 'w2016'))
    add(r'A 股腿夏普 (?P<g1>-?\d+(?:\.\d+)?) 的牛市', ('long_ash', 'w2019'))
    add(r'最优 w = (?P<g1>-?\d+(?:\.\d+)?)~(?P<g2>-?\d+(?:\.\d+)?)（全仓 A 股腿）',
        ('multi', ('cw', 'abs_sharpe', 'w2019'), ('cw', 'alpha_gz', 'w2019')))
    add(r'B 组：(?P<g1>-?\d+(?:\.\d+)?) / (?P<g2>-?\d+(?:\.\d+)?) / (?P<g3>-?\d+(?:\.\d+)?) / (?P<g4>-?\d+(?:\.\d+)?)，跨度 (?P<g5>-?\d+(?:\.\d+)?)',
        ('multi', ('e2b_uniform', 'alpha_gz'), ('e2b_uniform', 'alpha_hs300'),
         ('e2b_uniform', 'abs_sharpe'), ('e2b_uniform', 'alpha_ew'),
         ('e2b_span',)))
    add(r'同组内迁移命中率\*\*：绝对夏普 (?P<g1>-?\d+(?:\.\d+)?)% / α_gz (?P<g2>-?\d+(?:\.\d+)?)% / α_ew (?P<g3>-?\d+(?:\.\d+)?)% / α_hs300 (?P<g4>-?\d+(?:\.\d+)?)%',
        ('multi', ('e6_same', 'abs_sharpe', 'pct'), ('e6_same', 'alpha_gz', 'pct'),
         ('e6_same', 'alpha_ew', 'pct'), ('e6_same', 'alpha_hs300', 'pct')))
    add(r'在绝大多数窗口排 (?P<g1>-?\d+(?:\.\d+)?)/21（中游），在 oos 的 α 口径下排 (?P<g2>-?\d+(?:\.\d+)?)/21（上游）',
        ('multi', ('e3_rank', 'alpha_gz', 'main'),
         ('e3_rank', 'alpha_gz', 'oos')))
    add(r'main (?P<g1>-?\d+(?:\.\d+)?)~(?P<g2>-?\d+(?:\.\d+)?)、oos (?P<g3>-?\d+(?:\.\d+)?)~(?P<g4>-?\d+(?:\.\d+)?)、\s*'
        r'w2013 (?P<g5>-?\d+(?:\.\d+)?)~(?P<g6>-?\d+(?:\.\d+)?)、w2016 (?P<g7>-?\d+(?:\.\d+)?)~(?P<g8>-?\d+(?:\.\d+)?)、\s*'
        r'w2019 (?P<g9>-?\d+(?:\.\d+)?)~(?P<g10>-?\d+(?:\.\d+)?)',
        ('multi', ('e3_loss_all_min', 'main'),
         ('e3_loss_all_max', 'main'),
         ('e3_loss_all_min', 'oos'),
         ('e3_loss_all_max', 'oos'),
         ('e3_loss_all_min', 'w2013'),
         ('e3_loss_all_max', 'w2013'),
         ('e3_loss_all_min', 'w2016'),
         ('e3_loss_all_max', 'w2016'),
         ('e3_loss_all_min', 'w2019'),
         ('e3_loss_all_max', 'w2019')))
    add(r'优势只有 (?P<g1>-?\d+(?:\.\d+)?) / (?P<g2>−0\.0007)',
        ('multi', ('e4gap', 'alpha_gz', 'oos'),
         ('e4gap', 'alpha_ew', 'oos')))

    # ── §八 结论 ──
    add(r'4 个评价口径给出 4 个不同的最优 w（跨度最大 (?P<g1>-?\d+(?:\.\d+)?)）',
        ('cw_spread', 'alpha_hs300'))
    add(r'5 个窗口给出跨度 (?P<g1>-?\d+(?:\.\d+)?) 的最优 w', ('cw_spread', 'alpha_gz'))
    add(r'拿到别的窗口，命中率 (?P<g1>-?\d+(?:\.\d+)?)%~(?P<g2>-?\d+(?:\.\d+)?)%，低于抛硬币',
        ('multi', ('e5_min', 'pct'), ('e5_max', 'pct')))
    add(r'只强 (?P<g1>-?\d+(?:\.\d+)?)（α_ew 甚至为负）', ('e4gap', 'alpha_gz', 'oos'))
    add(r'最优 w 照样散开（(?P<g1>-?\d+(?:\.\d+)?)~(?P<g2>-?\d+(?:\.\d+)?)）',
        ('multi', ('cw_bestw_min', 'alpha_gz'),
         ('cw_bestw_max', 'alpha_gz')))
    add(r'w2016 夏普 (?P<g1>-?\d+(?:\.\d+)?) / w2019 夏普 \+?(?P<g2>-?\d+(?:\.\d+)?)',
        ('multi', ('long_ash', 'w2016'), ('long_ash', 'w2019')))
    add(r'组合绝对夏普 (?P<g1>-?\d+(?:\.\d+)?)~(?P<g2>-?\d+(?:\.\d+)?)（定稿配置）',
        ('multi', ('comb_min',), ('comb_max',)))
    add(r'α 夏普 (?P<g1>-?\d+(?:\.\d+)?)~(?P<g2>-?\d+(?:\.\d+)?)（main/oos，定稿三类）',
        ('multi', ('ew_alpha_min',), ('ew_alpha_max',)))
    add(r'(?P<g1>-?\d+(?:\.\d+)?)~(?P<g2>-?\d+(?:\.\d+)?)（长历史 3 窗口，降级两类）',
        ('multi', ('long_sum', 'min'), ('long_sum', 'max')))
    add(r'组合 β 仅 (?P<g1>-?\d+(?:\.\d+)?)~(?P<g2>-?\d+(?:\.\d+)?)（对国证A指/等权持有），ρ (?P<g3>-?\d+(?:\.\d+)?)~(?P<g4>-?\d+(?:\.\d+)?)',
        ('multi', ('beta_min',), ('beta_max',), ('corr_min',), ('corr_max',)))

    # ── P4 / P5：定稿基线数字（曾因硬编码而与产物脱节）──
    P4 = 'P4_CLASS_WEIGHT'
    A.append((P4, r'组合夏普 \*\*main (?P<g1>-?\d+(?:\.\d+)?) / oos (?P<g2>-?\d+(?:\.\d+)?)\*\*，回撤 \*\*−(?P<g3>\d+(?:\.\d+)?)%\*\*',
              ('multi', ('p3_final', 'main'), ('p3_final', 'oos'),
               ('p3_final_dd', 'main'))))
    P5 = 'P5_CLASS_AUDIT'
    A.append((P5, r'剔除后组合夏普 main (?P<g1>-?\d+(?:\.\d+)?) → \*\*(?P<g2>-?\d+(?:\.\d+)?)\*\*、',
              ('multi', ('p3_base', 'main'), ('p3_final', 'main'))))
    A.append((P5, r'oos (?P<g1>-?\d+(?:\.\d+)?) → \*\*(?P<g2>-?\d+(?:\.\d+)?)\*\*，回撤 −(?P<g3>\d+(?:\.\d+)?)% → \*\*−(?P<g4>\d+(?:\.\d+)?)%\*\*',
              ('multi', ('p3_base', 'oos'), ('p3_final', 'oos'),
               ('p3_base_dd', 'main'), ('p3_final_dd', 'main'))))
    A.append((P5, r'一位都不变\*\*（main (?P<g1>-?\d+(?:\.\d+)?) / oos (?P<g2>-?\d+(?:\.\d+)?) / 回撤 −(?P<g3>\d+(?:\.\d+)?)%）',
              ('multi', ('p3_final', 'main'), ('p3_final', 'oos'),
               ('p3_final_dd', 'main'))))

    # ── P17：类内选标的 oracle 上界 + 袖套调仓频率 ──
    # 用**正文正则**而非表序号定位 —— 表序号定位在插表后会全乱。
    P17 = 'P17_CLASS_SELECT'
    A.append((P17, r'组合夏普上界只有 \*\*main \+(?P<g1>\d+(?:\.\d+)?)\*\* / \*\*oos \+(?P<g2>\d+(?:\.\d+)?)\*\*',
              ('multi', ('p17_gain', 'main'), ('p17_gain', 'oos'))))
    A.append((P17, r'oos 上随便随机选一组就能拿到 \+(?P<g1>\d+(?:\.\d+)?)',
              ('multi', ('p17_rmax', 'oos'))))
    A.append((P17, r'\*\*上界极小\*\*：main \+(?P<g1>\d+(?:\.\d+)?) / oos \+(?P<g2>\d+(?:\.\d+)?)',
              ('multi', ('p17_gain', 'main'), ('p17_gain', 'oos'))))
    A.append((P17, r'最高 \+(?P<g1>\d+(?:\.\d+)?) —— \*\*oracle 上界',
              ('multi', ('p17_rmax', 'oos'))))
    A.append((P17, r'\*\*main 最优是双周（(?P<g1>\d+(?:\.\d+)?)），oos 最优是月度（(?P<g2>\d+(?:\.\d+)?)）',
              ('multi', ('p17_freq', 'main', 'biweekly_10'),
               ('p17_freq', 'oos', 'monthly_21'))))

    # ── P18：42f 之上叠加 gbm 增量（分数级融合）──
    # 用**正文正则**定位（表序号定位在插表后会全乱，见 P17 的教训）。
    # ⚠️ 门控语义（2026-11-01 修正）：main 断言只要 main 产物存在就执行；
    # oos 断言只在 oos 产物存在时执行。原实现用 `and` 把整段一起跳过
    # → P18 文档数字长期无人审计。
    P18 = 'P18_FUSE_GBM'
    if P18_MAIN_READY:
        A.append((P18, r'同日截面相关中位数 = \*\*(?P<g1>\d+(?:\.\d+)?)\*\*',
                  ('multi', ('p18_corr', 'main'))))
        A.append((P18, r'复现门：orig42f 组合夏普 = \*\*(?P<g1>\d+(?:\.\d+)?)\*\*，'
                        r'fuse_w0\.00 组合夏普 = \*\*(?P<g2>\d+(?:\.\d+)?)\*\*',
                  ('multi', ('p18_arm', 'main', 'orig42f'),
                   ('p18_arm', 'main', 'fuse_w0.00'))))
        A.append((P18, r'候选臂 main 组合夏普 = \*\*(?P<g1>\d+(?:\.\d+)?)\*\*，'
                        r'基线 = \*\*(?P<g2>\d+(?:\.\d+)?)\*\*',
                  ('multi', ('p18_cand', 'main'), ('p18_base', 'main'))))
        A.append((P18, r'Δ组合（main）= \*\*(?P<g1>[+-]?\d+(?:\.\d+)?)\*\*',
                  ('multi', ('p18_delta', 'main'))))
    if P18_OOS_READY:
        A.append((P18, r'候选臂 oos 组合夏普 = \*\*(?P<g1>\d+(?:\.\d+)?)\*\*，'
                        r'oos 基线 = \*\*(?P<g2>\d+(?:\.\d+)?)\*\*',
                  ('multi', ('p18_cand', 'oos'), ('p18_base', 'oos'))))
        A.append((P18, r'Δ组合（oos）= \*\*(?P<g1>[+-]?\d+(?:\.\d+)?)\*\*',
                  ('multi', ('p18_delta', 'oos'))))
        A.append((P18, r'DSR（报告项）= \*\*(?P<g1>\d\.\d+)\*\* / 门槛 (?P<g2>\d+(?:\.\d+)?)',
                  ('multi', ('p18_dsr', 'oos'), ('p18_dsr_thr', 'oos'))))
    elif P18_MAIN_READY:
        # oos 不跑时，DSR 报告项在 §八 用 main 阶段的值列出（main 产物里有）
        A.append((P18, r'DSR（报告项）= \*\*(?P<g1>\d\.\d+)\*\* / 门槛 (?P<g2>\d+(?:\.\d+)?)',
                  ('multi', ('p18_dsr', 'main'), ('p18_dsr_thr', 'main'))))
    return A


def check_prose(texts, anchors):
    """正文数字核对：捕获到的数字必须与产物一致（按捕获文本的小数位舍入）。"""
    problems = []
    for doc_pref, pattern, spec in anchors:
        doc = next((k for k in texts if k.startswith(doc_pref)), None)
        if not doc:
            problems.append(f'{doc_pref}: 文档不存在')
            continue
        hits = list(re.finditer(pattern, texts[doc]))
        if len(hits) != 1:
            problems.append(
                f'{doc_pref}: 正则 /{pattern}/ 匹配 {len(hits)} 次（要求恰好 1 次）')
            continue
        specs = spec[1:] if spec[0] == 'multi' else (spec,)
        m = hits[0]
        gnames = sorted((g for g in m.groupdict() if g.startswith('g')),
                        key=lambda x: int(x[1:]))
        if len(gnames) != len(specs):
            problems.append(
                f'{doc_pref}: /{pattern}/ 捕获组 {len(gnames)} 个 vs 期望 {len(specs)} 个')
            continue
        for gname, sp in zip(gnames, specs):
            txt = m.group(gname).strip()
            # 百分比单位：捕获组只含数字，单位由 spec 声明（'pct'）
            pct = bool(sp[1:] and sp[-1] == 'pct')
            raw = txt.replace('−', '-')
            try:
                got = float(raw)
            except ValueError:
                problems.append(f'{doc_pref}: 组 {gname} 不是数字：{txt!r}')
                continue
            want = prose_value(sp)
            want = want * 100 if pct else want
            dec = len(raw.split('.')[1]) if '.' in raw else 0
            tol = 0.5 * (10 ** -dec) + 1e-9
            if abs(got - want) > tol:
                problems.append(
                    f'{doc_pref}: /{pattern}/ 组 {gname} 正文 {got} vs 产物 {want}'
                    f'（{dec} 位精度）')
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

# ── 按文档限定的旧值扫描 ─────────────────────────────────
# 为什么要有「限定范围」：STALE 是全局扫描，同一个数字在不同文档里
# 含义可能完全不同 —— 1.3767 在 FINAL_ROBUSTNESS 里是**合法的历史对照**
# （「修复前 1.3581 → 修复后 1.3767」），但在 P4/P5 里是**陈旧的当前状态**
# （引擎顺序修复后早已不是 1.3767）。全局白名单只能选一个，两头都不对。
#
# 引擎顺序不变性修复（948f6c55）前后的组合基线：
#   含可转债 base：main 1.3767→1.3786 / oos 1.0382→1.0391
#   剔除后定稿：  main 1.4092→1.4114 / oos 1.0685→1.0691
# 这些值曾硬编码在 gen_p4/gen_p5_doc.py 里，引擎修 bug 后没跟着变，
# 且旧值在任何当前产物里都不存在 → 文档与产物脱节而审计器抓不到。
SCOPED_STALE = {
    'P4_CLASS_WEIGHT_2026-10-31': {
        '1.4092': '旧（顺序修复前）组合基线 main',
        '1.0685': '旧（顺序修复前）组合基线 oos',
        '−7.95%': '旧（顺序修复前）定稿回撤',
    },
    'P5_CLASS_AUDIT_2026-10-31': {
        '1.4092': '旧（顺序修复前）组合基线 main',
        '1.0685': '旧（顺序修复前）组合基线 oos',
        '1.3767': '旧（顺序修复前）含可转债 base main',
        '1.0382': '旧（顺序修复前）含可转债 base oos',
        '−7.95%': '旧（顺序修复前）定稿回撤',
        '−8.70%': '旧（顺序修复前）含可转债回撤',
    },
}


def scan_stale(texts):
    hits = []
    for doc, body in texts.items():
        for old, why in STALE.items():
            for m in re.finditer(re.escape(old), body):
                ctx = body[max(0, m.start() - 70):m.start() + 50].replace('\n', '⏎')
                hits.append((doc, old, why, ctx))
        # 限定文档范围的旧值（同一数字在不同文档含义不同，见 SCOPED_STALE 注释）
        for pref, table in SCOPED_STALE.items():
            if not doc.startswith(pref):
                continue
            for old, why in table.items():
                for m in re.finditer(re.escape(old), body):
                    ctx = body[max(0, m.start() - 70):m.start() + 50].replace('\n', '⏎')
                    hits.append((doc, old, why + f'（{pref} 限定）', ctx))
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
    text_anchors = build_text_anchors()
    crit_anchors = build_crit_anchors()
    prose_anchors = build_prose_anchors()

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
    print(f'文本格核对：{len(text_anchors)} 条断言 × 持仓集字符串比对产物')
    print('=' * 78)
    tproblems = check_text_cells(texts, text_anchors)
    for p in tproblems:
        print(f'  ❌ {p}')
    if not tproblems:
        print('  ✅ 全部文本断言通过')

    print()
    print('=' * 78)
    print(f'判据符号核对：{len(crit_anchors)} 条断言 × ✅/❌/— 逐格比对产物')
    print('=' * 78)
    cproblems = check_crit_cells(texts, crit_anchors)
    for p in cproblems:
        print(f'  ❌ {p}')
    if not cproblems:
        print('  ✅ 全部判据符号通过')

    print()
    print('=' * 78)
    print(f'正文数字核对：{len(prose_anchors)} 条断言 × 正则定位 + 产物比对')
    print('=' * 78)
    pproblems = check_prose(texts, prose_anchors)
    for p in pproblems:
        print(f'  ❌ {p}')
    if not pproblems:
        print('  ✅ 全部正文数字断言通过')

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
            # 每个文档注入一处（seen_docs 去重）—— 上限按文档数放开，
            # 否则排在后面的新文档（如 P11）永远轮不到，自检变成盲区。
            if len(injected) >= 100:
                break
            doc_pref, tidx, row_label, col_hdr, spec = a[0], a[1], a[2], a[3], a[4]
            if doc_pref in seen_docs:
                continue
            doc = next((k for k in texts if k.startswith(doc_pref)), None)
            if not doc:
                continue
            try:
                tables = extract_tables(texts[doc])
                if isinstance(tidx, tuple) and tidx and tidx[0] == 'row':
                    # 行标签定位（与 check() 同语义）：先用行标签筛，再用列名筛
                    cands = [i for i, (h, rows) in enumerate(tables)
                             if any(r and norm(r[0]) == norm(tidx[1]) for r in rows)]
                    cands2 = [i for i in cands
                              if any(norm(c) == norm(col_hdr) for c in tables[i][0])]
                    if cands2:
                        cands = cands2
                    if not cands:
                        continue
                    tidx = cands[0]
                elif isinstance(tidx, str):
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

        # 文本格注入：破坏一个持仓集单元格（数字注入覆盖不到字符串格）
        # 每类注入用**独立副本**，否则一处注入会污染后续注入的定位，
        # 自检会互相掩盖（曾出现：正文注入改坏了判据符号 → 判据自检假绿）。
        t3 = dict(texts)
        tinj = []
        for a in build_text_anchors():
            doc_pref, tidx, row_label, col_hdr, spec = a
            doc = next((k for k in texts if k.startswith(doc_pref)), None)
            if not doc:
                continue
            tables = extract_tables(t3[doc])
            if tidx >= len(tables):
                continue
            header, data = tables[tidx]
            cell, err = find_cell(header, data, row_label, col_hdr)
            if err or not cell:
                continue
            bad = None
            for real in ('510170', '159981', '159985', '159980', '511260', '518880'):
                if real in cell and t3[doc].count(cell) == 1:
                    bad = cell.replace(real, '999999')
                    break
            if not bad:
                continue
            t3[doc] = t3[doc].replace(cell, bad, 1)
            tinj.append((doc, row_label, col_hdr, cell, bad))
            break
        for doc, rl, ch, oc, bc in tinj:
            print(f'  文本注入 [{doc}] 「{rl}」×「{ch}」: {oc} → {bc}')
        tp2 = check_text_cells(t3, build_text_anchors())
        print(f'  文本注入后断言失败数：{len(tp2)}（期望 > 0）')
        for p in tp2[:5]:
            print(f'      {p}')
        neg_ok = neg_ok and len(tp2) > 0

        # 判据符号注入：把 ✅ 改成 ❌（数字注入覆盖不到符号格）。
        # 注意符号格内容常重复（多个 ' ✅ '），**不能**用字符串唯一性判断，
        # 否则会静默跳过注入 → 自检变盲区。这里改用「行+列」定位重建整行。
        t4 = dict(texts)
        cinj = []
        for a in build_crit_anchors():
            doc_pref, tidx, row_label, col_hdr, spec = a
            doc = next((k for k in texts if k.startswith(doc_pref)), None)
            if not doc:
                continue
            tables = extract_tables(t4[doc])
            if tidx >= len(tables):
                continue
            header, data = tables[tidx]
            cell, err = find_cell(header, data, row_label, col_hdr)
            if err or not cell or '✅' not in cell:
                continue
            # 定位该格所在的 markdown 行：整行唯一 → 逐格替换
            hi = [norm(c) for c in header]
            ci = hi.index(norm(col_hdr))
            want_cols = [norm(x) for x in row_label] if isinstance(row_label, (tuple, list)) \
                else [norm(row_label)]
            target_line = None
            for line in t4[doc].splitlines():
                if not line.strip().startswith('|'):
                    continue
                cells = line.split('|')[1:-1]
                if len(cells) <= ci:
                    continue
                if not norm(cells[0]).startswith(want_cols[0]):
                    continue
                if all(norm(cells[i]) == want_cols[i] for i in range(1, len(want_cols))):
                    target_line = (line, cells)
                    break
            if target_line is None:
                continue
            line, cells = target_line
            bad_cell = cells[ci].replace('✅', '❌', 1)
            if bad_cell == cells[ci]:
                continue
            new_cells = list(cells)
            new_cells[ci] = bad_cell
            new_line = '|' + '|'.join(new_cells) + '|'
            t4[doc] = t4[doc].replace(line, new_line, 1)
            cinj.append((doc, row_label, col_hdr, cells[ci], bad_cell))
            break
        for doc, rl, ch, oc, bc in cinj:
            print(f'  判据注入 [{doc}] 「{rl[:18]}」×「{ch}」: {oc} → {bc}')
        # 正文数字注入：把正文里某个数字改错（表格断言覆盖不到正文）
        t5 = dict(texts)
        pp2 = []
        for doc_pref, pattern, spec in build_prose_anchors():
            doc = next((k for k in texts if k.startswith(doc_pref)), None)
            if not doc:
                continue
            m = re.search(pattern, texts[doc])
            if not m:
                continue
            gname = sorted((g for g in m.groupdict() if g.startswith('g')),
                           key=lambda x: int(x[1:]))[0]
            orig = m.group(gname)
            # 改最后一位数字
            digits = '0123456789'
            new_txt = None
            for i in range(len(orig) - 1, -1, -1):
                if orig[i] in digits:
                    nd = digits[(int(orig[i]) + 1) % 10]
                    new_txt = orig[:i] + nd + orig[i + 1:]
                    break
            if not new_txt or new_txt == orig:
                continue
            t5[doc] = texts[doc][:m.start(gname)] + new_txt + texts[doc][m.end(gname):]
            pp2.append((doc, pattern, orig, new_txt))
            break
        for doc, pat, o, n in pp2:
            print(f'  正文注入 [{doc}] /{pat}/: {o} → {n}')
        rp2 = check_prose(t5, build_prose_anchors())
        print(f'  正文注入后断言失败数：{len(rp2)}（期望 > 0）')
        for p in rp2[:5]:
            print(f'      {p}')
        neg_ok = neg_ok and len(rp2) > 0

        cp2 = check_crit_cells(t4, build_crit_anchors())
        print(f'  判据注入后断言失败数：{len(cp2)}（期望 > 0）')
        for p in cp2[:5]:
            print(f'      {p}')
        neg_ok = neg_ok and len(cp2) > 0

        # ── 逐锚点穷举自检 ───────────────────────────────────────────
        # 上面的注入是「每文档一处」，只能证明「这个文档被抓到」，
        # 不能证明**每一条锚点都真的生效**（写错行列 → 静默不比对 → 绿灯）。
        # 这里对每一条锚点单独注入一次，逐条确认审计失败。
        # 代价 = O(锚点数 × 文档解析)，只在 --neg 下跑。
        print()
        print('=' * 78)
        print('逐锚点穷举自检：每条锚点单独注入 → 必须逐条失败')
        print('=' * 78)
        miss_anchor, ok_anchor, skipped = [], 0, []
        for a in anchors:
            doc_pref, tidx, row_label, col_hdr, spec = a[0], a[1], a[2], a[3], a[4]
            doc = next((k for k in texts if k.startswith(doc_pref)), None)
            if not doc:
                skipped.append((doc_pref, row_label, col_hdr, '文档缺失'))
                continue
            try:
                tables = extract_tables(texts[doc])
                if isinstance(tidx, tuple) and tidx and tidx[0] == 'row':
                    cands = [i for i, (h, rows) in enumerate(tables)
                             if any(r and norm(r[0]) == norm(tidx[1]) for r in rows)]
                    cands2 = [i for i in cands
                              if any(norm(c) == norm(col_hdr) for c in tables[i][0])]
                    if cands2:
                        cands = cands2
                    if not cands:
                        skipped.append((doc_pref, row_label, col_hdr, '无此行'))
                        continue
                    tidx = cands[0]
                elif isinstance(tidx, str):
                    cands = [i for i, (h, _) in enumerate(tables)
                             if any(norm(col_hdr) == norm(c) for c in h)]
                    if not cands:
                        skipped.append((doc_pref, row_label, col_hdr, '无此列'))
                        continue
                    tidx = cands[0]
                if tidx >= len(tables):
                    skipped.append((doc_pref, row_label, col_hdr, '表越界'))
                    continue
                header, data = tables[tidx]
                cell, err = find_cell(header, data, row_label, col_hdr)
                if err or cell is None:
                    skipped.append((doc_pref, row_label, col_hdr, err or '无格'))
                    continue
                # **按 (行, 列) 索引定位**，不能按文本找：
                # 同一行里两列可能文本完全相同（如 P2 的「样本外」与「较差窗口」
                # 都可能是 0.8722）→ 按文本找会改错列 → 该锚点看起来「抓不到」。
                hi = [norm(c) for c in header]
                ci = next((k for k, c in enumerate(hi) if c == norm(col_hdr)), None)
                # 行键可能是元组（前 N 列依次匹配），语义与 find_cell 一致
                if isinstance(row_label, (tuple, list)):
                    wcols = [norm(x) for x in row_label]
                    nk = len(wcols)
                else:
                    wcols = [norm(row_label)]
                    nk = 1
                ri = None
                for k, r in enumerate(data):
                    if not r or len(r) < nk:
                        continue
                    if nk == 1:
                        if norm(r[0]) == wcols[0]:
                            ri = k
                            break
                    else:
                        if norm(r[0]).startswith(wcols[0]) and all(
                                norm(r[i]) == wcols[i] for i in range(1, nk)):
                            ri = k
                            break
                if ci is None or ri is None:
                    skipped.append((doc_pref, row_label, col_hdr, '行列索引定位失败'))
                    continue
                cell = data[ri][ci]
                nums = cell_num_positions(cell)
                if not nums:
                    skipped.append((doc_pref, row_label, col_hdr, '格内无数字'))
                    continue
                # 复合格里可能有多个数字，锚点可能对应第 idx 个 →
                # 逐个位置注入，只要**任一**位置能破坏该锚点就算这条锚点有效。
                # （只注入第 0 个会误判：锚点盯的是第 1 个数字时，第 0 个坏了也照样通过。）
                variants = []
                for n, s, e in nums:
                    bad = cell[:s] + f"{n + 0.5:.4f}" + cell[e:]
                    if bad != cell:
                        variants.append((n, s, e, bad))
                if not variants:
                    skipped.append((doc_pref, row_label, col_hdr, '注入值相同'))
                    continue
                old_row = data[ri][ci]
                # 找到对应 markdown 行：表 tidx 的第 (ri+2) 行（+1 表头 +1 分隔行）
                lines6 = texts[doc].splitlines(keepends=True)
                tbl_lines = []
                cur = []
                for ln in lines6:
                    if ln.strip().startswith('|'):
                        cur.append(ln)
                    else:
                        if len(cur) >= 2:
                            tbl_lines.append(cur)
                        cur = []
                if len(cur) >= 2:
                    tbl_lines.append(cur)
                if tidx >= len(tbl_lines):
                    skipped.append((doc_pref, row_label, col_hdr, '表行定位失败'))
                    continue
                raw = tbl_lines[tidx][ri + 2]  # +1 表头 +1 分隔行
                broke = False
                for n, s, e, bad in variants:
                    new_row = old_row[:s] + f"{n + 0.5:.4f}" + old_row[e:]
                    cells = raw.split('|')
                    cells[ci + 1] = new_row
                    new_raw = '|'.join(cells)
                    t6 = dict(texts)
                    t6[doc] = t6[doc].replace(raw, new_raw, 1)
                    if check(t6, [a]):
                        broke = True
                        break
                if broke:
                    ok_anchor += 1
                else:
                    miss_anchor.append((doc_pref, row_label, col_hdr))
            except Exception as ex:
                skipped.append((doc_pref, row_label, col_hdr, f'异常:{ex}'))
        print(f'  可注入锚点：{ok_anchor + len(miss_anchor)}，抓��成功：{ok_anchor}，漏抓：{len(miss_anchor)}')
        if miss_anchor:
            print('  ❌ 漏抓锚点（前 15 条）：')
            for m in miss_anchor[:15]:
                print(f'      {m[0]} 「{str(m[1])[:24]}」×「{str(m[2])[:16]}」')
        if skipped:
            print(f'  ℹ️  跳过（该格不含可注入数字等）：{len(skipped)} 条')
            for sk in skipped[:8]:
                print(f'      {sk[0]} 「{str(sk[1])[:20]}」×「{str(sk[2])[:14]}」 {sk[3]}')
        all_ok = neg_ok and not miss_anchor
        print(f'  逐锚点自检：{"✅ 全部锚点可被破坏并抓到" if not miss_anchor else "❌ 有锚点形同虚设"}')
        neg_ok = all_ok
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

    return 1 if (problems or tproblems or cproblems) else 0


if __name__ == '__main__':
    sys.exit(main())
