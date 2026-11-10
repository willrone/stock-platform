'use client';

/**
 * 多策略纸面盘（策略赛马）
 *
 * ⚠️ 2026-10-27 修正
 *   原来这里 `fetch('/paper-portfolio.json')` 读 public/ 下的静态文件，问题：
 *     1. **陈旧** —— 静态快照与实时账本不同步（实测 2026-10-23 vs 10-26）；
 *     2. **付费内容泄露** —— public/ 无需鉴权即可下载，等于免费公开多策略
 *        买卖清单。
 *   现在改为读**受保护的实时端点**：
 *     - GET /api/v1/trading/strategies  → 各条线的持仓数 / 最新净值
 *     - GET /api/v1/trading/signal?strategy=<key> → 该条线的实时买卖/持仓明细
 *   并加数据过期守卫（各条线调仓日不同步，过期的线单独标灰 + 标注）。
 *
 *   👉 仍需后端 / 运维处理：把 frontend/public/paper-portfolio.json
 *      移出 public 目录（不在本任务的可改文件清单内）。
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import { RefreshCw, ShieldAlert, Lock } from 'lucide-react';
import { ErrorState } from '@/components/mobile/ErrorState';
import { MobileTradeCard } from '@/components/mobile/MobileTradeCard';
import { ResponsiveTable } from '@/components/mobile/ResponsiveTable';
import StrategyRaceChart, { RaceNavPoint } from '@/components/charts/StrategyRaceChart';
import { MobileBottomNav } from '@/components/mobile/MobileBottomNav';
import { SafeAreaTop } from '@/components/mobile/SafeAreaTop';
import { BOTTOM_NAV_CONTENT_PADDING, MIN_TOUCH_TARGET } from '@/components/mobile/safeArea';
import { tradingGet, daysSince } from '@/components/mobile/tradingClient';

/**
 * 【P0-04】策略赛马（3 年全市场真实盘口径回测）的数据形状。
 * 与后端 app/services/trading/strategy_race.py 的返回结构一一对应。
 */
type RaceTurnover = {
  trades_per_year: number;
  turnover_pct: number;
  positions: number;
  frequency: '低频' | '中频' | '高频' | '未知';
  /** 'measured' = 实测持仓只数；'topk_fallback' = 降级用 topk 猜值 */
  positions_source?: 'measured' | 'topk_fallback' | 'unknown';
};

type RaceMetrics = {
  total_return: number | null;
  annualized_return: number | null;
  sharpe_ratio: number | null;
  max_drawdown: number | null;
  volatility: number | null;
  win_rate: number | null;
  total_trades: number;
  buy_trades?: number | null;
  sell_trades?: number | null;
  final_value: number | null;
};

type BenchmarkComparison = {
  benchmark_name?: string | null;
  benchmark_total_return?: number | null;
  benchmark_annualized_return?: number | null;
  benchmark_sharpe?: number | null;
  benchmark_max_drawdown?: number | null;
  mainline_total_return?: number | null;
  mainline_annualized_return?: number | null;
  mainline_sharpe?: number | null;
  mainline_max_drawdown?: number | null;
  excess_total_return_pct?: number | null;
  excess_annualized_return_pct?: number | null;
  beats_benchmark_total_return?: boolean;
  beats_benchmark_sharpe?: boolean;
  mainline_drawdown_better?: boolean;
  verdict?: string;
};

type RaceStrategy = {
  key: string;
  name: string;
  is_mainline: boolean;
  metrics: RaceMetrics;
  turnover: RaceTurnover;
  yearly?: Record<string, number>;
  nav: RaceNavPoint[];
  nav_basis?: string;
};

type RaceData = {
  status: string;
  window?: string;
  years?: number;
  universe_size?: number;
  methodology?: Record<string, unknown>;
  strategies: RaceStrategy[];
  benchmark?: {
    name?: string;
    symbol?: string;
    total_return?: number;
    annualized_return?: number;
    sharpe_ratio?: number;
    max_drawdown?: number;
  } | null;
  disclosure?: string[];
  benchmark_comparison?: BenchmarkComparison | null;
  locked_detail?: boolean;
  upgrade_hint?: string;
  message?: string;
};

type Trade = {
  code: string;
  name?: string;
  price?: number;
  shares?: number;
  amount?: number;
  commission?: number;
  stamp_tax?: number;
  slippage?: number;
  total_cost?: number;
  net?: number;
  score?: number | string;
  factors?: { factor: string; contribution: number; direction: string }[];
};

type LineSignal = {
  strategy: string;
  display_name: string;
  desc: string;
  status: string;
  message?: string;
  date?: string;
  topk: number;
  ndrop: number;
  cash: number;
  value: number;
  capital: number;
  total_return_pct: number;
  n_positions: number;
  buys: Trade[];
  sells: Trade[];
  holds: Array<{
    code: string;
    name?: string;
    shares: number;
    cost: number;
    cost_amount: number;
    buy_date: string;
  }>;
  cost_model?: {
    commission: number;
    stamp_tax: number;
    slippage: number;
    cash_reserve: number;
    board_lot: number;
  };
  fill_assumption?: string;
  upgrade_hint?: string;
  // ── 观测口径层（2026-11-02）：暴露度 / 投入段收益 / as-of / 成交笔数 / 状态 ──
  /** 暴露度 = 1 − cash/value（0~1） */
  exposure?: number | null;
  /** 投入段收益% = (value−initial)/(value−cash)×100 → 消除仓位量差异后的可比口径 */
  invested_return_pct?: number | null;
  /** 展示用 PnL 字符串；从未成交的线是 "—"（不是 "0.00%"） */
  pnl_display?: string | null;
  /** 该线估值所用的数据日 */
  as_of?: string | null;
  /** 未建仓 / 持仓中 / 空仓 */
  line_status?: string | null;
  /** 累计成交笔数 */
  n_trades?: number | null;
};

type StrategyInfo = {
  key: string;
  display_name: string;
  desc: string;
  factor_note: string;
  topk: number;
  ndrop: number;
  n_positions: number;
  last_signal: string | null;
  last_value: number | null;
  locked?: boolean;
  delay_days?: number;
  // ── 观测口径层（2026-11-02）──────────────────────────────────
  exposure?: number | null;
  invested_return_pct?: number | null;
  pnl_display?: string | null;
  as_of?: string | null;
  status?: string | null;
  n_trades?: number | null;
};

type StrategyListResponse = {
  strategies: StrategyInfo[];
  tier: string;
  tier_label: string;
  delay_days: number;
  include_execution_detail: boolean;
};

const LINE_COLORS: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'secondary'> = {
  '22f': 'info',
  rsi_71421: 'success',
  factor_ranking: 'warning',
  gbm_voltgt: 'secondary',
  honest_42f_neu: 'secondary',
  // 旗舰线（2026-11-01）：多资产组合是平台唯一的跨资产信号线 → 用主色
  multi_asset_final: 'primary',
  // 免费档主力退到辅色：主色留给旗舰
  honest_42f: 'secondary',
};

/** 数据超过这个天数就视为过期 */
const STALE_DAYS = 7;

const fmtMoney = (v?: number) =>
  typeof v === 'number' && Number.isFinite(v)
    ? `¥${v.toLocaleString('zh-CN', { maximumFractionDigits: 0 })}`
    : '暂无';

/** 百分比格式化（后端返回的是小数，0.3551 → +35.51%）。null → '—' */
const fmtPct = (v?: number | null) =>
  typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(2)}%` : '—';

const feeOf = (t: Trade) => (t.commission ?? 0) + (t.stamp_tax ?? 0) + (t.slippage ?? 0);

const fmtScore = (v?: number | string | null) => {
  if (v === undefined || v === null || v === '') {
    return '暂无';
  }
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(3) : String(v);
};

export default function PaperPortfolioPage() {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [lines, setLines] = useState<Record<string, LineSignal>>({});
  const [failed, setFailed] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [errorDetail, setErrorDetail] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [active, setActive] = useState<string>('');

  // ── 策略赛马（3 年真实盘口径回测）─────────────────────────
  const [race, setRace] = useState<RaceData | null>(null);
  const [raceError, setRaceError] = useState<string | null>(null);
  const [raceLoading, setRaceLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setErrorDetail(null);
    setFailed({});
    try {
      const list = await tradingGet<StrategyListResponse>('/trading/strategies');
      const infos = list.strategies || [];
      setStrategies(infos);

      // 各条线的实时明细并发拉取；单条线失败（403 付费墙等）不影响其它线
      const usable = infos.filter(s => !s.locked);
      const settled = await Promise.allSettled(
        usable.map(s =>
          tradingGet<LineSignal>(`/trading/signal?strategy=${encodeURIComponent(s.key)}`)
        )
      );
      const next: Record<string, LineSignal> = {};
      const nextFailed: Record<string, string> = {};
      settled.forEach((res, i) => {
        const key = usable[i].key;
        if (res.status === 'fulfilled') {
          next[key] = res.value;
        } else {
          const e = res.reason as { message?: string; detail?: unknown };
          nextFailed[key] = e?.message || '该条线加载失败';
        }
      });
      setLines(next);
      setFailed(nextFailed);

      const first = Object.keys(next)[0] || usable[0]?.key || '';
      setActive(prev => (prev && (next[prev] || nextFailed[prev]) ? prev : first));
    } catch (e) {
      setError(e instanceof Error ? e.message : '读取纸面组合失败');
      setErrorDetail((e as { detail?: unknown })?.detail ?? e);
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * 拉取策略赛马数据（与纸面盘接口独立 —— 纸面盘是**区间**收益，
   * 赛马是 **3 年累计**口径，两者不能混排）。
   */
  const loadRace = useCallback(async () => {
    setRaceLoading(true);
    setRaceError(null);
    try {
      const res = await tradingGet<RaceData>('/trading/strategy-race');
      setRace(res);
    } catch (e) {
      setRaceError(e instanceof Error ? e.message : '策略赛马数据加载失败');
    } finally {
      setRaceLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadRace();
  }, [loadRace]);

  useEffect(() => {
    void load();
  }, [load]);

  const line = active ? lines[active] : undefined;
  const activeInfo = strategies.find(s => s.key === active);
  const cap = 1_000_000;
  const ageDays = daysSince(line?.date);
  const isStale = ageDays !== null && ageDays > STALE_DAYS;

  const order = useMemo(
    () => [...strategies].sort((a, b) => (b.last_value ?? 0) - (a.last_value ?? 0)),
    [strategies]
  );

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2, md: 3 } }}>
      <SafeAreaTop />
      <MobileBottomNav />

      <Stack
        direction="row"
        justifyContent="space-between"
        alignItems="flex-start"
        sx={{ mb: 0.5 }}
      >
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="h5" gutterBottom>
            多策略纸面盘（策略赛马）
          </Typography>
          <Typography variant="body2" color="text.secondary">
            每条策略一条独立线，各 1,000,000 元、独立持仓与账本，同一全市场宇宙。
          </Typography>
        </Box>
        <Button
          startIcon={<RefreshCw size={16} />}
          onClick={() => void load()}
          disabled={loading}
          sx={{ minHeight: MIN_TOUCH_TARGET, flexShrink: 0 }}
        >
          刷新
        </Button>
      </Stack>

      {error && (
        <ErrorState
          title="纸面组合加载失败"
          message={error}
          detail={errorDetail}
          onRetry={() => void load()}
          retrying={loading}
        />
      )}

      <Alert severity="warning" icon={<ShieldAlert size={18} />} sx={{ mb: 2, mt: 1.5 }}>
        <Typography variant="body2">
          这是<strong>纸面虚拟盘</strong>，各条线独立 100 万初始资金、独立记账。 成本模型：佣金 3.0‱
          + 印花税 5.0‱（卖出） + 滑点 1.0‰ + 现金预留 5%，
          成交价按「信号日收盘出信号、下一交易日成交」（lag1）假设。
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          各条线的调仓日可能不同步（有的线停在 {STALE_DAYS} 天前），已单独标注。
          纸面成交不保证能在实盘复现。
        </Typography>
      </Alert>

      {loading && Object.keys(lines).length === 0 && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
          <CircularProgress />
        </Box>
      )}

      {Object.keys(failed).length > 0 && !error && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          以下条线加载失败（可能是订阅等级限制）：{' '}
          {Object.entries(failed)
            .map(([k, v]) => `${k}（${v}）`)
            .join('、')}
        </Alert>
      )}

      {/* ═══════════════════════════════════════════════════════════
          【P0-04】策略赛马：3 年真实盘口径的横向对比
          —— 夏普 / 最大回撤 / 年换手率 / 净值曲线 / 跑赢基准
          与上方「纸面盘」严格分开：纸面盘是**两次调仓之间**的区间收益，
          赛马是 **3 年累计**口径，并排显示会误导。
          ═══════════════════════════════════════════════════════════ */}
      <Card sx={{ mb: 3, mt: 2 }}>
        <CardContent>
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            justifyContent="space-between"
            alignItems={{ xs: 'flex-start', sm: 'center' }}
            spacing={1}
            sx={{ mb: 1.5 }}
          >
            <Box sx={{ minWidth: 0 }}>
              <Typography variant="h6" sx={{ fontWeight: 700 }}>
                策略赛马 · 3 年全市场回测对比
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {race?.window
                  ? `回测窗口 ${race.window} · 全市场 ${race.universe_size ?? '—'} 只 · `
                  : ''}
                真实盘口径（信号日收盘出信号 → 下一交易日成交 lag1 + 可成交性检查）
              </Typography>
            </Box>
            <Chip
              size="small"
              label="按夏普降序"
              color="primary"
              variant="outlined"
              sx={{ flexShrink: 0 }}
            />
          </Stack>

          {raceError && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              策略赛马数据加载失败：{raceError}
            </Alert>
          )}

          {raceLoading && !race && (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 3 }}>
              <CircularProgress size={28} />
            </Box>
          )}

          {race?.status === 'unavailable' && (
            <Alert severity="info" sx={{ mb: 2 }}>
              {race.message ?? '策略赛马数据尚未生成'}
            </Alert>
          )}

          {race && race.strategies.length > 0 && (
            <>
              {race.benchmark_comparison && (
                <Alert
                  severity={
                    race.benchmark_comparison.beats_benchmark_total_return ? 'success' : 'warning'
                  }
                  sx={{ mb: 2 }}
                >
                  <Typography variant="body2" sx={{ fontWeight: 700, mb: 0.5 }}>
                    与基准的诚实对照
                  </Typography>
                  <Typography variant="body2" component="div" sx={{ mb: 0.5 }}>
                    主线「综合因子」3 年累计{' '}
                    <strong>{race.benchmark_comparison.mainline_total_return?.toFixed(1)}%</strong>
                    ，基准（{race.benchmark_comparison.benchmark_name ?? '全市场等权买入持有'}）
                    <strong>{race.benchmark_comparison.benchmark_total_return?.toFixed(1)}%</strong>
                    ，超额{' '}
                    <strong>
                      {race.benchmark_comparison.excess_total_return_pct?.toFixed(1)} 个百分点
                    </strong>
                    。
                  </Typography>
                  <Typography variant="body2" component="div" sx={{ mb: 0.5 }}>
                    但主线最大回撤{' '}
                    <strong>{race.benchmark_comparison.mainline_max_drawdown?.toFixed(1)}%</strong>{' '}
                    vs 基准{' '}
                    <strong>{race.benchmark_comparison.benchmark_max_drawdown?.toFixed(1)}%</strong>
                    ，夏普 {race.benchmark_comparison.mainline_sharpe?.toFixed(2)} vs{' '}
                    {race.benchmark_comparison.benchmark_sharpe?.toFixed(2)}。
                  </Typography>
                  <Typography variant="body2" component="div" sx={{ fontWeight: 600 }}>
                    结论：{race.benchmark_comparison.verdict}
                  </Typography>
                </Alert>
              )}
              <ResponsiveTable minWidth={880}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 700 }}>策略</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 700 }}>
                        3 年累计
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 700 }}>
                        年化
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 700 }}>
                        夏普
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 700 }}>
                        最大回撤
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 700 }}>
                        逐笔胜率
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 700 }}>
                        年换手率
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: 'block', fontSize: 10, fontWeight: 400 }}
                        >
                          持仓位年换手次数
                        </Typography>
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 700 }}>
                        交易笔数
                      </TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {race.strategies.map(s => {
                      const m = s.metrics;
                      const t = s.turnover;
                      const pos = m.total_return ?? 0;
                      const freqColor =
                        t.frequency === '高频'
                          ? 'error'
                          : t.frequency === '中频'
                            ? 'warning'
                            : 'success';
                      return (
                        <TableRow
                          key={s.key}
                          sx={s.is_mainline ? { bgcolor: 'action.hover' } : undefined}
                        >
                          <TableCell>
                            <Stack direction="row" spacing={0.75} alignItems="center">
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                {s.name}
                              </Typography>
                              {s.is_mainline && (
                                <Chip
                                  size="small"
                                  label="主线"
                                  color="primary"
                                  sx={{ height: 18, fontSize: 10 }}
                                />
                              )}
                            </Stack>
                          </TableCell>
                          <TableCell align="right">
                            <Typography
                              variant="body2"
                              sx={{
                                fontWeight: 600,
                                color: pos >= 0 ? 'success.main' : 'error.main',
                              }}
                            >
                              {fmtPct(m.total_return)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography
                              variant="body2"
                              sx={{
                                color:
                                  (m.annualized_return ?? 0) >= 0 ? 'success.main' : 'error.main',
                              }}
                            >
                              {fmtPct(m.annualized_return)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography
                              variant="body2"
                              sx={{
                                fontWeight: 700,
                                color:
                                  m.sharpe_ratio === null
                                    ? 'text.secondary'
                                    : m.sharpe_ratio >= 0.8
                                      ? 'success.main'
                                      : m.sharpe_ratio >= 0
                                        ? 'text.primary'
                                        : 'error.main',
                              }}
                            >
                              {m.sharpe_ratio === null ? '—' : m.sharpe_ratio.toFixed(3)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography
                              variant="body2"
                              sx={{
                                color:
                                  m.max_drawdown === null
                                    ? 'text.secondary'
                                    : m.max_drawdown <= -0.2
                                      ? 'error.main'
                                      : 'text.primary',
                              }}
                            >
                              {fmtPct(m.max_drawdown)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2">
                              {m.win_rate === null ? '—' : `${(m.win_rate * 100).toFixed(1)}%`}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Stack
                              direction="row"
                              spacing={0.5}
                              justifyContent="flex-end"
                              alignItems="center"
                            >
                              <Typography variant="body2">
                                {t.turnover_pct > 0 ? `${t.turnover_pct.toFixed(0)}%` : '—'}
                              </Typography>
                              <Chip
                                size="small"
                                label={t.frequency}
                                color={freqColor}
                                variant="outlined"
                                sx={{ height: 18, fontSize: 10 }}
                              />
                            </Stack>
                            <Typography
                              variant="caption"
                              color={
                                t.positions_source === 'topk_fallback'
                                  ? 'warning.main'
                                  : 'text.secondary'
                              }
                              sx={{ display: 'block', fontSize: 10 }}
                            >
                              {t.positions_source === 'topk_fallback'
                                ? `持仓 ${t.positions}（topk 估算）`
                                : `持仓 ${t.positions} 只`}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2">{m.total_trades}</Typography>
                            {m.buy_trades != null && m.sell_trades != null && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{ display: 'block', fontSize: 10 }}
                              >
                                买 {m.buy_trades} / 卖 {m.sell_trades}
                              </Typography>
                            )}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                    {race.benchmark && (
                      <TableRow sx={{ bgcolor: 'background.default' }}>
                        <TableCell>
                          <Typography
                            variant="body2"
                            sx={{ fontWeight: 600, color: 'text.secondary' }}
                          >
                            基准 · {race.benchmark.name ?? '全市场等权'}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                            {fmtPct(race.benchmark.total_return)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                            {fmtPct(race.benchmark.annualized_return)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography
                            variant="body2"
                            sx={{ fontWeight: 700, color: 'text.secondary' }}
                          >
                            {race.benchmark.sharpe_ratio?.toFixed(3) ?? '—'}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                            {fmtPct(race.benchmark.max_drawdown)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">—</TableCell>
                        <TableCell align="right">—</TableCell>
                        <TableCell align="right">—</TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </ResponsiveTable>

              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.5 }}>
                  净值曲线对比
                </Typography>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ display: 'block', mb: 1 }}
                >
                  起点 = 1.0（期初 100 万）；曲线由分月收益复利还原 （月频 36
                  点，非日频精确净值，阶梯感是预期的）。
                </Typography>
                <StrategyRaceChart
                  series={race.strategies.map(s => ({
                    key: s.key,
                    name: s.name,
                    isMainline: s.is_mainline,
                    nav: s.nav,
                  }))}
                  height={320}
                />
              </Box>

              {race.locked_detail && (
                <Alert severity="info" sx={{ mt: 2 }}>
                  净值曲线与分年明细为信号版功能。{race.upgrade_hint}
                </Alert>
              )}

              {race.disclosure && race.disclosure.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.5 }}>
                    口径披露
                  </Typography>
                  <Box component="ul" sx={{ m: 0, pl: 2.5 }}>
                    {race.disclosure.map(d => (
                      <Typography key={d} component="li" variant="caption" color="text.secondary">
                        {d}
                      </Typography>
                    ))}
                  </Box>
                </Box>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* 各条线总览 */}
      <Stack direction="row" spacing={1.5} sx={{ mb: 2, flexWrap: 'wrap' }}>
        {order.map(s => {
          const l = lines[s.key];
          // 注意：不要 `?? 0` —— 取不到数据时显示 ¥0 会被读成"市值归零"，
          // 实际是"这条线没有数据"。fmtMoney(undefined) → 「暂无」。
          const value = l?.value ?? s.last_value;
          // 裸 PnL：只作为**兜底**（后端还没给 pnl_display 时用）；无市值 → null
          const pnlPct =
            typeof value === 'number' && Number.isFinite(value)
              ? ((value - cap) / cap) * 100
              : null;
          // ── 观测口径层（2026-11-02）──────────────────────────────
          // 7 条线的暴露度差 5 倍（17.5%~92.1%）→ **裸 PnL 不可比**：
          //   · 从未成交的线（0 持仓/100% 现金）裸 PnL = 0.00%，会被读成"打平"，
          //     实际它排在 4 条真亏钱的线前面；
          //   · 现金占比 82.5% 的线裸 PnL -0.05%（像打平），投入的部分实际亏 0.30%。
          // → 展示用 pnl_display（未成交 = "—"）+ 暴露度 + 投入段收益。
          const obsStatus = s.status ?? l?.line_status ?? null;
          const obsNeverBuilt = obsStatus === '未建仓';
          const obsExposure = s.exposure ?? l?.exposure ?? null;
          const obsInvRet = s.invested_return_pct ?? l?.invested_return_pct ?? null;
          const obsAsOf = s.as_of ?? l?.as_of ?? null;
          const pnlText =
            s.pnl_display ??
            l?.pnl_display ??
            (obsNeverBuilt || pnlPct === null
              ? '—'
              : `${pnlPct >= 0 ? '↑ +' : '↓ '}${pnlPct.toFixed(2)}%`);
          const stale = (() => {
            const d = daysSince(l?.date ?? s.last_signal);
            return d !== null && d > STALE_DAYS;
          })();
          const selected = active === s.key;
          return (
            <Card
              key={s.key}
              variant={selected ? 'elevation' : 'outlined'}
              onClick={() => setActive(s.key)}
              role="button"
              tabIndex={0}
              onKeyDown={e => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setActive(s.key);
                }
              }}
              sx={{
                cursor: 'pointer',
                minWidth: 210,
                // 【P1 · mobile】不限死宽度：策略描述里的长英文/数字片段
                // （如 "42 因子 z-score 先按申万一级行业去均值…"）min-content
                // 可达 480px，把卡片撑破容器 → 整页横向溢出 289px。
                maxWidth: '100%',
                flex: '0 1 210px',
                borderColor: selected ? 'primary.main' : 'divider',
                opacity: stale ? 0.72 : 1,
                // 触摸目标高度 ≥44px
                minHeight: MIN_TOUCH_TARGET,
              }}
            >
              <CardContent sx={{ py: 1.25, '&:last-child': { pb: 1.25 } }}>
                <Stack
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                  spacing={1}
                >
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    {s.key}
                  </Typography>
                  <Stack direction="row" spacing={0.5} alignItems="center">
                    {s.locked && <Lock size={13} color="#888" aria-label="需订阅" />}
                    <Chip
                      size="small"
                      color={LINE_COLORS[s.key] ?? 'primary'}
                      label={`${l?.n_positions ?? s.n_positions} 只`}
                      sx={{ height: 24 }}
                    />
                  </Stack>
                </Stack>
                <Typography variant="h6" sx={{ mt: 0.5, fontSize: '1.15rem' }}>
                  {fmtMoney(value)}
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    // 未建仓的线不给涨跌色（0.00% 不是"打平"，是"没做过"）
                    color:
                      obsNeverBuilt || pnlPct === null
                        ? 'text.secondary'
                        : pnlPct >= 0
                          ? 'success.main'
                          : 'error.main',
                    fontWeight: 600,
                  }}
                >
                  {pnlText}
                </Typography>
                {/* 观测口径层：暴露度 + 投入段收益 → 让仓位量不同的线可比 */}
                <Typography
                  variant="caption"
                  color="text.secondary"
                  // display:block —— MUI 的 caption 变体渲染成 inline <span>，
                  // 不加这行会跟下面那段策略描述**粘成一行**（实测：
                  // "数据日 2026-10-26RSI 7/14/21 超卖…"）。
                  sx={{ display: 'block', overflowWrap: 'anywhere' }}
                >
                  {obsNeverBuilt
                    ? '未建仓（从未成交 · 0.00% 不代表打平）'
                    : `暴露 ${
                        obsExposure != null ? `${(obsExposure * 100).toFixed(1)}%` : '—'
                      } · 投入段 ${obsInvRet != null ? `${obsInvRet.toFixed(2)}%` : '—'}${
                        obsAsOf ? ` · 数据日 ${obsAsOf}` : ''
                      }`}
                </Typography>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ overflowWrap: 'anywhere' }}
                >
                  {s.desc}
                </Typography>
                <Box sx={{ mt: 0.5 }}>
                  <Chip
                    size="small"
                    variant="outlined"
                    color={stale ? 'warning' : 'default'}
                    label={`调仓 ${l?.date ?? s.last_signal ?? '暂无'}${stale ? '（已过期）' : ''}`}
                    sx={{ height: 22, fontSize: 11 }}
                  />
                </Box>
              </CardContent>
            </Card>
          );
        })}
      </Stack>

      {line && (
        <>
          <Typography variant="h6" sx={{ mb: 0.5 }}>
            {line.strategy} — {line.date ?? '暂无日期'} 调仓明细
          </Typography>
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ mb: 2, overflowWrap: 'anywhere' }}
          >
            TopK={line.topk} / 每次换 {line.ndrop} 只 / 现金 {fmtMoney(line.cash)} / 成交假设：
            {line.fill_assumption ?? '暂无'}
          </Typography>

          {isStale && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              这条线的调仓日是 {line.date}（{ageDays} 天前），已超过 {STALE_DAYS} 天阈值，
              数据可能未反映最新一次调仓。
            </Alert>
          )}

          <Stack spacing={2}>
            {(
              [
                ['卖出', line.sells ?? [], 'error.main'],
                ['买入', line.buys ?? [], 'success.main'],
              ] as const
            ).map(([title, rows, color]) => (
              <Card key={title}>
                <CardContent>
                  <Typography variant="subtitle2" color={color} gutterBottom>
                    {title}（{rows.length}）
                  </Typography>
                  {rows.length === 0 ? (
                    <Typography variant="body2" color="text.secondary">
                      本次无{title}操作
                    </Typography>
                  ) : (
                    <>
                      <Box sx={{ display: { xs: 'block', md: 'none' } }}>
                        {rows.map((t: Trade) => (
                          <MobileTradeCard
                            key={`${title}-${t.code}`}
                            trade={{
                              code: t.code,
                              name: t.name,
                              price: t.price,
                              shares: t.shares,
                              amount: t.amount,
                              fee: feeOf(t),
                              score: t.score,
                              factors: t.factors,
                              side: title === '买入' ? 'buy' : 'sell',
                            }}
                          />
                        ))}
                      </Box>
                      <Box sx={{ display: { xs: 'none', md: 'block' } }}>
                        <ResponsiveTable minWidth={720}>
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>代码</TableCell>
                                <TableCell>名称</TableCell>
                                <TableCell align="right">价格</TableCell>
                                <TableCell align="right">股数</TableCell>
                                <TableCell align="right">金额</TableCell>
                                <TableCell align="right">费用</TableCell>
                                <TableCell align="right">分数</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {rows.map((t: Trade) => (
                                <TableRow key={`${title}-${t.code}`}>
                                  <TableCell sx={{ fontFamily: 'monospace' }}>{t.code}</TableCell>
                                  <TableCell>{t.name}</TableCell>
                                  <TableCell align="right">
                                    {t.price != null ? t.price.toFixed(2) : '暂无'}
                                  </TableCell>
                                  <TableCell align="right">
                                    {t.shares?.toLocaleString() ?? '暂无'}
                                  </TableCell>
                                  <TableCell align="right">{fmtMoney(t.amount)}</TableCell>
                                  <TableCell align="right">{feeOf(t).toFixed(2)}</TableCell>
                                  <TableCell align="right">{fmtScore(t.score)}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </ResponsiveTable>
                      </Box>
                    </>
                  )}
                </CardContent>
              </Card>
            ))}

            {activeInfo && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                {activeInfo.display_name} —— {activeInfo.factor_note}
              </Typography>
            )}
          </Stack>
        </>
      )}

      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 3 }}>
        数据仅供研究，不构成投资建议。纸面成交不保证实盘可复现。
      </Typography>

      <Box
        sx={{ height: BOTTOM_NAV_CONTENT_PADDING, display: { xs: 'block', md: 'none' } }}
        aria-hidden="true"
      />
    </Box>
  );
}
