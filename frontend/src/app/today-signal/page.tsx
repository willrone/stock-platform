'use client';

/**
 * 今日信号 —— 今天买哪几只、卖哪几只、为什么
 *
 * 2026-10-25 新增。这是付费用户真正买的东西。
 *
 * 数据来源：GET /api/v1/trading/signal?strategy=xxx
 *   信号在每个交易日收盘后由平台自动生成（约 16:10），
 *   买入建议按下一交易日可成交价执行。
 *
 * ⚠️ 口径声明（2026-10-27 修正）
 *   1. daily_signal 端点**不返回** sharpe_ratio / max_drawdown_pct / nav_points，
 *      风险指标只在 /trading/performance 里 → 本页合并读取两个端点。
 *   2. score（分数）与 factors（因子归因）后端在 rebuild_line_from_files()
 *      重建 portfolio.json 时**丢掉了**（ledger.csv 里有 score 但没被读出来），
 *      所以本页从 /trading/performance 的 recent_trades（ledger 行）里
 *      按「代码 + 成交日」回补 score；factors 后端暂未落盘 → 显示「暂无」。
 *   3. 信号是研究口径的**选股结果**，不含券商下单、资金风控与自动执行。
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  Grid,
  MenuItem,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from '@mui/material';
import {
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Info,
  Wallet,
  Target,
  ShieldAlert,
  LineChart,
} from 'lucide-react';
import { ErrorState } from '@/components/mobile/ErrorState';
import DailyNavChart, { type DailyNavData } from './DailyNavChart';
import { MobileTradeCard } from '@/components/mobile/MobileTradeCard';
import { ResponsiveTable } from '@/components/mobile/ResponsiveTable';
import { DeltaValue } from '@/components/mobile/DeltaValue';
import { MobileBottomNav } from '@/components/mobile/MobileBottomNav';
import { BOTTOM_NAV_CONTENT_PADDING, MIN_TOUCH_TARGET } from '@/components/mobile/safeArea';
import { SafeAreaTop } from '@/components/mobile/SafeAreaTop';
import { tradingGet, daysSince } from '@/components/mobile/tradingClient';
import { trackEventOnce } from '@/services/funnel';
import { primaryColors } from '@/theme/tokens';

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
  limit_pct?: number;
  /** 因子归因：为什么买它（42f 专用） */
  factors?: { factor: string; contribution: number; direction: string }[];
};

type Hold = {
  code: string;
  name?: string;
  shares: number;
  cost: number;
  cost_amount: number;
  buy_date: string;
};

type DailySignal = {
  strategy: string;
  display_name: string;
  desc: string;
  factor_note: string;
  status: string;
  message?: string;
  date?: string;
  topk: number;
  ndrop: number;
  cash: number;
  value: number;
  capital: number;
  total_return_pct: number;
  sharpe_ratio?: number | null;
  max_drawdown_pct?: number | null;
  trade_win_rate_pct?: number | null;
  nav_points?: number;
  nav_basis?: string;
  nav_points_required?: number;
  nav_insufficient?: boolean;
  nav_start?: string | null;
  nav_end?: string | null;
  n_positions: number;
  buys: Trade[];
  sells: Trade[];
  holds: Hold[];
  cost_model?: {
    commission: number;
    stamp_tax: number;
    slippage: number;
    cash_reserve: number;
    board_lot: number;
  };
  fill_assumption?: string;
  tier?: string;
  tier_label?: string;
  delay_days?: number;
  delayed?: boolean;
  upgrade_hint?: string;
  execution_detail_locked?: boolean;
  blocked_buys?: { code: string; name?: string; reason: string }[];
  blocked_sells?: { code: string; name?: string; reason: string }[];
};

type Performance = {
  status?: string;
  date?: string | null;
  sharpe_ratio?: number | null;
  max_drawdown_pct?: number | null;
  trade_win_rate_pct?: number | null;
  nav_points?: number;
  nav_basis?: string;
  nav_points_required?: number;
  nav_insufficient?: boolean;
  nav_start?: string | null;
  nav_end?: string | null;
  n_trades?: number;
  locked_detail?: boolean;
  /** 付费墙升级提示（免费档位命中时后端会带上） */
  upgrade_hint?: string | null;
  recent_trades?: Array<Record<string, string | number | undefined>>;
  /**
   * 日频盯市净值点（total_return@trade_day 口径，~12 点 / 1 月）。
   * ⚠️ 与周频 `nav_basis`（调仓日净值点）、以及 paper-portfolio 页的
   * 「月频复利还原」是**三套互不相同的口径**，不可混画。
   */
  nav_daily?: DailyNavData;
};

type SignalQuality = {
  status: string;
  message?: string;
  n_signals?: number;
  horizons?: Record<
    string,
    {
      n: number;
      win_rate: number;
      avg_excess: number;
      median_excess: number;
      worst: number;
      best: number;
    }
  >;
  recent_10?: {
    date: string;
    code: string;
    name?: string;
    excess_T1?: number | null;
    excess_T5?: number | null;
    excess_T10?: number | null;
    excess_T20?: number | null;
  }[];
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
};

type StrategyListResponse = {
  strategies: StrategyInfo[];
  tier: string;
  tier_label: string;
  allowed_strategies: string[];
  delay_days: number;
  include_execution_detail: boolean;
};

const fmtMoney = (v?: number) =>
  v === undefined || v === null || !Number.isFinite(v)
    ? '暂无'
    : `¥${v.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

const fmtShares = (v?: number) =>
  v === undefined || v === null || !Number.isFinite(v) ? '暂无' : v.toLocaleString('zh-CN');

/** 分数格式化：没有就明确说「暂无」，不再用「—」冒充 0 */
const fmtScore = (v?: number | string | null) => {
  if (v === undefined || v === null || v === '') {
    return '暂无';
  }
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(3) : String(v);
};

const feeOf = (t: Trade) => (t.commission ?? 0) + (t.stamp_tax ?? 0) + (t.slippage ?? 0);

export default function TodaySignalPage() {
  const router = useRouter();
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [active, setActive] = useState('honest_42f');
  const [signal, setSignal] = useState<DailySignal | null>(null);
  const [perf, setPerf] = useState<Performance | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [errorDetail, setErrorDetail] = useState<unknown>(null);
  // ⚠️ tier 初始为 null：加载失败时**不渲染**档位与付费墙，
  // 否则 enterprise 用户会被误判成「体验版」并被推去付费墙。
  const [tier, setTier] = useState<string | null>(null);
  const [tierLabel, setTierLabel] = useState<string | null>(null);
  const [delayDays, setDelayDays] = useState(0);
  const [hasDetail, setHasDetail] = useState(true);
  const [quality, setQuality] = useState<SignalQuality | null>(null);
  const [strategiesLoaded, setStrategiesLoaded] = useState(false);

  // ── 策略列表 ──────────────────────────────────────────────
  const loadStrategies = useCallback(async () => {
    try {
      const res = await tradingGet<StrategyListResponse>('/trading/strategies');
      setStrategies(res.strategies || []);
      setTier(res.tier ?? null);
      setTierLabel(res.tier_label ?? null);
      setDelayDays(res.delay_days ?? 0);
      setHasDetail(res.include_execution_detail !== false);
      setStrategiesLoaded(true);
      const usable = (res.strategies || []).filter(s => !s.locked);
      if (usable.length) {
        const withPaper = usable.find(s => s.n_positions > 0) || usable[0];
        setActive(withPaper.key);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载策略列表失败');
      setErrorDetail((e as { detail?: unknown })?.detail ?? e);
      setStrategiesLoaded(false);
    }
  }, []);

  useEffect(() => {
    const token = window.localStorage.getItem('access_token');
    if (!token) {
      router.replace('/login');
      return;
    }
    void loadStrategies();
  }, [router, loadStrategies]);

  // ── 今日信号 + 业绩（风险指标在这里）────────────────────────
  const loadSignal = useCallback(async () => {
    if (!active) {
      return;
    }
    setLoading(true);
    setError(null);
    setErrorDetail(null);
    try {
      const res = await tradingGet<DailySignal>(
        `/trading/signal?strategy=${encodeURIComponent(active)}`
      );
      setSignal(res);
      // 漏斗埋点：用户真的看到了一条信号（不是只进了页面）
      trackEventOnce('view_signal', { source: 'today-signal', strategy: active });
      // 档位以 /trading/signal 返回的为准（与 strategies 端点同源）
      if (res.tier !== undefined) {
        setTier(res.tier);
      }
      if (res.tier_label !== undefined) {
        setTierLabel(res.tier_label);
      }
      if (res.delay_days !== undefined) {
        setDelayDays(res.delay_days);
      }

      // 风险指标（夏普 / 回撤 / 净值点）只在 performance 端点 → 并行读
      try {
        const p = await tradingGet<Performance>(
          `/trading/performance?strategy=${encodeURIComponent(active)}`
        );
        setPerf(p);
      } catch {
        setPerf(null);
      }

      // 信号质量（付费墙可能 403，失败就静默）
      try {
        const q = await tradingGet<SignalQuality>(
          `/trading/signal-quality?strategy=${encodeURIComponent(active)}`
        );
        setQuality(q);
      } catch {
        setQuality(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载信号失败');
      setErrorDetail((e as { detail?: unknown })?.detail ?? e);
      setSignal(null);
      setPerf(null);
    } finally {
      setLoading(false);
    }
  }, [active]);

  useEffect(() => {
    void loadSignal();
  }, [loadSignal]);

  // ── score 回补：ledger 行（performance.recent_trades）里有 score ──
  const scoreByCode = useMemo(() => {
    const map = new Map<string, number>();
    const rows = perf?.recent_trades ?? [];
    const signalDate = signal?.date;
    for (const row of rows) {
      const code = String(row.code ?? '');
      const score = Number(row.score);
      if (!code || !Number.isFinite(score)) {
        continue;
      }
      const rowDate = String(row.date ?? '');
      // 优先匹配本次成交日的记录，否则取任何一条
      if (signalDate && rowDate === signalDate) {
        map.set(code, score);
      } else if (!map.has(code)) {
        map.set(code, score);
      }
    }
    return map;
  }, [perf, signal?.date]);

  const resolveScore = useCallback(
    (t: Trade): number | string | null => {
      if (t.score !== undefined && t.score !== null && t.score !== '') {
        return t.score;
      }
      const fromLedger = scoreByCode.get(t.code);
      return fromLedger !== undefined ? fromLedger : null;
    },
    [scoreByCode]
  );

  const totalBuyAmount = (signal?.buys ?? []).reduce((s, b) => s + (b.amount ?? 0), 0);
  const totalSellAmount = (signal?.sells ?? []).reduce((s, b) => s + (b.amount ?? 0), 0);
  const totalCost = (signal?.buys ?? []).reduce((s, b) => s + feeOf(b), 0);

  // 风险指标统一从 performance 读（signal 端点不返回）
  const sharpe = perf?.sharpe_ratio ?? null;
  const maxDd = perf?.max_drawdown_pct ?? null;
  const winRate = perf?.trade_win_rate_pct ?? null;
  const navPoints = perf?.nav_points ?? null;
  const navRequired = perf?.nav_points_required ?? 20;
  const navInsufficient = perf?.nav_insufficient ?? (navPoints !== null && navPoints < navRequired);
  const navBasis = perf?.nav_basis ?? null;

  // 信号日期过期守卫
  const signalAgeDays = daysSince(signal?.date);
  const isStale = signalAgeDays !== null && signalAgeDays > 7;

  const navCaption = navInsufficient
    ? `净值样本不足（${navPoints ?? 0}/${navRequired} 点）`
    : navBasis
      ? `${navBasis} ${navPoints ?? 0} 点`
      : navPoints !== null
        ? `${navPoints} 点`
        : '暂无净值样本';

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2, md: 3 }, pb: { xs: 10, md: 3 } }}>
      <SafeAreaTop />
      <MobileBottomNav />

      <Stack direction="row" justifyContent="space-between" alignItems="flex-start" sx={{ mb: 2 }}>
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="h5">今日信号</Typography>
          <Typography variant="body2" color="text.secondary">
            收盘后自动生成，含股数 / 金额 / 成本
          </Typography>
        </Box>
        <Button
          startIcon={<RefreshCw size={16} />}
          onClick={() => void loadSignal()}
          disabled={loading}
          sx={{ minHeight: MIN_TOUCH_TARGET, flexShrink: 0 }}
        >
          刷新
        </Button>
      </Stack>

      {/* 口径披露 —— 不美化数据，明确说明这不等于「照着下单就能赚钱」 */}
      <Alert severity="warning" icon={<ShieldAlert size={18} />} sx={{ mb: 2 }}>
        <Typography variant="body2" fontWeight={700}>
          当前口径与已知限制
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          信号为<strong>研究口径的选股结果</strong>：给出参考价、股数、金额与估算费用，但
          <strong>不含券商下单、资金风控与自动执行</strong>。实盘执行需自行核对成本与仓位 ——
          实际成交价会因开盘跳空、涨跌停、停牌与流动性上限而偏离参考价。
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>
          风险指标（夏普 / 最大回撤）来自<strong>周频调仓的纸面账本</strong>，回撤会被低估，
          且样本点不足 20 个时后端不返回该指标 —— 这两个数为空是刻意的，不是 bug。
        </Typography>
      </Alert>

      {error && (
        <ErrorState
          title="今日信号加载失败"
          message={error}
          detail={errorDetail}
          onRetry={() => void loadSignal()}
          retrying={loading}
        />
      )}

      {/* 订阅状态条 —— 仅在档位成功加载后渲染，避免把付费用户误判为体验版 */}
      {strategiesLoaded && tier !== null && (
        <Stack direction="row" spacing={1.5} sx={{ mb: 2, alignItems: 'center', flexWrap: 'wrap' }}>
          <Chip
            size="small"
            color={tier === 'free' ? 'default' : 'primary'}
            label={tierLabel ?? tier}
            sx={{ height: 28 }}
          />
          {delayDays > 0 && (
            <Chip
              size="small"
              color="warning"
              variant="outlined"
              label={`信号延迟 ${delayDays} 个交易日`}
              sx={{ height: 28 }}
            />
          )}
          {tier === 'free' && (
            <Button
              size="small"
              variant="contained"
              onClick={() => router.push('/pricing')}
              sx={{ ml: 'auto !important', minHeight: MIN_TOUCH_TARGET }}
            >
              升级解锁全部信号线
            </Button>
          )}
        </Stack>
      )}

      {signal?.upgrade_hint && tier === 'free' && (
        <Alert severity="info" sx={{ mb: 2 }}>
          {signal.upgrade_hint}
        </Alert>
      )}

      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        spacing={1.5}
        sx={{ mb: 2, alignItems: { sm: 'center' } }}
      >
        <Typography variant="body2" color="text.secondary" sx={{ flexShrink: 0 }}>
          策略
        </Typography>
        <Select
          size="small"
          value={active}
          onChange={e => setActive(e.target.value)}
          sx={{ minWidth: { xs: '100%', sm: 260 }, minHeight: MIN_TOUCH_TARGET }}
        >
          {strategies.map(s => (
            <MenuItem key={s.key} value={s.key} disabled={s.locked}>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ width: '100%' }}>
                <span>
                  {s.display_name}
                  {!s.locked && s.n_positions > 0 ? `（${s.n_positions} 只持仓）` : ''}
                </span>
                {s.locked && (
                  <Chip size="small" label="需信号版" color="warning" variant="outlined" />
                )}
              </Stack>
            </MenuItem>
          ))}
        </Select>
        {signal?.date && (
          <Chip
            size="small"
            color={isStale ? 'warning' : 'primary'}
            label={isStale ? `信号日期 ${signal.date}（已过期）` : `信号日期 ${signal.date}`}
            sx={{ height: 28 }}
          />
        )}
      </Stack>

      {isStale && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          这份信号已生成 {signalAgeDays} 天，不是今天的信号。平台在每个交易日收盘后（约 16:10）
          自动重算；如需当日信号请先确认后台调仓任务是否已运行。
        </Alert>
      )}

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
          <CircularProgress />
        </Box>
      )}

      {!loading && signal?.status === 'no_data' && <Alert severity="info">{signal.message}</Alert>}

      {!loading && !error && signal?.status === 'ok' && (
        <>
          <Grid container spacing={1.5} sx={{ mb: 2 }}>
            <Grid size={{ xs: 6, sm: 3 }}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                  <Typography variant="caption" color="text.secondary">
                    组合总值
                  </Typography>
                  <Typography variant="h6" sx={{ fontSize: { xs: '1.05rem', sm: '1.25rem' } }}>
                    {fmtMoney(signal.value)}
                  </Typography>
                  <DeltaValue value={signal.total_return_pct} variant="body2" />
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 6, sm: 3 }}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                  <Typography variant="caption" color="text.secondary">
                    最大回撤
                  </Typography>
                  <Typography
                    variant="h6"
                    sx={{ fontSize: { xs: '1.05rem', sm: '1.25rem' } }}
                    color={maxDd != null && maxDd < -20 ? 'error.main' : 'text.primary'}
                  >
                    {maxDd != null ? `${maxDd.toFixed(2)}%` : '暂无'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {navCaption}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    周频调仓口径，回撤会被低估 —— 以日频盯市曲线为准
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 6, sm: 3 }}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                  <Typography variant="caption" color="text.secondary">
                    年化夏普
                  </Typography>
                  <Typography variant="h6" sx={{ fontSize: { xs: '1.05rem', sm: '1.25rem' } }}>
                    {sharpe != null ? sharpe.toFixed(2) : '暂无'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {winRate != null ? `逐笔胜率 ${winRate.toFixed(0)}%` : '逐笔胜率待积累'}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 6, sm: 3 }}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                  <Typography variant="caption" color="text.secondary">
                    持仓
                  </Typography>
                  <Typography variant="h6" sx={{ fontSize: { xs: '1.05rem', sm: '1.25rem' } }}>
                    {signal.n_positions} 只
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    现金 {fmtMoney(signal.cash)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 6, sm: 3 }}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                  <Typography variant="caption" color="text.secondary">
                    本次操作
                  </Typography>
                  <Typography variant="h6" sx={{ fontSize: { xs: '1.05rem', sm: '1.25rem' } }}>
                    买 {signal.buys.length} · 卖 {signal.sells.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    买 {fmtMoney(totalBuyAmount)} / 卖 {fmtMoney(totalSellAmount)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 6, sm: 3 }}>
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                  <Typography variant="caption" color="text.secondary">
                    预计成本
                  </Typography>
                  <Typography variant="h6" sx={{ fontSize: { xs: '1.05rem', sm: '1.25rem' } }}>
                    {fmtMoney(totalCost)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    含佣金与滑点
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Alert severity="info" icon={<Info size={16} />} sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>{signal.display_name}</strong> —— {signal.desc}
            </Typography>
            <Typography variant="body2" sx={{ mt: 0.5 }}>
              {signal.factor_note} · 持仓上限 {signal.topk} 只 · 每次最多换 {signal.ndrop} 只
              {signal.fill_assumption ? ` · ${signal.fill_assumption}` : ''}
            </Typography>
          </Alert>

          {/* ── 日频盯市净值曲线（口径③）────────────────────────────
           * ⚠️ 与上方卡片里的周频「调仓日净值点」（口径②）、以及
           *    paper-portfolio 页的「月频复利还原」（口径①）是三套不同口径，
           *    绝不同框、不可直接对比。见 DailyNavChart 组件头注释。
           */}
          <Card sx={{ mb: 2 }} data-testid="daily-nav-card">
            <CardContent>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                <LineChart size={18} color={primaryColors.DEFAULT} />
                <Typography variant="subtitle1">日频盯市净值</Typography>
                <Chip
                  size="small"
                  variant="outlined"
                  label="每个交易日盯市"
                  sx={{ height: 22, fontSize: 11 }}
                />
              </Stack>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                纸面盘组合在<strong>每个交易日收盘后按总回报口径盯市</strong>算出的净值点。
                与上方「最大回撤 / 年化夏普」卡片使用的周频调仓日净值点
                <strong>不是同一口径</strong> —— 周频会漏掉持仓期内的回撤，回撤以本曲线为准。
              </Typography>
              <DailyNavChart
                data={perf?.nav_daily}
                perfLoaded={perf !== null}
                locked={perf?.locked_detail === true}
                upgradeHint={perf?.upgrade_hint}
                // 仅开发态开启「暂无日频数据」降级分支的前端级演示开关。
                // 生产构建里 process.env.NODE_ENV !== 'development' → 该 prop 恒为 false，
                // 且组件内部还会再判一次 isDev，页面上不会出现任何预览痕迹。
                previewUnavailable={process.env.NODE_ENV === 'development'}
              />
            </CardContent>
          </Card>

          {/* 信号质量：凭什么信这个信号 */}
          {quality && quality.status === 'ok' && quality.horizons && (
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                  <Target size={18} color="#6a1b9a" />
                  <Typography variant="subtitle1">信号质量</Typography>
                  <Typography variant="caption" color="text.secondary">
                    历史上 {quality.n_signals} 个买入信号发出后的实际表现（相对全市场等权）
                  </Typography>
                </Stack>
                <Grid container spacing={1.5}>
                  {Object.entries(quality.horizons).map(([h, v]) => (
                    <Grid size={{ xs: 6, sm: 3 }} key={h}>
                      <Box sx={{ p: 1.5, border: 1, borderColor: 'divider', borderRadius: 1.5 }}>
                        <Typography variant="caption" color="text.secondary">
                          {h} 超额
                        </Typography>
                        <DeltaValue value={v.avg_excess * 100} variant="h6" />
                        <Typography variant="caption" color="text.secondary">
                          胜率 {(v.win_rate * 100).toFixed(0)}% · {v.n} 次
                        </Typography>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ mt: 1, display: 'block' }}
                >
                  超额 = 该信号股票收益 − 同期全市场等权收益。中位{' '}
                  {Object.values(quality.horizons)[0]
                    ? (Object.values(quality.horizons)[0].median_excess * 100).toFixed(2)
                    : '0.00'}
                  % ，最差{' '}
                  {Object.values(quality.horizons)[0]
                    ? (Object.values(quality.horizons)[0].worst * 100).toFixed(1)
                    : '0.0'}
                  %
                </Typography>
              </CardContent>
            </Card>
          )}

          {/* 因可成交性未成交的信号 */}
          {((signal?.blocked_buys?.length ?? 0) > 0 ||
            (signal?.blocked_sells?.length ?? 0) > 0) && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              <Typography variant="body2" sx={{ fontWeight: 700, mb: 0.5 }}>
                有信号没成交（实盘同样买不进/卖不出）
              </Typography>
              {(signal?.blocked_buys ?? []).slice(0, 6).map(b => (
                <Typography variant="caption" key={`b${b.code}`} sx={{ display: 'block' }}>
                  买不进 {b.code} {b.name} —— {b.reason}
                </Typography>
              ))}
              {(signal?.blocked_sells ?? []).slice(0, 6).map(b => (
                <Typography variant="caption" key={`s${b.code}`} sx={{ display: 'block' }}>
                  卖不出 {b.code} {b.name} —— {b.reason}
                </Typography>
              ))}
            </Alert>
          )}

          <Stack spacing={2}>
            {/* ── 买入 ── */}
            <Card>
              <CardContent>
                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                  <TrendingUp size={18} color="#2e7d32" />
                  <Typography variant="subtitle1">买入（{signal.buys.length}）</Typography>
                </Stack>
                {signal.buys.length === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    本次无需买入 —— 持仓未掉出前列，等下次调仓
                  </Typography>
                ) : (
                  <>
                    {/* 移动端：卡片 */}
                    <Box sx={{ display: { xs: 'block', md: 'none' } }}>
                      {signal.buys.map(b => (
                        <MobileTradeCard
                          key={b.code}
                          trade={{
                            code: b.code,
                            name: b.name,
                            price: b.price,
                            shares: b.shares,
                            amount: b.amount,
                            fee: feeOf(b),
                            score: resolveScore(b),
                            factors: b.factors,
                            side: 'buy',
                          }}
                        />
                      ))}
                    </Box>
                    {/* 桌面端：表格（可横向滚动） */}
                    <Box sx={{ display: { xs: 'none', md: 'block' } }}>
                      <ResponsiveTable minWidth={720}>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>代码</TableCell>
                              <TableCell>名称</TableCell>
                              <TableCell align="right">参考价</TableCell>
                              {hasDetail && <TableCell align="right">股数</TableCell>}
                              {hasDetail && <TableCell align="right">金额</TableCell>}
                              {hasDetail && <TableCell align="right">费用</TableCell>}
                              <TableCell align="right">分数</TableCell>
                              <TableCell>为什么买它</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {signal.buys.map(b => (
                              <TableRow key={b.code}>
                                <TableCell sx={{ fontFamily: 'monospace' }}>{b.code}</TableCell>
                                <TableCell>{b.name}</TableCell>
                                <TableCell align="right">
                                  {b.price != null ? b.price.toFixed(2) : '暂无'}
                                </TableCell>
                                {hasDetail && (
                                  <TableCell align="right">{fmtShares(b.shares)}</TableCell>
                                )}
                                {hasDetail && (
                                  <TableCell align="right">{fmtMoney(b.amount)}</TableCell>
                                )}
                                {hasDetail && (
                                  <TableCell align="right">{feeOf(b).toFixed(2)}</TableCell>
                                )}
                                <TableCell align="right">{fmtScore(resolveScore(b))}</TableCell>
                                <TableCell sx={{ maxWidth: 320 }}>
                                  <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                                    {(b.factors ?? []).map(f => (
                                      <Tooltip
                                        key={f.factor}
                                        title={`${f.factor} 贡献 ${f.contribution > 0 ? '+' : ''}${
                                          f.contribution
                                        }`}
                                      >
                                        <Chip
                                          size="small"
                                          label={f.factor}
                                          variant="outlined"
                                          sx={{
                                            fontSize: 11,
                                            color:
                                              f.contribution > 0 ? 'success.main' : 'error.main',
                                            borderColor:
                                              f.contribution > 0 ? 'success.light' : 'error.light',
                                          }}
                                        />
                                      </Tooltip>
                                    ))}
                                    {(!b.factors || b.factors.length === 0) && (
                                      <Typography variant="caption" color="text.secondary">
                                        暂无
                                      </Typography>
                                    )}
                                  </Stack>
                                </TableCell>
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

            {/* ── 卖出 ── */}
            <Card>
              <CardContent>
                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                  <TrendingDown size={18} color="#c62828" />
                  <Typography variant="subtitle1">卖出（{signal.sells.length}）</Typography>
                </Stack>
                {signal.sells.length === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    本次无需卖出
                  </Typography>
                ) : (
                  <>
                    <Box sx={{ display: { xs: 'block', md: 'none' } }}>
                      {signal.sells.map(s => (
                        <MobileTradeCard
                          key={s.code}
                          trade={{
                            code: s.code,
                            name: s.name,
                            price: s.price,
                            shares: s.shares,
                            amount: s.amount,
                            fee: feeOf(s),
                            score: resolveScore(s),
                            factors: s.factors,
                            side: 'sell',
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
                              <TableCell align="right">参考价</TableCell>
                              {hasDetail && <TableCell align="right">股数</TableCell>}
                              {hasDetail && <TableCell align="right">金额</TableCell>}
                              {hasDetail && <TableCell align="right">费用</TableCell>}
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {signal.sells.map(s => (
                              <TableRow key={s.code}>
                                <TableCell sx={{ fontFamily: 'monospace' }}>{s.code}</TableCell>
                                <TableCell>{s.name}</TableCell>
                                <TableCell align="right">
                                  {s.price != null ? s.price.toFixed(2) : '暂无'}
                                </TableCell>
                                {hasDetail && (
                                  <TableCell align="right">{fmtShares(s.shares)}</TableCell>
                                )}
                                {hasDetail && (
                                  <TableCell align="right">{fmtMoney(s.amount)}</TableCell>
                                )}
                                {hasDetail && (
                                  <TableCell align="right">{feeOf(s).toFixed(2)}</TableCell>
                                )}
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

            {/* ── 当前持仓 ── */}
            <Card>
              <CardContent>
                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                  <Wallet size={18} color="#1565c0" />
                  <Typography variant="subtitle1">当前持仓（{signal.holds.length}）</Typography>
                </Stack>
                <Box sx={{ display: { xs: 'block', md: 'none' } }}>
                  {signal.holds.map(h => (
                    <MobileTradeCard
                      key={h.code}
                      trade={{
                        code: h.code,
                        name: h.name,
                        shares: h.shares,
                        cost: h.cost,
                        costAmount: h.cost_amount,
                        buyDate: h.buy_date,
                        side: 'hold',
                      }}
                    />
                  ))}
                </Box>
                <Box sx={{ display: { xs: 'none', md: 'block' } }}>
                  <ResponsiveTable minWidth={640}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>代码</TableCell>
                          <TableCell>名称</TableCell>
                          {hasDetail && <TableCell align="right">股数</TableCell>}
                          <TableCell align="right">成本价</TableCell>
                          {hasDetail && <TableCell align="right">成本金额</TableCell>}
                          <TableCell align="right">买入日</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {signal.holds.map(h => (
                          <TableRow key={h.code}>
                            <TableCell sx={{ fontFamily: 'monospace' }}>{h.code}</TableCell>
                            <TableCell>{h.name}</TableCell>
                            {hasDetail && (
                              <TableCell align="right">{fmtShares(h.shares)}</TableCell>
                            )}
                            <TableCell align="right">
                              {h.cost != null ? h.cost.toFixed(2) : '暂无'}
                            </TableCell>
                            {hasDetail && (
                              <TableCell align="right">{fmtMoney(h.cost_amount)}</TableCell>
                            )}
                            <TableCell align="right">{h.buy_date || '暂无'}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </ResponsiveTable>
                </Box>
              </CardContent>
            </Card>
          </Stack>

          {signal.cost_model && (
            <>
              <Divider sx={{ my: 2 }} />
              <Typography variant="caption" color="text.secondary">
                成本模型：佣金 {(signal.cost_model.commission * 10000).toFixed(1)}‱ · 印花税{' '}
                {(signal.cost_model.stamp_tax * 10000).toFixed(1)}‱（卖出） · 滑点{' '}
                {(signal.cost_model.slippage * 1000).toFixed(1)}‰（单边） · 现金预留{' '}
                {(signal.cost_model.cash_reserve * 100).toFixed(0)}% · {signal.cost_model.board_lot}{' '}
                股整手
              </Typography>
            </>
          )}
        </>
      )}

      {/* 主内容底部留白，避免最后一行被固定底栏盖住 */}
      <Box
        sx={{ height: BOTTOM_NAV_CONTENT_PADDING, display: { xs: 'block', md: 'none' } }}
        aria-hidden="true"
      />
    </Box>
  );
}
