'use client';

/**
 * 纸面信号（22f 对照线）
 *
 * ⚠️ 2026-10-27 修正
 *   原来这里 `fetch('/paper-signal.json')` 读 public/ 下的静态文件，问题有两个：
 *     1. **陈旧** —— 实测该文件是 2026-10-16 的快照，与「今日信号」页口径打架；
 *     2. **付费内容泄露** —— public/ 目录无需鉴权即可下载（任何人 curl 一下
 *        就能拿到买卖清单），等于把付费信号免费公开。
 *   现在改为读**受保护的实时端点** GET /api/v1/trading/signal?strategy=22f
 *   （与「今日信号」页同源、同一口径），并加数据过期守卫。
 *
 *   👉 仍需后端 / 运维处理：把 frontend/public/paper-signal.json 与
 *      frontend/public/paper-portfolio.json **移出 public 目录**
 *      （这两个文件由他人负责删除，不在本任务的可改文件清单内）。
 */

import React, { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import { RefreshCw, ShieldAlert } from 'lucide-react';
import { ErrorState } from '@/components/mobile/ErrorState';
import { MobileTradeCard } from '@/components/mobile/MobileTradeCard';
import { ResponsiveTable } from '@/components/mobile/ResponsiveTable';
import { MobileBottomNav } from '@/components/mobile/MobileBottomNav';
import { SafeAreaTop } from '@/components/mobile/SafeAreaTop';
import { BOTTOM_NAV_CONTENT_PADDING, MIN_TOUCH_TARGET } from '@/components/mobile/safeArea';
import { tradingGet, daysSince } from '@/components/mobile/tradingClient';

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
  n_positions: number;
  buys: Trade[];
  sells: Trade[];
  holds: Array<{ code: string; name?: string; shares: number; cost: number; cost_amount: number; buy_date: string }>;
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
  upgrade_hint?: string;
};

const STRATEGY = '22f';
/** 数据超过这个天数就视为过期 */
const STALE_DAYS = 7;

const money = (v?: number) =>
  typeof v === 'number' && Number.isFinite(v)
    ? `¥${v.toLocaleString('zh-CN', { maximumFractionDigits: 0 })}`
    : '暂无';

const feeOf = (t: Trade) => (t.commission ?? 0) + (t.stamp_tax ?? 0) + (t.slippage ?? 0);

const fmtScore = (v?: number | string | null) => {
  if (v === undefined || v === null || v === '') {
    return '暂无';
  }
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(3) : String(v);
};

const TradeTable: React.FC<{ rows: Trade[]; side: 'buy' | 'sell' }> = ({ rows, side }) => {
  if (!rows.length) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
        无
      </Typography>
    );
  }
  return (
    <>
      {/* 移动端：卡片 */}
      <Box sx={{ display: { xs: 'block', md: 'none' } }}>
        {rows.map((r) => (
          <MobileTradeCard
            key={`${side}-${r.code}`}
            trade={{
              code: r.code,
              name: r.name,
              price: r.price,
              shares: r.shares,
              amount: r.amount,
              fee: feeOf(r),
              score: r.score,
              factors: r.factors,
              side,
            }}
          />
        ))}
      </Box>
      {/* 桌面端：表格 */}
      <Box sx={{ display: { xs: 'none', md: 'block' } }}>
        <ResponsiveTable minWidth={680}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>代码</TableCell>
                <TableCell>名称</TableCell>
                <TableCell align="right">股数</TableCell>
                <TableCell align="right">价格</TableCell>
                <TableCell align="right">金额</TableCell>
                <TableCell align="right">{side === 'buy' ? '成本' : '净得'}</TableCell>
                <TableCell align="right">分数</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((r) => (
                <TableRow key={`${side}-${r.code}`} hover>
                  <TableCell sx={{ fontFamily: 'ui-monospace, Menlo, monospace' }}>{r.code}</TableCell>
                  <TableCell>{r.name || '—'}</TableCell>
                  <TableCell align="right">{r.shares?.toLocaleString?.() ?? r.shares ?? '暂无'}</TableCell>
                  <TableCell align="right">
                    {r.price != null ? r.price.toFixed(2) : '暂无'}
                  </TableCell>
                  <TableCell align="right">{money(r.amount)}</TableCell>
                  <TableCell align="right">
                    {money(side === 'buy' ? r.total_cost : r.net)}
                  </TableCell>
                  <TableCell align="right">{fmtScore(r.score)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </ResponsiveTable>
      </Box>
    </>
  );
};

export default function PaperSignalPage() {
  const [data, setData] = useState<DailySignal | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorDetail, setErrorDetail] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  const load = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    setErrorDetail(null);
    try {
      const res = await tradingGet<DailySignal>(
        `/trading/signal?strategy=${encodeURIComponent(STRATEGY)}`
      );
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : '纸面信号读取失败');
      setErrorDetail((e as { detail?: unknown })?.detail ?? e);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const stats = useMemo(() => {
    if (!data) {
      return [];
    }
    return [
      { label: '信号日期', value: data.date || '暂无' },
      { label: '组合总值', value: money(data.value) },
      { label: '现金', value: money(data.cash) },
      { label: '持仓数', value: String(data.n_positions ?? '暂无') },
      {
        label: '买 / 卖 / 持',
        value: `${data.buys?.length ?? 0} / ${data.sells?.length ?? 0} / ${data.holds?.length ?? 0}`,
      },
    ];
  }, [data]);

  const ageDays = daysSince(data?.date);
  const isStale = ageDays !== null && ageDays > STALE_DAYS;

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2, md: 3 } }}>
      <SafeAreaTop />
      <MobileBottomNav />

      <Stack direction="row" justifyContent="space-between" alignItems="flex-start" sx={{ mb: 0.5 }}>
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="h5" fontWeight={700}>
            纸面信号
          </Typography>
          <Stack direction="row" spacing={1} sx={{ mt: 0.5 }} flexWrap="wrap" useFlexGap>
            <Chip label="22f · 周频" color="primary" size="small" sx={{ height: 26 }} />
            <Chip label="虚拟盘" variant="outlined" size="small" sx={{ height: 26 }} />
            {data?.date && (
              <Chip
                label={isStale ? `${data.date}（已过期）` : data.date}
                color={isStale ? 'warning' : 'default'}
                size="small"
                sx={{ height: 26 }}
              />
            )}
          </Stack>
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

      {data && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          topk={data.topk} · 每次换 {data.ndrop} 只 · 成交假设：
          {data.fill_assumption || '暂无'} · 持仓上限 {data.topk} 只
        </Typography>
      )}

      {error && (
        <ErrorState
          title="纸面信号读取失败"
          message={error}
          detail={errorDetail}
          onRetry={() => void load()}
          retrying={loading}
        />
      )}

      {isStale && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          这份纸面信号已生成 {ageDays} 天（阈值 {STALE_DAYS} 天），不是最新一次调仓结果。
          平台在每个交易日收盘后自动重算 —— 若一直显示过期，说明后台调仓任务没跑成功。
        </Alert>
      )}

      {loading && !data && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
          <CircularProgress />
        </Box>
      )}

      {data && (
        <>
          <Alert severity="warning" icon={<ShieldAlert size={18} />} sx={{ mb: 2 }}>
            <Typography variant="body2">
              这是<strong>纸面虚拟盘</strong>，不是实盘建议。回测口径：22f 周频 / 含实盘成本；
              2021-2026 OOS 9.38% 年化 / -27.1% 回撤 / 0.685 夏普。先纸面 4-8 周，跑赢基准再谈真钱。
            </Typography>
            <Typography variant="body2" sx={{ mt: 0.5 }}>
              纸面信号<strong>不保证成交</strong>：实盘会受到涨跌停、停牌、流动性与开盘跳空影响，
              参考价与真实成交价可能差很多。
            </Typography>
          </Alert>

          <Stack direction="row" spacing={1.5} flexWrap="wrap" useFlexGap sx={{ mb: 2 }}>
            {stats.map((s) => (
              <Card key={s.label} variant="outlined" sx={{ minWidth: 140, flex: '0 0 auto' }}>
                <CardContent sx={{ py: 1.25, '&:last-child': { pb: 1.25 } }}>
                  <Typography variant="caption" color="text.secondary">
                    {s.label}
                  </Typography>
                  <Typography variant="h6" fontWeight={700}>
                    {s.value}
                  </Typography>
                </CardContent>
              </Card>
            ))}
          </Stack>

          <Card variant="outlined" sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 1 }}>
                买入（{data.buys?.length ?? 0}）
              </Typography>
              <Divider sx={{ mb: 1 }} />
              <TradeTable rows={data.buys ?? []} side="buy" />
            </CardContent>
          </Card>

          <Card variant="outlined" sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 1 }}>
                卖出（{data.sells?.length ?? 0}）
              </Typography>
              <Divider sx={{ mb: 1 }} />
              <TradeTable rows={data.sells ?? []} side="sell" />
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 1 }}>
                持有（{data.holds?.length ?? 0}）
              </Typography>
              <Divider sx={{ mb: 1.5 }} />
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                {(data.holds ?? []).map((h) => (
                  <Chip
                    key={h.code}
                    label={`${h.code}${h.name ? ` ${h.name}` : ''}`}
                    size="small"
                    sx={{ fontFamily: 'ui-monospace, Menlo, monospace', height: 30 }}
                  />
                ))}
                {(data.holds?.length ?? 0) === 0 && (
                  <Typography variant="body2" color="text.secondary">
                    无持仓
                  </Typography>
                )}
              </Stack>
            </CardContent>
          </Card>

          {data.cost_model && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
              成本模型：佣金 {(data.cost_model.commission * 10000).toFixed(1)}‱ · 印花税{' '}
              {(data.cost_model.stamp_tax * 10000).toFixed(1)}‱（卖出） · 滑点{' '}
              {(data.cost_model.slippage * 1000).toFixed(1)}‰（单边） · 现金预留{' '}
              {(data.cost_model.cash_reserve * 100).toFixed(0)}% · {data.cost_model.board_lot} 股整手
            </Typography>
          )}
        </>
      )}

      <Box sx={{ height: BOTTOM_NAV_CONTENT_PADDING, display: { xs: 'block', md: 'none' } }} aria-hidden="true" />
    </Box>
  );
}
