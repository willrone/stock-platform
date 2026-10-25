'use client';

/**
 * 今日信号 —— 可直接照着下单的清单
 *
 * 2026-10-25 新增。这是付费用户真正买的东西：不是回测报告，
 * 而是「今天买哪几只、卖哪几只、买多少股、多少钱、为什么」。
 *
 * 数据来源：GET /api/v1/trading/signal?strategy=xxx
 *   信号在每个交易日收盘后由平台自动生成（约 16:10），
 *   买入建议按下一交易日可成交价执行。
 */

import React, { useCallback, useEffect, useState } from 'react';
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
import { RefreshCw, TrendingUp, TrendingDown, Info, Wallet } from 'lucide-react';
import { apiRequest } from '@/services/api';

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
  v === undefined || v === null ? '—' : `¥${v.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

const fmtShares = (v?: number) => (v ? v.toLocaleString('zh-CN') : '—');

export default function TodaySignalPage() {
  const router = useRouter();
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [active, setActive] = useState('honest_42f');
  const [signal, setSignal] = useState<DailySignal | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tier, setTier] = useState('free');
  const [tierLabel, setTierLabel] = useState('体验版');
  const [delayDays, setDelayDays] = useState(0);
  const [hasDetail, setHasDetail] = useState(true);

  useEffect(() => {
    const token = window.localStorage.getItem('access_token');
    if (!token) {
      router.replace('/login');
      return;
    }
    (async () => {
      try {
        const res = await apiRequest.get<StrategyListResponse>('/trading/strategies');
        setStrategies(res.strategies || []);
        setTier(res.tier || 'free');
        setTierLabel(res.tier_label || '体验版');
        setDelayDays(res.delay_days ?? 0);
        setHasDetail(res.include_execution_detail !== false);
        const usable = (res.strategies || []).filter((s) => !s.locked);
        if (usable.length) {
          const withPaper = usable.find((s) => s.n_positions > 0) || usable[0];
          setActive(withPaper.key);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : '加载策略列表失败');
      }
    })();
  }, [router]);

  const loadSignal = useCallback(async () => {
    if (!active) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiRequest.get<DailySignal>(`/trading/signal?strategy=${active}`);
      setSignal(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载信号失败');
      setSignal(null);
    } finally {
      setLoading(false);
    }
  }, [active]);

  useEffect(() => {
    loadSignal();
  }, [loadSignal]);

  const totalBuyAmount = (signal?.buys ?? []).reduce((s, b) => s + (b.amount ?? 0), 0);
  const totalSellAmount = (signal?.sells ?? []).reduce((s, b) => s + (b.amount ?? 0), 0);
  const totalCost = (signal?.buys ?? []).reduce(
    (s, b) => s + (b.commission ?? 0) + (b.slippage ?? 0),
    0
  );

  return (
    <Box sx={{ p: 3 }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
        <Box>
          <Typography variant="h5">今日信号</Typography>
          <Typography variant="body2" color="text.secondary">
            收盘后自动生成，含股数 / 金额 / 成本，可直接照着下单
          </Typography>
        </Box>
        <Button startIcon={<RefreshCw size={16} />} onClick={loadSignal} disabled={loading}>
          刷新
        </Button>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* 订阅状态条 */}
      <Stack direction="row" spacing={1.5} sx={{ mb: 2, alignItems: 'center', flexWrap: 'wrap' }}>
        <Chip size="small" color={tier === 'free' ? 'default' : 'primary'} label={`${tierLabel}`} />
        {delayDays > 0 && (
          <Chip
            size="small"
            color="warning"
            variant="outlined"
            label={`信号延迟 ${delayDays} 个交易日`}
          />
        )}
        {tier === 'free' && (
          <Button
            size="small"
            variant="contained"
            onClick={() => router.push('/pricing')}
            sx={{ ml: 'auto !important' }}
          >
            升级解锁全部信号线
          </Button>
        )}
      </Stack>

      {signal?.upgrade_hint && tier === 'free' && (
        <Alert severity="info" sx={{ mb: 2 }}>
          {signal.upgrade_hint}
        </Alert>
      )}

      <Stack direction="row" spacing={2} sx={{ mb: 2, alignItems: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          策略
        </Typography>
        <Select
          size="small"
          value={active}
          onChange={(e) => setActive(e.target.value)}
          sx={{ minWidth: 260 }}
        >
          {strategies.map((s) => (
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
          <Chip size="small" color="primary" label={`信号日期 ${signal.date}`} />
        )}
      </Stack>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
          <CircularProgress />
        </Box>
      )}

      {!loading && signal?.status === 'no_data' && (
        <Alert severity="info">{signal.message}</Alert>
      )}

      {!loading && signal?.status === 'ok' && (
        <>
          <Stack direction="row" spacing={2} sx={{ mb: 2, flexWrap: 'wrap' }}>
            <Card sx={{ minWidth: 170 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  组合总值
                </Typography>
                <Typography variant="h6">{fmtMoney(signal.value)}</Typography>
                <Typography
                  variant="body2"
                  color={signal.total_return_pct >= 0 ? 'success.main' : 'error.main'}
                >
                  {signal.total_return_pct >= 0 ? '+' : ''}
                  {signal.total_return_pct?.toFixed(2)}%
                </Typography>
              </CardContent>
            </Card>
            <Card sx={{ minWidth: 150 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  持仓
                </Typography>
                <Typography variant="h6">{signal.n_positions} 只</Typography>
                <Typography variant="body2" color="text.secondary">
                  现金 {fmtMoney(signal.cash)}
                </Typography>
              </CardContent>
            </Card>
            <Card sx={{ minWidth: 190 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  本次操作
                </Typography>
                <Typography variant="h6">
                  买 {signal.buys.length} · 卖 {signal.sells.length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  买 {fmtMoney(totalBuyAmount)} / 卖 {fmtMoney(totalSellAmount)}
                </Typography>
              </CardContent>
            </Card>
            <Card sx={{ minWidth: 160 }}>
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  预计成本
                </Typography>
                <Typography variant="h6">{fmtMoney(totalCost)}</Typography>
                <Typography variant="body2" color="text.secondary">
                  含佣金与滑点
                </Typography>
              </CardContent>
            </Card>
          </Stack>

          <Alert severity="info" icon={<Info size={16} />} sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>{signal.display_name}</strong> —— {signal.desc}
            </Typography>
            <Typography variant="body2" sx={{ mt: 0.5 }}>
              {signal.factor_note} · 持仓上限 {signal.topk} 只 · 每次最多换 {signal.ndrop} 只
              {signal.fill_assumption ? ` · ${signal.fill_assumption}` : ''}
            </Typography>
          </Alert>

          <Stack spacing={2}>
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
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {signal.buys.map((b) => (
                        <TableRow key={b.code}>
                          <TableCell sx={{ fontFamily: 'monospace' }}>{b.code}</TableCell>
                          <TableCell>{b.name}</TableCell>
                          <TableCell align="right">{b.price?.toFixed(2)}</TableCell>
                          {hasDetail && <TableCell align="right">{fmtShares(b.shares)}</TableCell>}
                          {hasDetail && <TableCell align="right">{fmtMoney(b.amount)}</TableCell>}
                          {hasDetail && (
                            <TableCell align="right">
                              {((b.commission ?? 0) + (b.slippage ?? 0)).toFixed(2)}
                            </TableCell>
                          )}
                          <TableCell align="right">
                            {b.score !== undefined && b.score !== null && b.score !== ''
                              ? Number(b.score).toFixed(3)
                              : '—'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>

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
                      {signal.sells.map((s) => (
                        <TableRow key={s.code}>
                          <TableCell sx={{ fontFamily: 'monospace' }}>{s.code}</TableCell>
                          <TableCell>{s.name}</TableCell>
                          <TableCell align="right">{s.price?.toFixed(2)}</TableCell>
                          {hasDetail && <TableCell align="right">{fmtShares(s.shares)}</TableCell>}
                          {hasDetail && <TableCell align="right">{fmtMoney(s.amount)}</TableCell>}
                          {hasDetail && (
                            <TableCell align="right">
                              {((s.commission ?? 0) + (s.stamp_tax ?? 0) + (s.slippage ?? 0)).toFixed(2)}
                            </TableCell>
                          )}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                  <Wallet size={18} color="#1565c0" />
                  <Typography variant="subtitle1">当前持仓（{signal.holds.length}）</Typography>
                </Stack>
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
                    {signal.holds.map((h) => (
                      <TableRow key={h.code}>
                        <TableCell sx={{ fontFamily: 'monospace' }}>{h.code}</TableCell>
                        <TableCell>{h.name}</TableCell>
                        {hasDetail && <TableCell align="right">{fmtShares(h.shares)}</TableCell>}
                        <TableCell align="right">{h.cost?.toFixed(2)}</TableCell>
                        {hasDetail && <TableCell align="right">{fmtMoney(h.cost_amount)}</TableCell>}
                        <TableCell align="right">{h.buy_date}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
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
                {(signal.cost_model.cash_reserve * 100).toFixed(0)}% ·{' '}
                {signal.cost_model.board_lot} 股整手
              </Typography>
            </>
          )}
        </>
      )}
    </Box>
  );
}
