'use client';

import React, { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
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

type Trade = {
  code: string;
  name?: string;
  price: number;
  shares: number;
  amount: number;
  commission?: number;
  stamp_tax?: number;
  slippage?: number;
  total_cost?: number;
  net?: number;
  score?: number;
};

type Snapshot = {
  meta: {
    date: string;
    topk: number;
    n_drop: number;
    cash_reserve: number;
    fill_assumption: string;
    generated_at: string;
  };
  portfolio: { cash: number; value: number; n_positions: number };
  buys: Trade[];
  sells: Trade[];
  holds: string[];
};

const money = (v?: number) =>
  typeof v === 'number' && Number.isFinite(v)
    ? `¥${v.toLocaleString('zh-CN', { maximumFractionDigits: 0 })}`
    : '-';

const TradeTable: React.FC<{ rows: Trade[]; side: 'buy' | 'sell' }> = ({ rows, side }) => {
  if (!rows.length) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
        无
      </Typography>
    );
  }
  return (
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
            <TableCell>{r.name || '-'}</TableCell>
            <TableCell align="right">{r.shares?.toLocaleString?.() ?? r.shares}</TableCell>
            <TableCell align="right">{r.price?.toFixed(2)}</TableCell>
            <TableCell align="right">{money(r.amount)}</TableCell>
            <TableCell align="right">{money(side === 'buy' ? r.total_cost : r.net)}</TableCell>
            <TableCell align="right">{typeof r.score === 'number' ? r.score.toFixed(3) : '-'}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
};

export default function PaperSignalPage() {
  const [data, setData] = useState<Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetch('/paper-signal.json', { cache: 'no-store' })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, []);

  const stats = useMemo(() => {
    if (!data) return [];
    return [
      { label: '信号日期', value: data.meta?.date || '-' },
      { label: '组合总值', value: money(data.portfolio?.value) },
      { label: '现金', value: money(data.portfolio?.cash) },
      { label: '持仓数', value: String(data.portfolio?.n_positions ?? '-') },
      {
        label: '买 / 卖 / 持',
        value: `${data.buys?.length ?? 0} / ${data.sells?.length ?? 0} / ${data.holds?.length ?? 0}`,
      },
    ];
  }, [data]);

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">纸面信号读取失败：{error}。请先跑 weekly_signal.py。</Alert>
      </Box>
    );
  }

  if (!data) {
    return (
      <Box sx={{ p: 6, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 0.5 }}>
        <Typography variant="h5" fontWeight={700}>
          纸面信号
        </Typography>
        <Chip label="22f · 周频" color="primary" size="small" />
        <Chip label="虚拟盘" variant="outlined" size="small" />
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        topk={data.meta?.topk} · n_drop={data.meta?.n_drop} · 成交假设：
        {data.meta?.fill_assumption || '-'} · 生成于 {data.meta?.generated_at}
      </Typography>

      <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap sx={{ mb: 2 }}>
        {stats.map((s) => (
          <Card key={s.label} variant="outlined" sx={{ minWidth: 150, flex: '0 0 auto' }}>
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

      <Alert severity="warning" sx={{ mb: 2 }}>
        这是纸面虚拟盘，不是实盘建议。回测口径：22f 周频 / 实盘成本；2021-2026 OOS 9.38%
        年化 / -27.1% 回撤 / 0.685 夏普。先纸面 4-8 周，跑赢基准再谈真钱。
      </Alert>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 1 }}>
            买入（{data.buys.length}）
          </Typography>
          <Divider sx={{ mb: 1 }} />
          <TradeTable rows={data.buys} side="buy" />
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 1 }}>
            卖出（{data.sells.length}）
          </Typography>
          <Divider sx={{ mb: 1 }} />
          <TradeTable rows={data.sells} side="sell" />
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent>
          <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 1 }}>
            持有（{data.holds.length}）
          </Typography>
          <Divider sx={{ mb: 1.5 }} />
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            {data.holds.map((c) => (
              <Chip key={c} label={c} size="small" sx={{ fontFamily: 'ui-monospace, Menlo, monospace' }} />
            ))}
          </Stack>
        </CardContent>
      </Card>
    </Box>
  );
}
