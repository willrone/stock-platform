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
  score?: number | string;
};

type Line = {
  strategy: string;
  desc: string;
  date: string;
  topk: number;
  ndrop: number;
  cash: number;
  value: number;
  n_positions: number;
  target_weight: number;
  sells: Trade[];
  buys: Trade[];
  holds: string[];
};

type Portfolio = {
  meta: {
    date: string;
    data_latest?: string;
    capital_per_line: number;
    costs: {
      commission: number;
      stamp_tax: number;
      slippage: number;
      cash_reserve: number;
    };
    fill_assumption: string;
    generated_at: string;
  };
  lines: Record<string, Line>;
};

const LINE_COLORS: Record<string, 'primary' | 'success' | 'warning' | 'info'> = {
  '22f': 'info',
  honest_42f: 'primary',
  rsi_71421: 'success',
  factor_ranking: 'warning',
};

export default function PaperPortfolioPage() {
  const [data, setData] = useState<Portfolio | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [active, setActive] = useState<string>('');

  useEffect(() => {
    let cancelled = false;
    fetch('/paper-portfolio.json', { cache: 'no-store' })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((j: Portfolio) => {
        if (cancelled) return;
        setData(j);
        const first = Object.keys(j.lines ?? {})[0];
        if (first) setActive(first);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const lines = useMemo(() => data?.lines ?? {}, [data]);
  const line = active ? lines[active] : undefined;

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error || !data) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">
          读取纸面组合失败：{error ?? '数据为空'}。请先运行
          <code> backend/scripts/paper_portfolio.py </code>生成
          <code> frontend/public/paper-portfolio.json</code>。
        </Alert>
      </Box>
    );
  }

  const meta = data.meta;
  const cap = meta.capital_per_line;

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        多策略纸面盘（策略赛马）
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        每条策略一条独立线，各 {cap.toLocaleString('zh-CN')} 元、独立持仓与账本，
        同一调仓日（{meta.date}
        {meta.data_latest ? `，数据最新 ${meta.data_latest}` : ''}）、同一全市场宇宙。
        成本：佣金 {(meta.costs.commission * 10000).toFixed(1)}‱ + 印花税{' '}
        {(meta.costs.stamp_tax * 10000).toFixed(1)}‱ + 滑点{' '}
        {(meta.costs.slippage * 1000).toFixed(1)}‰ + 现金预留{' '}
        {(meta.costs.cash_reserve * 100).toFixed(0)}%。成交价假设：
        {meta.fill_assumption}。
      </Typography>

      <Divider sx={{ my: 2 }} />

      <Stack direction="row" spacing={2} sx={{ mb: 2, flexWrap: 'wrap' }}>
        {Object.values(lines).map((l) => {
          const pnl = l.value - cap;
          const pnlPct = (pnl / cap) * 100;
          return (
            <Card
              key={l.strategy}
              variant={active === l.strategy ? 'elevation' : 'outlined'}
              onClick={() => setActive(l.strategy)}
              sx={{
                cursor: 'pointer',
                minWidth: 230,
                borderColor:
                  active === l.strategy ? 'primary.main' : 'divider',
              }}
            >
              <CardContent>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Typography variant="subtitle1">{l.strategy}</Typography>
                  <Chip
                    size="small"
                    color={LINE_COLORS[l.strategy] ?? 'primary'}
                    label={`${l.n_positions} 只`}
                  />
                </Stack>
                <Typography variant="h6" sx={{ mt: 1 }}>
                  ¥{l.value.toLocaleString('zh-CN')}
                </Typography>
                <Typography
                  variant="body2"
                  color={pnl >= 0 ? 'success.main' : 'error.main'}
                >
                  {pnl >= 0 ? '+' : ''}
                  {pnlPct.toFixed(2)}%
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {l.desc}
                </Typography>
              </CardContent>
            </Card>
          );
        })}
      </Stack>

      {line && (
        <>
          <Typography variant="h6" sx={{ mb: 1 }}>
            {line.strategy} — {line.date} 调仓明细
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            TopK={line.topk} / 每次换 {line.ndrop} 只 / 单仓目标权重{' '}
            {(line.target_weight * 100).toFixed(2)}% / 现金 ¥
            {line.cash.toLocaleString('zh-CN')}
          </Typography>

          <Stack spacing={2}>
            {(
              [
                ['卖出', line.sells, 'error.main'],
                ['买入', line.buys, 'success.main'],
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
                        {rows.map((t) => (
                          <TableRow key={`${title}-${t.code}`}>
                            <TableCell sx={{ fontFamily: 'monospace' }}>{t.code}</TableCell>
                            <TableCell>{t.name}</TableCell>
                            <TableCell align="right">{t.price?.toFixed(2)}</TableCell>
                            <TableCell align="right">{t.shares?.toLocaleString()}</TableCell>
                            <TableCell align="right">
                              {t.amount?.toLocaleString('zh-CN')}
                            </TableCell>
                            <TableCell align="right">
                              {(
                                (t.commission ?? 0) +
                                (t.stamp_tax ?? 0) +
                                (t.slippage ?? 0)
                              ).toFixed(2)}
                            </TableCell>
                            <TableCell align="right">
                              {t.score !== undefined && t.score !== null
                                ? Number(t.score).toFixed(3)
                                : '-'}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
            ))}
          </Stack>
        </>
      )}

      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 3 }}>
        生成时间 {meta.generated_at} · 数据仅供研究，不构成投资建议
      </Typography>
    </Box>
  );
}
