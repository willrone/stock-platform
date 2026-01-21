'use client';

import React, { useEffect, useMemo, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material';
import { DataService, LatestSignalItem, SignalEvent } from '../../services/dataService';
import TradingViewChart from '@/components/charts/TradingViewChart';

export default function SignalsPage() {
  const [strategies, setStrategies] = useState<Array<{ key: string; name: string }>>([]);
  const [strategyName, setStrategyName] = useState<string>('');
  const [days, setDays] = useState<number>(60);
  const [source, setSource] = useState<'local' | 'remote'>('local');

  const [limit, setLimit] = useState<number>(200);
  const [offset, setOffset] = useState<number>(0);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [total, setTotal] = useState<number>(0);
  const [rows, setRows] = useState<LatestSignalItem[]>([]);
  const [failures, setFailures] = useState<string[]>([]);

  // 前端筛选：按信号类型、日期范围
  const [signalFilter, setSignalFilter] = useState<'ALL' | 'BUY' | 'SELL' | 'HOLD'>('ALL');
  const [dateFrom, setDateFrom] = useState<string>('');
  const [dateTo, setDateTo] = useState<string>('');

  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyStock, setHistoryStock] = useState<string>('');
  const [historyEvents, setHistoryEvents] = useState<SignalEvent[]>([]);

  // 将历史信号事件转换为价格图表可用的信号标记（与回测结果中的图表风格保持一致）
  const chartSignals = useMemo(
    () =>
      historyEvents.map((ev) => ({
        signal_id: `${ev.timestamp}-${ev.signal}-${ev.price}`,
        stock_code: historyStock,
        signal_type: ev.signal,
        price: ev.price,
        timestamp: ev.timestamp,
        executed: true,
      })),
    [historyEvents, historyStock],
  );

  // 为价格图表构造时间窗口：以当前时间往前扩展 days + 缓冲天数
  const chartStartDate = useMemo(() => {
    const end = new Date();
    const start = new Date(end);
    start.setDate(start.getDate() - (days + 60)); // 多给一些缓冲天数，避免实际交易日不足
    return start.toISOString().slice(0, 10);
  }, [days]);

  const chartEndDate = useMemo(() => {
    const end = new Date();
    return end.toISOString().slice(0, 10);
  }, []);

  const page = useMemo(() => Math.floor(offset / Math.max(1, limit)) + 1, [offset, limit]);

  const filteredRows = useMemo(() => {
    return rows.filter((r) => {
      // 信号类型筛选
      if (signalFilter !== 'ALL' && r.latest_signal !== signalFilter) {
        return false;
      }
      // 日期范围筛选
      if (r.signal_date) {
        const d = r.signal_date.slice(0, 10); // YYYY-MM-DD
        if (dateFrom && d < dateFrom) return false;
        if (dateTo && d > dateTo) return false;
      }
      return true;
    });
  }, [rows, signalFilter, dateFrom, dateTo]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const list = await DataService.getAvailableStrategies();
        if (!mounted) return;
        const simplified = list.map(s => ({ key: s.key, name: s.name || s.key }));
        setStrategies(simplified);
        if (!strategyName && simplified.length > 0) {
          setStrategyName(simplified[0].key);
        }
      } catch (e: any) {
        if (!mounted) return;
        setError(e?.message || '加载策略列表失败');
      }
    })();
    return () => {
      mounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchLatest = async (customOffset?: number) => {
    const realOffset = customOffset ?? offset;
    if (!strategyName) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await DataService.getLatestSignals({
        strategy_name: strategyName,
        days,
        source,
        limit,
        offset: realOffset,
      });
      setRows(resp.signals || []);
      setTotal(resp.pagination?.total ?? 0);
      setFailures(resp.failures || []);
      setOffset(realOffset);
    } catch (e: any) {
      setError(e?.message || '获取信号失败');
    } finally {
      setLoading(false);
    }
  };

  const openHistory = async (stockCode: string) => {
    setHistoryOpen(true);
    setHistoryStock(stockCode);
    setHistoryEvents([]);
    setHistoryError(null);
    setHistoryLoading(true);
    try {
      const resp = await DataService.getSignalHistory({
        stock_code: stockCode,
        strategy_name: strategyName,
        days,
      });
      setHistoryEvents(resp.events || []);
    } catch (e: any) {
      setHistoryError(e?.message || '获取信号历史失败');
    } finally {
      setHistoryLoading(false);
    }
  };

  return (
    <Stack spacing={2}>
      <Box>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          策略信号
        </Typography>
        <Typography variant="body2" color="text.secondary">
          选择策略后，生成全市场（分页）最近N个交易日窗口内的“最新信号”，点击某只股票可查看近N日信号事件历史。
        </Typography>
      </Box>

      <Card>
        <CardContent>
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems={{ md: 'center' }}>
            <FormControl sx={{ minWidth: 220 }}>
              <InputLabel id="strategy-select-label">策略</InputLabel>
              <Select
                labelId="strategy-select-label"
                label="策略"
                value={strategyName}
                onChange={(e) => {
                  setStrategyName(String(e.target.value));
                  setOffset(0);
                }}
              >
                {strategies.map(s => (
                  <MenuItem key={s.key} value={s.key}>
                    {s.name}（{s.key}）
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl sx={{ minWidth: 160 }}>
              <InputLabel id="source-select-label">股票池来源</InputLabel>
              <Select
                labelId="source-select-label"
                label="股票池来源"
                value={source}
                onChange={(e) => {
                  setSource(e.target.value as any);
                  setOffset(0);
                }}
              >
                <MenuItem value="local">本地（parquet）</MenuItem>
                <MenuItem value="remote">远端数据服务</MenuItem>
              </Select>
            </FormControl>

            <TextField
              label="最近N个交易日"
              type="number"
              value={days}
              onChange={(e) => setDays(Math.max(5, Math.min(365, Number(e.target.value || 60))))}
              sx={{ width: 160 }}
              inputProps={{ min: 5, max: 365 }}
            />

            <TextField
              label="分页大小"
              type="number"
              value={limit}
              onChange={(e) => {
                const v = Math.max(1, Math.min(2000, Number(e.target.value || 200)));
                setLimit(v);
                setOffset(0);
              }}
              sx={{ width: 140 }}
              inputProps={{ min: 1, max: 2000 }}
            />

            <Button
              variant="contained"
              onClick={() => fetchLatest(0)}
              disabled={!strategyName || loading}
            >
              {loading ? '生成中...' : '生成信号'}
            </Button>
          </Stack>

          {/* 筛选条件：信号类型 + 日期范围 */}
          <Stack
            direction={{ xs: 'column', md: 'row' }}
            spacing={2}
            alignItems={{ md: 'center' }}
            sx={{ mt: 2 }}
          >
            <FormControl sx={{ minWidth: 160 }}>
              <InputLabel id="signal-filter-label">信号筛选</InputLabel>
              <Select
                labelId="signal-filter-label"
                label="信号筛选"
                value={signalFilter}
                onChange={(e) => setSignalFilter(e.target.value as any)}
              >
                <MenuItem value="ALL">全部</MenuItem>
                <MenuItem value="BUY">BUY</MenuItem>
                <MenuItem value="SELL">SELL</MenuItem>
                <MenuItem value="HOLD">HOLD</MenuItem>
              </Select>
            </FormControl>

            <TextField
              label="开始日期"
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              InputLabelProps={{ shrink: true }}
              sx={{ width: 180 }}
            />
            <TextField
              label="结束日期"
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              InputLabelProps={{ shrink: true }}
              sx={{ width: 180 }}
            />
          </Stack>

          {error && (
            <Box sx={{ mt: 2 }}>
              <Typography color="error">{error}</Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Stack direction="row" spacing={2} alignItems="center" justifyContent="space-between">
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              最新信号（第 {page} 页，{filteredRows.length} / {rows.length} / {total}）
            </Typography>
            <Stack direction="row" spacing={1}>
              <Button
                variant="outlined"
                disabled={loading || offset <= 0}
                onClick={() => {
                  if (offset <= 0) return;
                  const nextOffset = Math.max(0, offset - limit);
                  fetchLatest(nextOffset);
                }}
              >
                上一页
              </Button>
              <Button
                variant="outlined"
                disabled={loading || offset + limit >= total}
                onClick={() => {
                  if (offset + limit >= total) return;
                  const nextOffset = offset + limit;
                  fetchLatest(nextOffset);
                }}
              >
                下一页
              </Button>
            </Stack>
          </Stack>

          <Box sx={{ mt: 2, overflowX: 'auto' }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>股票</TableCell>
                  <TableCell>最新信号</TableCell>
                  <TableCell>信号日期</TableCell>
                  <TableCell align="right">强度</TableCell>
                  <TableCell align="right">价格</TableCell>
                  <TableCell>原因</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={6}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <CircularProgress size={18} />
                        <Typography variant="body2" color="text.secondary">
                          计算中（全市场建议分页/分批）...
                        </Typography>
                      </Stack>
                    </TableCell>
                  </TableRow>
                ) : rows.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={6}>
                      <Typography variant="body2" color="text.secondary">
                        暂无数据（先点击“生成信号”）
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : filteredRows.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={6}>
                      <Typography variant="body2" color="text.secondary">
                        当前筛选条件下无数据
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredRows.map((r) => (
                    <TableRow
                      key={r.stock_code}
                      hover
                      sx={{ cursor: 'pointer' }}
                      onClick={() => openHistory(r.stock_code)}
                    >
                      <TableCell>{r.stock_code}</TableCell>
                      <TableCell>{r.latest_signal}</TableCell>
                      <TableCell>{r.signal_date || '-'}</TableCell>
                      <TableCell align="right">{Number(r.strength || 0).toFixed(3)}</TableCell>
                      <TableCell align="right">{r.price ?? '-'}</TableCell>
                      <TableCell sx={{ maxWidth: 520 }}>
                        <Typography variant="body2" noWrap title={r.reason || ''}>
                          {r.reason || '-'}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </Box>

          {failures.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary">
                失败样例（最多20条）：{failures.slice(0, 5).join(' | ')}
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      <Dialog open={historyOpen} onClose={() => setHistoryOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          {historyStock} · {strategyName} · 近{days}个交易日信号事件
        </DialogTitle>
        <DialogContent>
          {historyLoading ? (
            <Stack direction="row" spacing={1} alignItems="center" sx={{ py: 2 }}>
              <CircularProgress size={18} />
              <Typography variant="body2" color="text.secondary">
                加载中...
              </Typography>
            </Stack>
          ) : historyError ? (
            <Typography color="error" sx={{ py: 1 }}>
              {historyError}
            </Typography>
          ) : historyEvents.length === 0 ? (
            <Typography variant="body2" color="text.secondary" sx={{ py: 1 }}>
              窗口内无 BUY/SELL 事件
            </Typography>
          ) : (
            <Stack spacing={2} sx={{ py: 1 }}>
              {/* 价格走势图（从后端/本地 parquet 读取K线数据），并叠加 BUY/SELL 信号标记 */}
              <TradingViewChart
                stockCode={historyStock}
                startDate={chartStartDate}
                endDate={chartEndDate}
                signals={chartSignals}
                showSignals
                showTrades={false}
                height={320}
              />

              {/* 信号明细表，与回测结果中的信号记录表现一致 */}
              <Box sx={{ overflowX: 'auto' }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>时间</TableCell>
                      <TableCell>信号</TableCell>
                      <TableCell align="right">强度</TableCell>
                      <TableCell align="right">价格</TableCell>
                      <TableCell>原因</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {historyEvents.map((ev, idx) => (
                      <TableRow key={`${ev.timestamp}-${idx}`}>
                        <TableCell>{ev.timestamp}</TableCell>
                        <TableCell>{ev.signal}</TableCell>
                        <TableCell align="right">{Number(ev.strength || 0).toFixed(3)}</TableCell>
                        <TableCell align="right">{ev.price}</TableCell>
                        <TableCell sx={{ maxWidth: 520 }}>
                          <Typography variant="body2" noWrap title={ev.reason || ''}>
                            {ev.reason || '-'}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Box>
            </Stack>
          )}
        </DialogContent>
      </Dialog>
    </Stack>
  );
}

