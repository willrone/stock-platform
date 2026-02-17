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
  FormControlLabel,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material';
import {
  DataService,
  MultiLatestSignalRow,
  MultiSignalHistoryResponse,
  SignalEvent,
} from '../../services/dataService';
import TradingViewChart from '@/components/charts/TradingViewChart';
import {
  StrategyConfigForm,
  StrategyParameter,
} from '../../components/backtest/StrategyConfigForm';
import {
  PortfolioStrategyConfig,
  PortfolioStrategyItem,
} from '../../components/backtest/PortfolioStrategyConfig';
import { StrategyConfig, StrategyConfigService } from '../../services/strategyConfigService';
import { MobileSignalCard } from '../../components/mobile/MobileSignalCard';

export default function SignalsPage() {
  const [strategies, setStrategies] = useState<
    Array<{
      key: string;
      name: string;
      description: string;
      parameters?: Record<string, StrategyParameter>;
    }>
  >([]);
  const [strategyType, setStrategyType] = useState<'single' | 'portfolio'>('single');
  const [selectedStrategyName, setSelectedStrategyName] = useState<string>('');
  const [strategyConfig, setStrategyConfig] = useState<Record<string, any>>({});
  const [portfolioConfig, setPortfolioConfig] = useState<{
    strategies: PortfolioStrategyItem[];
    integration_method: string;
  } | null>(null);
  const [savedConfigs, setSavedConfigs] = useState<StrategyConfig[]>([]);
  const [loadingConfigs, setLoadingConfigs] = useState(false);
  const [configFormKey, setConfigFormKey] = useState(0);
  const [portfolioConfigKey, setPortfolioConfigKey] = useState(0);
  const [selectedPortfolioConfigId, setSelectedPortfolioConfigId] = useState('');
  const [days, setDays] = useState<number>(60);
  const [source, setSource] = useState<'local' | 'remote'>('local');

  const [limit, setLimit] = useState<number>(200);
  const [offset, setOffset] = useState<number>(0);
  const [generateAll, setGenerateAll] = useState(true);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [total, setTotal] = useState<number>(0);
  const [rows, setRows] = useState<MultiLatestSignalRow[]>([]);
  const [failures, setFailures] = useState<string[]>([]);

  // 前端筛选：按信号类型、日期范围
  const [signalFilter, setSignalFilter] = useState<'ALL' | 'BUY' | 'SELL' | 'HOLD'>('ALL');
  const [dateFrom, setDateFrom] = useState<string>('');
  const [dateTo, setDateTo] = useState<string>('');

  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyStock, setHistoryStock] = useState<string>('');
  const [historyEventsByStrategy, setHistoryEventsByStrategy] = useState<
    MultiSignalHistoryResponse['events_by_strategy']
  >({});
  const [historyStrategies, setHistoryStrategies] = useState<string[]>([]);

  const selectedStrategyNames = useMemo(() => {
    if (strategyType === 'portfolio') {
      return ['portfolio'];
    }
    return selectedStrategyName ? [selectedStrategyName] : [];
  }, [strategyType, selectedStrategyName]);

  // 将历史信号事件转换为价格图表可用的信号标记（多策略）
  const chartSignals = useMemo(() => {
    const signals: Array<{
      signal_id: string;
      stock_code: string;
      signal_type: 'BUY' | 'SELL';
      price: number;
      timestamp: string;
      executed?: boolean;
      strategy_name?: string;
      strategy_id?: string;
    }> = [];
    Object.entries(historyEventsByStrategy || {}).forEach(([strategyName, events]) => {
      events.forEach((ev: SignalEvent) => {
        signals.push({
          signal_id: `${strategyName}-${ev.timestamp}-${ev.signal}-${ev.price}`,
          stock_code: historyStock,
          signal_type: ev.signal,
          price: ev.price,
          timestamp: ev.timestamp,
          executed: true,
          strategy_name: strategyName,
          strategy_id: strategyName,
        });
      });
    });
    return signals;
  }, [historyEventsByStrategy, historyStock]);

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
  const canGenerate = useMemo(() => {
    if (strategyType === 'portfolio') {
      return (portfolioConfig?.strategies?.length || 0) > 0;
    }
    return !!selectedStrategyName;
  }, [strategyType, selectedStrategyName, portfolioConfig]);
  const selectedStrategy = useMemo(
    () => strategies.find(s => s.key === selectedStrategyName),
    [strategies, selectedStrategyName]
  );

  const filteredRows = useMemo(() => {
    if (!rows.length) {
      return [];
    }
    return rows.filter(row => {
      const perStrategy = row.per_strategy || {};
      const entries = Object.entries(perStrategy).filter(([name]) =>
        selectedStrategyNames.includes(name)
      );
      if (!entries.length) {
        return false;
      }

      // 只要有任意一个策略在当前筛选条件下命中，就保留该股票
      return entries.some(([, val]) => {
        if (!val) {
          return false;
        }
        if (signalFilter !== 'ALL' && val.latest_signal !== signalFilter) {
          return false;
        }
        if (val.signal_date) {
          const d = val.signal_date.slice(0, 10); // YYYY-MM-DD
          if (dateFrom && d < dateFrom) {
            return false;
          }
          if (dateTo && d > dateTo) {
            return false;
          }
        }
        return true;
      });
    });
  }, [rows, selectedStrategyNames, signalFilter, dateFrom, dateTo]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const list = await DataService.getAvailableStrategies();
        if (!mounted) {
          return;
        }
        const normalized = list.map(s => ({
          key: s.key,
          name: s.name || s.key,
          description: s.description || s.name || s.key || '无描述',
          parameters: s.parameters,
        }));
        setStrategies(normalized);
        if (normalized.length > 0) {
          setSelectedStrategyName(prev => (prev ? prev : normalized[0].key));
        }
      } catch (e: any) {
        if (!mounted) {
          return;
        }
        setError(e?.message || '加载策略列表失败');
      }
    })();
    return () => {
      mounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const loadSavedConfigs = async () => {
      const targetStrategyName = strategyType === 'portfolio' ? 'portfolio' : selectedStrategyName;
      if (!targetStrategyName) {
        setSavedConfigs([]);
        return;
      }
      setLoadingConfigs(true);
      try {
        const response = await StrategyConfigService.getConfigs(targetStrategyName);
        setSavedConfigs(response.configs);
      } catch (e) {
        console.error('加载已保存配置失败:', e);
        setSavedConfigs([]);
      } finally {
        setLoadingConfigs(false);
      }
    };

    loadSavedConfigs();
  }, [strategyType, selectedStrategyName]);

  useEffect(() => {
    setSelectedPortfolioConfigId('');
  }, [strategyType]);

  const fetchLatest = async (customOffset?: number) => {
    const realOffset = customOffset ?? offset;
    if (!selectedStrategyNames.length) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const pageSize = limit;
      const fetchPage = async (pageOffset: number) => {
        if (strategyType === 'portfolio') {
          const resp = await DataService.getLatestSignals({
            strategy_name: 'portfolio',
            strategy_config: portfolioConfig
              ? {
                  strategies: portfolioConfig.strategies,
                  integration_method: portfolioConfig.integration_method,
                }
              : undefined,
            days,
            source,
            limit: pageSize,
            offset: pageOffset,
          });
          const mappedRows = (resp.signals || []).map(item => ({
            stock_code: item.stock_code,
            per_strategy: {
              portfolio: {
                latest_signal: item.latest_signal,
                signal_date: item.signal_date,
                strength: item.strength,
                price: item.price,
                reason: item.reason,
              },
            },
          }));
          return {
            rows: mappedRows,
            total: resp.pagination?.total ?? 0,
            failures: resp.failures || [],
          };
        }

        const resp = await DataService.getLatestSignals({
          strategy_name: selectedStrategyName,
          strategy_config: strategyConfig,
          days,
          source,
          limit: pageSize,
          offset: pageOffset,
        });
        const mappedRows = (resp.signals || []).map(item => ({
          stock_code: item.stock_code,
          per_strategy: {
            [selectedStrategyName]: {
              latest_signal: item.latest_signal,
              signal_date: item.signal_date,
              strength: item.strength,
              price: item.price,
              reason: item.reason,
            },
          },
        }));
        return {
          rows: mappedRows,
          total: resp.pagination?.total ?? 0,
          failures: resp.failures || [],
        };
      };

      if (generateAll) {
        let nextOffset = 0;
        let totalCount = 0;
        const allRows: MultiLatestSignalRow[] = [];
        const allFailures: string[] = [];

        let hasMore = true;
        while (hasMore) {
          const pageResp = await fetchPage(nextOffset);
          totalCount = pageResp.total || totalCount || nextOffset + pageResp.rows.length;
          allRows.push(...pageResp.rows);
          allFailures.push(...pageResp.failures);

          if (
            totalCount === 0 ||
            nextOffset + pageSize >= totalCount ||
            pageResp.rows.length === 0
          ) {
            hasMore = false;
          } else {
            nextOffset += pageSize;
          }
        }

        setRows(allRows);
        setTotal(totalCount);
        setFailures(allFailures);
        setOffset(0);
        return;
      }

      if (strategyType === 'portfolio') {
        const pageResp = await fetchPage(realOffset);
        setRows(pageResp.rows);
        setTotal(pageResp.total);
        setFailures(pageResp.failures);
        setOffset(realOffset);
      } else {
        const pageResp = await fetchPage(realOffset);
        setRows(pageResp.rows);
        setTotal(pageResp.total);
        setFailures(pageResp.failures);
        setOffset(realOffset);
      }
    } catch (e: any) {
      setError(e?.message || '获取信号失败');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadConfig = async (configId: string) => {
    try {
      const config = await StrategyConfigService.getConfig(configId);
      setStrategyConfig(config.parameters || {});
      setConfigFormKey(prev => prev + 1);
    } catch (e) {
      console.error('加载策略配置失败:', e);
    }
  };

  const handleLoadPortfolioConfig = async (configId: string) => {
    try {
      const config = await StrategyConfigService.getConfig(configId);
      const parameters = config.parameters || {};
      const strategies = Array.isArray(parameters.strategies) ? parameters.strategies : [];
      const integrationMethod = parameters.integration_method || 'weighted_voting';
      setPortfolioConfig({
        strategies,
        integration_method: integrationMethod,
      });
      setPortfolioConfigKey(prev => prev + 1);
    } catch (e) {
      console.error('加载组合策略配置失败:', e);
    }
  };

  const openHistory = async (stockCode: string) => {
    setHistoryOpen(true);
    setHistoryStock(stockCode);
    setHistoryEventsByStrategy({});
    setHistoryStrategies([]);
    setHistoryError(null);
    setHistoryLoading(true);
    try {
      if (strategyType === 'portfolio') {
        const resp = await DataService.getSignalHistory({
          stock_code: stockCode,
          strategy_name: 'portfolio',
          strategy_config: portfolioConfig
            ? {
                strategies: portfolioConfig.strategies,
                integration_method: portfolioConfig.integration_method,
              }
            : undefined,
          days,
        });
        setHistoryEventsByStrategy({ portfolio: resp.events || [] });
        setHistoryStrategies(['portfolio']);
      } else {
        const resp = await DataService.getSignalHistory({
          stock_code: stockCode,
          strategy_name: selectedStrategyName,
          strategy_config: strategyConfig,
          days,
        });
        setHistoryEventsByStrategy({
          [selectedStrategyName]: resp.events || [],
        });
        setHistoryStrategies([selectedStrategyName]);
      }
    } catch (e: any) {
      setHistoryError(e?.message || '获取信号历史失败');
    } finally {
      setHistoryLoading(false);
    }
  };

  const getStrategyLabel = (name: string) => {
    if (name === 'portfolio') {
      return '组合策略';
    }
    const meta = strategies.find(s => s.key === name);
    return meta ? `${meta.name}（${meta.key}）` : name;
  };

  return (
    <Stack spacing={2}>
      <Box>
        <Typography
          variant="h4"
          sx={{ fontWeight: 700, fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2.125rem' } }}
        >
          策略信号
        </Typography>
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
        >
          选择一个或多个策略后，生成全市场最近N个交易日窗口内的&quot;最新信号&quot;，点击某只股票可查看信号事件历史与价格走势。
        </Typography>
      </Box>

      <Card>
        <CardContent>
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems={{ md: 'center' }}>
            <FormControl sx={{ minWidth: 160 }}>
              <InputLabel id="strategy-type-select-label">策略类型</InputLabel>
              <Select
                labelId="strategy-type-select-label"
                label="策略类型"
                value={strategyType}
                onChange={e => {
                  const next = e.target.value as 'single' | 'portfolio';
                  setStrategyType(next);
                  setOffset(0);
                }}
              >
                <MenuItem value="single">单策略</MenuItem>
                <MenuItem value="portfolio">组合策略</MenuItem>
              </Select>
            </FormControl>

            {strategyType === 'single' && (
              <FormControl sx={{ minWidth: 220 }}>
                <InputLabel id="strategy-select-label">策略</InputLabel>
                <Select
                  labelId="strategy-select-label"
                  label="策略"
                  value={selectedStrategyName}
                  onChange={e => {
                    setSelectedStrategyName(e.target.value as string);
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
            )}

            <FormControl sx={{ minWidth: 160 }}>
              <InputLabel id="source-select-label">股票池来源</InputLabel>
              <Select
                labelId="source-select-label"
                label="股票池来源"
                value={source}
                onChange={e => {
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
              onChange={e => setDays(Math.max(5, Math.min(365, Number(e.target.value || 60))))}
              sx={{ width: 160 }}
              inputProps={{ min: 5, max: 365 }}
            />

            <TextField
              label="分页大小"
              type="number"
              value={limit}
              onChange={e => {
                const v = Math.max(1, Math.min(2000, Number(e.target.value || 200)));
                setLimit(v);
                setOffset(0);
              }}
              sx={{ width: 140 }}
              inputProps={{ min: 1, max: 2000 }}
            />

            <FormControlLabel
              control={
                <Switch
                  checked={generateAll}
                  onChange={e => setGenerateAll(e.target.checked)}
                  color="primary"
                />
              }
              label="全量生成"
            />

            <Button
              variant="contained"
              onClick={() => fetchLatest(0)}
              disabled={!canGenerate || loading}
            >
              {loading ? '生成中...' : '生成信号'}
            </Button>
          </Stack>

          {strategyType === 'single' ? (
            selectedStrategy?.parameters ? (
              <Box sx={{ mt: 2 }}>
                <StrategyConfigForm
                  key={`${selectedStrategyName}-${configFormKey}`}
                  strategyName={selectedStrategyName}
                  parameters={selectedStrategy.parameters}
                  values={
                    configFormKey > 0 && Object.keys(strategyConfig).length > 0
                      ? strategyConfig
                      : undefined
                  }
                  onChange={newConfig => {
                    setStrategyConfig(newConfig);
                  }}
                  onLoadConfig={handleLoadConfig}
                  savedConfigs={savedConfigs.map(c => ({
                    config_id: c.config_id,
                    config_name: c.config_name,
                    created_at: c.created_at,
                  }))}
                  loading={loadingConfigs}
                />
              </Box>
            ) : null
          ) : (
            <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
              <FormControl fullWidth disabled={loadingConfigs || savedConfigs.length === 0}>
                <InputLabel>已保存组合配置</InputLabel>
                <Select
                  value={selectedPortfolioConfigId}
                  label="已保存组合配置"
                  onChange={e => {
                    const configId = e.target.value as string;
                    setSelectedPortfolioConfigId(configId);
                    if (configId) {
                      handleLoadPortfolioConfig(configId);
                    }
                  }}
                >
                  <MenuItem value="">不使用</MenuItem>
                  {savedConfigs.map(config => (
                    <MenuItem key={config.config_id} value={config.config_id}>
                      {config.config_name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <PortfolioStrategyConfig
                key={`portfolio-config-${portfolioConfigKey}`}
                availableStrategies={strategies}
                portfolioConfig={portfolioConfig || undefined}
                onChange={config => setPortfolioConfig(config)}
                constraints={{
                  maxWeight: 0.5,
                  grossLeverage: 1.0,
                  minStrategies: 1,
                  maxStrategies: 10,
                }}
              />
            </Box>
          )}

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
                onChange={e => setSignalFilter(e.target.value as any)}
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
              onChange={e => setDateFrom(e.target.value)}
              InputLabelProps={{ shrink: true }}
              sx={{ width: 180 }}
            />
            <TextField
              label="结束日期"
              type="date"
              value={dateTo}
              onChange={e => setDateTo(e.target.value)}
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
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={2}
            alignItems={{ xs: 'flex-start', sm: 'center' }}
            justifyContent="space-between"
          >
            <Typography
              variant="h6"
              sx={{
                fontWeight: 600,
                fontSize: { xs: '0.875rem', sm: '1.25rem' },
                wordBreak: 'break-word',
              }}
            >
              最新信号（第 {page} 页，{filteredRows.length} / {rows.length} / {total}，已选策略：
              {selectedStrategyNames.map(name => getStrategyLabel(name)).join('，') || '无'}）
            </Typography>
            <Stack direction="row" spacing={1}>
              <Button
                variant="outlined"
                disabled={generateAll || loading || offset <= 0}
                onClick={() => {
                  if (offset <= 0) {
                    return;
                  }
                  const nextOffset = Math.max(0, offset - limit);
                  fetchLatest(nextOffset);
                }}
              >
                上一页
              </Button>
              <Button
                variant="outlined"
                disabled={generateAll || loading || offset + limit >= total}
                onClick={() => {
                  if (offset + limit >= total) {
                    return;
                  }
                  const nextOffset = offset + limit;
                  fetchLatest(nextOffset);
                }}
              >
                下一页
              </Button>
            </Stack>
          </Stack>

          {/* 移动端：卡片列表 */}
          <Box sx={{ display: { xs: 'block', md: 'none' }, mt: 2 }}>
            {loading ? (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <CircularProgress size={24} />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  计算中...
                </Typography>
              </Box>
            ) : filteredRows.length === 0 ? (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="body2" color="text.secondary">
                  暂无信号数据
                </Typography>
              </Box>
            ) : (
              filteredRows.map((row, idx) => {
                // 合并所有策略的信号（移动端简化显示）
                const signals = selectedStrategyNames.map(name => row[name]).filter(Boolean);
                if (signals.length === 0) {
                  return null;
                }

                return signals.map((sig, sidx) => (
                  <MobileSignalCard
                    key={`${idx}-${sidx}`}
                    signal={{
                      stock_code: row.stock_code,
                      stock_name: row.stock_name,
                      signal: sig.signal,
                      price: sig.price,
                      change_percent: sig.change_pct,
                      signal_time: sig.signal_time,
                      strategy: selectedStrategyNames[sidx],
                    }}
                  />
                ));
              })
            )}
          </Box>

          {/* 桌面端：表格 */}
          <Box sx={{ display: { xs: 'none', md: 'block' }, mt: 2, overflowX: 'auto' }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>股票</TableCell>
                  {selectedStrategyNames.map(name => {
                    const label = getStrategyLabel(name);
                    return <TableCell key={name}>{label} · 最新信号</TableCell>;
                  })}
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
                        暂无数据（先点击&quot;生成信号&quot;）
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
                  filteredRows.map(row => (
                    <TableRow
                      key={row.stock_code}
                      hover
                      sx={{ cursor: 'pointer' }}
                      onClick={() => openHistory(row.stock_code)}
                    >
                      <TableCell>{row.stock_code}</TableCell>
                      {selectedStrategyNames.map(name => {
                        const val = row.per_strategy?.[name] || null;
                        if (!val) {
                          return <TableCell key={name}>-</TableCell>;
                        }
                        const shortDate = val.signal_date ? val.signal_date.slice(0, 10) : '-';
                        const desc = `${val.latest_signal} @ ${shortDate}${
                          val.price != null ? ` · ${val.price}` : ''
                        }`;
                        return (
                          <TableCell key={name}>
                            <Typography
                              variant="body2"
                              noWrap
                              title={val.reason ? `${desc}\n原因：${val.reason}` : desc}
                            >
                              {desc}
                            </Typography>
                          </TableCell>
                        );
                      })}
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

      <Dialog
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
        maxWidth="md"
        fullWidth
        sx={{ '& .MuiDialog-paper': { m: { xs: 1, sm: 2 } } }}
      >
        <DialogTitle>
          {historyStock} ·{' '}
          {selectedStrategyNames.map(name => getStrategyLabel(name)).join('，') || '策略'} · 近
          {days}个交易日信号事件
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
          ) : !historyStrategies.length ? (
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
                      <TableCell>策略</TableCell>
                      <TableCell>信号</TableCell>
                      <TableCell align="right">强度</TableCell>
                      <TableCell align="right">价格</TableCell>
                      <TableCell>原因</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {historyStrategies.flatMap(strategyName => {
                      const events = historyEventsByStrategy?.[strategyName] || [];
                      return events.map((ev: SignalEvent, idx: number) => (
                        <TableRow key={`${strategyName}-${ev.timestamp}-${idx}`}>
                          <TableCell>{ev.timestamp}</TableCell>
                          <TableCell>{getStrategyLabel(strategyName)}</TableCell>
                          <TableCell>{ev.signal}</TableCell>
                          <TableCell align="right">{Number(ev.strength || 0).toFixed(3)}</TableCell>
                          <TableCell align="right">{ev.price}</TableCell>
                          <TableCell sx={{ maxWidth: 520 }}>
                            <Typography variant="body2" noWrap title={ev.reason || ''}>
                              {ev.reason || '-'}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ));
                    })}
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
