'use client';

/**
 * 策略信号 —— 全市场最新信号
 *
 * ⚠️ 2026-10-27 移动端 / 大数据量改造
 *   1. **5594 行全量渲染 → 浏览器 OOM**（实测文档高度 186,030px，移动端
 *      423,040px）。现在：
 *        - 默认走**服务端分页**（generateAll = false），每页 100 行；
 *        - 勾选「全量生成」时仍然串行翻页，但加了**进度条 + 百分比 + 停止
 *          按钮 + 单页超时**，并且结果走**客户端分页**渲染，DOM 节点数固定。
 *   2. **陈旧信号与当日信号混在一张表**：现在表头标注口径，陈旧行灰显 +
 *      「陈旧」标签，并提供「只看近 5 日」筛选。
 *   3. 长任务无反馈 → LinearProgress + 已翻页数/总页数 + 停止按钮。
 *   4. 错误态只有一句文案 → 换成带「重试」的错误态组件，并透传后端 detail。
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  FormControl,
  FormControlLabel,
  InputLabel,
  LinearProgress,
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
import { ErrorState } from '../../components/mobile/ErrorState';
import { MobileBottomNav } from '../../components/mobile/MobileBottomNav';
import { SafeAreaTop } from '../../components/mobile/SafeAreaTop';
import { Pagination } from '../../components/mobile/Pagination';
import { ResponsiveTable } from '../../components/mobile/ResponsiveTable';
import { BOTTOM_NAV_CONTENT_PADDING, MIN_TOUCH_TARGET } from '../../components/mobile/safeArea';
import { daysSince } from '../../components/mobile/tradingClient';

/** 服务端分页每页行数 */
const SERVER_PAGE_SIZE = 100;
/** 全量模式下客户端分页每页行数 */
const CLIENT_PAGE_SIZE = 100;
/** 单页请求超时（毫秒） */
const PAGE_TIMEOUT_MS = 60_000;
/** 「陈旧」判定阈值（天） */
const STALE_DAYS = 5;

/** 给单页请求加超时 —— apiRequest 层的 5 分钟超时太长，卡住时用户干等 */
const withTimeout = <T,>(promise: Promise<T>, ms: number, label: string): Promise<T> =>
  new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`${label}超时（${Math.round(ms / 1000)}s），已中止本次请求`));
    }, ms);
    promise.then(
      v => {
        clearTimeout(timer);
        resolve(v);
      },
      e => {
        clearTimeout(timer);
        reject(e);
      }
    );
  });

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

  const [limit, setLimit] = useState<number>(SERVER_PAGE_SIZE);
  const [offset, setOffset] = useState<number>(0);
  // 默认关闭「全量生成」：全量拉 5594 行会拖垮浏览器（实测 OOM）
  const [generateAll, setGenerateAll] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [errorDetail, setErrorDetail] = useState<unknown>(null);

  const [total, setTotal] = useState<number>(0);
  const [rows, setRows] = useState<MultiLatestSignalRow[]>([]);
  const [failures, setFailures] = useState<string[]>([]);

  // 全量生成的进度：已翻页数 / 总页数
  const [progress, setProgress] = useState<{ done: number; totalPages: number } | null>(null);
  const stopRef = useRef(false);
  // 全量模式下的客户端分页
  const [clientPage, setClientPage] = useState(0);

  // 前端筛选：按信号类型、日期范围
  const [signalFilter, setSignalFilter] = useState<'ALL' | 'BUY' | 'SELL' | 'HOLD'>('ALL');
  const [dateFrom, setDateFrom] = useState<string>('');
  const [dateTo, setDateTo] = useState<string>('');
  // 只看近 5 日
  const [recentOnly, setRecentOnly] = useState(false);

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

  const serverPage = useMemo(() => Math.floor(offset / Math.max(1, limit)) + 1, [offset, limit]);
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
          if (recentOnly) {
            const age = daysSince(d);
            if (age !== null && age > STALE_DAYS) {
              return false;
            }
          }
        }
        return true;
      });
    });
  }, [rows, selectedStrategyNames, signalFilter, dateFrom, dateTo, recentOnly]);

  // 全量模式下，渲染只取当前客户端分页的一页
  const visibleRows = useMemo(() => {
    if (!generateAll) {
      return filteredRows;
    }
    const start = clientPage * CLIENT_PAGE_SIZE;
    return filteredRows.slice(start, start + CLIENT_PAGE_SIZE);
  }, [generateAll, filteredRows, clientPage]);

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
        setErrorDetail(e?.response?.data ?? e);
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

    void loadSavedConfigs();
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
    setErrorDetail(null);
    stopRef.current = false;
    try {
      const pageSize = limit;
      const fetchPage = async (pageOffset: number) => {
        const label = `第 ${Math.floor(pageOffset / pageSize) + 1} 页`;
        if (strategyType === 'portfolio') {
          const resp = await withTimeout(
            DataService.getLatestSignals({
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
            }),
            PAGE_TIMEOUT_MS,
            label
          );
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

        const resp = await withTimeout(
          DataService.getLatestSignals({
            strategy_name: selectedStrategyName,
            strategy_config: strategyConfig,
            days,
            source,
            limit: pageSize,
            offset: pageOffset,
          }),
          PAGE_TIMEOUT_MS,
          label
        );
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
        let pageNo = 0;
        while (hasMore) {
          if (stopRef.current) {
            break;
          }
          const pageResp = await fetchPage(nextOffset);
          pageNo += 1;
          totalCount = pageResp.total || totalCount || nextOffset + pageResp.rows.length;
          allRows.push(...pageResp.rows);
          allFailures.push(...pageResp.failures);

          const totalPages = Math.max(1, Math.ceil(totalCount / pageSize));
          setProgress({ done: pageNo, totalPages });

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

        const stopped = stopRef.current;
        setRows(allRows);
        setTotal(totalCount);
        setFailures(allFailures);
        setOffset(0);
        setClientPage(0);
        setProgress(null);
        if (stopped) {
          setError(
            `已停止：共拉取 ${allRows.length} / ${totalCount} 条（${pageNo} 页）。已加载部分仍可查看。`
          );
        }
        return;
      }

      const pageResp = await fetchPage(realOffset);
      setRows(pageResp.rows);
      setTotal(pageResp.total);
      setFailures(pageResp.failures);
      setOffset(realOffset);
      setClientPage(0);
      setProgress(null);
    } catch (e: any) {
      setError(e?.message || '获取信号失败');
      setErrorDetail(e?.response?.data ?? e?.detail ?? e);
    } finally {
      setLoading(false);
      setProgress(null);
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

  /** 单个策略信号的时效：返回 { age, stale } */
  const signalAge = (signalDate?: string | null) => {
    if (!signalDate) {
      return { age: null as number | null, stale: true };
    }
    const age = daysSince(signalDate.slice(0, 10));
    return { age, stale: age !== null && age > STALE_DAYS };
  };

  const progressPct =
    progress && progress.totalPages > 0
      ? Math.min(100, Math.round((progress.done / progress.totalPages) * 100))
      : 0;

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2, md: 3 } }}>
      <SafeAreaTop />
      <MobileBottomNav />

      <Box sx={{ mb: 2 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, fontSize: { xs: '1.5rem', sm: '2rem' } }}>
          策略信号
        </Typography>
        <Typography variant="body2" color="text.secondary">
          选择一个或多个策略后，生成全市场（分页）最近 N
          个交易日窗口内的「最新信号」，点击某只股票可查看多策略近 N 日信号事件历史与价格走势。
        </Typography>
      </Box>

      <Card sx={{ mb: 2 }}>
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
                const v = Math.max(1, Math.min(2000, Number(e.target.value || SERVER_PAGE_SIZE)));
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
                  onChange={e => {
                    setGenerateAll(e.target.checked);
                    setOffset(0);
                    setClientPage(0);
                  }}
                  color="primary"
                />
              }
              label="全量生成"
            />

            <Button
              variant="contained"
              onClick={() => void fetchLatest(0)}
              disabled={!canGenerate || loading}
              sx={{ minHeight: MIN_TOUCH_TARGET }}
            >
              {loading ? '生成中...' : '生成信号'}
            </Button>
          </Stack>

          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
            「最新信号」= 近 {days} 个交易日窗口内该股票最后一次信号，
            <strong>不等于今天的信号</strong>。 勾选「全量生成」会串行翻完全市场（{' '}
            {total ? total.toLocaleString('zh-CN') : '数千'} 条），耗时较长且会占用较多内存 ——
            移动端建议保持关闭，用分页浏览。
          </Typography>

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
                      void handleLoadPortfolioConfig(configId);
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

          {/* 筛选条件：信号类型 + 日期范围 + 只看近 5 日 */}
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

            <FormControlLabel
              control={
                <Checkbox
                  checked={recentOnly}
                  onChange={e => setRecentOnly(e.target.checked)}
                  color="primary"
                  sx={{ p: 1 }}
                />
              }
              label={`只看近 ${STALE_DAYS} 日`}
              sx={{ minHeight: MIN_TOUCH_TARGET }}
            />
          </Stack>

          {/* 全量生成进度 */}
          {progress && (
            <Box sx={{ mt: 2 }} role="status" aria-live="polite">
              <Stack
                direction="row"
                justifyContent="space-between"
                alignItems="center"
                sx={{ mb: 0.75 }}
                spacing={1}
              >
                <Typography variant="body2" color="text.secondary">
                  正在翻页：第 {progress.done} / {progress.totalPages} 页（{progressPct}%）
                </Typography>
                <Button
                  size="small"
                  variant="outlined"
                  color="error"
                  onClick={() => {
                    stopRef.current = true;
                  }}
                  sx={{ minHeight: MIN_TOUCH_TARGET }}
                >
                  停止
                </Button>
              </Stack>
              <LinearProgress
                variant="determinate"
                value={progressPct}
                sx={{ height: 8, borderRadius: 4 }}
              />
            </Box>
          )}

          {error && (
            <Box sx={{ mt: 2 }}>
              <ErrorState
                title="信号生成失败"
                message={error}
                detail={errorDetail}
                onRetry={() => void fetchLatest(0)}
                retrying={loading}
              />
            </Box>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={2}
            alignItems={{ sm: 'center' }}
            justifyContent="space-between"
          >
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              {generateAll
                ? `最新信号（全量 ${rows.length.toLocaleString('zh-CN')} 条，已选策略：${
                    selectedStrategyNames.map(name => getStrategyLabel(name)).join('，') || '无'
                  }）`
                : `最新信号（第 ${serverPage} 页，本页 ${
                    filteredRows.length
                  } 条 / 共 ${total.toLocaleString('zh-CN')} 条，已选策略：${
                    selectedStrategyNames.map(name => getStrategyLabel(name)).join('，') || '无'
                  }）`}
            </Typography>
            {!generateAll && (
              <Stack direction="row" spacing={1}>
                <Button
                  variant="outlined"
                  disabled={loading || offset <= 0}
                  onClick={() => {
                    if (offset <= 0) {
                      return;
                    }
                    const nextOffset = Math.max(0, offset - limit);
                    void fetchLatest(nextOffset);
                  }}
                  sx={{ minHeight: MIN_TOUCH_TARGET }}
                >
                  上一页
                </Button>
                <Button
                  variant="outlined"
                  disabled={loading || offset + limit >= total}
                  onClick={() => {
                    if (offset + limit >= total) {
                      return;
                    }
                    const nextOffset = offset + limit;
                    void fetchLatest(nextOffset);
                  }}
                  sx={{ minHeight: MIN_TOUCH_TARGET }}
                >
                  下一页
                </Button>
              </Stack>
            )}
          </Stack>

          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
            口径：每行是「近 {days} 个交易日内该策略对这只股票的<strong>最后一次</strong>信号」。
            超过 {STALE_DAYS} 天的行会灰显并标注「陈旧」—— 它们<strong>不是</strong>今天的信号，
            请只把近 {STALE_DAYS} 日的行当作可执行参考。
          </Typography>

          {/* 移动端：卡片列表 */}
          <Box sx={{ display: { xs: 'block', md: 'none' }, mt: 2 }}>
            {loading ? (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <CircularProgress size={24} />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  计算中...
                </Typography>
              </Box>
            ) : visibleRows.length === 0 ? (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="body2" color="text.secondary">
                  暂无信号数据
                </Typography>
              </Box>
            ) : (
              visibleRows.map((row, idx) => {
                // 合并所有策略的信号（移动端简化显示）
                const signals = selectedStrategyNames
                  .map(name => ({ name, value: row.per_strategy?.[name] || null }))
                  .filter(
                    (
                      entry
                    ): entry is {
                      name: string;
                      value: NonNullable<(typeof entry)['value']>;
                    } => entry.value !== null
                  );
                if (signals.length === 0) {
                  return null;
                }

                return signals.map(({ name, value }, sidx) => {
                  const { age, stale } = signalAge(value.signal_date);
                  return (
                    <Box
                      key={`${idx}-${sidx}`}
                      sx={{ opacity: stale ? 0.6 : 1, position: 'relative' }}
                    >
                      <MobileSignalCard
                        signal={{
                          stock_code: row.stock_code,
                          stock_name: row.stock_name || undefined,
                          signal: value.latest_signal,
                          price: value.price ?? 0,
                          signal_time: value.signal_date || '',
                          strategy: name,
                        }}
                      />
                      {stale && (
                        <Typography
                          variant="caption"
                          color="warning.main"
                          sx={{ display: 'block', mt: -0.75, mb: 1, pl: 1.5 }}
                        >
                          ⚠ 陈旧信号（{age} 天前，非今日信号）
                        </Typography>
                      )}
                    </Box>
                  );
                });
              })
            )}
          </Box>

          {/* 桌面端：表格 */}
          <Box sx={{ display: { xs: 'none', md: 'block' }, mt: 2 }}>
            <ResponsiveTable minWidth={560}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>股票</TableCell>
                    {selectedStrategyNames.map(name => {
                      const label = getStrategyLabel(name);
                      return (
                        <TableCell key={name}>
                          {label} · 最新信号
                          <Typography variant="caption" color="text.secondary" display="block">
                            信号日期 / 强度
                          </Typography>
                        </TableCell>
                      );
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
                    visibleRows.map(row => (
                      <TableRow
                        key={row.stock_code}
                        hover
                        sx={{ cursor: 'pointer' }}
                        onClick={() => void openHistory(row.stock_code)}
                      >
                        <TableCell>{row.stock_code}</TableCell>
                        {selectedStrategyNames.map(name => {
                          const val = row.per_strategy?.[name] || null;
                          if (!val) {
                            return <TableCell key={name}>-</TableCell>;
                          }
                          const shortDate = val.signal_date ? val.signal_date.slice(0, 10) : '-';
                          const { age, stale } = signalAge(val.signal_date);
                          const desc = `${val.latest_signal} @ ${shortDate}${
                            val.price != null ? ` · ${val.price}` : ''
                          }`;
                          return (
                            <TableCell key={name} sx={{ opacity: stale ? 0.55 : 1 }}>
                              <Typography
                                variant="body2"
                                noWrap
                                title={val.reason ? `${desc}\n原因：${val.reason}` : desc}
                              >
                                {desc}
                              </Typography>
                              {stale && (
                                <Typography variant="caption" color="warning.main">
                                  ⚠ 陈旧（{age} 天前）
                                </Typography>
                              )}
                            </TableCell>
                          );
                        })}
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </ResponsiveTable>
          </Box>

          {/* 分页（全量模式下用客户端分页） */}
          {generateAll && filteredRows.length > 0 && (
            <Pagination
              total={filteredRows.length}
              pageSize={CLIENT_PAGE_SIZE}
              page={clientPage}
              onPageChange={setClientPage}
              label={`已筛选（原始 ${rows.length.toLocaleString('zh-CN')} 条）`}
            />
          )}

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
            <Box sx={{ py: 1 }}>
              <ErrorState
                title="信号历史加载失败"
                message={historyError}
                onRetry={() => void openHistory(historyStock)}
                retrying={historyLoading}
              />
            </Box>
          ) : !historyStrategies.length ? (
            <Typography variant="body2" color="text.secondary" sx={{ py: 1 }}>
              窗口内无 BUY/SELL 事件
            </Typography>
          ) : (
            <Stack spacing={2} sx={{ py: 1 }}>
              {/* 价格走势图（从后端/本地 parquet 读取K线数据），并叠加 BUY/SELL 信号标记 */}
              <Box
                role="img"
                aria-label={`${historyStock} 的价格走势与买卖信号标记图，共 ${chartSignals.length} 个信号事件`}
              >
                <TradingViewChart
                  stockCode={historyStock}
                  startDate={chartStartDate}
                  endDate={chartEndDate}
                  signals={chartSignals}
                  showSignals
                  showTrades={false}
                  height={320}
                />
              </Box>

              {/* 信号明细表，与回测结果中的信号记录表现一致 */}
              <ResponsiveTable minWidth={720}>
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
                          <TableCell>
                            <Box
                              component="span"
                              sx={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: 0.25,
                                color: ev.signal === 'BUY' ? 'success.main' : 'error.main',
                                fontWeight: 600,
                              }}
                            >
                              <span aria-hidden="true">{ev.signal === 'BUY' ? '↑' : '↓'}</span>
                              {ev.signal}
                            </Box>
                          </TableCell>
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
              </ResponsiveTable>
            </Stack>
          )}
        </DialogContent>
      </Dialog>

      <Box
        sx={{ height: BOTTOM_NAV_CONTENT_PADDING, display: { xs: 'block', md: 'none' } }}
        aria-hidden="true"
      />
    </Box>
  );
}
