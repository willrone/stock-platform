'use client';

/**
 * 仪表板页面
 *
 * 显示系统概览信息，包括：
 * - 任务统计
 * - 系统状态
 * - 最近活动
 * - 快速操作
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Button,
  Chip,
  Avatar,
  Box,
  Stack,
  Typography,
  LinearProgress,
  Alert,
  Skeleton,
} from '@mui/material';
import {
  Bot,
  CheckCircle,
  Clock,
  AlertTriangle,
  TrendingUp,
  Database,
  Wifi,
  Plus,
  Eye,
  Activity,
  Signal,
} from 'lucide-react';
import { useRouter } from 'next/navigation';

import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { TaskService } from '../../services/taskService';
import { apiRequest } from '../../services/api';
import { useSnackbarStore } from '../../stores/useSnackbarStore';
import { MobileBottomNav } from '../../components/mobile/MobileBottomNav';
import { SafeAreaTop } from '../../components/mobile/SafeAreaTop';
import { DeltaValue } from '../../components/mobile/DeltaValue';
import { BOTTOM_NAV_CONTENT_PADDING, MIN_TOUCH_TARGET } from '../../components/mobile/safeArea';
import { tradingGet, daysSince } from '../../components/mobile/tradingClient';

/** 后端 /monitoring/health 里单个服务的原始结构 */
type ServiceHealthRaw = {
  healthy?: boolean;
  status_label?: string;
  response_time_ms?: number;
  core?: boolean;
  optional?: boolean;
  error_message?: string | null;
  service_url?: string;
};

/** 解析后的服务健康状态（前端展示用） */
type ServiceHealth = {
  key: string;
  healthy: boolean;
  label: string;
  responseTimeMs?: number;
  core: boolean;
  errorMessage?: string;
};

/** 服务 key → 展示名（后端 key 是技术命名，UI 用中文） */
const SERVICE_LABEL: Record<string, string> = {
  remote_data_service: '数据服务',
  indicators_service: '指标服务',
  parquet_manager: '数据仓库',
  api_service: 'API服务',
  model_service: '模型服务',
};

export default function DashboardPage() {
  const router = useRouter();

  const showSnackbar = useSnackbarStore();

  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [recentTasks, setRecentTasks] = useState<any[]>([]);
  const [systemStats, setSystemStats] = useState({
    totalTasks: 0,
    runningTasks: 0,
    completedTasks: 0,
    failedTasks: 0,
    dataFiles: 0,
    systemHealth: 'good' as 'good' | 'warning' | 'error',
  });

  /**
   * 【P2-12】真实服务健康状态。
   * 后端 GET /monitoring/health 的 data.services 是
   * `{ [serviceKey]: { healthy, status_label, response_time_ms, core, ... } }`
   * —— 注意不是字符串，旧的 `s === 'unhealthy'` 比较永远为 false，
   * 导致 systemHealth 恒为 'good'，而 UI 又是写死的字面量，
   * 后端全挂了页面照样显示「正常」。现在按对象契约解析。
   */
  const [services, setServices] = useState<ServiceHealth[]>([]);

  // ── 今日信号摘要（2026-10-27）──
  // 之前仪表板与「今天买什么」零关联：用户进平台第一眼看不到今日信号。
  // 这里在顶部加一条摘要：买入 / 卖出 / 持有 数量 + 跳转 /today-signal。
  const [todaySummary, setTodaySummary] = useState<{
    strategy: string;
    displayName: string;
    date?: string;
    buys: number;
    sells: number;
    holds: number;
    value?: number;
    totalReturnPct?: number;
  } | null>(null);
  const [todaySummaryError, setTodaySummaryError] = useState<string | null>(null);
  const [todaySummaryLoading, setTodaySummaryLoading] = useState(false);

  const loadTodaySummary = React.useCallback(async () => {
    setTodaySummaryLoading(true);
    setTodaySummaryError(null);
    try {
      const list = await tradingGet<{
        strategies: Array<{ key: string; locked?: boolean; n_positions: number }>;
      }>('/trading/strategies');
      const usable = (list.strategies || []).filter(s => !s.locked);
      const target = usable.find(s => s.n_positions > 0) || usable[0];
      if (!target) {
        setTodaySummary(null);
        return;
      }
      const sig = await tradingGet<{
        display_name: string;
        date?: string;
        buys?: unknown[];
        sells?: unknown[];
        holds?: unknown[];
        value?: number;
        total_return_pct?: number;
      }>(`/trading/signal?strategy=${encodeURIComponent(target.key)}`);
      setTodaySummary({
        strategy: target.key,
        displayName: sig.display_name || target.key,
        date: sig.date,
        buys: sig.buys?.length ?? 0,
        sells: sig.sells?.length ?? 0,
        holds: sig.holds?.length ?? 0,
        value: sig.value,
        totalReturnPct: sig.total_return_pct,
      });
    } catch (e) {
      setTodaySummary(null);
      setTodaySummaryError(e instanceof Error ? e.message : '加载今日信号摘要失败');
    } finally {
      setTodaySummaryLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadTodaySummary();
  }, [loadTodaySummary]);

  /**
   * 加载仪表板数据。
   * 【P2-12】必须同时拉 /monitoring/health —— 之前的版本在错误重试分支
   * 里另抄了一份只拉 tasks+stats 的逻辑，并把 systemHealth 写死成 'good'、
   * dataFiles 写死成 156，于是「重试」后系统状态反而永远是好的。
   * 现在单一 loader，重试与首屏走同一条路径。
   */
  const loadDashboardData = useCallback(async () => {
      try {
        setLoading(true);

        // 并行请求：最近任务 + 统计数据 + 系统健康状态
        const [tasksResult, statsResult, healthResult] = await Promise.all([
          TaskService.getTasks(undefined, 5, 0),
          TaskService.getTaskStats(),
          apiRequest
            .get<{ overall_healthy?: boolean; services?: Record<string, ServiceHealthRaw> }>(
              '/monitoring/health'
            )
            .catch(() => null),
        ]);

        setRecentTasks(tasksResult.tasks);

        // 【P2-12】从后端取真实健康状态（apiRequest 已拆信封，services 在顶层）
        const rawServices = healthResult?.services ?? {};
        const parsed = Object.entries(rawServices).map(([key, s]) => ({
          key,
          healthy: s?.healthy !== false,
          label: s?.status_label ?? (s?.healthy === false ? '异常' : '正常'),
          responseTimeMs: s?.response_time_ms,
          core: s?.core === true,
          errorMessage: s?.error_message ?? undefined,
        }));
        setServices(parsed);

        const hasError = parsed.some(s => !s.healthy);
        const hasWarning = parsed.some(s => s.healthy && s.label !== '正常');

        setSystemStats({
          totalTasks: statsResult.total,
          runningTasks: statsResult.running,
          completedTasks: statsResult.completed,
          failedTasks: statsResult.failed,
          dataFiles: 0,
          systemHealth: hasError ? 'error' : hasWarning ? 'warning' : 'good',
        });
      } catch (error) {
        const errMsg = error instanceof Error ? error.message : '加载仪表板数据失败';
        console.error('加载仪表板数据失败:', error);
        setLoadError(errMsg);
        showSnackbar.showError(errMsg);
      } finally {
        setLoading(false);
      }
  }, [showSnackbar]);

  useEffect(() => {
    void loadDashboardData();
  }, [loadDashboardData]);

  // 获取任务状态颜色
  const getTaskStatusColor = (status: string): 'primary' | 'success' | 'error' | 'default' => {
    switch (status) {
      case 'running':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  // 获取任务状态文本
  const getTaskStatusText = (status: string) => {
    switch (status) {
      case 'running':
        return '运行中';
      case 'completed':
        return '已完成';
      case 'failed':
        return '失败';
      default:
        return '已创建';
    }
  };

  // 获取任务状态图标
  const getTaskStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Clock size={16} />;
      case 'completed':
        return <CheckCircle size={16} />;
      case 'failed':
        return <AlertTriangle size={16} />;
      default:
        return <Bot size={16} />;
    }
  };

  if (loading) {
    return <LoadingSpinner text="加载仪表板数据..." />;
  }

  if (loadError) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <Box>
          <Typography variant="h4" component="h1" sx={{ fontWeight: 600, mb: 1 }}>
            仪表板
          </Typography>
          <Typography variant="body2" color="text.secondary">
            系统概览和快速操作
          </Typography>
        </Box>
        <Alert
          severity="error"
          variant="filled"
          sx={{ borderRadius: 2 }}
          action={
            <Button
              color="inherit"
              size="small"
              onClick={() => {
                setLoadError(null);
                setLoading(true);
                void loadDashboardData();
              }}
            >
              重试
            </Button>
          }
        >
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            加载失败
          </Typography>
          <Typography variant="caption">{loadError}</Typography>
        </Alert>
        {/* 骨架屏：错误时显示占位 */}
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' },
            gap: 2,
          }}
        >
          {[1, 2, 3, 4].map(i => (
            <Card key={i}>
              <CardContent>
                <Skeleton variant="circular" width={40} height={40} />
                <Skeleton variant="text" width="60%" sx={{ mt: 1 }} />
                <Skeleton variant="text" width="40%" />
              </CardContent>
            </Card>
          ))}
        </Box>
      </Box>
    );
  }

  const todayAge = daysSince(todaySummary?.date);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, pb: { xs: 10, md: 3 } }}>
      <SafeAreaTop />
      <MobileBottomNav />

      {/* ── 今日信号摘要（进平台第一眼就能看到「今天买什么」）── */}
      <Card sx={{ borderLeft: 4, borderLeftColor: 'primary.main' }}>
        <CardContent>
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={1.5}
            alignItems={{ sm: 'center' }}
            justifyContent="space-between"
          >
            <Box sx={{ minWidth: 0 }}>
              <Typography variant="subtitle1" fontWeight={700}>
                今日信号
                {todaySummary?.date ? `（${todaySummary.date}）` : ''}
              </Typography>
              {todaySummaryLoading && (
                <Typography variant="body2" color="text.secondary">
                  加载中…
                </Typography>
              )}
              {todaySummary && (
                <Stack direction="row" spacing={1} sx={{ mt: 0.5 }} flexWrap="wrap" useFlexGap>
                  <Chip
                    size="small"
                    color="success"
                    label={`↑ 买入 ${todaySummary.buys}`}
                    sx={{ height: 28 }}
                  />
                  <Chip
                    size="small"
                    color="error"
                    label={`↓ 卖出 ${todaySummary.sells}`}
                    sx={{ height: 28 }}
                  />
                  <Chip size="small" variant="outlined" label={`持有 ${todaySummary.holds}`} sx={{ height: 28 }} />
                  {todaySummary.totalReturnPct !== undefined && (
                    <DeltaValue value={todaySummary.totalReturnPct} variant="body2" />
                  )}
                  {todayAge !== null && todayAge > 7 && (
                    <Chip size="small" color="warning" label={`已过期 ${todayAge} 天`} sx={{ height: 28 }} />
                  )}
                </Stack>
              )}
              {todaySummaryError && (
                <Typography variant="caption" color="error" sx={{ display: 'block', mt: 0.5 }}>
                  {todaySummaryError}
                </Typography>
              )}
            </Box>
            <Stack direction="row" spacing={1}>
              <Button
                size="small"
                onClick={() => void loadTodaySummary()}
                disabled={todaySummaryLoading}
                sx={{ minHeight: MIN_TOUCH_TARGET }}
              >
                刷新
              </Button>
              <Button
                size="small"
                variant="contained"
                onClick={() => router.push('/today-signal')}
                sx={{ minHeight: MIN_TOUCH_TARGET }}
              >
                查看完整清单
              </Button>
            </Stack>
          </Stack>
        </CardContent>
      </Card>

      {/* 页面标题 */}
      <Box>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600, mb: 1 }}>
          仪表板
        </Typography>
        <Typography variant="body2" color="text.secondary">
          系统概览和快速操作
        </Typography>
      </Box>

      {/* 统计卡片 */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' },
          gap: 2,
        }}
      >
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box
                sx={{
                  p: 1.5,
                  bgcolor: 'primary.light',
                  borderRadius: 1,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <Bot size={24} color="white" />
              </Box>
              <Box>
                <Typography variant="h4" sx={{ fontWeight: 600 }}>
                  {systemStats.totalTasks}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  总任务数
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box
                sx={{
                  p: 1.5,
                  bgcolor: 'success.light',
                  borderRadius: 1,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <CheckCircle size={24} color="white" />
              </Box>
              <Box>
                <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                  {systemStats.completedTasks}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  已完成
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box
                sx={{
                  p: 1.5,
                  bgcolor: 'warning.light',
                  borderRadius: 1,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <Clock size={24} color="white" />
              </Box>
              <Box>
                <Typography variant="h4" sx={{ fontWeight: 600, color: 'warning.main' }}>
                  {systemStats.runningTasks}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  运行中
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box
                sx={{
                  p: 1.5,
                  bgcolor: 'error.light',
                  borderRadius: 1,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <AlertTriangle size={24} color="white" />
              </Box>
              <Box>
                <Typography variant="h4" sx={{ fontWeight: 600, color: 'error.main' }}>
                  {systemStats.failedTasks}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  失败
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Box>

      <Box
        sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '2fr 1fr' }, gap: 3, mt: 3 }}
      >
        {/* 最近任务 */}
        <Box>
          <Card>
            <CardHeader
              title="最近任务"
              action={
                <Button size="small" onClick={() => router.push('/tasks')}>
                  查看全部
                </Button>
              }
            />
            <CardContent>
              {recentTasks.length === 0 ? (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Bot size={48} color="#ccc" style={{ margin: '0 auto 16px' }} />
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    暂无任务
                  </Typography>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={<Plus size={16} />}
                    onClick={() => router.push('/tasks/create')}
                  >
                    创建任务
                  </Button>
                </Box>
              ) : (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {recentTasks.map(task => (
                    <Box
                      key={task.task_id}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        p: 2,
                        border: 1,
                        borderColor: 'divider',
                        borderRadius: 1,
                        cursor: 'pointer',
                        '&:hover': { bgcolor: 'action.hover' },
                      }}
                      onClick={() => router.push(`/tasks/${task.task_id}`)}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Avatar sx={{ bgcolor: 'grey.200', width: 32, height: 32 }}>
                          {getTaskStatusIcon(task.status)}
                        </Avatar>
                        <Box>
                          <Typography variant="body1" sx={{ fontWeight: 500 }}>
                            {task.task_name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {task.stock_codes?.length || 0} 只股票
                          </Typography>
                        </Box>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Chip
                          label={getTaskStatusText(task.status)}
                          color={getTaskStatusColor(task.status)}
                          size="small"
                        />
                        {task.status === 'running' && (
                          <Box sx={{ width: 64 }}>
                            <LinearProgress variant="determinate" value={task.progress} />
                          </Box>
                        )}
                      </Box>
                    </Box>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Box>

        {/* 系统状态和快速操作 */}
        <Box>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* 系统状态 */}
            <Card>
              <CardHeader title="系统状态" />
              <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {/* 【P2-12】服务状态来自 /monitoring/health 真实响应。
                    后端挂了 / 探针失败 → 显示「未知」，不谎报「正常」。 */}
                {services.length === 0 ? (
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Activity size={16} color="#9e9e9e" />
                      <Typography variant="body2">服务状态</Typography>
                    </Box>
                    <Chip label="未知" size="small" />
                  </Box>
                ) : (
                  services.map(s => (
                    <Box
                      key={s.key}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {s.key.includes('data') || s.key.includes('parquet') ? (
                          <Database
                            size={16}
                            color={s.healthy ? '#2e7d32' : '#d32f2f'}
                          />
                        ) : s.key.includes('indicator') ? (
                          <Activity
                            size={16}
                            color={s.healthy ? '#2e7d32' : '#d32f2f'}
                          />
                        ) : (
                          <Wifi size={16} color={s.healthy ? '#2e7d32' : '#d32f2f'} />
                        )}
                        <Typography variant="body2">
                          {SERVICE_LABEL[s.key] ?? s.key}
                        </Typography>
                      </Box>
                      <Chip
                        label={s.label}
                        color={s.healthy ? 'success' : 'error'}
                        size="small"
                        title={
                          s.errorMessage ??
                          (s.responseTimeMs != null
                            ? `响应 ${s.responseTimeMs.toFixed(0)}ms`
                            : undefined)
                        }
                      />
                    </Box>
                  ))
                )}

                {/* 【P2-12】系统负载：后端未提供真实指标时显示「—」，
                    不用 65% 这类假值糊弄。 */}
                <Box sx={{ pt: 2, borderTop: 1, borderColor: 'divider' }}>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      mb: 1,
                    }}
                  >
                    <Typography variant="body2" color="text.secondary">
                      系统状态
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {services.length === 0
                        ? '—'
                        : systemStats.systemHealth === 'good'
                          ? '全部正常'
                          : systemStats.systemHealth === 'warning'
                            ? '部分降级'
                            : '存在异常'}
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={
                      services.length === 0
                        ? 0
                        : systemStats.systemHealth === 'good'
                          ? 100
                          : systemStats.systemHealth === 'warning'
                            ? 60
                            : 20
                    }
                    color={
                      systemStats.systemHealth === 'good'
                        ? 'success'
                        : systemStats.systemHealth === 'warning'
                          ? 'warning'
                          : 'error'
                    }
                  />
                </Box>
              </CardContent>
            </Card>

            {/* 快速操作 */}
            <Card>
              <CardHeader title="快速操作" />
              <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<Signal size={16} />}
                  onClick={() => router.push('/today-signal')}
                  fullWidth
                  sx={{ minHeight: MIN_TOUCH_TARGET }}
                >
                  今日信号
                </Button>

                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<Plus size={16} />}
                  onClick={() => router.push('/tasks/create')}
                  fullWidth
                  sx={{ minHeight: MIN_TOUCH_TARGET }}
                >
                  创建预测任务
                </Button>

                <Button
                  variant="outlined"
                  startIcon={<Eye size={16} />}
                  onClick={() => router.push('/tasks')}
                  fullWidth
                  sx={{ minHeight: MIN_TOUCH_TARGET }}
                >
                  查看所有任务
                </Button>

                <Button
                  variant="outlined"
                  startIcon={<Database size={16} />}
                  onClick={() => router.push('/data')}
                  fullWidth
                  sx={{ minHeight: MIN_TOUCH_TARGET }}
                >
                  数据管理
                </Button>

                <Button
                  variant="outlined"
                  startIcon={<TrendingUp size={16} />}
                  onClick={() => router.push('/predictions')}
                  fullWidth
                  sx={{ minHeight: MIN_TOUCH_TARGET }}
                >
                  预测分析
                </Button>
              </CardContent>
            </Card>
          </Box>
        </Box>
      </Box>

      <Box sx={{ height: BOTTOM_NAV_CONTENT_PADDING, display: { xs: 'block', md: 'none' } }} aria-hidden="true" />
    </Box>
  );
}
