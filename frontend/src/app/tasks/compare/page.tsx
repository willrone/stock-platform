'use client';

/**
 * 配置对比页面
 *
 * 提供多任务配置对比功能：
 * - 选择 2–5 个已完成的回测任务
 * - 调取后端 compare-configs API 对比配置
 * - diff/full 模式切换
 * - 差异行高亮展示
 */

import React, { useEffect, useState, useMemo } from 'react';
import {
  Autocomplete,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';
import { ArrowLeft, GitCompare } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { apiRequest } from '@/services/api';
import type { Task, TaskListResponse } from '@/types/task';

/* ============================================================
   Types
   ============================================================ */

interface TaskBrief {
  task_id: string;
  task_name: string;
  created_at: string;
  status: string;
}

interface ParameterValue {
  task_id: string;
  value: unknown;
}

interface ParameterComparison {
  /** 参数内部 key，用于 React key */
  param_key: string;
  /** 展示用参数名 */
  param_name: string;
  /** 每个任务的对应值 */
  values: ParameterValue[];
  /** 该参数在各任务间是否存在差异 */
  is_diff: boolean;
}

interface CompareResponse {
  tasks: TaskBrief[];
  parameters: ParameterComparison[];
  diff_count: number;
  total_count: number;
}

/* ============================================================
   Helpers
   ============================================================ */

/** 将任意值转为可读字符串 */
function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

/** 找出 values 中与"多数值"不同的 task_id 集合 */
function getMinorityTaskIds(values: ParameterValue[]): Set<string> {
  if (values.length <= 1) return new Set();

  // 序列化后做频次统计
  const freq = new Map<string, number>();
  for (const v of values) {
    const key = JSON.stringify(v.value);
    freq.set(key, (freq.get(key) ?? 0) + 1);
  }

  // 出现次数最多的 key 就是"多数值"
  let majorityKey = '';
  let maxCount = 0;
  freq.forEach((count, key) => {
    if (count > maxCount) {
      maxCount = count;
      majorityKey = key;
    }
  });

  // 与多数值不同的视为 minority
  const minority = new Set<string>();
  for (const v of values) {
    if (JSON.stringify(v.value) !== majorityKey) {
      minority.add(v.task_id);
    }
  }

  return minority;
}

/** 获取状态标签色 */
const statusColorMap: Record<
  string,
  'success' | 'error' | 'warning' | 'info' | 'primary' | 'secondary' | 'default'
> = {
  completed: 'success',
  failed: 'error',
  running: 'primary',
  created: 'default',
  queued: 'info',
  cancelled: 'warning',
  paused: 'warning',
};

/* ============================================================
   Main Component
   ============================================================ */

const MAX_SELECTION = 5;

export default function ComparePage() {
  const router = useRouter();

  /* ---------- state ---------- */

  const [allBacktestTasks, setAllBacktestTasks] = useState<Task[]>([]);
  const [loadingTasks, setLoadingTasks] = useState(true);
  const [selectedTasks, setSelectedTasks] = useState<Task[]>([]);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [comparing, setComparing] = useState(false);
  const [mode, setMode] = useState<'diff' | 'full'>('diff');
  const [error, setError] = useState<string | null>(null);

  /* ---------- fetch tasks ---------- */

  useEffect(() => {
    let cancelled = false;

    const fetchTasks = async () => {
      setLoadingTasks(true);
      try {
        // 拉取已完成任务列表，前端过滤出 backtest 类型
        const result = await apiRequest.get<TaskListResponse>('/tasks', {
          status: 'completed',
          limit: 100,
        });
        if (!cancelled) {
          const backtestTasks = (result?.tasks ?? []).filter(
            (t: Task) => t.task_type === 'backtest'
          );
          setAllBacktestTasks(backtestTasks);
        }
      } catch {
        if (!cancelled) setError('加载任务列表失败');
      } finally {
        if (!cancelled) setLoadingTasks(false);
      }
    };

    fetchTasks();
    return () => {
      cancelled = true;
    };
  }, []);

  /* ---------- compare ---------- */

  const handleCompare = async () => {
    if (selectedTasks.length < 2) return;

    setComparing(true);
    setError(null);
    setCompareResult(null);

    try {
      const result = await apiRequest.post<CompareResponse>('/tasks/compare-configs', {
        task_ids: selectedTasks.map(t => t.task_id),
      });
      setCompareResult(result);
    } catch {
      setError('对比请求失败，请稍后重试');
    } finally {
      setComparing(false);
    }
  };

  /* ---------- ui handlers ---------- */

  const handleModeChange = (_: React.MouseEvent<HTMLElement>, newMode: 'diff' | 'full') => {
    if (newMode) setMode(newMode);
  };

  const handleSelectionChange = (_: React.SyntheticEvent, value: Task[]) => {
    setSelectedTasks(value.slice(0, MAX_SELECTION));
  };

  /* ---------- derived data ---------- */

  const allParameters = compareResult?.parameters ?? [];

  const displayedParameters = useMemo(() => {
    if (mode === 'diff') {
      return allParameters.filter(p => p.is_diff);
    }
    return allParameters;
  }, [allParameters, mode]);

  // 预计算每个 diff parameter 的 minority task ids，用于单元格级高亮
  const minorityMap = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const param of allParameters) {
      if (param.is_diff) {
        map.set(param.param_key, getMinorityTaskIds(param.values));
      }
    }
    return map;
  }, [allParameters]);

  const numTotal = compareResult?.total_count ?? 0;
  const numDiffs = compareResult?.diff_count ?? 0;

  /* ============================================================
     Render
     ============================================================ */

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* ---- Header ---- */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <IconButton onClick={() => router.push('/tasks')}>
          <ArrowLeft size={20} />
        </IconButton>
        <Box>
          <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
            配置对比
          </Typography>
          <Typography variant="body2" color="text.secondary">
            选择 2–5 个已完成回测任务，对比配置参数差异
          </Typography>
        </Box>
      </Box>

      {/* ---- Task Selector ---- */}
      <Card>
        <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Autocomplete<Task, true, false, false>
            multiple
            options={allBacktestTasks}
            loading={loadingTasks}
            value={selectedTasks}
            onChange={handleSelectionChange}
            getOptionLabel={option => option.task_name}
            isOptionEqualToValue={(o, v) => o.task_id === v.task_id}
            filterSelectedOptions
            noOptionsText={loadingTasks ? '加载中…' : '没有符合条件的已完成回测任务'}
            renderInput={params => (
              <TextField
                {...params}
                label="选择对比任务"
                placeholder="搜索任务名称"
                helperText={
                  selectedTasks.length >= MAX_SELECTION
                    ? `已达到选择上限（${MAX_SELECTION}个）`
                    : `已选择 ${selectedTasks.length}/${MAX_SELECTION} 个任务`
                }
                InputProps={{
                  ...params.InputProps,
                  endAdornment: (
                    <>
                      {loadingTasks ? <CircularProgress size={20} /> : null}
                      {params.InputProps.endAdornment}
                    </>
                  ),
                }}
              />
            )}
            renderTags={(value, getTagProps) =>
              value.map((option, index) => (
                <Chip
                  label={option.task_name}
                  size="small"
                  {...getTagProps({ index })}
                  key={option.task_id}
                />
              ))
            }
          />

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Button
              variant="contained"
              color="primary"
              startIcon={
                comparing ? (
                  <CircularProgress size={16} color="inherit" />
                ) : (
                  <GitCompare size={16} />
                )
              }
              onClick={handleCompare}
              disabled={selectedTasks.length < 2 || comparing}
            >
              {comparing ? '对比中…' : `对比配置 (${selectedTasks.length})`}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* ---- Error Banner ---- */}
      {error && (
        <Box
          sx={{
            p: 2,
            bgcolor: 'error.main',
            color: 'error.contrastText',
            borderRadius: 1,
          }}
        >
          <Typography variant="body2">{error}</Typography>
        </Box>
      )}

      {/* ---- Results ---- */}
      {compareResult && (
        <>
          {/* === Task Info Cards === */}
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: {
                xs: '1fr',
                sm: 'repeat(2, 1fr)',
                md: 'repeat(3, 1fr)',
              },
              gap: 2,
            }}
          >
            {compareResult.tasks.map(task => (
              <Card key={task.task_id} variant="outlined">
                <CardContent>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                    {task.task_name}
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      ID: {task.task_id}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      创建时间: {new Date(task.created_at).toLocaleString()}
                    </Typography>
                    <Chip
                      label={task.status}
                      size="small"
                      color={statusColorMap[task.status] ?? 'default'}
                      sx={{ width: 'fit-content' }}
                    />
                  </Box>
                </CardContent>
              </Card>
            ))}
          </Box>

          {/* === Summary + Mode Toggle === */}
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: 1,
            }}
          >
            <Typography variant="body1" sx={{ fontWeight: 500 }}>
              共{' '}
              <Typography component="span" sx={{ fontWeight: 700 }}>
                {numTotal}
              </Typography>{' '}
              个参数，其中{' '}
              <Typography
                component="span"
                color={numDiffs > 0 ? 'warning.main' : 'success.main'}
                sx={{ fontWeight: 700 }}
              >
                {numDiffs}
              </Typography>{' '}
              个存在差异
            </Typography>

            <ToggleButtonGroup value={mode} exclusive onChange={handleModeChange} size="small">
              <ToggleButton value="diff">仅差异</ToggleButton>
              <ToggleButton value="full">全部</ToggleButton>
            </ToggleButtonGroup>
          </Box>

          {/* === Parameter Comparison Table === */}
          <TableContainer component={Card}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell
                    sx={{
                      fontWeight: 700,
                      minWidth: 220,
                      position: 'sticky',
                      left: 0,
                      bgcolor: 'background.paper',
                      zIndex: 1,
                    }}
                  >
                    参数名
                  </TableCell>
                  {compareResult.tasks.map(task => (
                    <TableCell key={task.task_id} sx={{ fontWeight: 700, minWidth: 160 }}>
                      {task.task_name}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {displayedParameters.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={compareResult.tasks.length + 1}
                      align="center"
                      sx={{ py: 4 }}
                    >
                      <Typography variant="body2" color="text.secondary">
                        {mode === 'diff' ? '所有参数均相同，无差异' : '暂无可显示的参数'}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  displayedParameters.map(param => {
                    const isDiff = param.is_diff;
                    const minorityIds = minorityMap.get(param.param_key) ?? new Set();

                    return (
                      <TableRow
                        key={param.param_key}
                        hover
                        sx={{
                          bgcolor: isDiff ? 'warning.light' : undefined,
                          '&:hover': {
                            bgcolor: isDiff ? '#fff3cd' : 'action.hover',
                          },
                        }}
                      >
                        <TableCell
                          sx={{
                            position: 'sticky',
                            left: 0,
                            bgcolor: isDiff ? 'warning.light' : 'background.paper',
                            zIndex: 1,
                          }}
                        >
                          <Typography
                            variant="body2"
                            sx={{
                              fontWeight: 500,
                              fontFamily: 'monospace',
                              fontSize: '0.8125rem',
                            }}
                          >
                            {param.param_name}
                          </Typography>
                        </TableCell>
                        {compareResult.tasks.map(task => {
                          const pv = param.values.find(v => v.task_id === task.task_id);
                          const cellIsMinority = minorityIds.has(task.task_id);
                          const displayVal = formatValue(pv?.value);

                          return (
                            <TableCell
                              key={task.task_id}
                              sx={{
                                bgcolor: cellIsMinority ? 'error.light' : undefined,
                                color: cellIsMinority ? 'error.contrastText' : undefined,
                              }}
                            >
                              <Typography
                                variant="body2"
                                sx={{
                                  fontFamily: 'monospace',
                                  fontSize: '0.8125rem',
                                  fontWeight: cellIsMinority ? 700 : 400,
                                  wordBreak: 'break-all',
                                }}
                              >
                                {displayVal}
                              </Typography>
                            </TableCell>
                          );
                        })}
                      </TableRow>
                    );
                  })
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      )}
    </Box>
  );
}
