/**
 * 性能监控标签页内容
 */

import React from 'react';
import {
  Box,
  Card,
  CardHeader,
  CardContent,
  Typography,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TableContainer,
  Paper,
} from '@mui/material';
import { Activity } from 'lucide-react';
import { Task } from '@/stores/useTaskStore';

interface PerformanceMonitorProps {
  task: Task;
}

export function PerformanceMonitor({ task }: PerformanceMonitorProps) {
  const backtestData = task.result || task.results?.backtest_results || task.backtest_results;
  const perf = backtestData?.performance_analysis as any;

  if (!backtestData) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Activity size={48} color="#999" style={{ margin: '0 auto 16px' }} />
        <Typography variant="body2" color="text.secondary">
          暂无回测结果数据，无法展示性能分析。
        </Typography>
      </Box>
    );
  }

  if (!perf) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Activity size={48} color="#999" style={{ margin: '0 auto 16px' }} />
        <Typography variant="body2" color="text.secondary">
          当前回测未启用性能监控，或后端尚未写入性能报告。
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          请确认后端创建回测执行器时已开启 enable_performance_profiling，并返回 performance_analysis
          字段。
        </Typography>
      </Box>
    );
  }

  const summary = perf.summary || {};
  const stages = perf.stages || {};
  const functionCalls = perf.function_calls || {};
  const parallel = perf.parallel_efficiency || {};

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* 总览卡片 */}
      <Card>
        <CardHeader title="整体性能概要" />
        <CardContent>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', sm: 'repeat(3, 1fr)' },
              gap: 2,
            }}
          >
            <Box>
              <Typography variant="caption" color="text.secondary">
                总执行时间
              </Typography>
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 600,
                  fontSize: { xs: '1rem', sm: '1.25rem' },
                  wordBreak: 'break-word',
                }}
              >
                {(summary.total_time || 0).toFixed(2)} 秒
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                总信号数 / 交易数
              </Typography>
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 600,
                  fontSize: { xs: '1rem', sm: '1.25rem' },
                  wordBreak: 'break-word',
                }}
              >
                {(summary.total_signals || 0).toLocaleString()} /{' '}
                {(summary.total_trades || 0).toLocaleString()}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                处理速度
              </Typography>
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 600,
                  fontSize: { xs: '1rem', sm: '1.25rem' },
                  wordBreak: 'break-word',
                }}
              >
                {(summary.days_per_second || 0).toFixed(2)} 天/秒
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* 阶段耗时表 */}
      <Card>
        <CardHeader title="阶段耗时与资源占用" />
        <CardContent sx={{ p: { xs: 1, sm: 2 } }}>
          <TableContainer component={Paper} sx={{ overflowX: 'auto' }}>
            <Table size="small" sx={{ minWidth: 500 }}>
              <TableHead>
                <TableRow>
                  <TableCell
                    sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                  >
                    阶段
                  </TableCell>
                  <TableCell
                    align="right"
                    sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                  >
                    耗时 (秒)
                  </TableCell>
                  <TableCell
                    align="right"
                    sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                  >
                    占比
                  </TableCell>
                  <TableCell
                    align="right"
                    sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                  >
                    峰值内存 (MB)
                  </TableCell>
                  <TableCell
                    align="right"
                    sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                  >
                    平均 CPU (%)
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(stages).map(([name, data]: [string, any]) => (
                  <TableRow key={name}>
                    <TableCell
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      <Typography variant="body2" sx={{ fontSize: 'inherit' }}>
                        {name === 'total_backtest' ? '整体回测' : name}
                      </Typography>
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      {(data.duration || 0).toFixed(2)}
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      {(data.percentage || 0).toFixed(1)}%
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      {(data.memory_peak_mb ?? data.memory_after_mb ?? 0).toFixed(2)}
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      {(data.cpu_avg_percent || 0).toFixed(1)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* 函数调用 Top N */}
      {Object.keys(functionCalls).length > 0 && (
        <Card>
          <CardHeader title="最耗时的函数 (Top 10)" />
          <CardContent sx={{ p: { xs: 1, sm: 2 } }}>
            <TableContainer component={Paper} sx={{ overflowX: 'auto' }}>
              <Table size="small" sx={{ minWidth: 450 }}>
                <TableHead>
                  <TableRow>
                    <TableCell
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      函数名
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      调用次数
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      总耗时 (秒)
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      平均耗时 (毫秒)
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(functionCalls)
                    .slice(0, 10)
                    .map(([name, data]: [string, any]) => (
                      <TableRow key={name}>
                        <TableCell
                          sx={{
                            fontSize: { xs: '0.75rem', sm: '0.875rem' },
                            p: { xs: 0.75, sm: 1 },
                            wordBreak: 'break-word',
                            maxWidth: 200,
                          }}
                        >
                          {name}
                        </TableCell>
                        <TableCell
                          align="right"
                          sx={{
                            fontSize: { xs: '0.75rem', sm: '0.875rem' },
                            p: { xs: 0.75, sm: 1 },
                          }}
                        >
                          {data.call_count || 0}
                        </TableCell>
                        <TableCell
                          align="right"
                          sx={{
                            fontSize: { xs: '0.75rem', sm: '0.875rem' },
                            p: { xs: 0.75, sm: 1 },
                          }}
                        >
                          {(data.total_time || 0).toFixed(4)}
                        </TableCell>
                        <TableCell
                          align="right"
                          sx={{
                            fontSize: { xs: '0.75rem', sm: '0.875rem' },
                            p: { xs: 0.75, sm: 1 },
                          }}
                        >
                          {((data.avg_time || 0) * 1000).toFixed(2)}
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* 并行化效率 */}
      {Object.keys(parallel).length > 0 && (
        <Card>
          <CardHeader title="并行化效率" />
          <CardContent sx={{ p: { xs: 1, sm: 2 } }}>
            <TableContainer component={Paper} sx={{ overflowX: 'auto' }}>
              <Table size="small" sx={{ minWidth: 550 }}>
                <TableHead>
                  <TableRow>
                    <TableCell
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      操作
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      顺序时间 (秒)
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      并行时间 (秒)
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      加速比
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      效率 (%)
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                    >
                      Worker 数
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(parallel).map(([name, data]: [string, any]) => (
                    <TableRow key={name}>
                      <TableCell
                        sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                      >
                        {name}
                      </TableCell>
                      <TableCell
                        align="right"
                        sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                      >
                        {(data.sequential_time || 0).toFixed(4)}
                      </TableCell>
                      <TableCell
                        align="right"
                        sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                      >
                        {(data.parallel_time || 0).toFixed(4)}
                      </TableCell>
                      <TableCell
                        align="right"
                        sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                      >
                        {(data.speedup || 0).toFixed(2)}x
                      </TableCell>
                      <TableCell
                        align="right"
                        sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                      >
                        {(data.efficiency_percent || 0).toFixed(1)}
                      </TableCell>
                      <TableCell
                        align="right"
                        sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' }, p: { xs: 0.75, sm: 1 } }}
                      >
                        {data.worker_count || 0}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
