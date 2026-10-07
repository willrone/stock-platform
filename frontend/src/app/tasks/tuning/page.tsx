'use client';

/**
 * 参数调优台页面
 *
 * 基于已完成任务的配置快照进行参数修改，预览后提交新回测任务。
 * 流程：选任务 → 加载配置 → 修改参数 → 预览 → 提交
 */

import React, { useEffect, useState, useCallback } from 'react';
import {
  Autocomplete,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  CircularProgress,
  Divider,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Snackbar,
  TextField,
  Typography,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import { ArrowLeft, ChevronDown, Play, Eye, CheckCircle, AlertCircle } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { apiRequest } from '@/services/api';

/* ── Types ──────────────────────────────────────────────────────────── */

interface TaskBrief {
  task_id: string;
  task_name: string;
  created_at: string;
  status: string;
}

interface ChangeItem {
  path: string;
  old: unknown;
  new: unknown;
}

interface PreviewResponse {
  base_task: TaskBrief;
  resolved_config: Record<string, unknown>;
  changes: ChangeItem[];
}

interface SubmitResponse {
  task_id: string;
  task_name: string;
}

/* ── Group config fields for display ────────────────────────────────── */

const FIELD_GROUPS: Record<string, string> = {
  prediction_config: '预测配置',
  backtest_config: '回测配置',
  stock_codes: '股票列表',
  task_name: '任务名称',
};

function getFieldGroup(key: string): string {
  // e.g. "backtest_config.strategy_name" -> "回测配置"
  const prefix = key.split('.')[0];
  return FIELD_GROUPS[prefix] || prefix || '其他';
}

/** Render a value for display — handles nested objects, arrays, primitives */
function renderValue(v: unknown): string {
  if (v === null || v === undefined) return '—';
  if (Array.isArray(v)) {
    if (v.length === 0) return '[]';
    if (typeof v[0] === 'string') return v.join(', ');
    return JSON.stringify(v);
  }
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

/** Pretty-print a JSON-style value as a compact string */
function formatValue(v: unknown): string {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number') {
    if (Number.isFinite(v)) {
      // 保留最多 6 位小数
      const s = v.toPrecision(6);
      return parseFloat(s).toString();
    }
    return '—';
  }
  if (Array.isArray(v)) {
    if (v.every(x => typeof x === 'string')) return v.join(', ');
    return JSON.stringify(v);
  }
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

/** Flatten nested object into dot-path keys */
function flattenObject(obj: Record<string, unknown>, prefix = ''): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${k}` : k;
    if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
      Object.assign(result, flattenObject(v as Record<string, unknown>, path));
    } else {
      result[path] = v;
    }
  }
  return result;
}

/* ============================================================
   Page Component
   ============================================================ */

export default function TuningPage() {
  const router = useRouter();

  /* ── State ──────────────────────────────────────────────────── */
  const [tasks, setTasks] = useState<TaskBrief[]>([]);
  const [tasksLoading, setTasksLoading] = useState(true);

  const [selectedTask, setSelectedTask] = useState<TaskBrief | null>(null);
  const [configLoading, setConfigLoading] = useState(false);

  // 展平的配置参数（路径 → 值）
  const [flatConfig, setFlatConfig] = useState<Record<string, unknown>>({});
  // 覆盖的参数（路径 → 新值）
  const [overrides, setOverrides] = useState<Record<string, string>>({});

  // 预览结果
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  // 提交结果
  const [submitLoading, setSubmitLoading] = useState(false);
  const [submitResult, setSubmitResult] = useState<SubmitResponse | null>(null);

  const [newTaskName, setNewTaskName] = useState('');
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info';
  }>({ open: false, message: '', severity: 'info' });

  /* ── Load task list ────────────────────────────────────────── */
  useEffect(() => {
    let cancelled = false;
    setTasksLoading(true);
    apiRequest
      .get<TaskBrief[]>('/api/v1/tasks?limit=200&status=completed')
      .then(data => {
        if (cancelled) return;
        // data 可能是 {tasks, total, ...} 结构，按两种形态归一为列表
        const list: TaskBrief[] = Array.isArray(data)
          ? data
          : ((data as Record<string, unknown>).tasks as TaskBrief[]) || [];
        setTasks(list.filter(t => t.status === 'completed' || t.status === 'success'));
      })
      .catch(() => {
        // fallback: 可能是列表响应变成了分页结构
      })
      .finally(() => {
        if (!cancelled) setTasksLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  /* ── Load config snapshot for selected task ────────────────── */
  const loadConfig = useCallback(async (task: TaskBrief) => {
    setConfigLoading(true);
    setPreview(null);
    setOverrides({});
    setSubmitResult(null);
    setNewTaskName('');
    try {
      const resp = await apiRequest.get<Record<string, unknown>>(
        `/api/v1/tasks/${task.task_id}/config`
      );
      const snapshot = resp;
      if (snapshot && typeof snapshot === 'object') {
        setFlatConfig(flattenObject(snapshot as Record<string, unknown>));
      } else {
        setFlatConfig({});
      }
    } catch {
      setSnackbar({
        open: true,
        message: '加载配置失败，该任务可能没有配置快照',
        severity: 'error',
      });
      setFlatConfig({});
    } finally {
      setConfigLoading(false);
    }
  }, []);

  /* ── Handle task selection ─────────────────────────────────── */
  const handleTaskChange = (_: unknown, task: TaskBrief | null) => {
    setSelectedTask(task);
    if (task) {
      loadConfig(task);
    } else {
      setFlatConfig({});
      setPreview(null);
      setOverrides({});
      setSubmitResult(null);
    }
  };

  /* ── Handle override input change ──────────────────────────── */
  const handleOverrideChange = (path: string, value: string) => {
    setOverrides(prev => {
      const next = { ...prev };
      if (value === '' || value === formatValue(flatConfig[path])) {
        delete next[path];
      } else {
        next[path] = value;
      }
      return next;
    });
  };

  /* ── Preview ───────────────────────────────────────────────── */
  const handlePreview = async () => {
    if (!selectedTask) return;
    setPreviewLoading(true);
    setPreview(null);
    setSubmitResult(null);

    // 构建 overrides payload — 自动类型转换
    const typedOverrides: Record<string, unknown> = {};
    for (const [path, strVal] of Object.entries(overrides)) {
      const orig = flatConfig[path];
      if (typeof orig === 'number') {
        const n = Number(strVal);
        typedOverrides[path] = Number.isNaN(n) ? strVal : n;
      } else if (typeof orig === 'boolean') {
        typedOverrides[path] = strVal === 'true' || strVal === '1';
      } else {
        typedOverrides[path] = strVal;
      }
    }

    try {
      const resp = await apiRequest.post<PreviewResponse>('/api/v1/tasks/tuning/preview-config', {
        base_task_id: selectedTask.task_id,
        overrides: typedOverrides,
      });
      setPreview(resp);
      if (!newTaskName) {
        setNewTaskName(`调优-${selectedTask.task_name}`);
      }
    } catch {
      setSnackbar({
        open: true,
        message: '预览生成失败',
        severity: 'error',
      });
    } finally {
      setPreviewLoading(false);
    }
  };

  /* ── Submit ────────────────────────────────────────────────── */
  const handleSubmit = async () => {
    if (!selectedTask || !preview) return;
    setSubmitLoading(true);

    const typedOverrides: Record<string, unknown> = {};
    for (const [path, strVal] of Object.entries(overrides)) {
      const orig = flatConfig[path];
      if (typeof orig === 'number') {
        const n = Number(strVal);
        typedOverrides[path] = Number.isNaN(n) ? strVal : n;
      } else if (typeof orig === 'boolean') {
        typedOverrides[path] = strVal === 'true' || strVal === '1';
      } else {
        typedOverrides[path] = strVal;
      }
    }

    try {
      const resp = await apiRequest.post<SubmitResponse>('/api/v1/tasks/tuning/submit', {
        base_task_id: selectedTask.task_id,
        task_name: newTaskName,
        overrides: typedOverrides,
      });
      setSubmitResult(resp);
      setSnackbar({
        open: true,
        message: `调优任务已提交: ${resp.task_name} (${resp.task_id})`,
        severity: 'success',
      });
    } catch {
      setSnackbar({
        open: true,
        message: '提交调优任务失败',
        severity: 'error',
      });
    } finally {
      setSubmitLoading(false);
    }
  };

  /* ── Compute grouped fields with override values ───────────── */
  const fieldsWithOverrides = React.useMemo(() => {
    const entries = Object.entries(flatConfig);
    // filter out internal keys
    const filtered = entries.filter(([k]) => !k.startsWith('_') && k !== 'task_name');

    // group by prefix
    const groups: Record<
      string,
      Array<{ path: string; orig: unknown; newVal: string | null }>
    > = {};
    for (const [path, orig] of filtered) {
      const group = getFieldGroup(path);
      if (!groups[group]) groups[group] = [];
      const overrideVal = overrides[path];
      groups[group].push({
        path,
        orig,
        newVal: overrideVal ?? null,
      });
    }
    return groups;
  }, [flatConfig, overrides]);

  /* ── Render ────────────────────────────────────────────────── */
  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3, gap: 1 }}>
        <IconButton onClick={() => router.push('/tasks')}>
          <ArrowLeft />
        </IconButton>
        <Typography variant="h5" fontWeight={600}>
          参数调优台 🎛️
        </Typography>
      </Box>

      {/* Step 1: Select task */}
      <Card sx={{ mb: 3 }}>
        <CardHeader
          title="① 选择基准任务"
          titleTypographyProps={{ variant: 'subtitle1', fontWeight: 600 }}
        />
        <CardContent>
          <Autocomplete<TaskBrief, false, false, false>
            options={tasks}
            loading={tasksLoading}
            value={selectedTask}
            onChange={handleTaskChange}
            getOptionLabel={opt => `${opt.task_name} (${opt.task_id.slice(0, 8)}...)`}
            renderInput={params => (
              <TextField
                {...params}
                label="选择已完成回测任务"
                placeholder="搜索任务..."
                slotProps={{
                  input: {
                    ...params.InputProps,
                    endAdornment: (
                      <>
                        {tasksLoading ? <CircularProgress size={20} /> : null}
                        {params.InputProps.endAdornment}
                      </>
                    ),
                  },
                }}
              />
            )}
            isOptionEqualToValue={(opt, val) => opt.task_id === val.task_id}
            fullWidth
            size="small"
          />
        </CardContent>
      </Card>

      {/* Step 2: Edit parameters */}
      {selectedTask && (
        <Card sx={{ mb: 3 }}>
          <CardHeader
            title="② 修改参数"
            titleTypographyProps={{ variant: 'subtitle1', fontWeight: 600 }}
            action={
              <Chip
                label={`${Object.keys(overrides).length} 个修改`}
                color={Object.keys(overrides).length > 0 ? 'warning' : 'default'}
                size="small"
              />
            }
          />
          <CardContent>
            {configLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : Object.keys(fieldsWithOverrides).length === 0 ? (
              <Typography color="text.secondary">该任务没有可编辑的配置参数</Typography>
            ) : (
              Object.entries(fieldsWithOverrides).map(
                ([group, fields]) =>
                  fields.length > 0 && (
                    <Accordion key={group} defaultExpanded>
                      <AccordionSummary expandIcon={<ChevronDown size={18} />}>
                        <Typography variant="subtitle2" fontWeight={600}>
                          {group}
                          <Chip label={`${fields.length} 项`} size="small" sx={{ ml: 1 }} />
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        {fields.map(f => (
                          <Box
                            key={f.path}
                            sx={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: 1.5,
                              mb: 1.5,
                              '&:last-child': { mb: 0 },
                            }}
                          >
                            <Typography
                              variant="body2"
                              sx={{
                                minWidth: 220,
                                maxWidth: 260,
                                fontFamily: 'monospace',
                                fontSize: 12,
                                color: 'text.secondary',
                                wordBreak: 'break-all',
                              }}
                            >
                              {f.path}
                            </Typography>
                            <Typography
                              variant="body2"
                              sx={{
                                minWidth: 100,
                                color: f.newVal ? 'text.disabled' : 'text.primary',
                                textDecoration: f.newVal ? 'line-through' : 'none',
                                fontFamily: 'monospace',
                                fontSize: 13,
                              }}
                            >
                              {formatValue(f.orig)}
                            </Typography>
                            <TextField
                              size="small"
                              variant="outlined"
                              placeholder={formatValue(f.orig)}
                              value={f.newVal ?? ''}
                              onChange={e => handleOverrideChange(f.path, e.target.value)}
                              sx={{ flex: 1, minWidth: 160 }}
                              slotProps={{
                                input: {
                                  sx: { fontFamily: 'monospace', fontSize: 13 },
                                },
                              }}
                            />
                          </Box>
                        ))}
                      </AccordionDetails>
                    </Accordion>
                  )
              )
            )}
          </CardContent>
        </Card>
      )}

      {/* Step 3: Preview & Submit */}
      {selectedTask && !configLoading && (
        <Card sx={{ mb: 3 }}>
          <CardHeader
            title="③ 预览并提交"
            titleTypographyProps={{ variant: 'subtitle1', fontWeight: 600 }}
          />
          <CardContent>
            <TextField
              label="新任务名称"
              value={newTaskName}
              onChange={e => setNewTaskName(e.target.value)}
              fullWidth
              size="small"
              sx={{ mb: 2 }}
            />

            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button
                variant="outlined"
                startIcon={previewLoading ? <CircularProgress size={16} /> : <Eye size={16} />}
                onClick={handlePreview}
                disabled={!selectedTask || previewLoading || submitLoading}
              >
                {previewLoading ? '生成中...' : '预览配置'}
              </Button>
              <Button
                variant="contained"
                color="primary"
                startIcon={
                  submitLoading ? (
                    <CircularProgress size={16} color="inherit" />
                  ) : (
                    <Play size={16} />
                  )
                }
                onClick={handleSubmit}
                disabled={!preview || submitLoading || !newTaskName.trim()}
              >
                {submitLoading ? '提交中...' : '提交新回测'}
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Preview Results */}
      {preview && (
        <Card sx={{ mb: 3 }}>
          <CardHeader
            title="变更预览"
            titleTypographyProps={{ variant: 'subtitle1', fontWeight: 600 }}
            avatar={<AlertCircle size={20} />}
          />
          <CardContent>
            {preview.changes.length === 0 ? (
              <Alert severity="info" sx={{ mb: 1 }}>
                无参数变更，提交将使用基准任务的全部原配置
              </Alert>
            ) : (
              <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 600 }}>参数路径</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>原值</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>新值</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {preview.changes.map(c => (
                      <TableRow key={c.path} sx={{ '&:hover': { bgcolor: 'action.hover' } }}>
                        <TableCell
                          sx={{
                            fontFamily: 'monospace',
                            fontSize: 12,
                            maxWidth: 260,
                            wordBreak: 'break-all',
                          }}
                        >
                          {c.path}
                        </TableCell>
                        <TableCell
                          sx={{
                            fontFamily: 'monospace',
                            fontSize: 12,
                            color: 'text.secondary',
                            textDecoration: 'line-through',
                          }}
                        >
                          {formatValue(c.old)}
                        </TableCell>
                        <TableCell
                          sx={{
                            fontFamily: 'monospace',
                            fontSize: 12,
                            color: 'success.main',
                            fontWeight: 600,
                          }}
                        >
                          {formatValue(c.new)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}

            <Accordion>
              <AccordionSummary expandIcon={<ChevronDown size={18} />}>
                <Typography variant="subtitle2">完整解析配置</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box
                  component="pre"
                  sx={{
                    fontFamily: 'monospace',
                    fontSize: 11,
                    lineHeight: 1.5,
                    bgcolor: 'grey.100',
                    p: 2,
                    borderRadius: 1,
                    maxHeight: 400,
                    overflow: 'auto',
                  }}
                >
                  {JSON.stringify(preview.resolved_config, null, 2)}
                </Box>
              </AccordionDetails>
            </Accordion>
          </CardContent>
        </Card>
      )}

      {/* Submit Result */}
      {submitResult && (
        <Card
          sx={{
            mb: 3,
            borderColor: 'success.main',
            border: 1,
          }}
        >
          <CardHeader
            title="✅ 提交成功"
            titleTypographyProps={{ variant: 'subtitle1', fontWeight: 600 }}
            avatar={<CheckCircle color="success" size={24} />}
          />
          <CardContent>
            <Typography>
              新任务 <strong>{submitResult.task_name}</strong> 已创建并提交执行。
            </Typography>
            <Button
              variant="text"
              size="small"
              sx={{ mt: 1 }}
              onClick={() => router.push(`/tasks/${submitResult.task_id}`)}
            >
              查看任务详情
            </Button>
            <Button
              variant="text"
              size="small"
              sx={{ mt: 1, ml: 1 }}
              onClick={() => {
                setPreview(null);
                setSubmitResult(null);
                setOverrides({});
                setNewTaskName('');
                setFlatConfig({});
                setSelectedTask(null);
              }}
            >
              继续调优
            </Button>
          </CardContent>
        </Card>
      )}

      <Snackbar
        open={snackbar.open}
        autoHideDuration={5000}
        onClose={() => setSnackbar(s => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          severity={snackbar.severity}
          onClose={() => setSnackbar(s => ({ ...s, open: false }))}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}
