import React from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Collapse,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Typography,
} from '@mui/material';
import { AlertTriangle, ArrowLeft, ChevronDown, ChevronRight, Settings } from 'lucide-react';

import { LoadingSpinner } from '../../../components/common/LoadingSpinner';
import { SaveStrategyConfigDialog } from '../../../components/backtest/SaveStrategyConfigDialog';
import { getStatusChip } from './taskDetailUtils';
import { TaskDetailActionPanel } from './TaskDetailActionPanel';
import { TaskDetailContent } from './TaskDetailContent';
import type { TaskDetailPageModel } from './types';

/**
 * 参数快照卡片组件
 * 从后端的 GET /tasks/{id}/config 获取完整的配置快照，以只读 JSON 展示
 */
function ConfigSnapshotCard({
  configSnapshot,
  loading,
}: {
  configSnapshot: Record<string, any>;
  loading: boolean;
}): React.ReactNode {
  const [expanded, setExpanded] = React.useState(false);

  const hasSnapshot = !loading && Object.keys(configSnapshot).length > 0;

  return (
    <Card>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 2,
          py: 1.5,
          cursor: 'pointer',
          '&:hover': { bgcolor: 'action.hover' },
        }}
        onClick={() => setExpanded(prev => !prev)}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Settings size={18} />
          <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
            参数快照
          </Typography>
          {loading && (
            <Typography variant="caption" color="text.secondary">
              加载中...
            </Typography>
          )}
          {!expanded && hasSnapshot && (
            <Typography variant="caption" color="text.secondary">
              ({Object.keys(configSnapshot).length} 个字段)
            </Typography>
          )}
        </Box>
        <IconButton size="small">
          {expanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
        </IconButton>
      </Box>
      <Collapse in={expanded}>
        <CardContent sx={{ pt: 0 }}>
          {loading ? (
            <Typography variant="body2" color="text.secondary" sx={{ py: 2, textAlign: 'center' }}>
              加载中...
            </Typography>
          ) : hasSnapshot ? (
            <Box
              component="pre"
              sx={{
                fontSize: '0.75rem',
                lineHeight: 1.6,
                fontFamily: '"SF Mono", "Fira Code", "Consolas", monospace',
                color: 'text.secondary',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                m: 0,
                p: 2,
                borderRadius: 1,
                bgcolor: 'grey.50',
                border: 1,
                borderColor: 'divider',
                maxHeight: 480,
                overflow: 'auto',
              }}
            >
              {JSON.stringify(configSnapshot, null, 2)}
            </Box>
          ) : (
            <Typography variant="body2" color="text.secondary" sx={{ py: 2, textAlign: 'center' }}>
              暂无配置快照
            </Typography>
          )}
        </CardContent>
      </Collapse>
    </Card>
  );
}

interface TaskDetailPageViewProps {
  model: TaskDetailPageModel;
}

export function TaskDetailPageView({ model }: TaskDetailPageViewProps): React.ReactNode {
  const { currentTask } = model;

  if (model.loading) {
    return <LoadingSpinner text="加载任务详情..." />;
  }

  if (!currentTask) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 384,
          gap: 2,
        }}
      >
        <Typography variant="body2" color="text.secondary">
          任务不存在或已被删除
        </Typography>
        <Button variant="contained" color="primary" onClick={model.handleBack}>
          返回任务列表
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <IconButton onClick={model.handleBack} size="small">
            <ArrowLeft size={20} />
          </IconButton>
          <Box>
            <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
              {currentTask.task_name}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              任务ID: {currentTask.task_id}
            </Typography>
          </Box>
          {getStatusChip(currentTask.status)}
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <TaskDetailActionPanel
            task={currentTask}
            refreshing={model.refreshing}
            onRefresh={() => void model.handleRefresh()}
            onRetry={() => void model.handleRetry()}
            onExport={() => void model.handleExport()}
            onRebuild={model.handleRebuild}
            onDelete={model.openDeleteDialog}
          />
        </Box>
      </Box>

      {/* 参数快照面板 */}
      <ConfigSnapshotCard
        configSnapshot={model.configSnapshot}
        loading={model.configSnapshotLoading}
      />

      <TaskDetailContent model={model} />

      {model.strategyConfigInfo && (
        <SaveStrategyConfigDialog
          isOpen={model.isSaveConfigOpen}
          onClose={model.closeSaveConfigDialog}
          strategyName={model.strategyConfigInfo.strategyName}
          parameters={model.strategyConfigInfo.parameters}
          onSave={model.handleSaveConfig}
          loading={model.savingConfig}
        />
      )}

      <Dialog open={model.isDeleteOpen} onClose={model.closeDeleteDialog}>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AlertTriangle size={20} color="#d32f2f" />
            <Typography variant="h6" component="span">
              确认删除
            </Typography>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            确定要删除这个任务吗？此操作不可撤销。
          </Typography>
          {currentTask.status === 'running' && (
            <Box
              sx={{
                mt: 2,
                p: 2,
                bgcolor: 'warning.light',
                border: 1,
                borderColor: 'warning.main',
                borderRadius: 1,
              }}
            >
              <Typography variant="body2" sx={{ color: 'warning.dark', mb: 1 }}>
                ⚠️ 该任务当前正在运行中
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <input
                  type="checkbox"
                  checked={model.deleteForce}
                  onChange={event => model.setDeleteForce(event.target.checked)}
                  style={{ width: 16, height: 16 }}
                />
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  强制删除（将中断正在运行的任务）
                </Typography>
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button
            variant="outlined"
            onClick={() => {
              model.setDeleteForce(false);
              model.closeDeleteDialog();
            }}
          >
            取消
          </Button>
          <Button
            variant="contained"
            color="error"
            onClick={() => {
              void model.handleDelete();
              model.closeDeleteDialog();
            }}
          >
            {model.deleteForce ? '强制删除' : '删除'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
