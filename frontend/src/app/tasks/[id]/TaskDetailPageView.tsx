import React from 'react';
import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Typography,
} from '@mui/material';
import { AlertTriangle, ArrowLeft } from 'lucide-react';

import { LoadingSpinner } from '../../../components/common/LoadingSpinner';
import { SaveStrategyConfigDialog } from '../../../components/backtest/SaveStrategyConfigDialog';
import { getStatusChip } from './taskDetailUtils';
import { TaskDetailActionPanel } from './TaskDetailActionPanel';
import { TaskDetailContent } from './TaskDetailContent';
import type { TaskDetailPageModel } from './types';

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
