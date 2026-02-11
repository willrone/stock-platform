/**
 * 删除任务确认对话框
 */

import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
} from '@mui/material';
import { AlertTriangle } from 'lucide-react';
import { Task } from '@/stores/useTaskStore';

interface DeleteTaskDialogProps {
  isOpen: boolean;
  task: Task | null;
  deleteForce: boolean;
  onClose: () => void;
  onConfirm: () => void;
  onForceChange: (force: boolean) => void;
}

export function DeleteTaskDialog({
  isOpen,
  task,
  deleteForce,
  onClose,
  onConfirm,
  onForceChange,
}: DeleteTaskDialogProps) {
  return (
    <Dialog open={isOpen} onClose={onClose} fullWidth maxWidth="sm">
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
        {task?.status === 'running' && (
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
                checked={deleteForce}
                onChange={e => onForceChange(e.target.checked)}
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
            onForceChange(false);
            onClose();
          }}
        >
          取消
        </Button>
        <Button variant="contained" color="error" onClick={onConfirm}>
          {deleteForce ? '强制删除' : '删除'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
