/**
 * 同步进度模态框组件
 *
 * 显示数据同步的实时进度，包括：
 * - 整体进度条
 * - 当前同步的股票
 * - 成功/失败统计
 * - 预估剩余时间
 * - 实时状态更新
 */

'use client';

import React, { useEffect, useState, useCallback } from 'react';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Button,
  LinearProgress,
  Chip,
  Divider,
  Box,
  Typography,
  CircularProgress,
} from '@mui/material';
import { Clock, CheckCircle, XCircle, Activity, Pause, Play, RotateCcw } from 'lucide-react';
import { DataService } from '../../services/dataService';

interface SyncProgressData {
  sync_id: string;
  total_stocks: number;
  completed_stocks: number;
  failed_stocks: number;
  current_stock: string | null;
  progress_percentage: number;
  estimated_remaining_time_seconds: number | null;
  start_time: string;
  status: string;
  last_update: string;
}

interface SyncProgressModalProps {
  isOpen: boolean;
  onClose: () => void;
  syncId: string | null;
  onSyncComplete?: () => void;
}

export function SyncProgressModal({
  isOpen,
  onClose,
  syncId,
  onSyncComplete,
}: SyncProgressModalProps) {
  const [progressData, setProgressData] = useState<SyncProgressData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadProgress = useCallback(async () => {
    if (!syncId) {
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const data = await DataService.getSyncProgress(syncId);
      setProgressData(data);

      // 如果同步完成，通知父组件
      if (data.status === 'completed' && onSyncComplete) {
        onSyncComplete();
      }
    } catch (err) {
      console.error('获取同步进度失败:', err);
      setError('获取同步进度失败');
    } finally {
      setLoading(false);
    }
  }, [syncId, onSyncComplete]);

  useEffect(() => {
    if (isOpen && syncId) {
      loadProgress();

      // 每2秒更新一次进度
      const interval = setInterval(loadProgress, 2000);
      return () => clearInterval(interval);
    }
  }, [isOpen, syncId, loadProgress]);

  const handleRetry = async () => {
    if (!syncId) {
      return;
    }

    try {
      setLoading(true);
      await DataService.retrySyncFailed(syncId);
      await loadProgress(); // 重新加载进度
    } catch (err) {
      console.error('重试同步失败:', err);
      setError('重试同步失败');
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (seconds: number | null) => {
    if (!seconds) {
      return '--';
    }

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}小时${minutes}分钟`;
    } else if (minutes > 0) {
      return `${minutes}分钟${secs}秒`;
    } else {
      return `${secs}秒`;
    }
  };

  const getStatusColor = (status: string): 'primary' | 'success' | 'error' | 'warning' => {
    switch (status) {
      case 'running':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'paused':
        return 'warning';
      default:
        return 'primary';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'running':
        return '同步中';
      case 'completed':
        return '已完成';
      case 'failed':
        return '失败';
      case 'paused':
        return '已暂停';
      default:
        return '未知';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Activity size={16} className="animate-pulse" />;
      case 'completed':
        return <CheckCircle size={16} />;
      case 'failed':
        return <XCircle size={16} />;
      case 'paused':
        return <Pause size={16} />;
      default:
        return <Clock size={16} />;
    }
  };

  return (
    <Dialog open={isOpen} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Activity size={20} color="#1976d2" />
            <span>数据同步进度</span>
          </Box>
          {syncId && (
            <Typography variant="caption" color="text.secondary">
              同步ID: {syncId}
            </Typography>
          )}
        </Box>
      </DialogTitle>

      <DialogContent>
        {error ? (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <XCircle size={48} color="#d32f2f" style={{ margin: '0 auto 16px' }} />
            <Typography variant="body2" color="error" sx={{ fontWeight: 500, mb: 2 }}>
              {error}
            </Typography>
            <Button variant="outlined" color="primary" onClick={loadProgress}>
              重新加载
            </Button>
          </Box>
        ) : progressData ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* 状态和进度 */}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {getStatusIcon(progressData.status)}
                  <Chip
                    label={getStatusText(progressData.status)}
                    color={getStatusColor(progressData.status)}
                    size="small"
                  />
                </Box>
                <Typography variant="body2" color="text.secondary">
                  {progressData.progress_percentage.toFixed(1)}%
                </Typography>
              </Box>

              <LinearProgress
                variant="determinate"
                value={progressData.progress_percentage}
                color={getStatusColor(progressData.status) as any}
                sx={{ height: 10, borderRadius: 5 }}
              />
            </Box>

            {/* 当前状态 */}
            {progressData.current_stock && (
              <Box sx={{ p: 2, bgcolor: 'primary.light', borderRadius: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <Activity size={16} color="#1976d2" className="animate-pulse" />
                  <Typography variant="body2" sx={{ fontWeight: 500, color: 'primary.main' }}>
                    正在同步
                  </Typography>
                </Box>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  {progressData.current_stock}
                </Typography>
              </Box>
            )}

            <Divider />

            {/* 统计信息 */}
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 2 }}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" sx={{ fontWeight: 600 }}>
                  {progressData.total_stocks}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  总股票数
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                  {progressData.completed_stocks}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  已完成
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" sx={{ fontWeight: 600, color: 'error.main' }}>
                  {progressData.failed_stocks}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  失败
                </Typography>
              </Box>
            </Box>

            <Divider />

            {/* 时间信息 */}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body2" color="text.secondary">
                  开始时间
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {new Date(progressData.start_time).toLocaleString()}
                </Typography>
              </Box>

              {progressData.estimated_remaining_time_seconds && (
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    预估剩余时间
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {formatTime(progressData.estimated_remaining_time_seconds)}
                  </Typography>
                </Box>
              )}

              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body2" color="text.secondary">
                  最后更新
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {new Date(progressData.last_update).toLocaleTimeString()}
                </Typography>
              </Box>
            </Box>
          </Box>
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <CircularProgress size={48} sx={{ mb: 2 }} />
            <Typography variant="body2" color="text.secondary">
              加载同步进度中...
            </Typography>
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
          {(progressData?.failed_stocks || 0) > 0 && progressData?.status !== 'running' && (
            <Button
              color="warning"
              variant="outlined"
              startIcon={<RotateCcw size={16} />}
              onClick={handleRetry}
              disabled={loading}
            >
              重试失败项
            </Button>
          )}

          <Box sx={{ flex: 1 }} />

          {progressData?.status === 'completed' || progressData?.status === 'failed' ? (
            <Button variant="contained" color="primary" onClick={onClose}>
              关闭
            </Button>
          ) : (
            <Button variant="outlined" onClick={onClose}>
              后台运行
            </Button>
          )}
        </Box>
      </DialogActions>
    </Dialog>
  );
}
