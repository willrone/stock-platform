/**
 * 实时训练监控弹窗组件
 */

'use client';

import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Button,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Box,
  Typography,
} from '@mui/material';
import { TrendingUp } from 'lucide-react';
import { Model } from '../../stores/useDataStore';

interface LiveTrainingModalProps {
  isOpen: boolean;
  onClose: () => void;
  modelId: string | null;
  models: Model[];
  trainingProgress: Record<string, any>;
  getStageText: (stage: string) => string;
}

export function LiveTrainingModal({
  isOpen,
  onClose,
  modelId,
  models,
  trainingProgress,
  getStageText,
}: LiveTrainingModalProps) {
  return (
    <Dialog open={isOpen} onClose={onClose} maxWidth="xl" fullWidth
      sx={{ '& .MuiDialog-paper': { m: { xs: 1, sm: 2 } } }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TrendingUp size={20} />
          <span>实时训练监控</span>
          {modelId && (
            <Chip
              label={models.find(m => m.model_id === modelId)?.model_name || '未知模型'}
              color="primary"
              size="small"
            />
          )}
        </Box>
      </DialogTitle>
      <DialogContent>
        {modelId && trainingProgress[modelId] ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* 顶部：实时指标概览 */}
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' },
                gap: 3,
              }}
            >
              {/* 左侧：实时指标 */}
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  实时指标
                </Typography>
                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 1 }}>
                  {trainingProgress[modelId].metrics &&
                    Object.entries(trainingProgress[modelId].metrics).map(([key, value]) => (
                      <Card key={key}>
                        <CardContent sx={{ p: 1.5 }}>
                          <Typography variant="caption" color="text.secondary">
                            {key}
                          </Typography>
                          <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.875rem', sm: '1.25rem' } }}>
                            {typeof value === 'number' ? value.toFixed(4) : String(value)}
                          </Typography>
                        </CardContent>
                      </Card>
                    ))}
                </Box>

                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">总体进度</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {trainingProgress[modelId].progress?.toFixed(1)}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={trainingProgress[modelId].progress || 0}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    当前阶段: {getStageText(trainingProgress[modelId].stage)}
                  </Typography>
                  {trainingProgress[modelId].message && (
                    <Typography variant="caption" color="text.secondary">
                      {trainingProgress[modelId].message}
                    </Typography>
                  )}
                </Box>
              </Box>

              {/* 右侧：训练曲线占位符 */}
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  训练曲线
                </Typography>
                <Box
                  sx={{
                    height: 256,
                    bgcolor: 'grey.50',
                    borderRadius: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Box sx={{ textAlign: 'center' }}>
                    <TrendingUp size={32} color="#666" style={{ margin: '0 auto 8px' }} />
                    <Typography variant="body2" color="text.secondary">
                      训练曲线图
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      (开发中)
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </Box>

            {/* 训练日志占位符 */}
            <Box>
              <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                训练日志
              </Typography>
              <Box
                sx={{
                  bgcolor: '#1e1e1e',
                  color: '#4ade80',
                  p: 2,
                  borderRadius: 1,
                  height: 128,
                  overflowY: 'auto',
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                }}
              >
                <div>
                  [{new Date().toLocaleTimeString()}] 训练进度:{' '}
                  {trainingProgress[modelId].progress?.toFixed(1)}%
                </div>
                <div>
                  [{new Date().toLocaleTimeString()}] 当前阶段:{' '}
                  {getStageText(trainingProgress[modelId].stage)}
                </div>
                {trainingProgress[modelId].message && (
                  <div>
                    [{new Date().toLocaleTimeString()}] {trainingProgress[modelId].message}
                  </div>
                )}
                <div style={{ color: '#666' }}>更多日志功能开发中...</div>
              </Box>
            </Box>
          </Box>
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body2" color="text.secondary">
              {modelId ? '暂无训练数据' : '请选择一个训练中的模型'}
            </Typography>
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>关闭</Button>
        {modelId && trainingProgress[modelId] && (
          <Button
            color="error"
            variant="outlined"
            onClick={() => {
              // TODO: 实现停止训练功能
              alert('停止训练功能开发中...');
            }}
          >
            停止训练
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}
