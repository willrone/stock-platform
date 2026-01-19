/**
 * 模型列表表格组件
 */

'use client';

import React from 'react';
import {
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Button,
  LinearProgress,
  Tooltip,
  Box,
  Typography,
  IconButton,
} from '@mui/material';
import { Trash2 } from 'lucide-react';
import { Model } from '../../stores/useDataStore';

interface ModelListTableProps {
  models: Model[];
  trainingProgress: Record<string, any>;
  getStatusColor: (status: string) => 'success' | 'primary' | 'secondary' | 'warning' | 'error' | 'default';
  getStatusText: (status: string) => string;
  getStageText: (stage: string) => string;
  onShowTrainingReport: (modelId: string) => void;
  onShowLiveTraining: (modelId: string) => void;
  onDeleteModel: (modelId: string) => void;
  deleting: boolean;
}

export function ModelListTable({
  models,
  trainingProgress,
  getStatusColor,
  getStatusText,
  getStageText,
  onShowTrainingReport,
  onShowLiveTraining,
  onDeleteModel,
  deleting,
}: ModelListTableProps) {
  return (
    <Box sx={{ overflowX: 'auto' }}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>模型名称</TableCell>
            <TableCell>类型</TableCell>
            <TableCell>准确率</TableCell>
            <TableCell>状态</TableCell>
            <TableCell>创建时间</TableCell>
            <TableCell>操作</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {models.map((model) => (
            <TableRow key={model.model_id} hover>
              <TableCell>
                <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {model.model_name}
                  </Typography>
                  {model.description && (
                    <Typography variant="caption" color="text.secondary">
                      {model.description}
                    </Typography>
                  )}
                </Box>
              </TableCell>
              <TableCell>{model.model_type}</TableCell>
              <TableCell>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {(model.accuracy * 100).toFixed(1)}%
                </Typography>
              </TableCell>
              <TableCell>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Chip
                    label={getStatusText(model.status)}
                    color={getStatusColor(model.status)}
                    size="small"
                  />
                  
                  {/* 训练进度显示 */}
                  {model.status === 'training' && (
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={trainingProgress[model.model_id]?.progress || model.training_progress || 0}
                          sx={{ flex: 1, height: 6, borderRadius: 3 }}
                        />
                        <Typography variant="caption" color="text.secondary" sx={{ minWidth: 40 }}>
                          {(trainingProgress[model.model_id]?.progress || model.training_progress || 0).toFixed(0)}%
                        </Typography>
                      </Box>
                      
                      {/* 训练阶段和消息 */}
                      <Typography variant="caption" color="text.secondary">
                        {trainingProgress[model.model_id]?.message || 
                         getStageText(trainingProgress[model.model_id]?.stage || model.training_stage || '')}
                      </Typography>
                      
                      {/* 实时指标 */}
                      {trainingProgress[model.model_id]?.metrics && Object.keys(trainingProgress[model.model_id].metrics).length > 0 && (
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          {Object.entries(trainingProgress[model.model_id].metrics).slice(0, 2).map(([key, value]) => (
                            <Tooltip key={key} title={key}>
                              <Typography variant="caption" color="text.secondary">
                                {key}: {typeof value === 'number' ? value.toFixed(3) : String(value)}
                              </Typography>
                            </Tooltip>
                          ))}
                        </Box>
                      )}
                      
                      {/* 查看实时详情按钮 */}
                      <Button
                        size="small"
                        variant="outlined"
                        color="primary"
                        onClick={() => onShowLiveTraining(model.model_id)}
                        sx={{ height: 24, fontSize: '0.75rem' }}
                      >
                        查看实时详情
                      </Button>
                    </Box>
                  )}
                  
                  {/* 非训练状态的简单显示 */}
                  {model.status !== 'training' && model.training_stage && (
                    <Typography variant="caption" color="text.secondary">
                      {getStageText(model.training_stage)}
                    </Typography>
                  )}
                </Box>
              </TableCell>
              <TableCell>
                {new Date(model.created_at).toLocaleDateString('zh-CN')}
              </TableCell>
              <TableCell>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  {model.status === 'ready' && (
                    <Button
                      size="small"
                      variant="outlined"
                      color="primary"
                      onClick={() => onShowTrainingReport(model.model_id)}
                    >
                      查看报告
                    </Button>
                  )}
                  <Tooltip
                    title={
                      model.status === 'training'
                        ? '取消训练并删除该模型'
                        : '删除模型'
                    }
                  >
                    <span>
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => onDeleteModel(model.model_id)}
                        disabled={deleting}
                      >
                        <Trash2 size={16} />
                      </IconButton>
                    </span>
                  </Tooltip>
                </Box>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
}
