'use client';

import React from 'react';
import { 
  Card, 
  CardContent, 
  Box, 
  Typography, 
  Chip, 
  LinearProgress, 
  IconButton,
  Button,
  Tooltip,
} from '@mui/material';
import { Trash2, FileText, Activity } from 'lucide-react';
import { Model } from '../../stores/useDataStore';

interface MobileModelCardProps {
  model: Model;
  trainingProgress?: any;
  getStatusColor: (status: string) => 'success' | 'primary' | 'secondary' | 'warning' | 'error' | 'default';
  getStatusText: (status: string) => string;
  getStageText: (stage: string) => string;
  onShowTrainingReport: (modelId: string) => void;
  onShowLiveTraining: (modelId: string) => void;
  onDeleteModel: (modelId: string) => void;
  deleting: boolean;
}

export const MobileModelCard: React.FC<MobileModelCardProps> = ({ 
  model,
  trainingProgress,
  getStatusColor,
  getStatusText,
  getStageText,
  onShowTrainingReport,
  onShowLiveTraining,
  onDeleteModel,
  deleting,
}) => {
  const progress = trainingProgress?.progress || model.training_progress || 0;
  const stage = trainingProgress?.stage || model.training_stage || '';
  const message = trainingProgress?.message || '';
  const metrics = trainingProgress?.metrics || {};

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / 86400000);

    if (diffDays === 0) {
      return '今天';
    } else if (diffDays === 1) {
      return '昨天';
    } else if (diffDays < 7) {
      return `${diffDays}天前`;
    } else {
      return date.toLocaleDateString('zh-CN', {
        month: 'short',
        day: 'numeric',
      });
    }
  };

  return (
    <Card 
      sx={{ 
        mb: 2, 
        borderRadius: 3,
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
        transition: 'all 0.2s',
      }}
    >
      <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
        {/* 标题行 */}
        <Box 
          sx={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'flex-start', 
            mb: 1.5,
          }}
        >
          <Box sx={{ flex: 1, pr: 1 }}>
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 600,
                fontSize: '1.1rem',
                lineHeight: 1.3,
                mb: 0.5,
              }}
            >
              {model.model_name}
            </Typography>
            {model.description && (
              <Typography variant="caption" color="text.secondary">
                {model.description}
              </Typography>
            )}
          </Box>
          <Chip 
            label={getStatusText(model.status)} 
            color={getStatusColor(model.status)} 
            size="small"
            sx={{ 
              fontWeight: 600,
              fontSize: '0.75rem',
            }}
          />
        </Box>

        {/* 信息行 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1.5, flexWrap: 'wrap' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Typography variant="body2" color="text.secondary">
              类型:
            </Typography>
            <Typography variant="body2" fontWeight={500}>
              {model.model_type}
            </Typography>
          </Box>
          
          <Typography variant="body2" color="text.secondary">
            •
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Typography variant="body2" color="text.secondary">
              准确率:
            </Typography>
            <Typography variant="body2" fontWeight={600} color="success.main">
              {(model.accuracy * 100).toFixed(1)}%
            </Typography>
          </Box>
          
          <Typography variant="body2" color="text.secondary">
            •
          </Typography>
          
          <Typography variant="body2" color="text.secondary">
            {formatDate(model.created_at)}
          </Typography>
        </Box>

        {/* 训练进度 */}
        {model.status === 'training' && (
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
              <Typography variant="caption" color="text.secondary">
                {message || getStageText(stage)}
              </Typography>
              <Typography variant="caption" fontWeight={600} color="primary">
                {progress.toFixed(0)}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={progress} 
              sx={{ 
                height: 6, 
                borderRadius: 3,
                bgcolor: 'action.hover',
              }}
            />
            
            {/* 实时指标 */}
            {Object.keys(metrics).length > 0 && (
              <Box sx={{ display: 'flex', gap: 1.5, mt: 1, flexWrap: 'wrap' }}>
                {Object.entries(metrics).slice(0, 3).map(([key, value]) => (
                  <Typography key={key} variant="caption" color="text.secondary">
                    {key}: {typeof value === 'number' ? value.toFixed(3) : String(value)}
                  </Typography>
                ))}
              </Box>
            )}
          </Box>
        )}

        {/* 操作按钮 */}
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          {model.status === 'training' && (
            <Button
              variant="outlined"
              size="medium"
              startIcon={<Activity size={18} />}
              onClick={() => onShowLiveTraining(model.model_id)}
              sx={{ 
                flex: '1 1 auto',
                borderRadius: 2,
                minHeight: 44,
                textTransform: 'none',
                fontWeight: 500,
              }}
            >
              实时详情
            </Button>
          )}

          {model.status === 'ready' && (
            <Button
              variant="outlined"
              size="medium"
              startIcon={<FileText size={18} />}
              onClick={() => onShowTrainingReport(model.model_id)}
              sx={{ 
                flex: '1 1 auto',
                borderRadius: 2,
                minHeight: 44,
                textTransform: 'none',
                fontWeight: 500,
              }}
            >
              查看报告
            </Button>
          )}

          <Tooltip title={model.status === 'training' ? '取消训练并删除' : '删除模型'}>
            <span>
              <IconButton
                size="medium"
                onClick={() => onDeleteModel(model.model_id)}
                disabled={deleting}
                sx={{ 
                  border: 1,
                  borderColor: 'error.main',
                  color: 'error.main',
                  borderRadius: 2,
                  minHeight: 44,
                  minWidth: 44,
                  '&:hover': {
                    bgcolor: 'error.light',
                    borderColor: 'error.dark',
                  },
                  '&:disabled': {
                    borderColor: 'action.disabled',
                    color: 'action.disabled',
                  },
                }}
              >
                <Trash2 size={20} />
              </IconButton>
            </span>
          </Tooltip>
        </Box>
      </CardContent>
    </Card>
  );
};
