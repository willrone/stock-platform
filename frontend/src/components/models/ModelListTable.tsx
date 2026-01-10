/**
 * 模型列表表格组件
 */

'use client';

import React from 'react';
import {
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Button,
  Progress,
  Tooltip,
} from '@heroui/react';
import { Trash2 } from 'lucide-react';
import { Model } from '../../stores/useDataStore';

interface ModelListTableProps {
  models: Model[];
  trainingProgress: Record<string, any>;
  getStatusColor: (status: string) => 'success' | 'primary' | 'secondary' | 'warning' | 'danger' | 'default';
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
    <Table aria-label="模型列表">
      <TableHeader>
        <TableColumn>模型名称</TableColumn>
        <TableColumn>类型</TableColumn>
        <TableColumn>准确率</TableColumn>
        <TableColumn>状态</TableColumn>
        <TableColumn>创建时间</TableColumn>
        <TableColumn>操作</TableColumn>
      </TableHeader>
      <TableBody>
        {models.map((model) => (
          <TableRow key={model.model_id}>
            <TableCell>
              <div className="flex flex-col">
                <span className="font-medium">{model.model_name}</span>
                {model.description && (
                  <span className="text-xs text-default-500">
                    {model.description}
                  </span>
                )}
              </div>
            </TableCell>
            <TableCell>{model.model_type}</TableCell>
            <TableCell>
              <span className="font-medium">
                {(model.accuracy * 100).toFixed(1)}%
              </span>
            </TableCell>
            <TableCell>
              <div className="flex flex-col gap-1">
                <Chip
                  size="sm"
                  color={getStatusColor(model.status)}
                  variant="flat"
                >
                  {getStatusText(model.status)}
                </Chip>
                
                {/* 训练进度显示 */}
                {model.status === 'training' && (
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Progress 
                        value={trainingProgress[model.model_id]?.progress || model.training_progress || 0}
                        size="sm"
                        className="flex-1"
                        color="primary"
                      />
                      <span className="text-xs text-default-500 min-w-[40px]">
                        {(trainingProgress[model.model_id]?.progress || model.training_progress || 0).toFixed(0)}%
                      </span>
                    </div>
                    
                    {/* 训练阶段和消息 */}
                    <div className="text-xs text-default-400">
                      {trainingProgress[model.model_id]?.message || 
                       getStageText(trainingProgress[model.model_id]?.stage || model.training_stage)}
                    </div>
                    
                    {/* 实时指标 */}
                    {trainingProgress[model.model_id]?.metrics && Object.keys(trainingProgress[model.model_id].metrics).length > 0 && (
                      <div className="flex gap-2 text-xs">
                        {Object.entries(trainingProgress[model.model_id].metrics).slice(0, 2).map(([key, value]) => (
                          <Tooltip key={key} content={key}>
                            <span className="text-default-500">
                              {key}: {typeof value === 'number' ? value.toFixed(3) : String(value)}
                            </span>
                          </Tooltip>
                        ))}
                      </div>
                    )}
                    
                    {/* 查看实时详情按钮 */}
                    <Button
                      size="sm"
                      variant="light"
                      color="primary"
                      className="text-xs h-6"
                      onPress={() => onShowLiveTraining(model.model_id)}
                    >
                      查看实时详情
                    </Button>
                  </div>
                )}
                
                {/* 非训练状态的简单显示 */}
                {model.status !== 'training' && model.training_stage && (
                  <span className="text-xs text-default-400">
                    {getStageText(model.training_stage)}
                  </span>
                )}
              </div>
            </TableCell>
            <TableCell>
              {new Date(model.created_at).toLocaleDateString('zh-CN')}
            </TableCell>
            <TableCell>
              <div className="flex gap-2">
                {model.status === 'ready' && (
                  <Button
                    size="sm"
                    variant="light"
                    color="primary"
                    onPress={() => onShowTrainingReport(model.model_id)}
                  >
                    查看报告
                  </Button>
                )}
                {model.status !== 'training' && (
                  <Button
                    size="sm"
                    variant="light"
                    color="danger"
                    startContent={<Trash2 className="w-4 h-4" />}
                    onPress={() => onDeleteModel(model.model_id)}
                    isDisabled={deleting}
                  >
                    删除
                  </Button>
                )}
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

