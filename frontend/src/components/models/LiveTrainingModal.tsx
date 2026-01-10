/**
 * 实时训练监控弹窗组件
 */

'use client';

import React from 'react';
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
  Card,
  Chip,
  Progress,
} from '@heroui/react';
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
    <Modal isOpen={isOpen} onClose={onClose} size="5xl" scrollBehavior="inside">
      <ModalContent>
        <ModalHeader>
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-5 h-5" />
            <span>实时训练监控</span>
            {modelId && (
              <Chip size="sm" color="primary" variant="flat">
                {models.find(m => m.model_id === modelId)?.model_name || '未知模型'}
              </Chip>
            )}
          </div>
        </ModalHeader>
        <ModalBody>
          {modelId && trainingProgress[modelId] ? (
            <div className="space-y-6">
              {/* 顶部：实时指标概览 */}
              <div className="grid grid-cols-2 gap-6">
                {/* 左侧：实时指标 */}
                <div className="space-y-4">
                  <h4 className="font-semibold">实时指标</h4>
                  <div className="grid grid-cols-2 gap-4">
                    {trainingProgress[modelId].metrics && Object.entries(trainingProgress[modelId].metrics).map(([key, value]) => (
                      <Card key={key} className="p-3">
                        <div className="text-sm text-default-500">{key}</div>
                        <div className="text-lg font-semibold">
                          {typeof value === 'number' ? value.toFixed(4) : String(value)}
                        </div>
                      </Card>
                    ))}
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>总体进度</span>
                      <span>{trainingProgress[modelId].progress?.toFixed(1)}%</span>
                    </div>
                    <Progress value={trainingProgress[modelId].progress || 0} />
                    <div className="text-xs text-default-500">
                      当前阶段: {getStageText(trainingProgress[modelId].stage)}
                    </div>
                    {trainingProgress[modelId].message && (
                      <div className="text-xs text-default-400">
                        {trainingProgress[modelId].message}
                      </div>
                    )}
                  </div>
                </div>
                
                {/* 右侧：训练曲线占位符 */}
                <div className="space-y-4">
                  <h4 className="font-semibold">训练曲线</h4>
                  <div className="h-64 bg-default-50 rounded-lg flex items-center justify-center">
                    <div className="text-center text-default-500">
                      <TrendingUp className="w-8 h-8 mx-auto mb-2" />
                      <p>训练曲线图</p>
                      <p className="text-sm">(开发中)</p>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* 训练日志占位符 */}
              <div>
                <h4 className="font-semibold mb-2">训练日志</h4>
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg h-32 overflow-y-auto text-sm font-mono">
                  <div>[{new Date().toLocaleTimeString()}] 训练进度: {trainingProgress[modelId].progress?.toFixed(1)}%</div>
                  <div>[{new Date().toLocaleTimeString()}] 当前阶段: {getStageText(trainingProgress[modelId].stage)}</div>
                  {trainingProgress[modelId].message && (
                    <div>[{new Date().toLocaleTimeString()}] {trainingProgress[modelId].message}</div>
                  )}
                  <div className="text-gray-500">更多日志功能开发中...</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="text-default-500">
                {modelId ? '暂无训练数据' : '请选择一个训练中的模型'}
              </div>
            </div>
          )}
        </ModalBody>
        <ModalFooter>
          <Button variant="light" onPress={onClose}>
            关闭
          </Button>
          {modelId && trainingProgress[modelId] && (
            <Button 
              color="danger" 
              variant="light"
              onPress={() => {
                // TODO: 实现停止训练功能
                alert('停止训练功能开发中...');
              }}
            >
              停止训练
            </Button>
          )}
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}

