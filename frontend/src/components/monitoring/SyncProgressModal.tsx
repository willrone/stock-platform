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
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
  Progress,
  Chip,
  Divider,
} from '@heroui/react';
import {
  Clock,
  CheckCircle,
  XCircle,
  Activity,
  Pause,
  Play,
  RotateCcw,
} from 'lucide-react';
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
  onSyncComplete 
}: SyncProgressModalProps) {
  const [progressData, setProgressData] = useState<SyncProgressData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadProgress = useCallback(async () => {
    if (!syncId) return;
    
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
    if (!syncId) return;
    
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
    if (!seconds) return '--';
    
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

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'danger';
      case 'paused':
        return 'warning';
      default:
        return 'default';
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
        return <Activity className="w-4 h-4 animate-pulse" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4" />;
      case 'failed':
        return <XCircle className="w-4 h-4" />;
      case 'paused':
        return <Pause className="w-4 h-4" />;
      default:
        return <Clock className="w-4 h-4" />;
    }
  };

  return (
    <Modal 
      isOpen={isOpen} 
      onClose={onClose}
      size="lg"
      isDismissable={progressData?.status === 'completed' || progressData?.status === 'failed'}
      hideCloseButton={progressData?.status === 'running'}
    >
      <ModalContent>
        {(onModalClose) => (
          <>
            <ModalHeader className="flex flex-col gap-1">
              <div className="flex items-center space-x-2">
                <Activity className="w-5 h-5 text-primary" />
                <span>数据同步进度</span>
              </div>
              {syncId && (
                <p className="text-sm text-default-500 font-normal">
                  同步ID: {syncId}
                </p>
              )}
            </ModalHeader>
            
            <ModalBody>
              {error ? (
                <div className="text-center py-8">
                  <XCircle className="w-12 h-12 text-danger mx-auto mb-4" />
                  <p className="text-danger font-medium">{error}</p>
                  <Button
                    color="primary"
                    variant="light"
                    onPress={loadProgress}
                    className="mt-4"
                  >
                    重新加载
                  </Button>
                </div>
              ) : progressData ? (
                <div className="space-y-6">
                  {/* 状态和进度 */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(progressData.status)}
                        <Chip
                          color={getStatusColor(progressData.status) as any}
                          variant="flat"
                          size="sm"
                        >
                          {getStatusText(progressData.status)}
                        </Chip>
                      </div>
                      <span className="text-sm text-default-500">
                        {progressData.progress_percentage.toFixed(1)}%
                      </span>
                    </div>
                    
                    <Progress
                      value={progressData.progress_percentage}
                      color={getStatusColor(progressData.status) as any}
                      className="w-full"
                      size="lg"
                    />
                  </div>

                  {/* 当前状态 */}
                  {progressData.current_stock && (
                    <div className="p-4 bg-primary-50 rounded-lg">
                      <div className="flex items-center space-x-2 mb-2">
                        <Activity className="w-4 h-4 text-primary animate-pulse" />
                        <span className="text-sm font-medium text-primary">正在同步</span>
                      </div>
                      <p className="text-lg font-semibold">{progressData.current_stock}</p>
                    </div>
                  )}

                  <Divider />

                  {/* 统计信息 */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-default-700">
                        {progressData.total_stocks}
                      </p>
                      <p className="text-sm text-default-500">总股票数</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-success">
                        {progressData.completed_stocks}
                      </p>
                      <p className="text-sm text-default-500">已完成</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-danger">
                        {progressData.failed_stocks}
                      </p>
                      <p className="text-sm text-default-500">失败</p>
                    </div>
                  </div>

                  <Divider />

                  {/* 时间信息 */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-default-500">开始时间</span>
                      <span className="text-sm font-medium">
                        {new Date(progressData.start_time).toLocaleString()}
                      </span>
                    </div>
                    
                    {progressData.estimated_remaining_time_seconds && (
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-default-500">预估剩余时间</span>
                        <span className="text-sm font-medium">
                          {formatTime(progressData.estimated_remaining_time_seconds)}
                        </span>
                      </div>
                    )}
                    
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-default-500">最后更新</span>
                      <span className="text-sm font-medium">
                        {new Date(progressData.last_update).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <Activity className="w-12 h-12 text-primary mx-auto mb-4 animate-spin" />
                  <p className="text-default-500">加载同步进度中...</p>
                </div>
              )}
            </ModalBody>
            
            <ModalFooter>
              <div className="flex items-center space-x-2 w-full">
                {(progressData?.failed_stocks || 0) > 0 && progressData?.status !== 'running' && (
                  <Button
                    color="warning"
                    variant="light"
                    startContent={<RotateCcw className="w-4 h-4" />}
                    onPress={handleRetry}
                    isLoading={loading}
                  >
                    重试失败项
                  </Button>
                )}
                
                <div className="flex-1" />
                
                {progressData?.status === 'completed' || progressData?.status === 'failed' ? (
                  <Button color="primary" onPress={onModalClose}>
                    关闭
                  </Button>
                ) : (
                  <Button variant="light" onPress={onModalClose}>
                    后台运行
                  </Button>
                )}
              </div>
            </ModalFooter>
          </>
        )}
      </ModalContent>
    </Modal>
  );
}