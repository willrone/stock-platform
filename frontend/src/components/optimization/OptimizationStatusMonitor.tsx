/**
 * 优化任务状态监控组件
 */

'use client';

import React from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Progress,
  Chip,
} from '@heroui/react';
import { OptimizationStatus } from '../../services/optimizationService';

interface OptimizationStatusMonitorProps {
  status: OptimizationStatus;
  task: any;
}

export default function OptimizationStatusMonitor({
  status,
  task,
}: OptimizationStatusMonitorProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'running':
        return 'primary';
      case 'failed':
        return 'danger';
      default:
        return 'default';
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardBody>
            <div className="text-center">
              <p className="text-2xl font-bold">
                {status.completed_trials || 0} / {status.n_trials || 0}
              </p>
              <p className="text-sm text-default-500">已完成试验</p>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardBody>
            <div className="text-center">
              <p className="text-2xl font-bold text-primary">
                {status.running_trials || 0}
              </p>
              <p className="text-sm text-default-500">运行中</p>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardBody>
            <div className="text-center">
              <p className="text-2xl font-bold text-warning">
                {status.pruned_trials || 0}
              </p>
              <p className="text-sm text-default-500">已剪枝</p>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardBody>
            <div className="text-center">
              <p className="text-2xl font-bold text-danger">
                {status.failed_trials || 0}
              </p>
              <p className="text-sm text-default-500">失败</p>
            </div>
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <h3 className="text-lg font-semibold">优化进度</h3>
        </CardHeader>
        <CardBody className="space-y-4">
          <Progress
            value={status.progress || 0}
            color={getStatusColor(status.status)}
            size="lg"
            label={`${(status.progress || 0).toFixed(1)}%`}
            showValueLabel
          />
          <div className="flex items-center gap-2">
            <span className="text-sm text-default-600">状态:</span>
            <Chip color={getStatusColor(status.status)} size="sm">
              {status.status}
            </Chip>
          </div>
        </CardBody>
      </Card>

      {status.best_score !== undefined && status.best_score !== null && (
        <Card>
          <CardHeader>
            <h3 className="text-lg font-semibold">最佳结果</h3>
          </CardHeader>
          <CardBody>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-default-600">最佳得分:</span>
                <span className="font-bold text-success text-lg">
                  {(status.best_score || 0).toFixed(4)}
                </span>
              </div>
              {status.best_trial_number !== undefined && (
                <div className="flex justify-between">
                  <span className="text-default-600">最佳试验编号:</span>
                  <span className="font-medium">#{status.best_trial_number}</span>
                </div>
              )}
              {status.best_params && (
                <div className="mt-4">
                  <p className="text-sm font-medium mb-2">最佳参数:</p>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(status.best_params).map(([key, value]) => (
                      <div key={key} className="flex justify-between text-sm">
                        <span className="text-default-600">{key}:</span>
                        <span className="font-medium">{String(value)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}

