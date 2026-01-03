'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardBody, CardHeader } from '@heroui/card';
import { Button } from '@heroui/button';
import { Chip } from '@heroui/chip';
import { Modal, ModalContent, ModalHeader, ModalBody, ModalFooter, useDisclosure } from '@heroui/modal';
import { 
  ExclamationTriangleIcon,
  XCircleIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  BellIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';

export interface BacktestNotification {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: Date;
  taskId?: string;
  autoClose?: boolean;
  duration?: number; // 自动关闭时间（毫秒）
}

interface BacktestErrorHandlerProps {
  notifications: BacktestNotification[];
  onDismiss: (id: string) => void;
  onClearAll: () => void;
  className?: string;
}

const notificationIcons = {
  info: <InformationCircleIcon className="w-5 h-5 text-primary-500" />,
  warning: <ExclamationTriangleIcon className="w-5 h-5 text-warning-500" />,
  error: <XCircleIcon className="w-5 h-5 text-danger-500" />,
  success: <CheckCircleIcon className="w-5 h-5 text-success-500" />
};

const notificationColors = {
  info: 'primary',
  warning: 'warning',
  error: 'danger',
  success: 'success'
} as const;

export default function BacktestErrorHandler({
  notifications,
  onDismiss,
  onClearAll,
  className = ''
}: BacktestErrorHandlerProps) {
  const [expandedNotifications, setExpandedNotifications] = useState<Set<string>>(new Set());
  const { isOpen: isModalOpen, onOpen: onModalOpen, onClose: onModalClose } = useDisclosure();

  // 自动关闭通知
  useEffect(() => {
    const timers: NodeJS.Timeout[] = [];

    notifications.forEach(notification => {
      if (notification.autoClose && notification.duration) {
        const timer = setTimeout(() => {
          onDismiss(notification.id);
        }, notification.duration);
        timers.push(timer);
      }
    });

    return () => {
      timers.forEach(timer => clearTimeout(timer));
    };
  }, [notifications, onDismiss]);

  // 切换通知展开状态
  const toggleExpanded = (id: string) => {
    const newExpanded = new Set(expandedNotifications);
    if (newExpanded.has(id)) {
      newExpanded.delete(id);
    } else {
      newExpanded.add(id);
    }
    setExpandedNotifications(newExpanded);
  };

  // 格式化时间
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('zh-CN', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // 获取通知统计
  const getNotificationStats = () => {
    const stats = {
      total: notifications.length,
      error: notifications.filter(n => n.type === 'error').length,
      warning: notifications.filter(n => n.type === 'warning').length,
      info: notifications.filter(n => n.type === 'info').length,
      success: notifications.filter(n => n.type === 'success').length
    };
    return stats;
  };

  const stats = getNotificationStats();

  if (notifications.length === 0) {
    return null;
  }

  return (
    <>
      <Card className={className}>
        <CardHeader className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            <BellIcon className="w-5 h-5" />
            <h3 className="text-lg font-semibold">通知中心</h3>
            <Chip size="sm" color="default" variant="flat">
              {stats.total}
            </Chip>
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="light"
              onPress={onModalOpen}
            >
              查看全部
            </Button>
            
            {notifications.length > 0 && (
              <Button
                size="sm"
                color="danger"
                variant="light"
                onPress={onClearAll}
              >
                清空
              </Button>
            )}
          </div>
        </CardHeader>

        <CardBody className="space-y-3">
          {/* 统计概览 */}
          <div className="flex gap-2 flex-wrap">
            {stats.error > 0 && (
              <Chip size="sm" color="danger" variant="flat">
                错误 {stats.error}
              </Chip>
            )}
            {stats.warning > 0 && (
              <Chip size="sm" color="warning" variant="flat">
                警告 {stats.warning}
              </Chip>
            )}
            {stats.success > 0 && (
              <Chip size="sm" color="success" variant="flat">
                成功 {stats.success}
              </Chip>
            )}
            {stats.info > 0 && (
              <Chip size="sm" color="primary" variant="flat">
                信息 {stats.info}
              </Chip>
            )}
          </div>

          {/* 最近的通知（最多显示3个） */}
          <div className="space-y-2">
            {notifications.slice(0, 3).map(notification => (
              <div
                key={notification.id}
                className={`
                  p-3 rounded-lg border cursor-pointer transition-colors
                  ${notification.type === 'error' ? 'bg-danger-50 border-danger-200 hover:bg-danger-100' :
                    notification.type === 'warning' ? 'bg-warning-50 border-warning-200 hover:bg-warning-100' :
                    notification.type === 'success' ? 'bg-success-50 border-success-200 hover:bg-success-100' :
                    'bg-primary-50 border-primary-200 hover:bg-primary-100'
                  }
                `}
                onClick={() => toggleExpanded(notification.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-2 flex-1 min-w-0">
                    {notificationIcons[notification.type]}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-sm truncate">
                          {notification.title}
                        </h4>
                        <span className="text-xs text-default-500 ml-2">
                          {formatTime(notification.timestamp)}
                        </span>
                      </div>
                      
                      <p className={`
                        text-sm mt-1
                        ${expandedNotifications.has(notification.id) ? '' : 'line-clamp-2'}
                        ${notification.type === 'error' ? 'text-danger-700' :
                          notification.type === 'warning' ? 'text-warning-700' :
                          notification.type === 'success' ? 'text-success-700' :
                          'text-primary-700'
                        }
                      `}>
                        {notification.message}
                      </p>
                      
                      {notification.taskId && (
                        <p className="text-xs text-default-500 mt-1">
                          任务ID: {notification.taskId}
                        </p>
                      )}
                    </div>
                  </div>
                  
                  <Button
                    isIconOnly
                    size="sm"
                    variant="light"
                    className="ml-2"
                    onPress={(e) => {
                      e.stopPropagation();
                      onDismiss(notification.id);
                    }}
                  >
                    <XMarkIcon className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>

          {notifications.length > 3 && (
            <div className="text-center">
              <Button
                size="sm"
                variant="light"
                onPress={onModalOpen}
              >
                查看更多 ({notifications.length - 3} 条)
              </Button>
            </div>
          )}
        </CardBody>
      </Card>

      {/* 全部通知模态框 */}
      <Modal 
        isOpen={isModalOpen} 
        onClose={onModalClose}
        size="2xl"
        scrollBehavior="inside"
      >
        <ModalContent>
          <ModalHeader className="flex justify-between items-center">
            <span>所有通知</span>
            <div className="flex items-center gap-2">
              <Chip size="sm" color="default" variant="flat">
                共 {notifications.length} 条
              </Chip>
            </div>
          </ModalHeader>
          
          <ModalBody>
            <div className="space-y-3">
              {notifications.map(notification => (
                <div
                  key={notification.id}
                  className={`
                    p-4 rounded-lg border
                    ${notification.type === 'error' ? 'bg-danger-50 border-danger-200' :
                      notification.type === 'warning' ? 'bg-warning-50 border-warning-200' :
                      notification.type === 'success' ? 'bg-success-50 border-success-200' :
                      'bg-primary-50 border-primary-200'
                    }
                  `}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3 flex-1">
                      {notificationIcons[notification.type]}
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium">
                            {notification.title}
                          </h4>
                          <Chip 
                            size="sm" 
                            color={notificationColors[notification.type]}
                            variant="flat"
                          >
                            {notification.type === 'error' ? '错误' :
                             notification.type === 'warning' ? '警告' :
                             notification.type === 'success' ? '成功' : '信息'}
                          </Chip>
                        </div>
                        
                        <p className={`
                          text-sm mb-2
                          ${notification.type === 'error' ? 'text-danger-700' :
                            notification.type === 'warning' ? 'text-warning-700' :
                            notification.type === 'success' ? 'text-success-700' :
                            'text-primary-700'
                          }
                        `}>
                          {notification.message}
                        </p>
                        
                        <div className="flex items-center justify-between text-xs text-default-500">
                          <span>
                            {notification.timestamp.toLocaleString('zh-CN')}
                          </span>
                          {notification.taskId && (
                            <span>任务ID: {notification.taskId}</span>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <Button
                      isIconOnly
                      size="sm"
                      variant="light"
                      className="ml-2"
                      onPress={() => onDismiss(notification.id)}
                    >
                      <XMarkIcon className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </ModalBody>
          
          <ModalFooter>
            <Button variant="light" onPress={onModalClose}>
              关闭
            </Button>
            <Button color="danger" variant="light" onPress={onClearAll}>
              清空所有通知
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </>
  );
}

/**
 * 通知管理Hook
 */
export function useBacktestNotifications() {
  const [notifications, setNotifications] = useState<BacktestNotification[]>([]);

  // 添加通知
  const addNotification = (notification: Omit<BacktestNotification, 'id' | 'timestamp'>) => {
    const newNotification: BacktestNotification = {
      ...notification,
      id: `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      autoClose: notification.autoClose ?? (notification.type === 'success' || notification.type === 'info'),
      duration: notification.duration ?? 5000
    };

    setNotifications(prev => [newNotification, ...prev]);
    return newNotification.id;
  };

  // 移除通知
  const removeNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  // 清空所有通知
  const clearAllNotifications = () => {
    setNotifications([]);
  };

  // 添加不同类型的通知的便捷方法
  const notifySuccess = (title: string, message: string, taskId?: string) => {
    return addNotification({ type: 'success', title, message, taskId });
  };

  const notifyError = (title: string, message: string, taskId?: string) => {
    return addNotification({ type: 'error', title, message, taskId, autoClose: false });
  };

  const notifyWarning = (title: string, message: string, taskId?: string) => {
    return addNotification({ type: 'warning', title, message, taskId, autoClose: false });
  };

  const notifyInfo = (title: string, message: string, taskId?: string) => {
    return addNotification({ type: 'info', title, message, taskId });
  };

  return {
    notifications,
    addNotification,
    removeNotification,
    clearAllNotifications,
    notifySuccess,
    notifyError,
    notifyWarning,
    notifyInfo
  };
}