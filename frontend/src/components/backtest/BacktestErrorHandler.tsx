'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Button,
  Chip,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Box,
  Typography,
  IconButton,
} from '@mui/material';
import {
  ExclamationTriangleIcon,
  XCircleIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  BellIcon,
  XMarkIcon,
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
  success: <CheckCircleIcon className="w-5 h-5 text-success-500" />,
};

const notificationColors = {
  info: 'primary',
  warning: 'warning',
  error: 'error',
  success: 'success',
} as const;

export default function BacktestErrorHandler({
  notifications,
  onDismiss,
  onClearAll,
  className = '',
}: BacktestErrorHandlerProps) {
  const [expandedNotifications, setExpandedNotifications] = useState<Set<string>>(new Set());
  const [isModalOpen, setIsModalOpen] = useState(false);

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
      second: '2-digit',
    });
  };

  // 获取通知统计
  const getNotificationStats = () => {
    const stats = {
      total: notifications.length,
      error: notifications.filter(n => n.type === 'error').length,
      warning: notifications.filter(n => n.type === 'warning').length,
      info: notifications.filter(n => n.type === 'info').length,
      success: notifications.filter(n => n.type === 'success').length,
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
        <CardHeader
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <BellIcon className="w-5 h-5" />
              <Typography
                variant="h6"
                component="span"
                sx={{ fontSize: { xs: '0.95rem', md: '1.25rem' } }}
              >
                通知中心
              </Typography>
              <Chip label={stats.total} size="small" />
            </Box>
          }
          action={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Button size="small" variant="outlined" onClick={() => setIsModalOpen(true)}>
                查看全部
              </Button>

              {notifications.length > 0 && (
                <Button size="small" color="error" variant="outlined" onClick={onClearAll}>
                  清空
                </Button>
              )}
            </Box>
          }
        />

        <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {/* 统计概览 */}
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {stats.error > 0 && <Chip label={`错误 ${stats.error}`} color="error" size="small" />}
            {stats.warning > 0 && (
              <Chip label={`警告 ${stats.warning}`} color="warning" size="small" />
            )}
            {stats.success > 0 && (
              <Chip label={`成功 ${stats.success}`} color="success" size="small" />
            )}
            {stats.info > 0 && <Chip label={`信息 ${stats.info}`} color="primary" size="small" />}
          </Box>

          {/* 最近的通知（最多显示3个） */}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {notifications.slice(0, 3).map(notification => (
              <Box
                key={notification.id}
                onClick={() => toggleExpanded(notification.id)}
                sx={{
                  p: 1.5,
                  borderRadius: 1,
                  border: 1,
                  cursor: 'pointer',
                  transition: 'background-color 0.2s',
                  ...(notification.type === 'error'
                    ? {
                        bgcolor: 'error.light',
                        borderColor: 'error.main',
                        '&:hover': { bgcolor: 'error.lighter' },
                      }
                    : notification.type === 'warning'
                      ? {
                          bgcolor: 'warning.light',
                          borderColor: 'warning.main',
                          '&:hover': { bgcolor: 'warning.lighter' },
                        }
                      : notification.type === 'success'
                        ? {
                            bgcolor: 'success.light',
                            borderColor: 'success.main',
                            '&:hover': { bgcolor: 'success.lighter' },
                          }
                        : {
                            bgcolor: 'primary.light',
                            borderColor: 'primary.main',
                            '&:hover': { bgcolor: 'primary.lighter' },
                          }),
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    justifyContent: 'space-between',
                  }}
                >
                  <Box
                    sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, flex: 1, minWidth: 0 }}
                  >
                    {notificationIcons[notification.type]}
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                        }}
                      >
                        <Typography
                          variant="body2"
                          sx={{
                            fontWeight: 500,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {notification.title}
                        </Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                          {formatTime(notification.timestamp)}
                        </Typography>
                      </Box>

                      <Typography
                        variant="body2"
                        sx={{
                          mt: 0.5,
                          ...(!expandedNotifications.has(notification.id) && {
                            display: '-webkit-box',
                            WebkitLineClamp: 2,
                            WebkitBoxOrient: 'vertical',
                            overflow: 'hidden',
                          }),
                          ...(notification.type === 'error'
                            ? { color: 'error.dark' }
                            : notification.type === 'warning'
                              ? { color: 'warning.dark' }
                              : notification.type === 'success'
                                ? { color: 'success.dark' }
                                : { color: 'primary.dark' }),
                        }}
                      >
                        {notification.message}
                      </Typography>

                      {notification.taskId && (
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ mt: 0.5, display: 'block' }}
                        >
                          任务ID: {notification.taskId}
                        </Typography>
                      )}
                    </Box>
                  </Box>

                  <IconButton
                    size="small"
                    onClick={e => {
                      e.stopPropagation();
                      onDismiss(notification.id);
                    }}
                    sx={{ ml: 1 }}
                  >
                    <XMarkIcon className="w-4 h-4" />
                  </IconButton>
                </Box>
              </Box>
            ))}
          </Box>

          {notifications.length > 3 && (
            <Box sx={{ textAlign: 'center' }}>
              <Button size="small" variant="outlined" onClick={() => setIsModalOpen(true)}>
                查看更多 ({notifications.length - 3} 条)
              </Button>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* 全部通知模态框 */}
      <Dialog open={isModalOpen} onClose={() => setIsModalOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>所有通知</span>
            <Chip label={`共 ${notifications.length} 条`} size="small" />
          </Box>
        </DialogTitle>

        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {notifications.map(notification => (
              <Box
                key={notification.id}
                sx={{
                  p: 2,
                  borderRadius: 1,
                  border: 1,
                  ...(notification.type === 'error'
                    ? {
                        bgcolor: 'error.light',
                        borderColor: 'error.main',
                      }
                    : notification.type === 'warning'
                      ? {
                          bgcolor: 'warning.light',
                          borderColor: 'warning.main',
                        }
                      : notification.type === 'success'
                        ? {
                            bgcolor: 'success.light',
                            borderColor: 'success.main',
                          }
                        : {
                            bgcolor: 'primary.light',
                            borderColor: 'primary.main',
                          }),
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    justifyContent: 'space-between',
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5, flex: 1 }}>
                    {notificationIcons[notification.type]}
                    <Box sx={{ flex: 1 }}>
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          mb: 1,
                        }}
                      >
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {notification.title}
                        </Typography>
                        <Chip
                          label={
                            notification.type === 'error'
                              ? '错误'
                              : notification.type === 'warning'
                                ? '警告'
                                : notification.type === 'success'
                                  ? '成功'
                                  : '信息'
                          }
                          color={notificationColors[notification.type]}
                          size="small"
                        />
                      </Box>

                      <Typography
                        variant="body2"
                        sx={{
                          mb: 1,
                          ...(notification.type === 'error'
                            ? { color: 'error.dark' }
                            : notification.type === 'warning'
                              ? { color: 'warning.dark' }
                              : notification.type === 'success'
                                ? { color: 'success.dark' }
                                : { color: 'primary.dark' }),
                        }}
                      >
                        {notification.message}
                      </Typography>

                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                        }}
                      >
                        <Typography variant="caption" color="text.secondary">
                          {notification.timestamp.toLocaleString('zh-CN')}
                        </Typography>
                        {notification.taskId && (
                          <Typography variant="caption" color="text.secondary">
                            任务ID: {notification.taskId}
                          </Typography>
                        )}
                      </Box>
                    </Box>
                  </Box>

                  <IconButton
                    size="small"
                    onClick={() => onDismiss(notification.id)}
                    sx={{ ml: 1 }}
                  >
                    <XMarkIcon className="w-4 h-4" />
                  </IconButton>
                </Box>
              </Box>
            ))}
          </Box>
        </DialogContent>

        <DialogActions>
          <Button onClick={() => setIsModalOpen(false)}>关闭</Button>
          <Button color="error" onClick={onClearAll}>
            清空所有通知
          </Button>
        </DialogActions>
      </Dialog>
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
      autoClose:
        notification.autoClose ?? (notification.type === 'success' || notification.type === 'info'),
      duration: notification.duration ?? 5000,
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
    notifyInfo,
  };
}
