/**
 * 错误边界组件
 *
 * 捕获和处理React组件错误，包括：
 * - 组件渲染错误
 * - 生命周期错误
 * - 错误信息展示
 * - 错误恢复机制
 */

'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Card, CardContent, Button, Box, Typography, Alert } from '@mui/material';
import { AlertTriangle, RefreshCw, Home, RotateCcw } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    // 更新state以显示错误UI
    return {
      hasError: true,
      error,
      errorInfo: null,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // 记录错误信息
    console.error('ErrorBoundary捕获到错误:', error, errorInfo);

    this.setState({
      error,
      errorInfo,
    });

    // 调用外部错误处理函数
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // 可以在这里上报错误到监控系统
    // reportError(error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      // 如果提供了自定义fallback，使用它
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // 默认错误UI
      return (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '100vh',
            p: 3,
          }}
        >
          <Card sx={{ maxWidth: 800, width: '100%' }}>
            <CardContent
              sx={{ textAlign: 'center', display: 'flex', flexDirection: 'column', gap: 3 }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                <AlertTriangle size={64} color="#d32f2f" />
              </Box>

              <Box>
                <Typography
                  variant="h4"
                  component="h1"
                  sx={{ fontWeight: 600, color: 'error.main', mb: 1 }}
                >
                  页面出现错误
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  抱歉，页面遇到了一些问题。您可以尝试刷新页面或返回首页。
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 1.5 }}>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<RefreshCw size={16} />}
                  onClick={this.handleReload}
                >
                  刷新页面
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Home size={16} />}
                  onClick={this.handleGoHome}
                >
                  返回首页
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<RotateCcw size={16} />}
                  onClick={this.handleRetry}
                >
                  重试
                </Button>
              </Box>

              {/* 开发环境下显示详细错误信息 */}
              {process.env.NODE_ENV === 'development' && this.state.error && (
                <Card sx={{ mt: 3 }}>
                  <CardContent>
                    <Box
                      sx={{ textAlign: 'left', display: 'flex', flexDirection: 'column', gap: 2 }}
                    >
                      <Typography variant="h6" sx={{ fontWeight: 600, color: 'error.main' }}>
                        错误详情（仅开发环境显示）
                      </Typography>

                      <Box>
                        <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                          错误信息：
                        </Typography>
                        <Alert severity="error" sx={{ fontFamily: 'monospace' }}>
                          {this.state.error.message}
                        </Alert>
                      </Box>

                      <Box>
                        <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                          错误堆栈：
                        </Typography>
                        <Box
                          component="pre"
                          sx={{
                            fontSize: '0.75rem',
                            maxHeight: 200,
                            overflow: 'auto',
                            bgcolor: 'grey.100',
                            p: 2,
                            borderRadius: 1,
                            fontFamily: 'monospace',
                          }}
                        >
                          {this.state.error.stack}
                        </Box>
                      </Box>

                      {this.state.errorInfo && (
                        <Box>
                          <Typography variant="body2" sx={{ fontWeight: 500, mb: 1 }}>
                            组件堆栈：
                          </Typography>
                          <Box
                            component="pre"
                            sx={{
                              fontSize: '0.75rem',
                              maxHeight: 200,
                              overflow: 'auto',
                              bgcolor: 'grey.100',
                              p: 2,
                              borderRadius: 1,
                              fontFamily: 'monospace',
                            }}
                          >
                            {this.state.errorInfo.componentStack}
                          </Box>
                        </Box>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </Box>
      );
    }

    return this.props.children;
  }
}

// 高阶组件：为组件添加错误边界
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  fallback?: ReactNode,
  onError?: (error: Error, errorInfo: ErrorInfo) => void
) {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary fallback={fallback} onError={onError}>
      <Component {...props} />
    </ErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;

  return WrappedComponent;
}

// 简化的错误显示组件
export const SimpleErrorFallback: React.FC<{
  error?: Error;
  onRetry?: () => void;
}> = ({ error, onRetry }) => (
  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', p: 3 }}>
    <Card>
      <CardContent sx={{ textAlign: 'center', display: 'flex', flexDirection: 'column', gap: 2 }}>
        <AlertTriangle size={48} color="#d32f2f" style={{ margin: '0 auto' }} />
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 600, color: 'error.main' }}>
            出现错误
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {error?.message || '页面加载失败'}
          </Typography>
        </Box>
        {onRetry && (
          <Button variant="contained" color="primary" onClick={onRetry}>
            重试
          </Button>
        )}
      </CardContent>
    </Card>
  </Box>
);
