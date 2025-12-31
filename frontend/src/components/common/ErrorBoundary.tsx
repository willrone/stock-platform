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
import { Card, CardBody, Button } from '@heroui/react';
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
        <div className="flex items-center justify-center min-h-screen p-6">
          <Card className="max-w-2xl w-full">
            <CardBody className="text-center space-y-6">
              <div className="flex justify-center">
                <AlertTriangle className="w-16 h-16 text-danger" />
              </div>
              
              <div>
                <h1 className="text-2xl font-bold text-danger mb-2">页面出现错误</h1>
                <p className="text-default-600">
                  抱歉，页面遇到了一些问题。您可以尝试刷新页面或返回首页。
                </p>
              </div>

              <div className="flex flex-wrap justify-center gap-3">
                <Button
                  color="primary"
                  startContent={<RefreshCw className="w-4 h-4" />}
                  onPress={this.handleReload}
                >
                  刷新页面
                </Button>
                <Button
                  variant="light"
                  startContent={<Home className="w-4 h-4" />}
                  onPress={this.handleGoHome}
                >
                  返回首页
                </Button>
                <Button
                  variant="light"
                  startContent={<RotateCcw className="w-4 h-4" />}
                  onPress={this.handleRetry}
                >
                  重试
                </Button>
              </div>

              {/* 开发环境下显示详细错误信息 */}
              {process.env.NODE_ENV === 'development' && this.state.error && (
                <Card className="mt-6">
                  <CardBody>
                    <div className="text-left space-y-4">
                      <h3 className="text-lg font-semibold text-danger">
                        错误详情（仅开发环境显示）
                      </h3>
                      
                      <div>
                        <p className="font-medium mb-2">错误信息：</p>
                        <code className="block p-2 bg-danger-50 text-danger rounded text-sm">
                          {this.state.error.message}
                        </code>
                      </div>
                      
                      <div>
                        <p className="font-medium mb-2">错误堆栈：</p>
                        <pre className="text-xs max-h-48 overflow-auto bg-default-100 p-3 rounded">
                          {this.state.error.stack}
                        </pre>
                      </div>

                      {this.state.errorInfo && (
                        <div>
                          <p className="font-medium mb-2">组件堆栈：</p>
                          <pre className="text-xs max-h-48 overflow-auto bg-default-100 p-3 rounded">
                            {this.state.errorInfo.componentStack}
                          </pre>
                        </div>
                      )}
                    </div>
                  </CardBody>
                </Card>
              )}
            </CardBody>
          </Card>
        </div>
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
  <div className="flex items-center justify-center p-6">
    <Card>
      <CardBody className="text-center space-y-4">
        <AlertTriangle className="w-12 h-12 text-danger mx-auto" />
        <div>
          <h3 className="text-lg font-semibold text-danger">出现错误</h3>
          <p className="text-default-600">{error?.message || '页面加载失败'}</p>
        </div>
        {onRetry && (
          <Button color="primary" onPress={onRetry}>
            重试
          </Button>
        )}
      </CardBody>
    </Card>
  </div>);