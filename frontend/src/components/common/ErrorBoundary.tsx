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
import { Result, Button, Typography, Space, Alert } from 'antd';
import { BugOutlined, ReloadOutlined, HomeOutlined } from '@ant-design/icons';

const { Paragraph, Text } = Typography;

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
        <div style={{ padding: '50px 24px', textAlign: 'center' }}>
          <Result
            status="error"
            icon={<BugOutlined />}
            title="页面出现错误"
            subTitle="抱歉，页面遇到了一些问题。您可以尝试刷新页面或返回首页。"
            extra={
              <Space direction="vertical" size="large">
                <Space>
                  <Button type="primary" icon={<ReloadOutlined />} onClick={this.handleReload}>
                    刷新页面
                  </Button>
                  <Button icon={<HomeOutlined />} onClick={this.handleGoHome}>
                    返回首页
                  </Button>
                  <Button onClick={this.handleRetry}>
                    重试
                  </Button>
                </Space>

                {/* 开发环境下显示详细错误信息 */}
                {process.env.NODE_ENV === 'development' && this.state.error && (
                  <Alert
                    message="错误详情（仅开发环境显示）"
                    description={
                      <div style={{ textAlign: 'left' }}>
                        <Paragraph>
                          <Text strong>错误信息：</Text>
                          <br />
                          <Text code>{this.state.error.message}</Text>
                        </Paragraph>
                        
                        <Paragraph>
                          <Text strong>错误堆栈：</Text>
                          <br />
                          <pre style={{ 
                            fontSize: 12, 
                            maxHeight: 200, 
                            overflow: 'auto',
                            background: '#f5f5f5',
                            padding: 8,
                            borderRadius: 4,
                          }}>
                            {this.state.error.stack}
                          </pre>
                        </Paragraph>

                        {this.state.errorInfo && (
                          <Paragraph>
                            <Text strong>组件堆栈：</Text>
                            <br />
                            <pre style={{ 
                              fontSize: 12, 
                              maxHeight: 200, 
                              overflow: 'auto',
                              background: '#f5f5f5',
                              padding: 8,
                              borderRadius: 4,
                            }}>
                              {this.state.errorInfo.componentStack}
                            </pre>
                          </Paragraph>
                        )}
                      </div>
                    }
                    type="error"
                    showIcon
                    style={{ marginTop: 16 }}
                  />
                )}
              </Space>
            }
          />
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
  <div style={{ padding: 24, textAlign: 'center' }}>
    <Result
      status="error"
      title="出现错误"
      subTitle={error?.message || '页面加载失败'}
      extra={
        onRetry && (
          <Button type="primary" onClick={onRetry}>
            重试
          </Button>
        )
      }
    />
  </div>
);