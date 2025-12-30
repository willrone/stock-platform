<<<<<<< HEAD
/**
 * 加载动画组件
 * 
 * 提供统一的加载状态显示，包括：
 * - 页面级加载
 * - 组件级加载
 * - 按钮加载状态
 * - 自定义加载文本
 */

'use client';

import React from 'react';
import { Spin, Space, Typography } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';

const { Text } = Typography;

interface LoadingSpinnerProps {
  /** 加载文本 */
  text?: string;
  /** 尺寸大小 */
  size?: 'small' | 'default' | 'large';
  /** 是否显示文本 */
  showText?: boolean;
  /** 是否居中显示 */
  centered?: boolean;
  /** 自定义样式 */
  style?: React.CSSProperties;
  /** 是否使用自定义图标 */
  useCustomIcon?: boolean;
}

// 自定义加载图标
const customIcon = <LoadingOutlined style={{ fontSize: 24 }} spin />;

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  text = '加载中...',
  size = 'default',
  showText = true,
  centered = true,
  style,
  useCustomIcon = false,
}) => {
  const spinProps = {
    size,
    indicator: useCustomIcon ? customIcon : undefined,
  };

  const content = (
    <Space direction="vertical" align="center" size="middle">
      <Spin {...spinProps} />
      {showText && (
        <Text type="secondary" style={{ fontSize: 14 }}>
          {text}
        </Text>
      )}
    </Space>
  );

  if (centered) {
    return (
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: 200,
          ...style,
        }}
      >
        {content}
      </div>
    );
  }

  return <div style={style}>{content}</div>;
};

// 页面级加载组件
export const PageLoading: React.FC<{ text?: string }> = ({ text = '页面加载中...' }) => (
  <LoadingSpinner
    text={text}
    size="large"
    useCustomIcon
    style={{
      minHeight: '60vh',
    }}
  />
);

// 内容区域加载组件
export const ContentLoading: React.FC<{ text?: string }> = ({ text = '内容加载中...' }) => (
  <LoadingSpinner
    text={text}
    size="default"
    style={{
      minHeight: 300,
    }}
  />
);

// 小型加载组件
export const InlineLoading: React.FC<{ text?: string }> = ({ text = '加载中...' }) => (
  <LoadingSpinner
    text={text}
    size="small"
    centered={false}
    style={{
      display: 'inline-flex',
      alignItems: 'center',
    }}
  />
=======
/**
 * 加载动画组件
 * 
 * 提供统一的加载状态显示，包括：
 * - 旋转动画
 * - 自定义文本
 * - 不同尺寸
 * - 居中布局
 */

'use client';

import React from 'react';
import { Spinner } from '@heroui/react';
import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  text?: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  fullScreen?: boolean;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  text = '加载中...',
  size = 'lg',
  className = '',
  fullScreen = false,
}) => {
  const containerClass = fullScreen
    ? 'fixed inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm z-50'
    : 'flex flex-col items-center justify-center p-8';

  return (
    <div className={`${containerClass} ${className}`}>
      <Spinner size={size} />
      {text && (
        <p className="mt-4 text-default-600 text-center">{text}</p>
      )}
    </div>
  );
};

// 页面级加载组件
export const PageLoading: React.FC<{ text?: string }> = ({ text }) => (
  <LoadingSpinner text={text} fullScreen />
);

// 内联加载组件
export const InlineLoading: React.FC<{ text?: string; size?: 'sm' | 'md' | 'lg' }> = ({ 
  text, 
  size = 'md' 
}) => (
  <div className="flex items-center space-x-2">
    <Spinner size={size} />
    {text && <span className="text-default-600">{text}</span>}
  </div>
);

// 按钮加载状态
export const ButtonLoading: React.FC = () => (
  <Loader2 className="w-4 h-4 animate-spin" />
>>>>>>> a6754c6 (feat(platform): Complete stock prediction platform with deployment and monitoring)
);