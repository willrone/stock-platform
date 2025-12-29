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
);