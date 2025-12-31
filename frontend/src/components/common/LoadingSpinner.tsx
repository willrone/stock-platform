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
  <Loader2 className="w-4 h-4 animate-spin" />);