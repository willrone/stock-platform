'use client';

/**
 * 顶部安全区占位
 *
 * 背景：顶部吸顶栏（AppBar）在 src/components/layout/AppLayout.tsx 里，
 * 该文件由其他代理负责，我不能改。页面侧只能提供一个**兜底**占位：
 * 在 iOS 刘海屏 / 灵动岛机型上，AppBar 默认会被系统状态栏压住，
 * 这里在页面内容顶部补一段 safe-area 高度，让正文不被系统栏遮挡。
 *
 * 👉 更好的做法（AppLayout 负责人实现）：
 *     给 AppLayout 的 <AppBar> 加
 *       sx={{ paddingTop: SAFE_TOP }}
 *     并给 <Toolbar> 加
 *       sx={{ minHeight: 'calc(56px + var(--safe-top, env(safe-area-inset-top, 0px)))' }}
 *     —— 做完之后，本组件可以整体删除（它只影响内容区的起点）。
 *
 * 用法：<SafeAreaTop /> 放在页面根容器内部第一行。
 */

import React from 'react';
import { Box } from '@mui/material';
import { SAFE_TOP } from './safeArea';

export const SafeAreaTop: React.FC = () => (
  <Box
    aria-hidden="true"
    sx={{
      display: { xs: 'block', sm: 'none' },
      height: SAFE_TOP,
      flexShrink: 0,
      pointerEvents: 'none',
    }}
  />
);

export default SafeAreaTop;
