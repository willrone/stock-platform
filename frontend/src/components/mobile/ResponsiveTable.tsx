'use client';

/**
 * 响应式表格容器
 *
 * 解决 P0-1：明细表被外层 Card 的 overflow:hidden 硬裁 —— 在 390px 视口下
 * 「金额 / 费用 / 分数 / 为什么买它」等右列**永久不可见且无法横向滚动**。
 *
 * 做法：
 *   - 外层 Box 加 overflowX:'auto' + WebkitOverflowScrolling:'touch'
 *   - Table 加 minWidth（默认 560px，移动端需要能左右滑）
 *   - 表格下方加一条「← 左右滑动查看完整列」的提示（移动端可见）
 *
 * 用法：
 *   <ResponsiveTable minWidth={640}>
 *     <Table>…</Table>
 *   </ResponsiveTable>
 */

import React from 'react';
import { Box, Typography } from '@mui/material';
import { MoveHorizontal } from 'lucide-react';

interface ResponsiveTableProps {
  children: React.ReactNode;
  /** 表格最小宽度（px）。列多时建议 560-720。 */
  minWidth?: number;
  /** 是否显示「左右滑动」提示（默认仅窄屏显示） */
  hint?: boolean;
  /** 额外样式 */
  sx?: Record<string, unknown>;
}

export const ResponsiveTable: React.FC<ResponsiveTableProps> = ({
  children,
  minWidth = 560,
  hint = true,
  sx,
}) => (
  <Box sx={{ width: '100%' }}>
    <Box
      sx={{
        width: '100%',
        maxWidth: '100%',
        overflowX: 'auto',
        overflowY: 'visible',
        WebkitOverflowScrolling: 'touch',
        // 让表格在窄屏下保持 minWidth 而不是被压缩
        '& table': { minWidth: `${minWidth}px` },
        ...sx,
      }}
    >
      {children}
    </Box>
    {hint && (
      <Box
        sx={{
          display: { xs: 'flex', md: 'none' },
          alignItems: 'center',
          gap: 0.5,
          mt: 0.75,
          color: 'text.secondary',
        }}
      >
        <MoveHorizontal size={13} />
        <Typography variant="caption" color="text.secondary">
          左右滑动查看完整列
        </Typography>
      </Box>
    )}
  </Box>
);

export default ResponsiveTable;
