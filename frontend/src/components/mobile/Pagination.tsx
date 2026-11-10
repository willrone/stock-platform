'use client';

/**
 * 客户端分页控件
 *
 * 解决 P0-2 / P0-3：一次渲染 5594 行直接把浏览器渲染进程拖垮
 * （实测桌面页高 29.8 万 px、窄屏 51 万 px → OOM）。
 *
 * 为什么选「客户端分页」而不是虚拟滚动：
 *   - 数据本来就一次性拿到了（5594 条 ≈ 1MB JSON），虚拟滚动需要固定行高 +
 *     绝对定位 + 滚动补偿，对 MUI Table 来说复杂且容易出错；
 *   - 分页对移动端更友好：用户能明确知道「第 3 / 112 页」，可以跳回
 *     某一条，不用在几千行里滚动定位；
 *   - DOM 节点数固定在 pageSize 量级（50-100），渲染开销可预测。
 *
 * 每页行数选 50：移动端一屏大约能看到 4-6 张卡片 / 12-15 行表格，
 * 翻页不至于太频繁；DOM 节点数远低于任何移动浏览器的压力阈值。
 */

import React from 'react';
import { Box, Button, Typography } from '@mui/material';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { MIN_TOUCH_TARGET } from './safeArea';

interface PaginationProps {
  /** 总条数 */
  total: number;
  /** 每页条数 */
  pageSize: number;
  /** 当前页（0 基） */
  page: number;
  onPageChange: (page: number) => void;
  /** 显示额外的说明文案（如「已筛选」） */
  label?: string;
  /** 每页可选条数（默认关闭） */
  allowPageSizeChange?: boolean;
  onPageSizeChange?: (size: number) => void;
}

export const Pagination: React.FC<PaginationProps> = ({
  total,
  pageSize,
  page,
  onPageChange,
  label,
  allowPageSizeChange = false,
  onPageSizeChange,
}) => {
  const pageCount = Math.max(1, Math.ceil(total / Math.max(1, pageSize)));
  const safePage = Math.min(page, pageCount - 1);
  const from = total === 0 ? 0 : safePage * pageSize + 1;
  const to = Math.min(total, (safePage + 1) * pageSize);

  return (
    <Box
      sx={{
        mt: 1.5,
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 1,
      }}
    >
      <Typography variant="caption" color="text.secondary">
        {total === 0
          ? '无数据'
          : `显示 ${from}-${to} / 共 ${total.toLocaleString('zh-CN')} 条${
              label ? ` · ${label}` : ''
            }`}
      </Typography>

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
        {allowPageSizeChange && onPageSizeChange && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mr: 0.5 }}>
            {[50, 100, 200].map(size => (
              <Button
                key={size}
                size="small"
                variant={pageSize === size ? 'contained' : 'text'}
                onClick={() => onPageSizeChange(size)}
                sx={{ minHeight: MIN_TOUCH_TARGET, minWidth: 40, px: 1 }}
              >
                {size}
              </Button>
            ))}
          </Box>
        )}
        <Button
          variant="outlined"
          size="small"
          disabled={safePage <= 0}
          onClick={() => onPageChange(safePage - 1)}
          aria-label="上一页"
          sx={{ minHeight: MIN_TOUCH_TARGET, minWidth: MIN_TOUCH_TARGET, px: 1 }}
        >
          <ChevronLeft size={18} />
        </Button>
        <Typography
          variant="body2"
          sx={{ minWidth: 74, textAlign: 'center', fontWeight: 600 }}
          aria-live="polite"
        >
          {safePage + 1} / {pageCount}
        </Typography>
        <Button
          variant="outlined"
          size="small"
          disabled={safePage >= pageCount - 1}
          onClick={() => onPageChange(safePage + 1)}
          aria-label="下一页"
          sx={{ minHeight: MIN_TOUCH_TARGET, minWidth: MIN_TOUCH_TARGET, px: 1 }}
        >
          <ChevronRight size={18} />
        </Button>
      </Box>
    </Box>
  );
};

export default Pagination;
