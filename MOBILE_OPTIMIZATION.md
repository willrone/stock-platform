# Willrone 移动端优化方案

**日期**: 2026-02-01  
**目标**: 提升手机竖屏浏览和操作体验

## 当前问题

1. **表格横向溢出** - Table 组件在小屏幕需要横向滚动
2. **点击目标过小** - 部分按钮和链接在触摸屏不好点
3. **字体偏小** - 移动端阅读费劲
4. **图表不适配** - 图表在小屏幕显示拥挤
5. **导航占用空间** - 顶部导航在竖屏占比过高

## 优化方案

### 1. 响应式表格 → 卡片布局

**位置**: 
- `src/app/tasks/page.tsx`
- `src/app/models/page.tsx`  
- `src/app/data/page.tsx`

**改造前**:
```tsx
<Table>
  <TableHead>...</TableHead>
  <TableBody>
    {tasks.map(task => (
      <TableRow>
        <TableCell>{task.name}</TableCell>
        <TableCell>{task.status}</TableCell>
        ...
      </TableRow>
    ))}
  </TableBody>
</Table>
```

**改造后**:
```tsx
{/* 桌面端：表格 */}
<Box sx={{ display: { xs: 'none', md: 'block' } }}>
  <Table>...</Table>
</Box>

{/* 移动端：卡片 */}
<Box sx={{ display: { xs: 'block', md: 'none' } }}>
  {tasks.map(task => (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="h6">{task.name}</Typography>
          <Chip label={task.status} size="small" />
        </Box>
        <Typography variant="body2" color="text.secondary">
          {task.stock_codes?.length || 0} 只股票
        </Typography>
        {task.progress && (
          <LinearProgress variant="determinate" value={task.progress} sx={{ mt: 1 }} />
        )}
        <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
          <Button size="small" onClick={() => router.push(`/tasks/${task.id}`)}>
            查看
          </Button>
          <IconButton size="small" onClick={() => handleDelete(task.id)}>
            <Trash2 size={16} />
          </IconButton>
        </Box>
      </CardContent>
    </Card>
  ))}
</Box>
```

### 2. 触摸优化

**最小点击区域**: 44x44px（Apple/Material Design 推荐）

```tsx
// 之前
<IconButton size="small">
  <Eye size={16} />
</IconButton>

// 优化后
<IconButton 
  size="small"
  sx={{ 
    minWidth: 44, 
    minHeight: 44,
    '@media (pointer: coarse)': { // 触摸设备
      minWidth: 48,
      minHeight: 48,
    }
  }}
>
  <Eye size={20} />  {/* 图标也加大 */}
</IconButton>
```

### 3. 字体和间距优化

**创建移动端主题覆盖** (`src/theme/mobileOverrides.ts`):

```tsx
import { Theme } from '@mui/material/styles';

export const getMobileOverrides = (theme: Theme) => ({
  components: {
    MuiTypography: {
      styleOverrides: {
        h4: {
          [theme.breakpoints.down('sm')]: {
            fontSize: '1.5rem', // 从 2.125rem 降到 1.5rem
          },
        },
        h5: {
          [theme.breakpoints.down('sm')]: {
            fontSize: '1.25rem',
          },
        },
        h6: {
          [theme.breakpoints.down('sm')]: {
            fontSize: '1.125rem',
          },
        },
        body1: {
          [theme.breakpoints.down('sm')]: {
            fontSize: '0.95rem', // 稍微加大
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          [theme.breakpoints.down('sm')]: {
            minHeight: 44, // 触摸友好
            fontSize: '0.95rem',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          [theme.breakpoints.down('sm')]: {
            borderRadius: 12, // 圆角更明显
          },
        },
      },
    },
  },
});
```

### 4. 图表适配

**位置**: `src/components/charts/*`

**改进**:

```tsx
// 动态调整图表高度
const chartHeight = useMediaQuery(theme.breakpoints.down('sm')) ? 250 : 400;

<ResponsiveContainer width="100%" height={chartHeight}>
  <LineChart data={data}>
    <XAxis 
      dataKey="date"
      tick={{ fontSize: 12 }} // 移动端字体
      angle={-45}  // 倾斜标签避免重叠
      textAnchor="end"
      height={60}
    />
    <YAxis tick={{ fontSize: 12 }} />
    <Tooltip 
      contentStyle={{ 
        fontSize: 12,
        borderRadius: 8,
      }}
    />
    <Line type="monotone" dataKey="value" strokeWidth={2} />
  </LineChart>
</ResponsiveContainer>
```

**TradingView 图表** (`src/components/charts/TradingViewChart.tsx`):
- 移动端隐藏部分工具栏
- 简化图例显示
- 启用触摸缩放

```tsx
const widgetOptions = {
  ...baseOptions,
  // 移动端优化
  ...(isMobile && {
    toolbar_bg: '#f1f3f6',
    hide_side_toolbar: true, // 隐藏侧边栏
    hide_top_toolbar: false,
    studies_overrides: {}, // 简化指标
  }),
};
```

### 5. 布局优化

**AppLayout 移动端改进** (`src/components/layout/AppLayout.tsx`):

```tsx
// 主内容区域 - 减少移动端 padding
<Box
  component="main"
  sx={{ 
    flexGrow: 1, 
    p: { xs: 1.5, sm: 2, md: 3 },  // 响应式 padding
    maxWidth: '1400px', 
    mx: 'auto', 
    width: '100%' 
  }}
>
  {children}
</Box>

// 顶部导航 - 移动端紧凑模式
<AppBar
  position="sticky"
  sx={{ 
    bgcolor: 'background.paper', 
    color: 'text.primary', 
    boxShadow: 1,
    '& .MuiToolbar-root': {
      minHeight: { xs: 56, sm: 64 }, // 移动端降低高度
      px: { xs: 1, sm: 2 },
    },
  }}
>
```

### 6. 移动端专用组件

**创建 `src/components/mobile/MobileTaskCard.tsx`**:

```tsx
'use client';

import React from 'react';
import { Card, CardContent, Box, Typography, Chip, LinearProgress, IconButton } from '@mui/material';
import { Eye, Trash2, Play, Pause } from 'lucide-react';
import { useRouter } from 'next/navigation';

interface MobileTaskCardProps {
  task: {
    task_id: string;
    task_name: string;
    status: string;
    progress?: number;
    stock_codes?: string[];
    created_at: string;
  };
  onDelete?: (id: string) => void;
  onToggle?: (id: string) => void;
}

export const MobileTaskCard: React.FC<MobileTaskCardProps> = ({ task, onDelete, onToggle }) => {
  const router = useRouter();

  const statusColor = {
    running: 'primary',
    completed: 'success',
    failed: 'error',
    created: 'default',
  }[task.status] || 'default';

  return (
    <Card 
      sx={{ 
        mb: 2, 
        borderRadius: 3,
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
      }}
    >
      <CardContent>
        {/* 标题行 */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
          <Typography 
            variant="h6" 
            sx={{ 
              fontWeight: 600,
              fontSize: '1.1rem',
              flex: 1,
              pr: 1,
            }}
          >
            {task.task_name}
          </Typography>
          <Chip label={task.status} color={statusColor as any} size="small" />
        </Box>

        {/* 信息行 */}
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          {task.stock_codes?.length || 0} 只股票 · {new Date(task.created_at).toLocaleDateString()}
        </Typography>

        {/* 进度条 */}
        {task.status === 'running' && task.progress != null && (
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
              <Typography variant="caption" color="text.secondary">
                进度
              </Typography>
              <Typography variant="caption" fontWeight={600}>
                {task.progress}%
              </Typography>
            </Box>
            <LinearProgress variant="determinate" value={task.progress} />
          </Box>
        )}

        {/* 操作按钮 */}
        <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
          <IconButton 
            size="medium"
            onClick={() => router.push(`/tasks/${task.task_id}`)}
            sx={{ 
              flex: 1,
              border: 1,
              borderColor: 'divider',
              borderRadius: 2,
              minHeight: 44,
            }}
          >
            <Eye size={18} />
            <Typography variant="body2" sx={{ ml: 1 }}>查看</Typography>
          </IconButton>

          {task.status === 'running' && onToggle && (
            <IconButton
              size="medium"
              onClick={() => onToggle(task.task_id)}
              sx={{ 
                border: 1,
                borderColor: 'warning.main',
                borderRadius: 2,
                minHeight: 44,
                minWidth: 44,
              }}
            >
              <Pause size={18} />
            </IconButton>
          )}

          {onDelete && (
            <IconButton
              size="medium"
              onClick={() => onDelete(task.task_id)}
              sx={{ 
                border: 1,
                borderColor: 'error.main',
                borderRadius: 2,
                minHeight: 44,
                minWidth: 44,
              }}
            >
              <Trash2 size={18} />
            </IconButton>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};
```

### 7. Viewport 配置

**检查 `src/app/layout.tsx` 的 metadata**:

```tsx
export const metadata: Metadata = {
  title: '股票预测平台',
  description: '基于AI的股票预测和任务管理系统',
  keywords: ['股票预测', 'AI', '机器学习', '量化交易', '投资分析'],
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 5, // 允许缩放
    userScalable: true,
  },
};
```

## 实施计划

### 阶段 1: 核心布局优化 (2小时)
- [x] 检查当前响应式设计
- [ ] 优化主题配置（字体/间距/触摸目标）
- [ ] 更新 AppLayout 移动端样式
- [ ] 添加 viewport meta

### 阶段 2: 表格 → 卡片改造 (3小时)
- [ ] 创建 MobileTaskCard 组件
- [ ] 改造任务列表页面 (`tasks/page.tsx`)
- [ ] 改造模型列表页面 (`models/page.tsx`)
- [ ] 改造数据管理页面 (`data/page.tsx`)

### 阶段 3: 图表优化 (2小时)
- [ ] 调整 Recharts 响应式配置
- [ ] 优化 TradingView 图表移动端
- [ ] 测试所有图表组件

### 阶段 4: 测试和调优 (1小时)
- [ ] iPhone SE (375px) 测试
- [ ] iPhone 14 Pro (393px) 测试
- [ ] Android 中等屏幕 (360px) 测试
- [ ] 横屏适配测试

## 预期效果

- ✅ 竖屏无需横向滚动
- ✅ 所有操作按钮触摸友好 (≥44px)
- ✅ 文字清晰可读
- ✅ 图表自适应屏幕尺寸
- ✅ 导航不阻挡内容
- ✅ 加载性能良好

## 技术栈

- Next.js 14+ (App Router)
- Material-UI v5
- Recharts / TradingView
- TypeScript

## 参考

- [Material Design - Touch targets](https://m3.material.io/foundations/interaction/gestures#c16c1ad6-3f1f-42dc-9b90-80dc35da43e7)
- [Apple HIG - Touch targets](https://developer.apple.com/design/human-interface-guidelines/layout#Best-practices)
- [Next.js - Responsive Design](https://nextjs.org/docs/pages/building-your-application/optimizing/fonts#responsive-design)
