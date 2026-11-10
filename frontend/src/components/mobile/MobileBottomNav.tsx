'use client';

/**
 * 移动端底部 Tab 栏
 *
 * 解决 P0-5：核心页面在手机上只有顶栏，没有底部快捷入口。
 *
 * 设计：
 *   - 5 个入口：仪表板 / 今日信号 / 策略信号 / 任务 / 更多
 *   - fixed 定位，底部加 safe-area padding（env(safe-area-inset-bottom)）
 *   - 每个入口的点击区域 ≥ 44×44（iOS HIG）
 *   - 当前路由高亮
 *   - 「更多」弹出抽屉，列出次级入口（纸面信号 / 多策略纸面盘 / 数据 / 预测 / 回测 / 定价 / 设置）
 *
 * 接入说明：
 *   本组件已在 7 个目标页面（today-signal / paper-signal / paper-portfolio /
 *   signals / data / dashboard / predictions）内直接渲染，并给页面主内容
 *   加了 BOTTOM_NAV_CONTENT_PADDING 的 paddingBottom。
 *   👉 如果 AppLayout 负责人改为**全局**接入（推荐，只写一次）：
 *        - 在 AppLayout 的 <Box component="main"> 内 children 之后加 <MobileBottomNav />
 *        - 并把 main 的 paddingBottom 改成 BOTTOM_NAV_CONTENT_PADDING
 *        - 同时删除各页面里手动渲染的 <MobileBottomNav />（避免重复渲染）
 */

import React from 'react';
import {
  Box,
  ButtonBase,
  Divider,
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
} from '@mui/material';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  BarChart3,
  Signal,
  Activity,
  Bot,
  MoreHorizontal,
  Layers,
  Database,
  TrendingUp,
  CreditCard,
  Settings,
  LineChart,
  type LucideIcon,
} from 'lucide-react';
import { BOTTOM_NAV_HEIGHT, MIN_TOUCH_TARGET, SAFE_BOTTOM } from './safeArea';

interface TabItem {
  key: string;
  label: string;
  href: string;
  icon: LucideIcon;
}

const PRIMARY_TABS: TabItem[] = [
  { key: 'dashboard', label: '仪表板', href: '/', icon: BarChart3 },
  { key: 'today-signal', label: '今日信号', href: '/today-signal', icon: Signal },
  { key: 'signals', label: '策略信号', href: '/signals', icon: Activity },
  { key: 'tasks', label: '任务', href: '/tasks', icon: Bot },
];

const MORE_ITEMS: TabItem[] = [
  { key: 'paper-signal', label: '纸面信号', href: '/paper-signal', icon: Signal },
  { key: 'paper-portfolio', label: '多策略纸面盘', href: '/paper-portfolio', icon: Layers },
  { key: 'data', label: '数据管理', href: '/data', icon: Database },
  { key: 'predictions', label: '预测分析', href: '/predictions', icon: TrendingUp },
  { key: 'backtest', label: '策略回测', href: '/backtest', icon: LineChart },
  { key: 'pricing', label: '订阅与定价', href: '/pricing', icon: CreditCard },
  { key: 'settings', label: '系统设置', href: '/settings', icon: Settings },
];

const isActive = (pathname: string, href: string): boolean => {
  if (href === '/') {
    return pathname === '/';
  }
  return pathname === href || pathname.startsWith(`${href}/`);
};

export const MobileBottomNav: React.FC = () => {
  // 【P1 · mobile】用 usePathname 而非 useRouter：
  // useRouter 在没有挂载 Next App Router 的环境（单测 / 部分 SSR 边缘）
  // 会抛 "invariant expected app router to be mounted"，把整个页面带崩。
  // 导航本身用 <Link> 完成，无需 router.push。
  const pathname = usePathname() || '/';
  const [moreOpen, setMoreOpen] = React.useState(false);

  return (
    <>
      <Box
        component="nav"
        aria-label="主导航"
        sx={{
          position: 'fixed',
          left: 0,
          right: 0,
          bottom: 0,
          // 【P1 · mobile】必须显式给宽度。
          // position:fixed + display:flex + width:auto 时，flex 容器的自动宽度
          // 按 **fit-content** 解析（不是 left/right 约束的包含块宽度），
          // 实测底栏被撑到 679px → 移动端整页横向溢出 289px。
          // 100% 相对包含块（视口）= 100vw，再用 maxWidth 兜底。
          width: '100%',
          maxWidth: '100vw',
          boxSizing: 'border-box',
          zIndex: theme => theme.zIndex.drawer,
          display: { xs: 'flex', md: 'none' },
          backgroundColor: 'background.paper',
          borderTop: 1,
          borderColor: 'divider',
          boxShadow: '0 -2px 8px rgba(0,0,0,0.08)',
          paddingBottom: SAFE_BOTTOM,
          // 让 safe-area 里的内容也有最小高度，避免在无 home indicator 的
          // 机型上底栏过矮
          minHeight: `calc(${BOTTOM_NAV_HEIGHT}px + ${SAFE_BOTTOM})`,
        }}
      >
        {PRIMARY_TABS.map(item => {
          const Icon = item.icon;
          const active = isActive(pathname, item.href);
          return (
            <ButtonBase
              key={item.key}
              component={Link}
              href={item.href}
              aria-current={active ? 'page' : undefined}
              sx={{
                flex: 1,
                minWidth: 0,
                overflow: 'hidden',
                minHeight: BOTTOM_NAV_HEIGHT,
                height: BOTTOM_NAV_HEIGHT,
                flexDirection: 'column',
                gap: 0.25,
                color: active ? 'primary.main' : 'text.secondary',
                py: 0.5,
                textDecoration: 'none',
              }}
            >
              <Icon size={20} color={active ? 'primary.main' : 'inherit'} />
              <Typography
                variant="caption"
                sx={{
                  fontSize: 10.5,
                  lineHeight: 1.1,
                  fontWeight: active ? 700 : 400,
                  whiteSpace: 'nowrap',
                }}
              >
                {item.label}
              </Typography>
            </ButtonBase>
          );
        })}

        <ButtonBase
          onClick={() => setMoreOpen(true)}
          aria-label="更多入口"
          sx={{
            flex: 1,
            minWidth: 0,
            overflow: 'hidden',
            minHeight: BOTTOM_NAV_HEIGHT,
            height: BOTTOM_NAV_HEIGHT,
            flexDirection: 'column',
            gap: 0.25,
            color: 'text.secondary',
            py: 0.5,
          }}
        >
          <MoreHorizontal size={20} />
          <Typography variant="caption" sx={{ fontSize: 10.5, lineHeight: 1.1 }}>
            更多
          </Typography>
        </ButtonBase>
      </Box>

      <Drawer
        anchor="bottom"
        open={moreOpen}
        onClose={() => setMoreOpen(false)}
        slotProps={{ paper: { sx: { paddingBottom: SAFE_BOTTOM } } }}
      >
        <Box sx={{ px: 2, pt: 1.5, pb: 0.5 }}>
          <Typography variant="subtitle1" fontWeight={700}>
            更多入口
          </Typography>
        </Box>
        <Divider />
        <List sx={{ py: 0.5 }}>
          {MORE_ITEMS.map(item => {
            const Icon = item.icon;
            const active = isActive(pathname, item.href);
            return (
              <ListItemButton
                key={item.key}
                component={Link}
                href={item.href}
                selected={active}
                onClick={() => setMoreOpen(false)}
                sx={{ minHeight: MIN_TOUCH_TARGET }}
              >
                <ListItemIcon sx={{ minWidth: 40 }}>
                  <Icon size={18} color={active ? 'primary.main' : 'inherit'} />
                </ListItemIcon>
                <ListItemText primary={item.label} />
              </ListItemButton>
            );
          })}
        </List>
      </Drawer>
    </>
  );
};

export default MobileBottomNav;
