'use client';

/**
 * 应用主布局组件（导航壳重构版，2026-10-27）
 *
 * 结构：
 * - 桌面端（md+）：permanent 侧边栏（分组可折叠）+ 顶栏（面包屑 + 账号菜单）
 * - 移动端（<md）：temporary 抽屉，宽度 100%（上限 320px），整纸可滚动、头部 sticky
 * - 顶栏不再平铺导航按钮，只留：菜单按钮（移动端）、logo、面包屑、连接状态、账号菜单
 * - 选中态：前缀匹配（见 navGroups.isActive），嵌套路由只高亮最深的一条
 * - 退出登录：clearStoredAuth() 清 access_token/refresh_token/user_info + setUser(null) + replace(/login)
 * - 系统警告（ws system:alert）直接进 GlobalSnackbar，不再留无后端支撑的铃铛死控件
 * - 所有纯图标按钮带 aria-label，交互元素带 focus-visible 指示
 */

import React, { useEffect, useState } from 'react';
import {
  AppBar,
  Avatar,
  Box,
  Breadcrumbs,
  Chip,
  Divider,
  Drawer,
  IconButton,
  Link as MuiLink,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Menu,
  Toolbar,
  Typography,
} from '@mui/material';
import {
  BarChart3,
  ChevronDown,
  ChevronRight,
  LogOut,
  Menu as MenuIcon,
  Settings,
  User,
  Wallet,
  Wifi,
  WifiOff,
  X,
} from 'lucide-react';
import { useRouter, usePathname } from 'next/navigation';
import { useAppStore } from '../../stores/useAppStore';
import { useSnackbarStore } from '../../stores/useSnackbarStore';
import { wsService } from '../../services/websocket';
import {
  findActiveKey,
  findNavGroup,
  getBreadcrumbs,
  navGroups,
} from './navGroups';

interface AppLayoutProps {
  children: React.ReactNode;
}

const DESKTOP_DRAWER_WIDTH = 240;

/** 交互元素的键盘焦点指示（主题层的 focusVisible 补齐见审计报告 P1-14） */
const focusOutline = {
  '&:focus-visible': {
    outline: '2px solid primary.main',
    outlineOffset: 2,
  },
};

/**
 * 清登录态。与 services/api.ts 的 clearStoredAuth() 行为一致
 * （删 access_token / refresh_token / user_info 三个真实 key；
 * 旧代码删的是不存在的 auth_token，登出后 token 残留、会话未失效）。
 * 注：api.ts 的 clearStoredAuth 目前未 export（由其他代理维护），
 * 故在此内联同实现；待 api.ts 导出后可直接改回 import。
 */
const clearStoredAuth = (): void => {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem('access_token');
  window.localStorage.removeItem('refresh_token');
  window.localStorage.removeItem('user_info');
};

/** 折叠状态的本地配置 */
type CollapsedMap = Record<string, boolean>;

export const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const router = useRouter();
  const pathname = usePathname() ?? '/';
  const { user, setUser } = useAppStore();
  const showWarning = useSnackbarStore(s => s.showWarning);
  const showError = useSnackbarStore(s => s.showError);

  const [mobileOpen, setMobileOpen] = useState(false);
  const [collapsed, setCollapsed] = useState<CollapsedMap>({});
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);
  const [wsConnected, setWsConnected] = useState(false);
  /**
   * 是否已完成首次客户端挂载。
   *
   * user 来自 localStorage 持久化（zustand persist），SSR 阶段读不到，
   * 客户端 hydrate 时已经恢复 → 若直接用 user 渲染，SSR HTML 与客户端
   * 首次渲染不一致（管理员入口「充值订单」SSR 里没有、客户端有），
   * React 会抛 hydration mismatch 并整页重渲染。
   *
   * 所以 mount 之前一律按「未登录 / 非管理员」渲染（与 SSR 一致），
   * mount 之后才用真实的 user。
   */
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const isAdmin = mounted && Boolean(user?.is_admin);
  const activeKey = findActiveKey(pathname, isAdmin);
  const breadcrumbs = getBreadcrumbs(pathname, isAdmin);

  // 实时连接状态 + 系统警告进 snackbar（替代原来点一下就清零的死铃铛）
  useEffect(() => {
    const checkConnection = () => setWsConnected(wsService.isConnected());
    checkConnection();
    const interval = setInterval(checkConnection, 5000);

    const handleSystemAlert = (data: {
      level: 'info' | 'warning' | 'error';
      message: string;
    }) => {
      if (data.level === 'error') showError(data.message);
      else if (data.level === 'warning') showWarning(data.message);
    };

    wsService.on('system:alert', handleSystemAlert);
    return () => {
      clearInterval(interval);
      wsService.off('system:alert', handleSystemAlert);
    };
  }, [showWarning, showError]);

  const toggleGroup = (id: string) => {
    setCollapsed(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const handleNav = (key: string) => {
    router.push(key);
    setMobileOpen(false);
  };

  // 退出登录：清掉真实的 token（access_token/refresh_token/user_info），
  // 再清 store 的 user，最后 replace 到登录页（replace 避免登录页前进又回到旧页）
  const handleLogout = () => {
    clearStoredAuth();
    setUser(null);
    setUserMenuAnchor(null);
    router.replace('/login');
  };

  /** 侧边栏内容（桌面 permanent 与移动 temporary 共用） */
  const renderSidebar = (mobile: boolean) => (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* 头部（移动端 sticky，滚动菜单时保持可见） */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          p: 2,
          borderBottom: 1,
          borderColor: 'divider',
          bgcolor: 'background.paper',
          position: mobile ? 'sticky' : 'static',
          top: 0,
          zIndex: 1,
          flexShrink: 0,
        }}
      >
        <BarChart3 size={24} color="#1976d2" aria-hidden />
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          股票预测平台
        </Typography>
        {mobile && (
          <IconButton
            onClick={() => setMobileOpen(false)}
            aria-label="关闭导航菜单"
            sx={{ ...focusOutline, ml: 'auto' }}
          >
            <X size={20} aria-hidden />
          </IconButton>
        )}
      </Box>

      {/* 菜单区：flex:1 自滚动，底部条目（如「系统设置」）不会被裁切 */}
      <Box
        component="nav"
        aria-label="主导航"
        sx={{ flex: 1, overflowY: 'auto', px: 1.5, py: 1 }}
      >
        <List sx={{ p: 0 }}>
          {navGroups.map(group => {
            const items = group.items.filter(item => !item.adminOnly || isAdmin);
            if (items.length === 0) return null;
            const isCollapsed = Boolean(collapsed[group.id]);
            const GroupIcon = isCollapsed ? ChevronRight : ChevronDown;
            return (
              <React.Fragment key={group.id}>
                <ListItemButton
                  onClick={() => toggleGroup(group.id)}
                  aria-expanded={!isCollapsed}
                  sx={{
                    ...focusOutline,
                    borderRadius: 1,
                    py: 0.5,
                    color: 'text.secondary',
                    minHeight: 36,
                  }}
                >
                  <GroupIcon size={16} aria-hidden />
                  <Typography
                    variant="overline"
                    component="span"
                    sx={{ ml: 0.5, fontWeight: 600 }}
                  >
                    {group.label}
                  </Typography>
                </ListItemButton>
                {!isCollapsed &&
                  items.map(item => {
                    const Icon = item.icon;
                    const isSelected = activeKey === item.key;
                    return (
                      <ListItem key={item.key} disablePadding>
                        <ListItemButton
                          selected={isSelected}
                          aria-current={isSelected ? 'page' : undefined}
                          onClick={() => handleNav(item.key)}
                          sx={{
                            ...focusOutline,
                            borderRadius: 1,
                            ml: 0.5,
                            pl: 2.5,
                            '&.Mui-selected': {
                              bgcolor: 'primary.main',
                              color: 'primary.contrastText',
                              '&:hover': { bgcolor: 'primary.dark' },
                            },
                          }}
                        >
                          <ListItemIcon sx={{ minWidth: 36 }}>
                            <Icon size={18} color={isSelected ? 'white' : 'inherit'} aria-hidden />
                          </ListItemIcon>
                          <ListItemText primary={item.label} />
                        </ListItemButton>
                      </ListItem>
                    );
                  })}
              </React.Fragment>
            );
          })}
        </List>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* 桌面端永久侧边栏 */}
      <Drawer
        variant="permanent"
        anchor="left"
        aria-label="主导航"
        sx={{
          display: { xs: 'none', md: 'block' },
          '& .MuiDrawer-paper': {
            width: DESKTOP_DRAWER_WIDTH,
            borderRight: 1,
            borderRightColor: 'divider',
          },
        }}
      >
        {renderSidebar(false)}
      </Drawer>

      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* 顶栏：菜单按钮 + logo + 面包屑 + 连接状态 + 账号菜单 */}
        <AppBar
          position="sticky"
          sx={{
            bgcolor: 'background.paper',
            color: 'text.primary',
            boxShadow: 1,
            // 【P0-18】吸顶栏整体下移 safe-top，避免被 iOS 状态栏 / 刘海屏压住
            paddingTop: 'var(--safe-top, 0px)',
          }}
        >
          <Toolbar>
            {/* 移动端菜单按钮 */}
            <IconButton
              edge="start"
              color="inherit"
              aria-label="打开导航菜单"
              onClick={() => setMobileOpen(true)}
              sx={{ ...focusOutline, display: { md: 'none' }, mr: 1 }}
            >
              <MenuIcon size={20} aria-hidden />
            </IconButton>

            {/* Logo */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexShrink: 0 }}>
              <BarChart3 size={22} color="#1976d2" aria-hidden />
              <Typography
                variant="subtitle1"
                component="span"
                sx={{ fontWeight: 600, display: { xs: 'none', sm: 'block' } }}
              >
                股票预测平台
              </Typography>
            </Box>

            {/* 面包屑：父级条目 + 当前页标题 */}
            <Breadcrumbs
              aria-label="面包屑"
              sx={{
                flexGrow: 1,
                minWidth: 0,
                pl: 1,
                '& .MuiBreadcrumbs-ol': { minWidth: 0 },
                '& .MuiBreadcrumbs-ol li': { minWidth: 0 },
                '& .MuiBreadcrumbs-ol li *': {
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  maxWidth: 240,
                },
              }}
            >
              {breadcrumbs.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  {pathname}
                </Typography>
              ) : (
                breadcrumbs.map((crumb, index) => {
                  const isLast = index === breadcrumbs.length - 1;
                  return isLast ? (
                    <Typography
                      key={crumb.key}
                      variant="body2"
                      color="text.primary"
                      sx={{ fontWeight: 600 }}
                    >
                      {crumb.label}
                    </Typography>
                  ) : (
                    <MuiLink
                      key={crumb.key}
                      component="button"
                      type="button"
                      onClick={() => handleNav(crumb.key)}
                      sx={{ ...focusOutline, color: 'text.secondary', display: 'inline-flex' }}
                    >
                      {crumb.label}
                    </MuiLink>
                  );
                })
              )}
            </Breadcrumbs>

            {/* 右侧：连接状态 + 账号菜单 */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexShrink: 0 }}>
              <Chip
                icon={wsConnected ? <Wifi size={16} aria-hidden /> : <WifiOff size={16} aria-hidden />}
                label={wsConnected ? '已连接' : '未连接'}
                color={wsConnected ? 'success' : 'error'}
                size="small"
                sx={{ display: { xs: 'none', sm: 'flex' } }}
              />

              {/* 账号菜单（原「个人资料/账户设置」已指向真实存在的 /settings，
                  「钱包」「充值订单」入口并入此处，见 P0-4 / P1-7） */}
              <IconButton
                onClick={e => setUserMenuAnchor(e.currentTarget)}
                aria-label="账户菜单"
                aria-haspopup="menu"
                sx={{ ...focusOutline, p: 0.5 }}
              >
                <Avatar
                  src={mounted ? user?.avatar || undefined : undefined}
                  sx={{ width: 32, height: 32 }}
                  aria-label={mounted && user?.name ? `账户：${user.name}` : '账户'}
                >
                  <User size={16} aria-hidden />
                </Avatar>
              </IconButton>

              <Menu
                anchorEl={userMenuAnchor}
                open={Boolean(userMenuAnchor)}
                onClose={() => setUserMenuAnchor(null)}
                aria-label="账户菜单"
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                transformOrigin={{ vertical: 'top', horizontal: 'right' }}
              >
                {/* 账户信息 */}
                <Box sx={{ px: 2, py: 1.5, display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {user?.name || '未登录'}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {user?.email}
                    {user?.subscription_tier ? ` · ${user.subscription_tier}` : ''}
                  </Typography>
                </Box>
                <Divider />
                <ListItemButton onClick={() => handleNav('/settings')} sx={{ mx: 0.5, borderRadius: 1, ...focusOutline }}>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <Settings size={18} aria-hidden />
                  </ListItemIcon>
                  <ListItemText primary="系统设置" />
                </ListItemButton>
                <ListItemButton onClick={() => handleNav('/wallet')} sx={{ mx: 0.5, borderRadius: 1, ...focusOutline }}>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <Wallet size={18} aria-hidden />
                  </ListItemIcon>
                  <ListItemText primary="钱包" />
                </ListItemButton>
                {isAdmin && (
                  <ListItemButton onClick={() => handleNav('/admin/payments')} sx={{ mx: 0.5, borderRadius: 1, ...focusOutline }}>
                    <ListItemIcon sx={{ minWidth: 36 }}>
                      <Wallet size={18} aria-hidden />
                    </ListItemIcon>
                    <ListItemText primary="充值订单" />
                  </ListItemButton>
                )}
                <Divider />
                <ListItemButton
                  onClick={handleLogout}
                  sx={{ mx: 0.5, my: 0.5, borderRadius: 1, color: 'error.main', ...focusOutline }}
                >
                  <ListItemIcon sx={{ minWidth: 36, color: 'error.main' }}>
                    <LogOut size={18} aria-hidden />
                  </ListItemIcon>
                  <ListItemText primary="退出登录" />
                </ListItemButton>
              </Menu>
            </Box>
          </Toolbar>
        </AppBar>

        {/* 主内容区域 */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: { xs: 1.5, sm: 2, md: 3 },
            maxWidth: '1400px',
            mx: 'auto',
            width: '100%',
          }}
        >
          {children}
        </Box>

        {/* 底部信息栏 */}
        <Box
          component="footer"
          sx={{
            borderTop: 1,
            borderColor: 'divider',
            bgcolor: 'background.paper',
            py: 2,
            mt: 'auto',
          }}
        >
          <Box sx={{ maxWidth: '1400px', mx: 'auto', px: 2, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              股票预测平台 ©2025 - 基于AI的智能投资决策系统
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* 移动端临时抽屉：宽度 100%（上限 320px），断点 down('lg')，
          整纸可滚动（overflowY auto），头部 sticky、菜单区 flex:1 自滚 */}
      <Drawer
        variant="temporary"
        anchor="left"
        open={mobileOpen}
        onClose={() => setMobileOpen(false)}
        aria-label="主导航"
        sx={{ display: { md: 'none' } }}
        PaperProps={{
          'aria-label': '主导航',
          sx: { width: '100%', maxWidth: 320, overflowY: 'auto' },
        }}
      >
        {renderSidebar(true)}
      </Drawer>
    </Box>
  );
};
