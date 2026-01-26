/**
 * 应用主布局组件
 *
 * 提供应用的整体布局结构，包括：
 * - 顶部导航栏
 * - 侧边菜单
 * - 主内容区域
 * - 底部信息栏
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  Button,
  Avatar,
  Menu,
  MenuItem,
  Badge,
  Chip,
  Drawer,
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  BarChart3,
  TrendingUp,
  Bot,
  Database,
  Settings,
  User,
  Bell,
  Menu as MenuIcon,
  X,
  LogOut,
  Wifi,
  WifiOff,
  Home,
  Activity,
  Brain,
  Sparkles,
  Signal,
} from 'lucide-react';
import { useRouter, usePathname } from 'next/navigation';
import { useAppStore } from '../../stores/useAppStore';
import { wsService } from '../../services/websocket';

interface AppLayoutProps {
  children: React.ReactNode;
}

// 菜单项配置
const menuItems = [
  {
    key: '/',
    icon: Home,
    label: '仪表板',
  },
  {
    key: '/tasks',
    icon: Bot,
    label: '任务管理',
  },
  {
    key: '/models',
    icon: Brain,
    label: '模型管理',
  },
  {
    key: '/data',
    icon: Database,
    label: '数据管理',
  },
  {
    key: '/monitoring',
    icon: BarChart3,
    label: '系统监控',
  },
  {
    key: '/predictions',
    icon: TrendingUp,
    label: '预测分析',
  },
  {
    key: '/backtest',
    icon: Activity,
    label: '策略回测',
  },
  {
    key: '/signals',
    icon: Signal,
    label: '策略信号',
  },
  {
    key: '/optimization',
    icon: Sparkles,
    label: '超参优化',
  },
  {
    key: '/settings',
    icon: Settings,
    label: '系统设置',
  },
];

export const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const router = useRouter();
  const pathname = usePathname();
  const { user } = useAppStore();

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [notifications, setNotifications] = useState(0);
  const [wsConnected, setWsConnected] = useState(false);
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);

  // 监听WebSocket连接状态
  useEffect(() => {
    const checkConnection = () => {
      setWsConnected(wsService.isConnected());
    };

    // 初始检查
    checkConnection();

    // 定期检查连接状态
    const interval = setInterval(checkConnection, 5000);

    // 监听系统警告
    const handleSystemAlert = (data: any) => {
      if (data.level === 'warning' || data.level === 'error') {
        setNotifications(prev => prev + 1);
      }
    };

    wsService.on('system:alert', handleSystemAlert);

    return () => {
      clearInterval(interval);
      wsService.off('system:alert', handleSystemAlert);
    };
  }, []);

  // 处理菜单点击
  const handleMenuClick = (key: string) => {
    router.push(key);
    setSidebarOpen(false);
  };

  // 处理用户菜单点击
  const handleUserAction = (key: string) => {
    setUserMenuAnchor(null);
    switch (key) {
      case 'profile':
        router.push('/profile');
        break;
      case 'settings':
        router.push('/account-settings');
        break;
      case 'logout':
        // 处理退出登录
        localStorage.removeItem('auth_token');
        router.push('/login');
        break;
    }
  };

  // 清除通知
  const clearNotifications = () => {
    setNotifications(0);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* 顶部导航栏 */}
      <AppBar
        position="sticky"
        sx={{ bgcolor: 'background.paper', color: 'text.primary', boxShadow: 1 }}
      >
        <Toolbar>
          {/* 移动端菜单按钮 */}
          <IconButton
            edge="start"
            color="inherit"
            aria-label="menu"
            onClick={() => setSidebarOpen(true)}
            sx={{ display: { xs: 'block', lg: 'none' }, mr: 2 }}
          >
            <MenuIcon size={20} />
          </IconButton>

          {/* Logo */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mr: 2 }}>
            <BarChart3 size={24} color="#1976d2" />
            <Typography
              variant="h6"
              component="div"
              sx={{ fontWeight: 600, display: { xs: 'none', sm: 'block' } }}
            >
              股票预测平台
            </Typography>
          </Box>

          {/* 桌面端菜单 */}
          <Box
            sx={{
              flexGrow: 1,
              display: { xs: 'none', lg: 'flex' },
              gap: 1,
              justifyContent: 'center',
            }}
          >
            {menuItems.map(item => {
              const Icon = item.icon;
              const isActive = pathname === item.key;
              return (
                <Button
                  key={item.key}
                  variant={isActive ? 'contained' : 'text'}
                  color={isActive ? 'primary' : 'inherit'}
                  startIcon={<Icon size={16} />}
                  onClick={() => handleMenuClick(item.key)}
                  sx={{ fontWeight: 500 }}
                >
                  {item.label}
                </Button>
              );
            })}
          </Box>

          {/* 右侧操作区 */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {/* 连接状态指示器 */}
            <Chip
              icon={wsConnected ? <Wifi size={16} /> : <WifiOff size={16} />}
              label={wsConnected ? '已连接' : '未连接'}
              color={wsConnected ? 'success' : 'error'}
              size="small"
              sx={{ display: { xs: 'none', sm: 'flex' } }}
            />

            {/* 通知铃铛 */}
            <IconButton color="inherit" onClick={clearNotifications}>
              <Badge badgeContent={notifications > 0 ? notifications : undefined} color="error">
                <Bell size={20} />
              </Badge>
            </IconButton>

            {/* 用户菜单 */}
            <IconButton onClick={e => setUserMenuAnchor(e.currentTarget)} sx={{ p: 0 }}>
              <Avatar src={user?.avatar} sx={{ width: 32, height: 32 }}>
                <User size={16} />
              </Avatar>
            </IconButton>
            <Menu
              anchorEl={userMenuAnchor}
              open={Boolean(userMenuAnchor)}
              onClose={() => setUserMenuAnchor(null)}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'right',
              }}
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
            >
              <MenuItem onClick={() => handleUserAction('profile')}>
                <ListItemIcon>
                  <User size={16} />
                </ListItemIcon>
                <ListItemText>个人资料</ListItemText>
              </MenuItem>
              <MenuItem onClick={() => handleUserAction('settings')}>
                <ListItemIcon>
                  <Settings size={16} />
                </ListItemIcon>
                <ListItemText>账户设置</ListItemText>
              </MenuItem>
              <Divider />
              <MenuItem onClick={() => handleUserAction('logout')} sx={{ color: 'error.main' }}>
                <ListItemIcon>
                  <LogOut size={16} color="inherit" />
                </ListItemIcon>
                <ListItemText>退出登录</ListItemText>
              </MenuItem>
            </Menu>
          </Box>
        </Toolbar>
      </AppBar>

      {/* 移动端侧边栏 */}
      <Drawer
        anchor="left"
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        sx={{ display: { lg: 'none' } }}
      >
        <Box sx={{ width: 256, p: 2 }}>
          <Box
            sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <BarChart3 size={24} color="#1976d2" />
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                股票预测平台
              </Typography>
            </Box>
            <IconButton onClick={() => setSidebarOpen(false)}>
              <X size={20} />
            </IconButton>
          </Box>

          <List>
            {menuItems.map(item => {
              const Icon = item.icon;
              const isActive = pathname === item.key;
              return (
                <ListItem key={item.key} disablePadding>
                  <ListItemButton
                    selected={isActive}
                    onClick={() => handleMenuClick(item.key)}
                    sx={{
                      borderRadius: 1,
                      '&.Mui-selected': {
                        bgcolor: 'primary.main',
                        color: 'primary.contrastText',
                        '&:hover': {
                          bgcolor: 'primary.dark',
                        },
                      },
                    }}
                  >
                    <ListItemIcon sx={{ minWidth: 40 }}>
                      <Icon size={18} color={isActive ? 'white' : 'inherit'} />
                    </ListItemIcon>
                    <ListItemText primary={item.label} />
                  </ListItemButton>
                </ListItem>
              );
            })}
          </List>
        </Box>
      </Drawer>

      {/* 主内容区域 */}
      <Box
        component="main"
        sx={{ flexGrow: 1, p: 3, maxWidth: '1400px', mx: 'auto', width: '100%' }}
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
  );
};
