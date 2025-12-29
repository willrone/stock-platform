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
import { Layout, Menu, Avatar, Dropdown, Badge, Button, Space, Typography } from 'antd';
import {
  DashboardOutlined,
  StockOutlined,
  RobotOutlined,
  BarChartOutlined,
  SettingOutlined,
  UserOutlined,
  BellOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  LogoutOutlined,
  DatabaseOutlined,
} from '@ant-design/icons';
import { useRouter, usePathname } from 'next/navigation';
import { useAppStore } from '../../stores/useAppStore';
import { wsService } from '../../services/websocket';

const { Header, Sider, Content, Footer } = Layout;
const { Text } = Typography;

interface AppLayoutProps {
  children: React.ReactNode;
}

// 菜单项配置
const menuItems = [
  {
    key: '/',
    icon: <DashboardOutlined />,
    label: '仪表板',
  },
  {
    key: '/tasks',
    icon: <RobotOutlined />,
    label: '任务管理',
  },
  {
    key: '/stocks',
    icon: <StockOutlined />,
    label: '股票数据',
  },
  {
    key: '/predictions',
    icon: <BarChartOutlined />,
    label: '预测分析',
  },
  {
    key: '/backtest',
    icon: <BarChartOutlined />,
    label: '策略回测',
  },
  {
    key: '/data-management',
    icon: <DatabaseOutlined />,
    label: '数据管理',
  },
  {
    key: '/settings',
    icon: <SettingOutlined />,
    label: '系统设置',
  },
];

export const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const router = useRouter();
  const pathname = usePathname();
  const { user, config } = useAppStore();
  
  const [collapsed, setCollapsed] = useState(false);
  const [notifications, setNotifications] = useState(0);
  const [wsConnected, setWsConnected] = useState(false);

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
  const handleMenuClick = ({ key }: { key: string }) => {
    router.push(key);
  };

  // 用户菜单
  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: '个人资料',
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: '账户设置',
    },
    {
      type: 'divider' as const,
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
      danger: true,
    },
  ];

  // 处理用户菜单点击
  const handleUserMenuClick = ({ key }: { key: string }) => {
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
    <Layout style={{ minHeight: '100vh' }}>
      {/* 侧边菜单 */}
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        width={240}
        style={{
          background: '#fff',
          boxShadow: '2px 0 8px 0 rgba(29,35,41,.05)',
        }}
      >
        {/* Logo区域 */}
        <div
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: collapsed ? 'center' : 'flex-start',
            padding: collapsed ? 0 : '0 24px',
            borderBottom: '1px solid #f0f0f0',
          }}
        >
          <StockOutlined style={{ fontSize: 24, color: '#1890ff' }} />
          {!collapsed && (
            <Text strong style={{ marginLeft: 12, fontSize: 16 }}>
              股票预测平台
            </Text>
          )}
        </div>

        {/* 菜单 */}
        <Menu
          mode="inline"
          selectedKeys={[pathname]}
          items={menuItems}
          onClick={handleMenuClick}
          style={{ border: 'none', marginTop: 16 }}
        />
      </Sider>

      <Layout>
        {/* 顶部导航栏 */}
        <Header
          style={{
            background: '#fff',
            padding: '0 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            boxShadow: '0 2px 8px 0 rgba(29,35,41,.05)',
            zIndex: 1,
          }}
        >
          {/* 左侧控制按钮 */}
          <Space>
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              style={{ fontSize: 16 }}
            />
            
            {/* 连接状态指示器 */}
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor: wsConnected ? '#52c41a' : '#ff4d4f',
                  marginRight: 8,
                }}
              />
              <Text type="secondary" style={{ fontSize: 12 }}>
                {wsConnected ? '实时连接正常' : '连接已断开'}
              </Text>
            </div>
          </Space>

          {/* 右侧用户信息 */}
          <Space size="middle">
            {/* 通知铃铛 */}
            <Badge count={notifications} size="small">
              <Button
                type="text"
                icon={<BellOutlined />}
                onClick={clearNotifications}
                style={{ fontSize: 16 }}
              />
            </Badge>

            {/* 用户头像和菜单 */}
            <Dropdown
              menu={{
                items: userMenuItems,
                onClick: handleUserMenuClick,
              }}
              placement="bottomRight"
            >
              <Space style={{ cursor: 'pointer' }}>
                <Avatar
                  size="small"
                  icon={<UserOutlined />}
                  src={user?.avatar}
                />
                <Text>{user?.name || '用户'}</Text>
              </Space>
            </Dropdown>
          </Space>
        </Header>

        {/* 主内容区域 */}
        <Content
          style={{
            margin: '24px',
            padding: '24px',
            background: '#fff',
            borderRadius: 8,
            minHeight: 'calc(100vh - 112px)',
          }}
        >
          {children}
        </Content>

        {/* 底部信息栏 */}
        <Footer
          style={{
            textAlign: 'center',
            background: '#f0f2f5',
            padding: '12px 24px',
          }}
        >
          <Text type="secondary" style={{ fontSize: 12 }}>
            股票预测平台 ©2025 - 基于AI的智能投资决策系统
          </Text>
        </Footer>
      </Layout>
    </Layout>
  );
};