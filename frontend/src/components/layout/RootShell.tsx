'use client';

/**
 * RootShell —— 根布局外壳（由 src/app/layout.tsx 的 server component 引用）
 *
 * 职责：按 pathname 决定是否套后台外壳（AppLayout）：
 * - 公开页（/、/login、/register、/reset-password、/pricing）不套 AppLayout，
 *   未登录时不会看到「仪表板…系统设置」这一整排后台菜单（P0-6）
 * - 其余页面套 AppLayout（侧边栏 + 顶栏）
 * - GlobalSnackbar 在这里挂载一次（无论公开页还是后台页都能弹通知），
 *   替代原来从未被引用的 Providers.tsx（见 P1-12 的方案说明）
 *
 * 用 usePathname 需要 client component，所以逻辑抽到这里，
 * layout.tsx 本身保持 server component（metadata 导出不受影响）。
 */

import React from 'react';
import { usePathname } from 'next/navigation';
import { Box } from '@mui/material';
import { AppLayout } from './AppLayout';
import { GlobalSnackbar } from '../common/GlobalSnackbar';
import { isPublicPath } from './navGroups';

export const RootShell: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const pathname = usePathname() ?? '/';

  if (isPublicPath(pathname)) {
    return (
      <>
        <Box component="main" sx={{ minHeight: '100vh' }}>
          {children}
        </Box>
        <GlobalSnackbar />
      </>
    );
  }

  return (
    <>
      <AppLayout>{children}</AppLayout>
      <GlobalSnackbar />
    </>
  );
};
