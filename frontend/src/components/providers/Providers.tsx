'use client';

/**
 * 客户端 Providers 边界组件
 *
 * 集中管理所有需要客户端运行时的 context provider，包括：
 * - MUI 主题
 * - 全局错误边界
 * - 全局 Snackbar 通知
 * - 应用布局
 *
 * 作为根 layout 的纯客户端入口，让 layout.tsx 保持 server component 身份。
 */

import { AppLayout } from '../layout/AppLayout';
import { ErrorBoundary } from '../common/ErrorBoundary';
import { GlobalSnackbar } from '../common/GlobalSnackbar';
import { MUIThemeProvider } from '../../theme/muiTheme';

export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <MUIThemeProvider>
      <ErrorBoundary>
        <AppLayout>
          {children}
        </AppLayout>
        <GlobalSnackbar />
      </ErrorBoundary>
    </MUIThemeProvider>
  );
}
