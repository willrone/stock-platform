import type { Metadata, Viewport } from 'next';
import { RootShell } from '../components/layout/RootShell';
import { ErrorBoundary } from '../components/common/ErrorBoundary';
import { MUIThemeProvider } from '../theme/muiTheme';
import './globals.css';

export const metadata: Metadata = {
  title: '股票预测平台',
  description: '基于AI的股票预测和任务管理系统',
  keywords: ['股票预测', 'AI', '机器学习', '量化交易', '投资分析'],
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
};

/**
 * 根布局保持 server component（metadata / viewport 导出依赖它）。
 * 按 pathname 决定是否套后台外壳的逻辑在 client 组件 RootShell 里
 * （见 src/components/layout/RootShell.tsx）：
 * 公开页（/login、/register 等）不渲染 AppLayout，不再出现后台菜单。
 * GlobalSnackbar 也在 RootShell 内挂载（替代未启用的 Providers.tsx）。
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <MUIThemeProvider>
          <ErrorBoundary>
            <RootShell>{children}</RootShell>
          </ErrorBoundary>
        </MUIThemeProvider>
      </body>
    </html>
  );
}
