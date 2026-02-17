import type { Metadata } from 'next';
import { AppLayout } from '../components/layout/AppLayout';
import { ErrorBoundary } from '../components/common/ErrorBoundary';
import { MUIThemeProvider } from '../theme/muiTheme';
import './globals.css';

export const metadata: Metadata = {
  title: '股票预测平台',
  description: '基于AI的股票预测和任务管理系统',
  keywords: ['股票预测', 'AI', '机器学习', '量化交易', '投资分析'],
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 5,
    userScalable: true,
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body
        style={{
          fontFamily:
            '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
        }}
      >
        <MUIThemeProvider>
          <ErrorBoundary>
            <AppLayout>{children}</AppLayout>
          </ErrorBoundary>
        </MUIThemeProvider>
      </body>
    </html>
  );
}
