import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { AppLayout } from '../components/layout/AppLayout';
import { ErrorBoundary } from '../components/common/ErrorBoundary';
import { MUIThemeProvider } from '../theme/muiTheme';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: '股票预测平台',
  description: '基于AI的股票预测和任务管理系统',
  keywords: ['股票预测', 'AI', '机器学习', '量化交易', '投资分析'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body className={inter.className}>
        <MUIThemeProvider>
          <ErrorBoundary>
            <AppLayout>{children}</AppLayout>
          </ErrorBoundary>
        </MUIThemeProvider>
      </body>
    </html>
  );}