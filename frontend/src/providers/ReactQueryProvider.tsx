/**
 * React Query 配置
 * 提供请求缓存、去重和 stale-while-revalidate 策略
 */

'use client';

import React from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';

const defaultQueryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000, // 1 分钟内数据视为新鲜
      cacheTime: 5 * 60 * 1000, // 5 分钟缓存
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export function ReactQueryProvider({ children }: { children: React.ReactNode }) {
  const [queryClient] = React.useState(() => defaultQueryClient);

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}
