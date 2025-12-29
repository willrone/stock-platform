'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { LoadingSpinner } from '../components/common/LoadingSpinner';

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    // 重定向到仪表板
    router.replace('/dashboard');
  }, [router]);

  return <LoadingSpinner text="正在跳转到仪表板..." />;
}