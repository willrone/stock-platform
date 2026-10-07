'use client';

import { useEffect } from 'react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('页面错误:', error);
  }, [error]);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '60vh',
      padding: 32,
      textAlign: 'center',
    }}>
      <h1 style={{ fontSize: 24, fontWeight: 600, marginBottom: 8, color: '#d32f2f' }}>
        出错了
      </h1>
      <p style={{ fontSize: 14, color: '#666', marginBottom: 24, maxWidth: 400 }}>
        {error.message || '页面加载时遇到错误'}
      </p>
      <button
        onClick={reset}
        style={{
          padding: '8px 24px',
          fontSize: 14,
          border: 'none',
          borderRadius: 8,
          backgroundColor: '#1976d2',
          color: '#fff',
          cursor: 'pointer',
        }}
      >
        重试
      </button>
    </div>
  );
}
