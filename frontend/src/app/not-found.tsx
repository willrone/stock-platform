import Link from 'next/link';

export default function NotFound() {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '60vh',
        padding: 32,
        textAlign: 'center',
      }}
    >
      <h1 style={{ fontSize: 48, fontWeight: 700, marginBottom: 8, color: '#d32f2f' }}>404</h1>
      <h2 style={{ fontSize: 20, fontWeight: 500, marginBottom: 16 }}>页面不存在</h2>
      <p style={{ fontSize: 14, color: '#666', marginBottom: 24 }}>您访问的页面不存在或已被移除</p>
      <Link
        href="/dashboard"
        style={{
          padding: '8px 24px',
          fontSize: 14,
          borderRadius: 8,
          backgroundColor: '#1976d2',
          color: '#fff',
          textDecoration: 'none',
        }}
      >
        返回仪表板
      </Link>
    </div>
  );
}
