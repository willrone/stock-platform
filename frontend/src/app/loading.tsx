export default function Loading() {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '60vh',
        padding: 32,
      }}
    >
      <div
        style={{
          width: 48,
          height: 48,
          border: '4px solid #e0e0e0',
          borderTopColor: '#1976d2',
          borderRadius: '50%',
          animation: 'loading-spin 0.8s linear infinite',
        }}
      />
      <style>{`@keyframes loading-spin { to { transform: rotate(360deg); } }`}</style>
      <p style={{ marginTop: 16, fontSize: 14, color: '#666' }}>加载中...</p>
    </div>
  );
}
