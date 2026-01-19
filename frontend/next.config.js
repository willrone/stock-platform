/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone',
  async rewrites() {
    // 从环境变量获取后端服务器地址，如果没有则使用默认值
    // 支持通过环境变量 NEXT_PUBLIC_BACKEND_HOST 配置后端地址
    // 如果未设置，使用 localhost（本地开发）或从 NEXT_PUBLIC_API_URL 提取
    const backendHost = process.env.NEXT_PUBLIC_BACKEND_HOST || 
                       process.env.NEXT_PUBLIC_API_URL?.replace(/^https?:\/\//, '').replace(/\/api\/v1.*$/, '') ||
                       'localhost:8000';
    
    // 确保有协议前缀
    const backendUrl = backendHost.startsWith('http') 
      ? backendHost 
      : `http://${backendHost}`;
    
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
  webpack: (config) => {
    // 处理 TradingView 图表库
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
    };
    return config;
  },
};
module.exports = nextConfig;