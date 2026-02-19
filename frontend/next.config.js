/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone',
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
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
      // 兼容前端直接调用后端的 /api/v1/... 路径（当前任务创建页会用到）
      {
        source: '/api/v1/:path*',
        destination: `${backendUrl}/api/v1/:path*`,
      },
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

// 使用 ANALYZE=true npm run build 分析 bundle 体积（需先安装 @next/bundle-analyzer）
module.exports =
  process.env.ANALYZE === 'true'
    ? require('@next/bundle-analyzer')({ enabled: true })(nextConfig)
    : nextConfig;