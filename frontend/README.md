# 股票预测平台 - 前端应用

基于React + TypeScript + Ant Design Pro的现代化前端应用。

## 技术栈

- **框架**: Next.js 14 (App Router)
- **语言**: TypeScript
- **UI库**: Ant Design 5.x
- **状态管理**: Zustand
- **HTTP客户端**: Axios
- **实时通信**: Socket.IO Client
- **图表库**: ECharts
- **样式**: CSS Modules + Tailwind CSS
- **测试**: Jest + Testing Library

## 项目结构

```
src/
├── app/                    # Next.js App Router页面
│   ├── dashboard/         # 仪表板页面
│   ├── layout.tsx         # 根布局
│   ├── page.tsx          # 首页
│   └── globals.css       # 全局样式
├── components/            # 可复用组件
│   ├── layout/           # 布局组件
│   └── common/           # 通用组件
├── services/             # API服务层
│   ├── api.ts           # HTTP客户端配置
│   ├── taskService.ts   # 任务管理服务
│   ├── dataService.ts   # 数据服务
│   └── websocket.ts     # WebSocket服务
├── stores/              # Zustand状态管理
│   ├── useAppStore.ts   # 应用全局状态
│   ├── useTaskStore.ts  # 任务管理状态
│   └── useDataStore.ts  # 数据管理状态
└── __tests__/           # 测试文件
```

## 已实现功能

### 基础架构 ✅
- [x] Next.js 14项目搭建
- [x] TypeScript配置
- [x] Ant Design Pro集成
- [x] 响应式布局系统
- [x] 路由配置

### 状态管理 ✅
- [x] Zustand状态管理配置
- [x] 应用全局状态（用户、配置、加载状态）
- [x] 任务管理状态（任务列表、分页、过滤）
- [x] 数据管理状态（股票数据缓存、模型信息、系统状态）

### 服务层 ✅
- [x] HTTP客户端封装（Axios）
- [x] 请求/响应拦截器
- [x] 错误处理和重试机制
- [x] 任务管理API服务
- [x] 数据获取API服务
- [x] WebSocket实时通信服务

### UI组件 ✅
- [x] 应用主布局（侧边栏 + 顶部导航）
- [x] 加载动画组件
- [x] 错误边界组件
- [x] 仪表板页面

### 开发工具 ✅
- [x] TypeScript类型检查
- [x] ESLint代码规范
- [x] Jest单元测试
- [x] 构建配置优化

## 快速开始

### 前置要求
- Node.js >= 18.0.0
- npm >= 8.0.0

### 安装步骤

1. **安装依赖**
   ```bash
   npm install
   ```

2. **配置环境变量（可选）**
   
   项目已包含默认配置，通常无需修改。如需自定义配置：
   ```bash
   cp .env.example .env.local
   # 根据需要修改 .env.local 中的配置
   ```
   
   环境变量说明：
   ```env
   # API服务地址（默认: http://localhost:8000/api/v1）
   NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
   
   # WebSocket服务地址（默认: ws://localhost:8000/ws）
   NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
   ```

3. **启动开发服务器**
   ```bash
   npm run dev
   ```
   
   应用将在 http://localhost:3000 启动

### 其他命令

```bash
# 类型检查
npm run type-check

# 运行测试
npm test

# 构建生产版本
npm run build

# 启动生产服务器
npm start
```

### 常见问题

**Q: 启动失败，提示模块未找到？**
- 确保已运行 `npm install` 安装所有依赖
- 检查 Node.js 版本是否符合要求（>= 18.0.0）
- 尝试删除 `node_modules` 和 `package-lock.json` 后重新安装

**Q: 端口已被占用？**
- 默认端口是 3000，可以通过环境变量 `PORT` 修改
- 或使用 `npm run dev -- -p 3001` 指定其他端口

**Q: API 请求失败？**
- 确保后端服务已启动（默认端口 8000）
- 检查 `.env.local` 中的 `NEXT_PUBLIC_API_URL` 配置是否正确

## 下一步计划

### 任务管理界面 (11.2)
- [ ] 任务创建表单
- [ ] 任务列表和状态显示
- [ ] 实时进度更新

### 数据展示界面 (11.4)
- [ ] TradingView图表集成
- [ ] ECharts技术指标展示
- [ ] 预测结果可视化

### 数据管理页面 (11.5)
- [ ] 数据服务状态监控
- [ ] 本地数据文件管理
- [ ] 数据同步控制

## 测试覆盖

当前测试覆盖了：
- ✅ 项目结构验证
- ✅ 依赖包导入测试
- ✅ 配置文件验证
- ✅ TypeScript类型检查

## 注意事项

1. **WebSocket连接**: 构建时的WebSocket连接错误是正常的，因为后端服务未启动
2. **API代理**: 开发环境下API请求会代理到后端服务
3. **状态持久化**: 使用Zustand的devtools中间件进行状态调试
4. **错误处理**: 全局错误边界捕获组件错误，API层处理网络错误

## 性能优化

- 使用Next.js的静态生成和服务端渲染
- 组件懒加载和代码分割
- 图片优化和资源压缩
- 状态管理优化，避免不必要的重渲染