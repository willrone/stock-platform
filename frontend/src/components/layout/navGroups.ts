/**
 * 导航配置：侧边栏分组 + 面包屑 + 公开路径 + 选中态判定
 *
 * 所有页面共享这份静态配置（不要改各页面文件，页面标题映射在这里维护）：
 * - navGroups：侧边栏按语义分组（概览 / 研究 / 信号 / 运营 / 账户）
 * - isActive：前缀匹配判定选中态（`/` 不能用作键，否则一切路径都高亮）
 * - getBreadcrumbs：当前页标题 + 父级条目（子页自动找到所属父条目）
 * - PUBLIC_PATHS：不套后台外壳（AppLayout）的公开页
 */

import {
  Activity,
  BarChart3,
  Bot,
  Brain,
  CalendarClock,
  CreditCard,
  Database,
  GitCompare,
  Home,
  Inbox,
  Layers,
  Plus,
  Settings,
  ShieldCheck,
  Signal,
  SlidersHorizontal,
  Sparkles,
  Star,
  TrendingUp,
  Wallet,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

export interface NavItem {
  /** 路由路径，用于前缀匹配选中态（注意：不要用 `/`，否则全部高亮） */
  key: string;
  label: string;
  icon: LucideIcon;
  /** 仅管理员可见 */
  adminOnly?: boolean;
}

export interface NavGroup {
  id: string;
  label: string;
  items: NavItem[];
}

/**
 * 侧边栏分组方案：
 * - 概览：仪表板
 * - 研究：预测分析 / 策略回测 / 超参优化 / 策略模板 / 模型管理 / 数据管理
 * - 信号：策略信号 / 今日信号 / 纸面信号 / 多策略纸面盘
 * - 运营：任务管理 / 创建任务 / 任务对比 / 参数调优 / 系统监控
 * - 账户：系统设置 / 订阅管理 / 钱包 / 充值订单（管理员）
 */
export const navGroups: NavGroup[] = [
  {
    id: 'overview',
    label: '概览',
    items: [{ key: '/dashboard', label: '仪表板', icon: Home }],
  },
  {
    id: 'research',
    label: '研究',
    items: [
      { key: '/predictions', label: '预测分析', icon: TrendingUp },
      { key: '/backtest', label: '策略回测', icon: Activity },
      { key: '/optimization', label: '超参优化', icon: Sparkles },
      { key: '/templates', label: '策略模板', icon: Layers },
      { key: '/models', label: '模型管理', icon: Brain },
      { key: '/data', label: '数据管理', icon: Database },
    ],
  },
  {
    id: 'signals',
    label: '信号',
    items: [
      { key: '/signals', label: '策略信号', icon: Signal },
      { key: '/today-signal', label: '今日信号', icon: CalendarClock },
      { key: '/paper-signal', label: '纸面信号', icon: Inbox },
      { key: '/paper-portfolio', label: '多策略纸面盘', icon: Layers },
    ],
  },
  {
    id: 'ops',
    label: '运营',
    items: [
      { key: '/tasks', label: '任务管理', icon: Bot },
      { key: '/tasks/create', label: '创建任务', icon: Plus },
      { key: '/tasks/compare', label: '任务对比', icon: GitCompare },
      { key: '/tasks/tuning', label: '参数调优', icon: SlidersHorizontal },
      { key: '/monitoring', label: '系统监控', icon: BarChart3 },
    ],
  },
  {
    id: 'account',
    label: '账户',
    items: [
      { key: '/settings', label: '系统设置', icon: Settings },
      { key: '/settings/subscription', label: '订阅管理', icon: Star },
      { key: '/wallet', label: '钱包', icon: Wallet },
      { key: '/admin', label: '管理后台', icon: ShieldCheck },
      { key: '/admin/payments', label: '充值订单', icon: CreditCard, adminOnly: true },
    ],
  },
];

/** 全部导航项（含管理员项），供面包屑与选中态计算使用 */
export const allNavItems: NavItem[] = navGroups.flatMap(group => group.items);

/** 前缀匹配：pathname === key，或 pathname 位于 key 的子路由下。`/` 不能作键。 */
export function isActive(pathname: string, key: string): boolean {
  return pathname === key || (key !== '/' && pathname.startsWith(`${key}/`));
}

/**
 * 当前 pathname 对应的导航键（取最长匹配）。
 * 嵌套路由（如 /tasks/create）会同时匹配 /tasks 与 /tasks/create，
 * 取最长的一条作为「当前选中」，避免父子项同时高亮。
 */
export function findActiveKey(pathname: string, isAdmin: boolean): string | null {
  const matching = allNavItems
    .filter(item => !item.adminOnly || isAdmin)
    .filter(item => isActive(pathname, item.key))
    .sort((a, b) => b.key.length - a.key.length);
  return matching.length > 0 ? matching[0].key : null;
}

export function findNavItem(key: string): NavItem | undefined {
  return allNavItems.find(item => item.key === key);
}

export function findNavGroup(groupId: string): NavGroup | undefined {
  return navGroups.find(group => group.id === groupId);
}

export interface BreadcrumbEntry {
  key: string;
  label: string;
}

/**
 * 面包屑：父级条目（如有）+ 当前页标题。
 * 例：/tasks/create → [任务管理, 创建任务]；/dashboard → [仪表板]。
 * /tasks/[id] 这类子页没有自己的菜单项，面包屑显示其父条目。
 */
export function getBreadcrumbs(pathname: string, isAdmin: boolean): BreadcrumbEntry[] {
  const activeKey = findActiveKey(pathname, isAdmin);
  if (!activeKey) {
    return [];
  }

  const current = findNavItem(activeKey);
  if (!current) {
    return [];
  }

  const parent = allNavItems
    .filter(
      item =>
        item !== current &&
        item.key !== activeKey &&
        (isActive(pathname, item.key) || item.key === activeKey.slice(0, item.key.length)) &&
        activeKey.startsWith(`${item.key}/`)
    )
    .sort((a, b) => b.key.length - a.key.length);

  const parentEntry = parent.length > 0 ? parent[0] : null;
  return parentEntry ? [parentEntry, current] : [current];
}

/** 不套后台外壳（AppLayout）的公开页路径 */
export const PUBLIC_PATHS = ['/', '/login', '/register', '/reset-password', '/pricing'] as const;

export type PublicPath = (typeof PUBLIC_PATHS)[number];

export function isPublicPath(pathname: string): boolean {
  return (PUBLIC_PATHS as readonly string[]).includes(pathname);
}
