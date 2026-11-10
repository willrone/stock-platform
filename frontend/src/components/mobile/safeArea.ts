/**
 * 安全区（safe-area）常量 —— 刘海屏 / 灵动岛 / 底部指示条适配。
 *
 * ⚠️ 背景：globals.css 由其他代理占用，本文件用 CSS 变量的方式把
 *    safe-area 的取值集中到一处：
 *
 *    - 变量未定义时 → 回退到 env(safe-area-inset-*)（浏览器原生值）
 *    - 变量已定义时 → 使用 globals.css 里定义的 --safe-top / --safe-bottom
 *
 *    👉 需在 src/app/globals.css 中补充（由 globals.css 负责人实现）：
 *
 *        :root {
 *          --safe-top: env(safe-area-inset-top, 0px);
 *          --safe-bottom: env(safe-area-inset-bottom, 0px);
 *        }
 *
 *    之所以要这一步：iOS 上 env(safe-area-inset-*) 在部分 WebView /
 *    桌面浏览器里返回 0，而顶部吸顶栏与底部固定栏仍会被系统栏遮挡，
 *    需要一个可以统一兜底的变量（例如给 --safe-top 加上最小值 12px）。
 */

/** 顶部安全区（用于吸顶栏 / 页头顶部留白） */
export const SAFE_TOP = 'var(--safe-top, env(safe-area-inset-top, 0px))';

/** 底部安全区（用于固定底栏 / 底部内容留白） */
export const SAFE_BOTTOM = 'var(--safe-bottom, env(safe-area-inset-bottom, 0px))';

/**
 * 移动端底部 Tab 栏的高度（含底部安全区）。
 * 页面主内容必须留出至少这么多 paddingBottom，否则最后一行会被底栏盖住。
 */
export const BOTTOM_NAV_CONTENT_PADDING = `calc(64px + ${SAFE_BOTTOM})`;

/** 底部 Tab 栏自身的高度（不含安全区） */
export const BOTTOM_NAV_HEIGHT = 64;

/**
 * 最小可点击区域 —— 44×44 CSS px（iOS HIG / WCAG 2.5.5）。
 * 页面里的按钮 / Chip / 图标按钮都要满足这个下限。
 */
export const MIN_TOUCH_TARGET = 44;
