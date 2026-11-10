/**
 * Tailwind 配置
 *
 * 【P1 · design】HeroUI 插件已移除。
 * 原因：全站 0 处引用 @heroui/react（唯一命中的 src/app/models/page.tsx.working
 * 是遗留工作稿，已删除），但插件会注入 128KB 死 CSS，占全局 CSS 的 56%。
 *
 * Tailwind 保留为「原子布局工具」：允许 flex/grid/gap/px 这类布局与间距原子类，
 * **禁止**使用 Tailwind 的颜色 / 阴影 / 圆角工具类 —— 那些必须走 MUI 主题
 * （src/theme/tokens.ts 是唯一真相源）。
 */

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        border: 'hsl(var(--border))',
      },
    },
  },
  // 【P1 · design】本项目只做浅色主题，移除 darkMode:'class'
  // （globals.css 的 .dark 已退化为浅色兼容层）。
  plugins: [],
};
