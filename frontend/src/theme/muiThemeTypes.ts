/**
 * MUI 主题类型扩展
 *
 * 背景：我们在 palette 上新增了 `semantics` 键（A 股涨跌语义色：红涨绿跌）。
 * MUI 的 `Palette` / `PaletteOptions` 是封闭接口，不做声明合并的话
 * `createTheme({ palette: { semantics: ... } })` 会直接报 TS2353。
 *
 * 【为什么是 .ts 而不是 .d.ts】
 * 本项目 tsconfig 的 include 用的是「两个星号 + 任意路径 + .ts」的模式，
 * 而 TypeScript 的这个模式**不匹配** .d.ts 文件（只匹配 .ts / .tsx），
 * 所以声明合并文件必须命名为 .ts 才会进入编译 program。
 * 这里没有任何运行时代码，全是 declare module（纯类型），不会产生任何 JS 输出。
 *
 * 值的唯一真相源在 src/theme/tokens.ts（semanticColors），这里只声明类型。
 *
 * 用法：
 *   sx={{ color: 'semantics.up' }}     // 涨（红）
 *   sx={{ color: 'semantics.down' }}   // 跌（绿）
 * 注意：Typography 的 color prop 只接受 palette 的一级键，
 *       所以语义色只能通过 sx 里的 color 用，不能用 color="semantics.up"。
 */

import '@mui/material/styles';

declare module '@mui/material/styles' {
  /** 涨跌语义色在 sx 中的取值 */
  interface PaletteSemantics {
    /** 涨 —— A 股习惯用红色 */
    up: string;
    /** 跌 —— A 股习惯用绿色 */
    down: string;
    /** 平盘 / 中性 */
    flat: string;
  }

  interface Palette {
    semantics: PaletteSemantics;
  }

  interface PaletteOptions {
    semantics?: PaletteSemantics;
  }
}

export {};
