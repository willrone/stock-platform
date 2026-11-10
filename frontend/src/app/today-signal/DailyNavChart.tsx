'use client';

/**
 * 纸面盘「日频盯市净值」曲线 —— today-signal 页专用
 *
 * ⚠️ 口径警告（本项目有**三套互不相同**的净值曲线，绝不可混画/直接对比）
 *   ① 月频复利还原（strategy_race._months_to_nav，36 点 / 3 年，历史回测口径）
 *      → 画在 paper-portfolio 页的 StrategyRaceChart，**与本组件无关**。
 *   ② 周频「调仓日净值点」（performance.nav_basis，~4 点 / 2 月）
 *      → 只在 today-signal 页以「点数」形式出现，**不画曲线**。
 *   ③ 日频盯市净值（nav_daily，total_return@trade_day，~12 点 / 1 月）
 *      → **本组件画的就是这一套**：每个交易日收盘后按总回报口径盯市，
 *        **非调仓日也在盯市**（这正是与口径②的差别）。
 *
 * 图表库：沿用项目现有栈 —— echarts 5.4.3（与 StrategyRaceChart 一致）。
 *   运行时 echarts 必须在 useEffect 里动态 import —— zrender 在模块加载期就做
 *   UA 探测（'ontouchstart' in window），Node ≥21 有全局 navigator 却没有 window
 *   → 顶层 import 会让静态预渲染该组件的页面构建失败。
 *   配色一律取自 src/theme/chartTokens.ts（图表色板唯一真相源），不硬编码。
 *
 * 数据来源：GET /api/v1/trading/performance?strategy=xxx → response.nav_daily
 *   {
 *     available: boolean,        // false → 明确提示「暂无日频盯市数据」，绝不画空图
 *     basis: string | null,      // 口径标识，如 'total_return@trade_day (...)'
 *     n_points: number,
 *     n_stale_points: number,    // 用了非当日收盘价的点数（后端已算好，直接用）
 *     updated_at: string | null, // 产物生成时间（数据新鲜度）
 *     initial_capital: number | null,
 *     points: [{date, value, return_pct, n_positions, stale_price}],
 *     note: string | null        // available=false 时说明为什么没有数据
 *   }
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
// ⚠️ 只做**类型**导入（编译期擦除，不进 server bundle）。运行时 import 见文件头注释。
import type * as EChartsNS from 'echarts';
import { Alert, Box, Button, Chip, Stack, Typography } from '@mui/material';
import { Info, LineChart } from 'lucide-react';
import { chartAxis, chartGrid, chartNeutral, chartSeries, chartTooltip } from '@/theme/chartTokens';

export interface DailyNavPoint {
  date: string;
  value: number;
  /** 相对初始资金的累计收益 % */
  return_pct?: number | null;
  /** 当日持仓数 */
  n_positions?: number | null;
  /**
   * 该点用的不是当日收盘价时，后端写入非空值。
   *
   * ⚠️ 实测有**三种形态**，判断陈旧必须显式判 null：
   *   ① `null`（JSON null）—— 该点用的是当日收盘价。
   *      ⚠️ **不是** false → 用 `if (p.stale_price)` 会把 null 与 false 一起吞掉，
   *      分不清「0 个陈旧点」与「未知」。
   *   ② `string[]` —— 该点有 N 只股票用了滞后价，数组元素是股票代码
   *      （实测 2026-11-03 的点：30 只全部滞后）。
   *   ③ `string` —— 后端写日期/说明文本的形态（防御性兼容）。
   *
   * ⚠️ 数组绝不能直接塞进模板渲染（会变成 "000157.SZ,600817.SH,…" 长串）。
   *    需要展示陈旧详情时只显示**数量**（如「30 只用滞后价」）。
   */
  stale_price?: string | string[] | number | boolean | null;
}

export interface DailyNavData {
  available?: boolean;
  basis?: string | null;
  n_points?: number;
  n_stale_points?: number;
  updated_at?: string | null;
  initial_capital?: number | null;
  points?: DailyNavPoint[];
  note?: string | null;
}

interface DailyNavChartProps {
  data?: DailyNavData | null;
  /**
   * 业绩接口是否成功返回。
   * false → 是「接口没拿到」，与「接口说没有日频数据」是**两回事**，
   * 不能都显示成「暂无日频盯市数据」（那是在替用户做假设）。
   */
  perfLoaded?: boolean;
  /**
   * 命中付费墙（后端对免费档位整段剥掉 nav_daily 等字段，locked_detail=true）。
   * 这**不是**「没有日频数据」—— 是「有但你看不到」，文案必须分开。
   */
  locked?: boolean;
  /** 付费墙升级提示（后端 upgrade_hint） */
  upgradeHint?: string | null;
  height?: number;
  /**
   * 开发态预览：强制走「暂无日频数据」降级分支。
   *
   * ⚠️ 这是**前端级演示**，不是真实降级 —— 生产环境 6 条策略都有 nav_daily 产物，
   *    available=false 在真实环境**无法自然触发**。所以这个开关只在 dev 构建里出现，
   *    且预览文案会明确标注「预览态 · 非真实数据」，避免被误当成真实降级截图。
   */
  previewUnavailable?: boolean;
}

/** 开发态「暂无日频数据」预览用的 fixture —— 文案自带「预览态」标记，绝不冒充真实降级。 */
const UNAVAILABLE_PREVIEW_FIXTURE: DailyNavData = {
  available: false,
  basis: null,
  n_points: 0,
  n_stale_points: 0,
  updated_at: null,
  initial_capital: null,
  points: [],
  note:
    '【预览态 · 非真实数据】这条提示用于演示「接口说该策略暂无日频盯市产物」时的降级渲染。' +
    '当前生产环境 6 条策略都有日频产物，真实 available=false 分支无法自然触发。',
};

/**
 * 是不是「陈旧价点」（用了非当日收盘价）—— 必须显式判 null。
 *
 * ⚠️ 实测三种形态：
 *   - `null` → 不是陈旧点（当日收盘价）
 *   - `string[]` → 陈旧点，数组长度 = 有多少只股票用了滞后价
 *   - `string` / 其它非空 → 陈旧点（后端写日期/说明的形态）
 */
const isStalePoint = (p: DailyNavPoint): boolean => {
  const v = p.stale_price;
  return v !== null && v !== undefined && v !== false && v !== 0;
};

/**
 * 该点里有多少只股票用了滞后价。
 *
 * ⚠️ 数组绝不能直接渲染（模板插值会变成 "000157.SZ,600817.SH,…" 长串），
 *    所以任何展示都只取**数量**。
 * 返回 null = 该点不是陈旧点（数量未知）。
 */
const staleCountOf = (p: DailyNavPoint): number | null => {
  if (!isStalePoint(p)) {
    return null;
  }
  const v = p.stale_price;
  if (Array.isArray(v)) {
    return v.length;
  }
  // 非数组形态：只知道「有滞后」，数量未知
  return null;
};

/** 2026-11-03T06:45:58.940076 → 2026-11-03 06:45 */
const fmtTimestamp = (ts?: string | null): string | null => {
  if (!ts) {
    return null;
  }
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) {
    return ts;
  }
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(
    d.getMinutes()
  )}`;
};

const fmtMoney = (v: number) =>
  `¥${v.toLocaleString('zh-CN', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;

const fmtPct = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;

/** Y 轴刻度：整数元（978,000 / 990,000 …） */
const fmtAxisMoney = (v: number) => v.toLocaleString('zh-CN', { maximumFractionDigits: 0 });

/**
 * 取「好看」的刻度步长：1 / 2 / 2.5 / 5 / 10 × 10^n。
 * 用于把 Y 轴上下界对齐到整数刻度 —— 否则 ECharts 会自己挑 978,345 这类
 * 读者对不上的刻度（实测会出现 965,976 / 1,003,645 这种两头不齐的轴）。
 */
const niceStep = (rawStep: number): number => {
  if (!Number.isFinite(rawStep) || rawStep <= 0) {
    return 1;
  }
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  let mult = 10;
  if (norm <= 1) {
    mult = 1;
  } else if (norm <= 2) {
    mult = 2;
  } else if (norm <= 2.5) {
    mult = 2.5;
  } else if (norm <= 5) {
    mult = 5;
  }
  return mult * mag;
};

export default function DailyNavChart({
  data,
  perfLoaded = true,
  locked = false,
  upgradeHint,
  height = 300,
  previewUnavailable = false,
}: DailyNavChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<EChartsNS.ECharts | null>(null);

  /**
   * 开发态预览「暂无日频数据」降级分支。
   * ⚠️ 仅 dev 构建可见，且渲染出的文案自带「预览态 · 非真实数据」标记 ——
   *    真实环境 6 条策略都有产物，available=false 不会自然发生。
   */
  const isDev = process.env.NODE_ENV === 'development';
  const [previewDegraded, setPreviewDegraded] = useState(false);
  const forceUnavailable = isDev && previewUnavailable && previewDegraded;

  /** 生效的数据：预览态下换成 fixture（走 available=false 分支）。 */
  const effectiveData = forceUnavailable ? UNAVAILABLE_PREVIEW_FIXTURE : data;

  const points = useMemo(() => {
    const raw = effectiveData?.points ?? [];
    return raw
      .filter(p => p && typeof p.date === 'string' && Number.isFinite(Number(p.value)))
      .map(p => ({ ...p, value: Number(p.value) }))
      .sort((a, b) => a.date.localeCompare(b.date));
  }, [effectiveData]);

  const capital = Number(effectiveData?.initial_capital) || 0;

  useEffect(() => {
    if (!chartRef.current) {
      return;
    }

    if (chartInstance.current) {
      chartInstance.current.dispose();
      chartInstance.current = null;
    }
    if (points.length === 0) {
      return;
    }

    let disposed = false;
    let chart: EChartsNS.ECharts | null = null;
    let onResize: (() => void) | null = null;

    void import('echarts').then(echarts => {
      if (disposed || !chartRef.current) {
        return;
      }

      chart = echarts.init(chartRef.current);
      chartInstance.current = chart;

      const axis = points.map(p => p.date);
      const values = points.map(p => p.value);
      const lineColor = chartSeries[0];

      // 陈旧价点单独一层（当前 6 条线实测 0 个，但代码必须支持并显式标记）
      const staleIdx = points.map((p, i) => (isStalePoint(p) ? i : -1)).filter(i => i >= 0);

      // Y 轴范围：把期初本金一起纳入，避免「只截取数据区间」把跌幅放大成断崖，
      // 也避免期初本金基准线被裁到画布外。
      const lo = capital > 0 ? Math.min(...values, capital) : Math.min(...values);
      const hi = capital > 0 ? Math.max(...values, capital) : Math.max(...values);
      const span = hi - lo || Math.abs(hi) || 1;
      // 上下留 12% 余量，再对齐到「整数刻度」——刻度值要读者能对上（970,000 / 980,000 …），
      // 不能出现 965,976 这种从数据里硬凑出来的边界值。
      const step = niceStep(span / 4);
      const yMin = Math.floor((lo - span * 0.12) / step) * step;
      const yMax = Math.ceil((hi + span * 0.12) / step) * step;

      const series: EChartsNS.SeriesOption[] = [
        {
          name: '日频盯市净值',
          type: 'line',
          data: values,
          connectNulls: true,
          symbol: 'circle',
          symbolSize: 6,
          smooth: false,
          lineStyle: { color: lineColor, width: 2.4 },
          itemStyle: { color: lineColor, borderColor: '#fff', borderWidth: 1.5 },
          emphasis: { focus: 'series' },
          z: 3,
        } as EChartsNS.SeriesOption,
      ];

      // 期初本金基准线（虚线）：标注「期初 100 万」，让读者一眼看到偏离幅度
      if (capital > 0) {
        series.push({
          name: '期初本金',
          type: 'line',
          data: axis.map(() => capital),
          symbol: 'none',
          lineStyle: { color: chartAxis.axisLine, width: 1, type: 'dashed' },
          tooltip: { show: false },
          z: 1,
        } as EChartsNS.SeriesOption);
      }

      if (staleIdx.length > 0) {
        series.push({
          name: '滞后价点（非当日收盘价）',
          type: 'scatter',
          data: staleIdx.map(i => [i, values[i]]),
          symbol: 'triangle',
          symbolSize: 11,
          itemStyle: { color: chartSeries[1], borderColor: '#fff', borderWidth: 1.5 },
          tooltip: { show: false },
          z: 5,
        } as EChartsNS.SeriesOption);
      }

      chart.setOption({
        animation: false,
        grid: { left: 8, right: 20, top: 34, bottom: 8, containLabel: true },
        legend: {
          show: staleIdx.length > 0,
          type: 'scroll',
          top: 0,
          ...(staleIdx.length > 0 ? ['滞后价点（非当日收盘价）'] : []),
          textStyle: { color: chartAxis.label, fontSize: 12 },
          inactiveColor: chartNeutral.iconMuted,
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'line', lineStyle: { color: chartAxis.axisLine } },
          backgroundColor: chartTooltip.background,
          borderColor: chartTooltip.border,
          textStyle: { color: chartTooltip.text, fontSize: 12 },
          extraCssText: chartTooltip.shadow,
          formatter: (params: unknown) => {
            const arr = Array.isArray(params) ? params : [params];
            // ⚠️ 找到「日频盯市净值」那一条 —— 它是 points 的原始下标，
            //    其它系列（基准线 / 陈旧价点 scatter）的 dataIndex 不是。
            const line = arr.find(
              p => (p as { seriesName?: string }).seriesName === '日频盯市净值'
            ) as { dataIndex?: number } | undefined;
            const i = line?.dataIndex ?? (arr[0] as { dataIndex?: number })?.dataIndex ?? 0;
            const pt = points[i];
            if (!pt) {
              return '';
            }
            const stale = isStalePoint(pt);
            const staleN = staleCountOf(pt);
            const rows = [
              `<div style="font-weight:600;margin-bottom:4px">${pt.date}${
                stale ? ' <span style="color:#B45309">⚠ 陈旧价点</span>' : ''
              }</div>`,
              `净值：${fmtMoney(pt.value)}`,
              capital > 0 ? `倍数：${(pt.value / capital).toFixed(4)}×` : '',
              typeof pt.return_pct === 'number' ? `累计收益：${fmtPct(pt.return_pct)}` : '',
              typeof pt.n_positions === 'number' ? `持仓：${pt.n_positions} 只` : '',
              // ⚠️ 只显示**数量**，绝不把代码数组铺开成一行长串
              stale
                ? staleN !== null
                  ? `⚠ ${staleN} 只用滞后价（该点市值截至上一有效收盘，非当日真实波动）`
                  : '⚠ 该点部分持仓用滞后价（市值截至上一有效收盘，非当日真实波动）'
                : '',
            ].filter(Boolean);
            return rows.join('<br/>');
          },
        },
        xAxis: {
          type: 'category',
          data: axis,
          boundaryGap: false,
          axisLine: { lineStyle: { color: chartAxis.axisLine } },
          axisLabel: {
            color: chartAxis.label,
            fontSize: 11,
            hideOverlap: true,
            formatter: (v: string) => (v.length >= 7 ? v.slice(2) : v),
          },
          axisTick: { show: false },
        },
        yAxis: {
          type: 'value',
          scale: true,
          min: yMin,
          max: yMax,
          splitNumber: 5,
          name: '组合净值（元）',
          nameTextStyle: { color: chartAxis.label, fontSize: 11 },
          axisLabel: {
            color: chartAxis.label,
            fontSize: 11,
            formatter: (v: number) => fmtAxisMoney(v),
          },
          splitLine: { lineStyle: { color: chartGrid.splitLine, type: 'dashed' } },
        },
        series,
      });

      onResize = () => chart?.resize();
      window.addEventListener('resize', onResize);
    });

    return () => {
      disposed = true;
      if (onResize) {
        window.removeEventListener('resize', onResize);
        onResize = null;
      }
      chart?.dispose();
      chart = null;
      chartInstance.current = null;
    };
  }, [points, capital]);

  const available = effectiveData?.available === true;
  const nPoints = effectiveData?.n_points ?? points.length;
  const nStale = effectiveData?.n_stale_points ?? 0;
  const updatedAt = fmtTimestamp(effectiveData?.updated_at);
  const dataThrough = points.length > 0 ? points[points.length - 1].date : null;

  /**
   * 持仓数轨迹。⚠️ 持仓只数变化 **不等于** 调仓日 —— 42f 这类固定 N 只持仓的
   *    策略调仓时换股但只数不变，所以这里只陈述「持仓只数」，不推断调仓日。
   */
  const posTrace = useMemo(() => {
    const withPos = points.filter(
      p => typeof p.n_positions === 'number' && Number.isFinite(Number(p.n_positions))
    );
    if (withPos.length === 0) {
      return null;
    }
    const nums = withPos.map(p => Number(p.n_positions));
    const min = Math.min(...nums);
    const max = Math.max(...nums);
    let changed = 0;
    for (let i = 1; i < withPos.length; i += 1) {
      if (Number(withPos[i].n_positions) !== Number(withPos[i - 1].n_positions)) {
        changed += 1;
      }
    }
    return { min, max, last: nums[nums.length - 1], changed, total: withPos.length };
  }, [points]);

  /**
   * 「水平末段」检测。
   *
   * ⚠️ 末点若整仓都用滞后价（stale_price 是非空数组），它的市值 = 上一有效收盘的
   *    市值 → 与前一点**完全相同** → 曲线末端是一段水平线。用户看到「净值一动不动」
   *    会误判成「没波动 / 卡住了」。必须主动点明这是滞后价造成的，不是真实波动。
   *
   * 这里判定的是「末点与前一点数值相同」这一**可观测事实**，再结合末点是否带滞后价
   * 来决定措辞，避免在数据正常时也乱喊「水平末段」。
   */
  const flatTail = useMemo(() => {
    if (points.length < 2) {
      return null;
    }
    const last = points[points.length - 1];
    const prev = points[points.length - 2];
    if (last.value !== prev.value) {
      return null;
    }
    const staleN = staleCountOf(last);
    return {
      date: last.date,
      prevDate: prev.date,
      staleN,
      isStale: isStalePoint(last),
    };
  }, [points]);

  /**
   * 开发态预览开关（只在 dev 构建、且页面显式开启时出现）。
   * 位置固定在组件最上方，让降级预览态也能切回来。
   */
  const devToggle =
    isDev && previewUnavailable ? (
      <Stack
        direction="row"
        spacing={1}
        alignItems="center"
        sx={{
          mb: 1,
          px: 1,
          py: 0.5,
          borderRadius: 1,
          border: '1px dashed',
          borderColor: 'warning.main',
          bgcolor: 'warning.50',
        }}
      >
        <Chip size="small" color="warning" label="预览态 · 非真实数据" />
        <Button
          size="small"
          variant="outlined"
          color="warning"
          onClick={() => setPreviewDegraded(v => !v)}
        >
          {previewDegraded ? '退出「暂无日频数据」预览' : '预览「暂无日频数据」降级态'}
        </Button>
        <Typography variant="caption" color="text.secondary">
          生产环境 6 条策略都有日频产物，真实 available=false 分支无法自然触发 ——
          此开关仅用于前端级演示。
        </Typography>
      </Stack>
    ) : null;

  // ── 降级 A：业绩接口本身没拿到 → 不能说成「暂无日频数据」 ──
  if (!perfLoaded) {
    return (
      <>
        {devToggle}
        <Alert severity="warning">
          <Typography variant="body2" fontWeight={700}>
            业绩数据未加载成功
          </Typography>
          <Typography variant="body2" sx={{ mt: 0.5 }}>
            无法判断该策略是否有日频盯市数据 —— 这是接口请求失败，不是「没有数据」。
            请点击右上角「刷新」重试。
          </Typography>
        </Alert>
      </>
    );
  }

  // ── 降级 B：命中付费墙（后端整段剥掉 nav_daily）→ 是「有但看不到」 ──
  if (locked) {
    return (
      <>
        {devToggle}
        <Alert severity="info" icon={<Info size={18} />}>
          <Typography variant="body2" fontWeight={700}>
            日频盯市净值曲线在当前订阅档位不可见
          </Typography>
          <Typography variant="body2" sx={{ mt: 0.5 }}>
            {upgradeHint ||
              '日频盯市净值属于信号版功能。当前档位只返回汇总数字，不返回逐日净值点。'}
            纸面盘周频「调仓日净值点」仍在运行（见上方点数），但那是
            <strong>另一套口径</strong>，不能用它的点数冒充日频盯市曲线。
          </Typography>
        </Alert>
      </>
    );
  }

  // ── 降级 C：接口说没有日频数据 → 明确提示 + 透出 note，绝不画空图 ──
  if (!available || points.length === 0) {
    return (
      <>
        {devToggle}
        <Alert severity="info" icon={<Info size={18} />}>
          <Typography variant="body2" fontWeight={700}>
            暂无日频盯市数据
          </Typography>
          <Typography variant="body2" sx={{ mt: 0.5 }}>
            {effectiveData?.note ||
              '暂无日频盯市数据（该策略的日频净值产物尚未生成）。日频盯市产物由定时任务在工作日 18:20 产出，产物生成后此处会自动出现曲线。'}
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
            提示：纸面盘周频「调仓日净值点」仍在运行（见上方点数），但那是
            <strong>另一套口径</strong>，不能用它的点数冒充日频盯市曲线。
          </Typography>
        </Alert>
      </>
    );
  }

  return (
    <Box>
      {devToggle}
      <Box
        ref={chartRef}
        role="img"
        aria-label="纸面盘日频盯市净值曲线"
        sx={{ width: '100%', height }}
      />

      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mt: 1.5, mb: 0.5 }}>
        <Chip
          size="small"
          icon={<LineChart size={14} />}
          label={`${nPoints} 个交易日盯市点`}
          sx={{ height: 24, fontSize: 12 }}
        />
        <Chip
          size="small"
          color={nStale > 0 ? 'warning' : 'default'}
          variant={nStale > 0 ? 'filled' : 'outlined'}
          label={nStale > 0 ? `${nStale} 个点用了非当日收盘价` : '全部为当日收盘价'}
          sx={{ height: 24, fontSize: 12 }}
        />
        {dataThrough && (
          <Chip
            size="small"
            variant="outlined"
            label={`数据截至 ${dataThrough}`}
            sx={{ height: 24, fontSize: 12 }}
          />
        )}
        {updatedAt && (
          <Chip
            size="small"
            variant="outlined"
            label={`产物生成于 ${updatedAt}`}
            sx={{ height: 24, fontSize: 12 }}
          />
        )}
      </Stack>

      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
        口径：<strong>total_return@trade_day</strong> —— 每个交易日收盘后按总回报盯市，
        <strong>非调仓日也在盯市</strong>；期初本金 {capital > 0 ? fmtMoney(capital) : '未知'}
        （虚线基准）。与「月频复利还原」（历史回测口径，36 点）
        <strong>不是同一口径，不可直接对比</strong>
        ；上方卡片用的周频「调仓日净值点」也是另一套口径，
        <strong>回撤以本曲线为准（周频会低估）</strong>。 本曲线点数不足，不做统计推断（不展示夏普 /
        最大回撤等结论）。
      </Typography>
      {effectiveData?.note && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
          {effectiveData.note}
        </Typography>
      )}
      {posTrace && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
          当日持仓{' '}
          {posTrace.min === posTrace.max ? posTrace.min : `${posTrace.min}–${posTrace.max}`} 只
          （最新 {posTrace.last} 只）；
          {posTrace.changed > 0 ? (
            <>
              其中 <strong>{posTrace.changed}</strong> / {posTrace.total} 个交易日持仓只数发生变化。
            </>
          ) : (
            <>
              {posTrace.total} 个交易日持仓只数未变 —— 策略换股但只数固定，
              <strong>非调仓日沿用上一交易日持仓继续盯市</strong>。
            </>
          )}
          日频与周频口径的差别就在这里。
        </Typography>
      )}
      {nStale > 0 && (
        <Typography variant="caption" color="warning.main" sx={{ display: 'block', mt: 0.5 }}>
          ⚠ 有 {nStale} 个净值点使用的不是当日收盘价（图中以 ▲ 标出），
          这些点的市值是按最近已知价（滞后价）估的 —— 反映的是
          <strong>截至上一有效收盘的市值</strong>，不是当日真实波动。
        </Typography>
      )}
      {flatTail && (
        <Typography variant="caption" color="warning.main" sx={{ display: 'block', mt: 0.5 }}>
          ⚠ 曲线末端有一段<strong>水平线</strong>：{flatTail.date} 的净值与 {flatTail.prevDate}{' '}
          完全相同（
          {fmtMoney(points[points.length - 1].value)}）。
          {flatTail.isStale
            ? flatTail.staleN !== null
              ? ` 该点 ${flatTail.staleN} 只持仓全部使用滞后价，`
              : ' 该点部分持仓使用滞后价，'
            : ''}
          所以这一段<strong>不代表「净值没有波动」</strong>，而是
          <strong>尚未拿到当日收盘价</strong>；下一个交易日的盯市点才会体现真实波动。
        </Typography>
      )}
    </Box>
  );
}
