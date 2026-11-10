'use client';

/**
 * 交易明细卡片（移动端降级方案）
 *
 * today-signal / paper-signal / paper-portfolio 的明细表在窄屏下有 7~8 列
 * （代码/名称/参考价/股数/金额/费用/分数/为什么买它），横向滚动体验很差。
 * 移动端改用卡片：每张卡片一只股票，把关键数字摊平成 label-value 行，
 * 「为什么买它」的因子用 Chip 平铺。
 *
 * 桌面端仍然用表格（见 ResponsiveTable），两套视图数据同源。
 */

import React from 'react';
import { Box, Chip, Stack, Tooltip, Typography } from '@mui/material';
import { TrendingUp, TrendingDown, Wallet, Info } from 'lucide-react';

export interface TradeFactor {
  factor: string;
  contribution: number;
  direction?: string;
}

export interface TradeCardData {
  code: string;
  name?: string;
  price?: number;
  shares?: number;
  amount?: number;
  /** 费用 = 佣金 + 印花税 + 滑点 */
  fee?: number;
  score?: number | string | null;
  factors?: TradeFactor[];
  /** 买入 / 卖出 / 持仓 */
  side: 'buy' | 'sell' | 'hold';
  /** 持仓用 */
  cost?: number;
  costAmount?: number;
  buyDate?: string;
}

const fmtMoney = (v?: number) =>
  v === undefined || v === null || !Number.isFinite(v)
    ? '暂无'
    : `¥${v.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

const fmtShares = (v?: number) =>
  v === undefined || v === null || !Number.isFinite(v) ? '暂无' : v.toLocaleString('zh-CN');

const fmtScore = (v?: number | string | null) => {
  if (v === undefined || v === null || v === '') {
    return '暂无';
  }
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(3) : String(v);
};

const sideMeta = {
  buy: { label: '买入', color: 'success.main' as const, Icon: TrendingUp },
  sell: { label: '卖出', color: 'error.main' as const, Icon: TrendingDown },
  hold: { label: '持仓', color: 'info.main' as const, Icon: Wallet },
};

const Row: React.FC<{ label: string; value: React.ReactNode }> = ({ label, value }) => (
  <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, alignItems: 'baseline' }}>
    <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
      {label}
    </Typography>
    <Typography variant="body2" sx={{ fontWeight: 600, textAlign: 'right' }}>
      {value}
    </Typography>
  </Box>
);

export const MobileTradeCard: React.FC<{ trade: TradeCardData }> = ({ trade }) => {
  const meta = sideMeta[trade.side];
  const { Icon } = meta;
  const hasFactors = (trade.factors?.length ?? 0) > 0;

  return (
    <Box
      sx={{
        mb: 1.25,
        p: 1.5,
        borderRadius: 2,
        border: 1,
        borderColor: 'divider',
        borderLeft: 4,
        borderLeftColor: meta.color,
        bgcolor: 'background.paper',
      }}
    >
      <Box
        sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 1 }}
      >
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="body1" fontWeight={700} sx={{ fontFamily: 'monospace' }}>
            {trade.code}
          </Typography>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis' }}
          >
            {trade.name || '—'}
          </Typography>
        </Box>
        <Chip
          icon={<Icon size={13} />}
          label={meta.label}
          size="small"
          sx={{
            color: meta.color,
            fontWeight: 700,
            fontSize: '0.72rem',
            flexShrink: 0,
          }}
        />
      </Box>

      <Stack spacing={0.25} sx={{ mt: 1 }}>
        {trade.side !== 'hold' && (
          <>
            <Row
              label="参考价"
              value={trade.price != null ? `¥${trade.price.toFixed(2)}` : '暂无'}
            />
            <Row label="股数" value={fmtShares(trade.shares)} />
            <Row label="金额" value={fmtMoney(trade.amount)} />
            <Row label="费用" value={trade.fee != null ? fmtMoney(trade.fee) : '暂无'} />
            <Row label="分数" value={fmtScore(trade.score)} />
          </>
        )}
        {trade.side === 'hold' && (
          <>
            <Row label="股数" value={fmtShares(trade.shares)} />
            <Row label="成本价" value={trade.cost != null ? `¥${trade.cost.toFixed(2)}` : '暂无'} />
            <Row label="成本金额" value={fmtMoney(trade.costAmount)} />
            <Row label="买入日" value={trade.buyDate || '暂无'} />
          </>
        )}
      </Stack>

      <Box sx={{ mt: 1, pt: 0.75, borderTop: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: hasFactors ? 0.75 : 0 }}>
          <Info size={13} color="#888" />
          <Typography variant="caption" color="text.secondary">
            为什么买它
          </Typography>
        </Box>
        {hasFactors ? (
          <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
            {trade.factors!.map(f => (
              <Tooltip
                key={f.factor}
                title={`${f.factor} 贡献 ${f.contribution > 0 ? '+' : ''}${f.contribution}`}
              >
                <Chip
                  size="small"
                  label={f.factor}
                  variant="outlined"
                  sx={{
                    fontSize: 11,
                    color: f.contribution > 0 ? 'success.main' : 'error.main',
                    borderColor: f.contribution > 0 ? 'success.light' : 'error.light',
                  }}
                />
              </Tooltip>
            ))}
          </Stack>
        ) : (
          <Typography variant="caption" color="text.secondary">
            暂无因子归因数据
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default MobileTradeCard;
