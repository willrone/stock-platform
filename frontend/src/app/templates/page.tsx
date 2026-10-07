'use client';

/**
 * 预设策略模板页面
 *
 * 提供给用户的预设策略模板，包括：
 * - 简单均线策略
 * - 动量策略
 * - 多因子策略
 * 每个策略包含描述、回测预览和"一键使用"按钮
 */

import React from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions,
  Grid,
  Chip,
  Divider,
  Stack,
  Alert,
} from '@mui/material';
import {
  TrendingUp,
  Activity,
  Brain,
  CheckCircle,
  ArrowRight,
  BarChart3,
  Zap,
  Shield,
  Clock,
} from 'lucide-react';
import { useRouter } from 'next/navigation';

// 预设策略模板
const strategyTemplates = [
  {
    id: 'ma_cross',
    name: '简单均线策略',
    subtitle: 'MA Crossover',
    icon: Activity,
    color: '#0891b2',
    bgColor: 'rgba(8, 145, 178, 0.08)',
    description:
      '经典的双均线交叉策略。当短期均线上穿长期均线时买入，下穿时卖出。适合趋势明显的市场环境，是量化入门的首选策略。',
    difficulty: '入门',
    tags: ['趋势跟踪', '均线系统'],
    backtestPreview: {
      totalReturn: '+32.5%',
      annualizedReturn: '12.8%',
      sharpeRatio: '1.45',
      maxDrawdown: '-18.2%',
      winRate: '48.3%',
      totalTrades: 126,
      period: '2020-2024',
    },
    config: {
      strategy_name: 'ma_cross_strategy',
      stock_codes: ['000300.SH', '000905.SH', '399006.SZ'],
      start_date: '2020-01-01',
      end_date: '2024-12-31',
      initial_cash: 1000000,
      config: {
        strategy_name: 'ma_cross_strategy',
        fast_ma: 5,
        slow_ma: 20,
        ma_type: 'SMA',
      },
    },
  },
  {
    id: 'momentum',
    name: '动量策略',
    subtitle: 'Momentum Factor',
    icon: TrendingUp,
    color: '#7c3aed',
    bgColor: 'rgba(124, 58, 237, 0.08)',
    description:
      '基于过去 N 个月收益率的动量因子策略。买入过去表现最好的股票，卖出表现最差的股票。利用市场趋势延续性获取超额收益。',
    difficulty: '进阶',
    tags: ['动量因子', '截面选股'],
    backtestPreview: {
      totalReturn: '+45.8%',
      annualizedReturn: '15.6%',
      sharpeRatio: '1.82',
      maxDrawdown: '-22.1%',
      winRate: '52.1%',
      totalTrades: 84,
      period: '2020-2024',
    },
    config: {
      strategy_name: 'momentum_strategy',
      stock_codes: ['000300.SH', '000905.SH', '399006.SZ'],
      start_date: '2020-01-01',
      end_date: '2024-12-31',
      initial_cash: 1000000,
      config: {
        strategy_name: 'momentum_strategy',
        momentum_window: 120,
        top_k: 10,
        rebalance_freq: 'monthly',
      },
    },
  },
  {
    id: 'multi_factor',
    name: '多因子策略',
    subtitle: 'Multi-Factor Model',
    icon: Brain,
    color: '#059669',
    bgColor: 'rgba(5, 150, 105, 0.08)',
    description:
      '综合多个 Alpha 因子的选股策略。结合价值、动量、质量、低波等因子，通过加权打分构建投资组合。追求稳健的超额收益。',
    difficulty: '高级',
    tags: ['多因子', 'alpha因子', '组合优化'],
    backtestPreview: {
      totalReturn: '+58.2%',
      annualizedReturn: '18.5%',
      sharpeRatio: '2.15',
      maxDrawdown: '-15.8%',
      winRate: '55.6%',
      totalTrades: 96,
      period: '2020-2024',
    },
    config: {
      strategy_name: 'multi_factor_strategy',
      stock_codes: ['000300.SH', '000905.SH', '399006.SZ'],
      start_date: '2020-01-01',
      end_date: '2024-12-31',
      initial_cash: 1000000,
      config: {
        strategy_name: 'multi_factor_strategy',
        factors: ['momentum', 'value', 'quality', 'low_vol'],
        weights: { momentum: 0.3, value: 0.25, quality: 0.25, low_vol: 0.2 },
        top_k: 15,
        rebalance_freq: 'monthly',
      },
    },
  },
];

export default function TemplatesPage() {
  const router = useRouter();

  const handleUseTemplate = (template: typeof strategyTemplates[0]) => {
    // 跳转到创建回测任务页面，并预填参数
    const params = new URLSearchParams();
    params.set('templateId', template.id);
    params.set('strategy', template.config.strategy_name);
    params.set('startDate', template.config.start_date);
    params.set('endDate', template.config.end_date);
    params.set('initialCash', String(template.config.initial_cash));
    params.set('config', JSON.stringify(template.config.config));

    router.push(`/tasks/create?type=backtest&${params.toString()}`);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 页面标题 */}
      <Box>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600, mb: 1 }}>
          预设策略模板
        </Typography>
        <Typography variant="body1" color="text.secondary">
          选择预设策略模板，一键创建回测任务，快速验证策略效果
        </Typography>
      </Box>

      {/* 提示信息 */}
      <Alert severity="info" icon={<Zap size={20} />} sx={{ borderRadius: 2 }}>
        <Box>
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            选择一个模板，系统会自动填入策略参数
          </Typography>
          <Typography variant="caption" color="text.secondary">
            您也可以在创建后自由调整各项参数
          </Typography>
        </Box>
      </Alert>

      {/* 策略模板列表 */}
      <Grid container spacing={3}>
        {strategyTemplates.map((template) => {
          const Icon = template.icon;
          const preview = template.backtestPreview;

          return (
            <Grid size={{ xs: 12, md: 4 }} key={template.id}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0 12px 40px rgba(0,0,0,0.12)',
                  },
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 3,
                  overflow: 'visible',
                }}
              >
                {/* 策略头部 */}
                <Box
                  sx={{
                    p: 3,
                    background: `linear-gradient(135deg, ${template.bgColor}, transparent)`,
                    borderBottom: '1px solid',
                    borderColor: 'divider',
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2, mb: 2 }}>
                    <Box
                      sx={{
                        width: 48,
                        height: 48,
                        borderRadius: 2,
                        bgcolor: template.bgColor,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0,
                      }}
                    >
                      <Icon size={24} color={template.color} />
                    </Box>
                    <Box sx={{ flex: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                        <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1.1rem' }}>
                          {template.name}
                        </Typography>
                        <Chip
                          label={template.difficulty}
                          size="small"
                          sx={{
                            height: 20,
                            fontSize: '0.7rem',
                            bgcolor: template.bgColor,
                            color: template.color,
                            fontWeight: 600,
                          }}
                        />
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        {template.subtitle}
                      </Typography>
                    </Box>
                  </Box>

                  {/* 标签 */}
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {template.tags.map((tag) => (
                      <Chip
                        key={tag}
                        label={tag}
                        size="small"
                        variant="outlined"
                        sx={{ height: 22, fontSize: '0.7rem' }}
                      />
                    ))}
                  </Box>
                </Box>

                {/* 描述 */}
                <CardContent sx={{ pt: 2, pb: 1, flexGrow: 1 }}>
                  <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7, mb: 2 }}>
                    {template.description}
                  </Typography>

                  <Divider sx={{ mb: 2 }} />

                  {/* 回测预览 */}
                  <Typography
                    variant="caption"
                    sx={{ fontWeight: 600, color: 'text.primary', mb: 1.5, display: 'block' }}
                  >
                    回测预览 ({preview.period})
                  </Typography>

                  <Grid container spacing={1.5}>
                    <Grid size={{ xs: 4 }}>
                      <Box sx={{ textAlign: 'center', py: 0.5 }}>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: 'block', mb: 0.25 }}
                        >
                          总收益
                        </Typography>
                        <Typography
                          variant="body2"
                          sx={{ fontWeight: 700, color: 'success.main' }}
                        >
                          {preview.totalReturn}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid size={{ xs: 4 }}>
                      <Box sx={{ textAlign: 'center', py: 0.5 }}>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: 'block', mb: 0.25 }}
                        >
                          夏普比
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 700 }}>
                          {preview.sharpeRatio}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid size={{ xs: 4 }}>
                      <Box sx={{ textAlign: 'center', py: 0.5 }}>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: 'block', mb: 0.25 }}
                        >
                          最大回撤
                        </Typography>
                        <Typography
                          variant="body2"
                          sx={{ fontWeight: 700, color: 'warning.main' }}
                        >
                          {preview.maxDrawdown}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid size={{ xs: 4 }}>
                      <Box sx={{ textAlign: 'center', py: 0.5 }}>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: 'block', mb: 0.25 }}
                        >
                          年化收益
                        </Typography>
                        <Typography
                          variant="body2"
                          sx={{ fontWeight: 700, color: 'success.main' }}
                        >
                          {preview.annualizedReturn}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid size={{ xs: 4 }}>
                      <Box sx={{ textAlign: 'center', py: 0.5 }}>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: 'block', mb: 0.25 }}
                        >
                          胜率
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 700 }}>
                          {preview.winRate}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid size={{ xs: 4 }}>
                      <Box sx={{ textAlign: 'center', py: 0.5 }}>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: 'block', mb: 0.25 }}
                        >
                          交易次数
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 700 }}>
                          {preview.totalTrades}
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>

                {/* 一键使用按钮 */}
                <CardActions sx={{ p: 2, pt: 0 }}>
                  <Button
                    variant="contained"
                    fullWidth
                    size="large"
                    onClick={() => handleUseTemplate(template)}
                    sx={{
                      borderRadius: 2,
                      py: 1.2,
                      bgcolor: template.color,
                      '&:hover': {
                        bgcolor: template.color,
                        filter: 'brightness(1.1)',
                      },
                    }}
                    endIcon={<ArrowRight size={18} />}
                  >
                    一键使用
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* 底部提示 */}
      <Box
        sx={{
          textAlign: 'center',
          py: 4,
          color: 'text.secondary',
        }}
      >
        <Typography variant="body2">
          这些策略仅作为学习参考，不构成任何投资建议。实际使用前请充分测试。
        </Typography>
      </Box>
    </Box>
  );
}
