'use client';

/**
 * Landing Page
 *
 * 平台首页，展示产品介绍和核心功能。
 * 已登录用户自动跳转到仪表板。
 */

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Box, Typography, Button, Card, CardContent, Container, Grid, Chip, Stack,
} from '@mui/material';
import {
  TrendingUp,
  BarChart3,
  Brain,
  Database,
  ArrowRight,
  Shield,
  Zap,
  Sparkles,
  LineChart,
  Activity,
  ShieldCheck,
  TrendingDown,
  Scale,
  ScrollText,
} from 'lucide-react';

// 功能特性配置
const features = [
  {
    icon: Brain,
    title: 'AI 因子挖掘',
    description: '基于深度学习的自动因子工程，从海量数据中挖掘有效 Alpha 因子，发现市场隐藏规律。',
    color: '#7c3aed',
    bgColor: 'rgba(124, 58, 237, 0.08)',
  },
  {
    icon: BarChart3,
    title: '回测引擎',
    description: '高性能事件驱动回测引擎，支持多品种、多周期、多策略并行回测，毫秒级信号处理。',
    color: '#0891b2',
    bgColor: 'rgba(8, 145, 178, 0.08)',
  },
  {
    icon: Sparkles,
    title: '策略管理',
    description: '一站式策略开发平台，支持自定义策略编写、参数优化、绩效归因与风险管理。',
    color: '#059669',
    bgColor: 'rgba(5, 150, 105, 0.08)',
  },
  {
    icon: Database,
    title: '数据服务',
    description: '全市场数据覆盖，实时行情、历史日线、财务指标、另类数据一站式接入。',
    color: '#d97706',
    bgColor: 'rgba(217, 119, 6, 0.08)',
  },
];

// 数据亮点 —— 全部为可复现的实测值，不写无法验证的指标
const highlights = [
  { icon: Database, value: '5,543', label: '全市场标的（含北交所）' },
  { icon: LineChart, value: '0.82', label: '42 因子 10 年真实夏普' },
  { icon: TrendingDown, value: '-80%', label: '周频换手 vs 日频' },
  { icon: Activity, value: '4 条', label: '独立信号线，各 100 万记账' },
];


// 故事板块：把行业不说的那部分摊开写（含对我们自己不利的）
const honesties = [
  {
    icon: Scale,
    color: '#c2410c',
    title: '股票池用的是含退市的全市场',
    body:
      '很多回测只用「现在还活着的股票」，等于是用未来视角挑股票 —— 那些退市的、暴雷的、被 ST 的，天然不会出现在样本里，收益凭空好看一大截。我们用 PIT 全市场口径，退市股票留在样本内。',
    evidence: 'PIT 宇宙 4,329 只 / 10 年，含已退市标的',
  },
  {
    icon: TrendingDown,
    color: '#be123c',
    title: '涨停那天，你买不进',
    body:
      '回测里最常见的作弊是「信号当天按收盘价成交」。但 A 股里信号最强的那些票，往往当天就涨停了 —— 你的成交价应该是涨停价，而且根本买不到。我们把涨停日的买入信号直接抹掉。',
    evidence: '涨停日买入信号置空，不做任何美化',
  },
  {
    icon: ScrollText,
    color: '#1d4ed8',
    title: '我们把对自己不利的数字撤掉了',
    body:
      '我们曾经有一个 0.976 的 10 年夏普，后来发现它来自一份已经丢失的因子映射文件，无法复现。于是我们把对外宣传改成了能复现的 0.8193 —— 数字更小，但它是真的。',
    evidence: '0.976（不可复现）→ 0.8193（PIT 可复现）',
  },
  {
    icon: Activity,
    color: '#0f766e',
    title: '周频比日频更赚钱，因为交易成本',
    body:
      '同样一组因子，日频调仓比周频多换 4~5 倍手。佣金、印花税、滑点加起来，把超额收益吃掉了大半。我们实测 4 条策略线周频全面优于日频 —— 这不是玄学，是算术。',
    evidence: '4/4 条策略线周频胜出，换手砍 4~5 倍',
  },
  {
    icon: ShieldCheck,
    color: '#7c2d12',
    title: 'ST 股票的 5% 涨停，掩码是失效的',
    body:
      '我们的涨停掩码按 9.5% 设定，但 ST 股票涨跌停只有 5%，掩码对它们不生效。这会高估实盘可执行性，所以我们在出信号时把 ST 直接剔除。',
    evidence: 'ST 标的在信号层剔除，不进入买入清单',
  },
  {
    icon: Shield,
    color: '#4338ca',
    title: '纸面账本公开，因为它也亏钱',
    body:
      '我们不做「只给你看赚钱那段」的回测报告。4 条信号线各自 100 万独立记账，每周更新净值，买卖价格按真实可成交价记，费用一分不少。亏钱的月份我们一样放出来。',
    evidence: '佣金 0.03% + 印花税 0.05% + 滑点 0.1%（单边）',
  },
];

export default function Home() {
  const router = useRouter();
  const [checkingAuth, setCheckingAuth] = useState(true);

  // 检查登录状态，如果已登录则跳转到仪表板
  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (token) {
      router.replace('/dashboard');
    } else {
      setCheckingAuth(false);
    }
  }, [router]);

  if (checkingAuth) {
    return null;
  }

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      {/* Hero 区域 */}
      <Box
        sx={{
          position: 'relative',
          overflow: 'hidden',
          pb: { xs: 8, md: 12 },
          pt: { xs: 6, md: 10 },
          background: 'linear-gradient(135deg, #0a1628 0%, #1a2332 50%, #0d2137 100%)',
          color: 'white',
        }}
      >
        {/* 装饰性背景元素 */}
        <Box
          sx={{
            position: 'absolute',
            top: '-20%',
            right: '-10%',
            width: '600px',
            height: '600px',
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(25,118,210,0.15) 0%, transparent 70%)',
            pointerEvents: 'none',
          }}
        />
        <Box
          sx={{
            position: 'absolute',
            bottom: '-30%',
            left: '-15%',
            width: '500px',
            height: '500px',
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(124,58,237,0.1) 0%, transparent 70%)',
            pointerEvents: 'none',
          }}
        />

        <Container maxWidth="lg">
          <Box
            sx={{
              display: 'flex',
              flexDirection: { xs: 'column', md: 'row' },
              alignItems: 'center',
              gap: 4,
              position: 'relative',
              zIndex: 1,
            }}
          >
            {/* 左侧文本 */}
            <Box sx={{ flex: 1, textAlign: { xs: 'center', md: 'left' } }}>
              <Chip
                icon={<Zap size={14} />}
                label="AI 驱动量化平台"
                size="small"
                sx={{
                  mb: 3,
                  bgcolor: 'rgba(25,118,210,0.15)',
                  color: '#90caf9',
                  border: '1px solid rgba(25,118,210,0.3)',
                  fontWeight: 500,
                }}
              />

              <Typography
                variant="h1"
                sx={{
                  fontSize: { xs: '2.25rem', sm: '2.75rem', md: '3.5rem' },
                  fontWeight: 800,
                  lineHeight: 1.15,
                  mb: 2,
                  letterSpacing: '-0.02em',
                }}
              >
                智能量化
                <Box component="span" sx={{ color: 'primary.light', display: 'inline' }}>
                  投资决策平台
                </Box>
              </Typography>

              <Typography
                variant="h5"
                sx={{
                  color: 'rgba(255,255,255,0.7)',
                  fontSize: { xs: '1rem', md: '1.2rem' },
                  fontWeight: 400,
                  lineHeight: 1.6,
                  mb: 1,
                  maxWidth: 540,
                  mx: { xs: 'auto', md: 0 },
                }}
              >
                基于先进机器学习的量化研究平台，提供因子挖掘、策略回测、
                模型训练与预测分析一体化解决方案。
              </Typography>

              <Typography
                variant="body1"
                sx={{
                  color: 'rgba(255,255,255,0.45)',
                  mb: 4,
                  maxWidth: 480,
                  mx: { xs: 'auto', md: 0 },
                }}
              >
                无需编程基础，快速构建和验证您的量化策略。
              </Typography>

              <Box
                sx={{
                  display: 'flex',
                  gap: 2,
                  justifyContent: { xs: 'center', md: 'flex-start' },
                  flexWrap: 'wrap',
                }}
              >
                <Button
                  variant="contained"
                  size="large"
                  onClick={() => router.push('/login')}
                  sx={{
                    px: 4,
                    py: 1.5,
                    fontSize: '1.05rem',
                    borderRadius: 2,
                    bgcolor: 'primary.main',
                    '&:hover': {
                      bgcolor: 'primary.dark',
                      transform: 'translateY(-2px)',
                      boxShadow: '0 8px 25px rgba(25,118,210,0.4)',
                    },
                    transition: 'all 0.2s ease',
                  }}
                  endIcon={<ArrowRight size={20} />}
                >
                  开始使用
                </Button>

                <Button
                  variant="outlined"
                  size="large"
                  onClick={() => router.push('/templates')}
                  sx={{
                    px: 4,
                    py: 1.5,
                    fontSize: '1.05rem',
                    borderRadius: 2,
                    borderColor: 'rgba(255,255,255,0.3)',
                    color: 'white',
                    '&:hover': {
                      borderColor: 'white',
                      bgcolor: 'rgba(255,255,255,0.08)',
                    },
                    transition: 'all 0.2s ease',
                  }}
                >
                  查看策略模板
                </Button>
              </Box>
            </Box>

            {/* 右侧装饰图 */}
            <Box
              sx={{
                flex: 1,
                display: { xs: 'none', md: 'flex' },
                justifyContent: 'center',
                alignItems: 'center',
                minHeight: 300,
              }}
            >
              <Box
                sx={{
                  width: 320,
                  height: 320,
                  borderRadius: 4,
                  background:
                    'linear-gradient(135deg, rgba(25,118,210,0.12), rgba(124,58,237,0.12))',
                  border: '1px solid rgba(255,255,255,0.08)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  position: 'relative',
                }}
              >
                <TrendingUp size={120} color="rgba(255,255,255,0.15)" />
                <Box
                  sx={{
                    position: 'absolute',
                    top: 16,
                    left: 16,
                    right: 16,
                    bottom: 16,
                    borderRadius: 3,
                    border: '1px dashed rgba(255,255,255,0.08)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <LineChart size={64} color="rgba(255,255,255,0.1)" />
                </Box>
              </Box>
            </Box>
          </Box>
        </Container>
      </Box>

      {/* 数据亮点 */}
      <Box
        sx={{
          py: 4,
          bgcolor: 'white',
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={3} justifyContent="center">
            {highlights.map(item => {
              const Icon = item.icon;
              return (
                <Grid size={{ xs: 6, sm: 3 }} key={item.label}>
                  <Box
                    sx={{
                      textAlign: 'center',
                      py: 1,
                    }}
                  >
                    <Icon size={28} style={{ color: '#1976d2', marginBottom: 8 }} />
                    <Typography
                      variant="h4"
                      sx={{ fontWeight: 700, color: 'text.primary', mb: 0.5 }}
                    >
                      {item.value}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {item.label}
                    </Typography>
                  </Box>
                </Grid>
              );
            })}
          </Grid>
        </Container>
      </Box>

      {/* 故事板块：我们把行业不说的那部分说出来了 */}
      <Box sx={{ py: { xs: 6, md: 10 }, bgcolor: '#f8fafc' }}>
        <Container maxWidth="lg">
          <Box sx={{ textAlign: 'center', mb: { xs: 4, md: 6 }, maxWidth: 760, mx: 'auto' }}>
            <Chip
              icon={<ShieldCheck size={14} />}
              label="我们先讲难听的那部分"
              size="small"
              sx={{ mb: 2, bgcolor: '#0f172a', color: 'white' }}
            />
            <Typography
              variant="h3"
              sx={{ fontWeight: 700, mb: 1.5, color: 'text.primary', lineHeight: 1.25 }}
            >
              量化平台最容易骗你的，不是策略，是口径
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
              同一个策略，回测夏普可以是 0.98，也可以是 0.82。差别不在策略，
              在于股票池、交易成本、以及「涨停那天你到底买不买得进」。
              我们把每一条口径都摊开写出来 —— 包括对我们自己不利的那些。
            </Typography>
          </Box>

          <Grid container spacing={3}>
            {honesties.map(item => (
              <Grid size={{ xs: 12, md: 4 }} key={item.title}>
                <Card
                  elevation={0}
                  sx={{
                    height: '100%',
                    border: 1,
                    borderColor: 'divider',
                    borderTop: `3px solid ${item.color}`,
                    borderRadius: 2,
                  }}
                >
                  <CardContent sx={{ p: 3 }}>
                    <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 2 }}>
                      <Box
                        sx={{
                          width: 36,
                          height: 36,
                          borderRadius: 1.5,
                          bgcolor: `${item.color}15`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                        }}
                      >
                        <item.icon size={18} style={{ color: item.color }} />
                      </Box>
                      <Typography variant="subtitle1" sx={{ fontWeight: 700, lineHeight: 1.3 }}>
                        {item.title}
                      </Typography>
                    </Stack>
                    <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.75, mb: 2 }}>
                      {item.body}
                    </Typography>
                    <Box
                      sx={{
                        bgcolor: `${item.color}0d`,
                        borderLeft: `3px solid ${item.color}`,
                        borderRadius: 1,
                        px: 1.5,
                        py: 1.2,
                      }}
                    >
                      <Typography variant="caption" sx={{ color: item.color, fontWeight: 700 }}>
                        {item.evidence}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>

          <Box
            sx={{
              mt: { xs: 4, md: 6 },
              p: { xs: 3, md: 4 },
              bgcolor: '#0f172a',
              borderRadius: 2,
              color: 'white',
            }}
          >
            <Grid container spacing={3} alignItems="center">
              <Grid size={{ xs: 12, md: 8 }}>
                <Typography variant="h5" sx={{ fontWeight: 700, mb: 1.5 }}>
                  所以我们给你什么
                </Typography>
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.78)', lineHeight: 1.85 }}>
                  不是一份漂亮的收益曲线，而是 4 条互相独立、每天更新、可直接照着下单的信号线
                  （42 因子等权 / LightGBM 元模型 / RSI 超跌 / 量价动量）。
                  每条线 100 万独立记账，每周调仓一次，成本按真实佣金 + 印花税 + 滑点扣。
                  组合亏钱了，我们跟你一起亏 —— 这也是我们敢把纸面账本公开的原因。
                </Typography>
              </Grid>
              <Grid size={{ xs: 12, md: 4 }}>
                <Stack spacing={1.5}>
                  <Button
                    variant="contained"
                    size="large"
                    onClick={() => router.push('/register')}
                    sx={{ bgcolor: '#1976d2', '&:hover': { bgcolor: '#1565c0' } }}
                  >
                    免费注册看今日信号
                  </Button>
                  <Button
                    variant="outlined"
                    size="large"
                    onClick={() => router.push('/pricing')}
                    sx={{ color: 'white', borderColor: 'rgba(255,255,255,0.4)' }}
                  >
                    看定价
                  </Button>
                </Stack>
              </Grid>
            </Grid>
          </Box>
        </Container>
      </Box>

      {/* 功能特性区域 */}
      <Box sx={{ py: { xs: 6, md: 10 } }}>
        <Container maxWidth="lg">
          <Box sx={{ textAlign: 'center', mb: { xs: 4, md: 6 } }}>
            <Typography
              variant="h3"
              sx={{
                fontWeight: 700,
                mb: 1.5,
                fontSize: { xs: '1.75rem', md: '2.5rem' },
              }}
            >
              核心功能
            </Typography>
            <Typography
              variant="body1"
              color="text.secondary"
              sx={{ maxWidth: 600, mx: 'auto', fontSize: '1.05rem' }}
            >
              为您提供从数据到策略的全流程量化研究工具链
            </Typography>
          </Box>

          <Grid container spacing={3}>
            {features.map(feature => {
              const Icon = feature.icon;
              return (
                <Grid size={{ xs: 12, sm: 6, md: 3 }} key={feature.title}>
                  <Card
                    sx={{
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        transform: 'translateY(-6px)',
                        boxShadow: '0 12px 40px rgba(0,0,0,0.12)',
                      },
                      border: '1px solid',
                      borderColor: 'divider',
                      borderRadius: 3,
                    }}
                  >
                    <CardContent
                      sx={{ p: 3, flexGrow: 1, display: 'flex', flexDirection: 'column' }}
                    >
                      <Box
                        sx={{
                          width: 56,
                          height: 56,
                          borderRadius: 2,
                          bgcolor: feature.bgColor,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          mb: 2,
                        }}
                      >
                        <Icon size={28} color={feature.color} />
                      </Box>

                      <Typography
                        variant="h6"
                        sx={{ fontWeight: 600, mb: 1, color: feature.color }}
                      >
                        {feature.title}
                      </Typography>

                      <Typography
                        variant="body2"
                        color="text.secondary"
                        sx={{ lineHeight: 1.7, flexGrow: 1 }}
                      >
                        {feature.description}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        </Container>
      </Box>

      {/* CTA 区域 */}
      <Box
        sx={{
          py: { xs: 6, md: 10 },
          background: 'linear-gradient(135deg, #0a1628 0%, #1a2332 100%)',
          color: 'white',
          textAlign: 'center',
        }}
      >
        <Container maxWidth="sm">
          <Typography
            variant="h3"
            sx={{
              fontWeight: 700,
              mb: 2,
              fontSize: { xs: '1.75rem', md: '2.25rem' },
            }}
          >
            准备好开始量化投资了吗？
          </Typography>
          <Typography
            variant="body1"
            sx={{ color: 'rgba(255,255,255,0.7)', mb: 4, lineHeight: 1.7 }}
          >
            立即注册，免费体验 AI 驱动的量化研究工具。 从策略构思到绩效分析，一站式完成。
          </Typography>
          <Button
            variant="contained"
            size="large"
            onClick={() => router.push('/login')}
            sx={{
              px: 5,
              py: 1.5,
              fontSize: '1.1rem',
              borderRadius: 2,
              bgcolor: 'primary.main',
              '&:hover': {
                bgcolor: 'primary.dark',
                transform: 'translateY(-2px)',
                boxShadow: '0 8px 25px rgba(25,118,210,0.4)',
              },
              transition: 'all 0.2s ease',
            }}
            endIcon={<ArrowRight size={20} />}
          >
            免费开始使用
          </Button>
        </Container>
      </Box>

      {/* 页脚 */}
      <Box
        sx={{
          py: 3,
          bgcolor: '#0d1117',
          color: 'rgba(255,255,255,0.5)',
          textAlign: 'center',
        }}
      >
        <Container maxWidth="lg">
          <Typography variant="body2">股票预测平台 © 2025 — 基于 AI 的智能投资决策系统</Typography>
        </Container>
      </Box>
    </Box>
  );
}
