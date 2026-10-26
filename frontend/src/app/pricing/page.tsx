'use client';

/**
 * 定价页（2026-10-25 重做）
 *
 * 改动逻辑：产品形态是**信号**（外部用户不跑回测，只要「今天买什么」），
 * 所以定价不该按「回测配额」分档，而该按**信号价值**分档：
 *   - 能看几条策略线
 *   - 推送到什么渠道、延迟多久
 *   - 有没有股数/金额的完整执行清单
 *   - 能不能看纸面账本（真实 track record）
 *
 * UI：原页是 antd（全站仅 5 个旧页面在用），改为 MUI —— 与首页故事板块、
 * 今日信号页视觉连贯（转化漏斗顶端）。register 仍是 antd，后续统一。
 *
 * 支付：原走 Stripe checkout（STRIPE_SECRET_KEY 为空 → 实际不可用）。
 * 改为引导到「钱包 → 充值」的人工对账订单制（2026-10-25 已落地）。
 */

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  Divider,
  Grid,
  Stack,
  Typography,
} from '@mui/material';
import {
  Check,
  Crown,
  Sparkles,
  Building2,
  Minus,
  Zap,
  LineChart,
  Shield,
  MessageSquare,
} from 'lucide-react';
import { apiRequest } from '@/services/api';

type Plan = {
  id: 'free' | 'pro' | 'enterprise';
  name: string;
  price: string;
  priceNote: string;
  description: string;
  accent: string;
  icon: typeof Crown;
  popular?: boolean;
  features: string[];
  notIncluded?: string[];
  cta: string;
};

const plans: Plan[] = [
  {
    id: 'free',
    name: '体验版',
    price: '¥0',
    priceNote: '永久免费',
    description: '先看看信号长什么样，再决定要不要为它付钱。我们不拦着你看。',
    accent: '#64748b',
    icon: Sparkles,
    cta: '免费注册',
    features: [
      '每日 1 条主策略线信号（25 因子等权复合分）',
      '延迟 1 个交易日',
      '今日信号页可查看完整清单',
      '策略逻辑与口径说明全公开',
      '每周纸面账本净值（延迟 1 周）',
    ],
    notIncluded: ['不含其余 3 条策略线', '不含股数与金额明细'],
  },
  {
    id: 'pro',
    name: '信号版',
    price: '¥199',
    priceNote: '每月',
    description: '全量 4 条独立信号线 + 可直接照着下单的执行清单 + 站内信号中心。',
    accent: '#1976d2',
    icon: Crown,
    popular: true,
    cta: '开通信号版',
    features: [
      '全部 4 条信号线（25 因子 / LightGBM / RSI 超跌 / 量价动量）',
      '站内信号中心 + 邮件通知，每个交易日 17:45 送达',
      '完整执行清单：代码、名称、价格、股数、金额、费用',
      '每周五收盘后调仓，实时持仓与成本',
      '全部纸面账本与历史交易记录',
      '每条线独立 100 万初始资金的表现对照',
    ],
    notIncluded: [],
  },
  {
    id: 'enterprise',
    name: '专业版',
    price: '¥599',
    priceNote: '每月',
    description: '给要按自己资金规模操作、并且要留痕的人。',
    accent: '#0f766e',
    icon: Building2,
    cta: '联系销售',
    features: [
      '包含信号版全部功能',
      '信号提前 30 分钟（收盘后即出，不等 17:45）',
      '自定义策略线：可指定你要跟的因子组合',
      '按你的资金规模换算股数与金额',
      '信号与成交记录导出（CSV / Excel）',
      '专属支持通道',
    ],
    notIncluded: [],
  },
];

const comparison = [
  { feature: '信号线数量', free: '1 条', pro: '4 条', ent: '4 条 + 自定义' },
  { feature: '通知渠道', free: '仅网页', pro: '网页 + 邮件', ent: '网页 + 邮件 + 短信' },
  { feature: '推送延迟', free: '1 个交易日', pro: '当日 17:45', ent: '当日收盘即出' },
  { feature: '股数 / 金额明细', free: false, pro: true, ent: true },
  { feature: '按你的资金规模换算', free: false, pro: false, ent: true },
  { feature: '纸面账本', free: '延迟 1 周', pro: '实时', ent: '实时' },
  { feature: '交易记录导出', free: false, pro: false, ent: true },
  { feature: '支持响应', free: '社区', pro: '邮件（1 个工作日）', ent: '专属通道' },
];

export default function PricingPage() {
  const router = useRouter();
  const [currentTier, setCurrentTier] = useState<Plan['id'] | null>(null);
  const [loggedIn, setLoggedIn] = useState(false);
  const [loadingPlan, setLoadingPlan] = useState<Plan['id'] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const token = window.localStorage.getItem('access_token');
    const storedUser = window.localStorage.getItem('user_info');
    setLoggedIn(Boolean(token));
    if (storedUser) {
      try {
        const user = JSON.parse(storedUser) as { subscription_tier?: Plan['id'] };
        setCurrentTier(user.subscription_tier || 'free');
      } catch {
        setCurrentTier('free');
      }
    }
  }, []);

  const choosePlan = async (plan: Plan) => {
    if (!loggedIn) {
      router.push('/login');
      return;
    }
    if (plan.id === currentTier) {
      return;
    }
    if (plan.id === 'enterprise') {
      router.push('/register');
      return;
    }

    setLoadingPlan(plan.id);
    setError(null);
    try {
      // Stripe 通道（STRIPE_SECRET_KEY 为空时不可用）→ 失败则回落到人工对账
      const result = await apiRequest.post<{ checkout_url?: string; url?: string }>(
        '/billing/checkout',
        { plan_id: plan.id, interval: 'monthly' }
      );
      const checkoutUrl = result?.checkout_url || result?.url;
      if (checkoutUrl) {
        window.location.assign(checkoutUrl);
        return;
      }
      setError('自动支付通道暂未开通，请到「钱包 → 充值」下单后联系客服确认。');
    } catch {
      setError('自动支付通道暂未开通，请到「钱包 → 充值」下单后联系客服确认。');
    } finally {
      setLoadingPlan(null);
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      {/* Header */}
      <Box sx={{ bgcolor: '#0f172a', color: 'white', py: { xs: 6, md: 9 } }}>
        <Container maxWidth="lg">
          <Box sx={{ textAlign: 'center', maxWidth: 720, mx: 'auto' }}>
            <Chip
              icon={<Zap size={14} />}
              label="按信号收费，不按回测次数收费"
              size="small"
              sx={{ mb: 2, bgcolor: 'rgba(255,255,255,0.12)', color: 'white' }}
            />
            <Typography variant="h3" sx={{ fontWeight: 700, mb: 2, lineHeight: 1.25 }}>
              你买的是信号，不是工具
            </Typography>
            <Typography variant="body1" sx={{ color: 'rgba(255,255,255,0.75)', lineHeight: 1.8 }}>
              工具谁都能装。我们卖的是每天收盘后那 20 分钟的判断 ——
              买哪几只、卖哪几只、买多少股、为什么，以及这些判断到目前为止赚没赚钱。
            </Typography>
          </Box>
        </Container>
      </Box>

      <Container maxWidth="lg" sx={{ py: { xs: 5, md: 8 } }}>
        {error && (
          <Alert severity="warning" sx={{ mb: 3 }} onClose={() => setError(null)}>
            {error}
            <Button size="small" sx={{ ml: 1.5 }} onClick={() => router.push('/wallet')}>
              去钱包充值
            </Button>
          </Alert>
        )}

        {loggedIn && currentTier && (
          <Alert
            severity="info"
            sx={{ mb: 3 }}
            action={
              <Button color="inherit" size="small" onClick={() => router.push('/settings/subscription')}>
                管理订阅
              </Button>
            }
          >
            当前套餐：
            <strong>
              {plans.find(p => p.id === currentTier)?.name || currentTier.toUpperCase()}
            </strong>
          </Alert>
        )}

        <Grid container spacing={3} alignItems="stretch">
          {plans.map(plan => {
            const Icon = plan.icon;
            const isCurrent = currentTier === plan.id;
            return (
              <Grid size={{ xs: 12, md: 4 }} key={plan.id}>
                <Card
                  elevation={0}
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    border: 2,
                    borderColor: plan.popular ? plan.accent : 'divider',
                    borderRadius: 2,
                    position: 'relative',
                  }}
                >
                  {plan.popular && (
                    <Chip
                      label="最受欢迎"
                      size="small"
                      sx={{
                        position: 'absolute',
                        top: -12,
                        left: '50%',
                        transform: 'translateX(-50%)',
                        bgcolor: plan.accent,
                        color: 'white',
                        fontWeight: 700,
                      }}
                    />
                  )}
                  <CardContent sx={{ p: 3.5, display: 'flex', flexDirection: 'column', flex: 1 }}>
                    <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 1.5 }}>
                      <Icon size={22} style={{ color: plan.accent }} />
                      <Typography variant="h6" sx={{ fontWeight: 700 }}>
                        {plan.name}
                      </Typography>
                    </Stack>

                    <Stack direction="row" alignItems="baseline" spacing={1} sx={{ mb: 0.5 }}>
                      <Typography variant="h4" sx={{ fontWeight: 700, color: plan.accent }}>
                        {plan.price}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {plan.priceNote}
                      </Typography>
                    </Stack>

                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ mb: 3, minHeight: 60, lineHeight: 1.7 }}
                    >
                      {plan.description}
                    </Typography>

                    <Divider sx={{ mb: 2 }} />

                    <Stack spacing={1.25} sx={{ mb: 3, flex: 1 }}>
                      {plan.features.map(f => (
                        <Stack direction="row" spacing={1} key={f} alignItems="flex-start">
                          <Check
                            size={16}
                            style={{ color: plan.accent, flexShrink: 0, marginTop: 3 }}
                          />
                          <Typography variant="body2" sx={{ lineHeight: 1.6 }}>
                            {f}
                          </Typography>
                        </Stack>
                      ))}
                      {plan.notIncluded?.map(f => (
                        <Stack direction="row" spacing={1} key={f} alignItems="flex-start">
                          <Minus
                            size={16}
                            style={{ color: 'text.disabled', flexShrink: 0, marginTop: 3 }}
                          />
                          <Typography variant="body2" color="text.disabled" sx={{ lineHeight: 1.6 }}>
                            {f}
                          </Typography>
                        </Stack>
                      ))}
                    </Stack>

                    <Button
                      fullWidth
                      size="large"
                      variant={plan.popular ? 'contained' : 'outlined'}
                      disabled={isCurrent || loadingPlan !== null}
                      onClick={() => choosePlan(plan)}
                      sx={
                        plan.popular
                          ? {
                              bgcolor: plan.accent,
                              '&:hover': { bgcolor: plan.accent, opacity: 0.9 },
                            }
                          : { borderColor: plan.accent, color: plan.accent }
                      }
                    >
                      {loadingPlan === plan.id ? (
                        <CircularProgress size={20} />
                      ) : isCurrent ? (
                        '当前套餐'
                      ) : !loggedIn ? (
                        '免费注册'
                      ) : (
                        plan.cta
                      )}
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>

        {/* 功能对照表 */}
        <Box sx={{ mt: { xs: 5, md: 8 } }}>
          <Typography variant="h5" sx={{ fontWeight: 700, mb: 1, textAlign: 'center' }}>
            逐项对照
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3, textAlign: 'center' }}>
            不藏功能，也不写「联系销售了解详情」
          </Typography>

          <Box sx={{ overflowX: 'auto' }}>
            <Box sx={{ minWidth: 640 }}>
              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: '1.4fr 1fr 1fr 1fr',
                  bgcolor: '#0f172a',
                  color: 'white',
                  px: 2,
                  py: 1.5,
                  borderRadius: '8px 8px 0 0',
                }}
              >
                {['功能', '体验版', '信号版', '专业版'].map((h, i) => (
                  <Typography
                    key={h}
                    variant="subtitle2"
                    sx={{ fontWeight: 700, textAlign: i === 0 ? 'left' : 'center' }}
                  >
                    {h}
                  </Typography>
                ))}
              </Box>

              {comparison.map((row, idx) => (
                <Box
                  key={row.feature}
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: '1.4fr 1fr 1fr 1fr',
                    px: 2,
                    py: 1.75,
                    bgcolor: idx % 2 === 0 ? 'background.paper' : 'rgba(15,23,42,0.02)',
                    borderBottom: 1,
                    borderColor: 'divider',
                  }}
                >
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {row.feature}
                  </Typography>
                  {[row.free, row.pro, row.ent].map((v, i) => (
                    <Box key={i} sx={{ display: 'flex', justifyContent: 'center' }}>
                      {typeof v === 'boolean' ? (
                        v ? (
                          <Check size={16} style={{ color: '#0f766e' }} />
                        ) : (
                          <Minus size={16} style={{ color: 'text.disabled' }} />
                        )
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          {v}
                        </Typography>
                      )}
                    </Box>
                  ))}
                </Box>
              ))}
            </Box>
          </Box>
        </Box>

        {/* 支付说明 */}
        <Alert severity="info" icon={<MessageSquare size={18} />} sx={{ mt: 4 }}>
          <Typography variant="body2" sx={{ fontWeight: 700, mb: 0.5 }}>
            关于付款方式
          </Typography>
          <Typography variant="body2" sx={{ lineHeight: 1.7 }}>
            目前支持人工对账：你在「钱包 → 充值」下单后按提示转账，把订单号发给我们，
            确认到账后即刻开通。Stripe 通道已预留，配置完成后会自动切换为自动支付。
            体验版不需要任何付款。
            <br />
            <br />
            充值到账后系统会**自动开通对应套餐**（free / pro / enterprise），
            不需要另外发邮件申请。
          </Typography>
        </Alert>

        {/* 信任背书 */}
        <Grid container spacing={3} sx={{ mt: 1 }}>
          {[
            {
              icon: LineChart,
              title: '纸面账本公开',
              body: '4 条线各自 100 万独立记账，每周更新净值，亏钱的月份照样放出来。',
            },
            {
              icon: Shield,
              title: '口径写清楚',
              body: '股票池、成本、涨停不可买、ST 剔除，全部在策略说明里写明，不藏着。',
            },
            {
              icon: Zap,
              title: '不夸大收益',
              body: '我们把 0.976 的夏普换成了能复现的 0.8193 —— 数字更小，但它是真的。',
            },
          ].map(item => (
            <Grid size={{ xs: 12, md: 4 }} key={item.title}>
              <Box
                sx={{
                  p: 2.5,
                  border: 1,
                  borderColor: 'divider',
                  borderRadius: 2,
                  height: '100%',
                }}
              >
                <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 1 }}>
                  <item.icon size={18} style={{ color: '#1976d2' }} />
                  <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                    {item.title}
                  </Typography>
                </Stack>
                <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                  {item.body}
                </Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
}
