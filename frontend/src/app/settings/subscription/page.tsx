'use client';

/**
 * 订阅与用量
 *
 * 【P0 · state】tier 兜底不再谎报 'free'。
 * 旧实现：`setTier(dashboard?.user?.subscription_tier || 'free')` —— 接口失败时
 * 静默回落成 free，付费用户被显示成 FREE、付费墙误判。
 * 现在 tier 初始为 null：加载失败 → 显示「无法确认套餐」+ 重试，
 * 绝不谎报档位。
 *
 * 【P1 · design】从 antd 迁移到 MUI，token 走 src/theme/tokens.ts。
 */

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  Grid,
  LinearProgress,
  Typography,
} from '@mui/material';
import { ArrowUpRight, CreditCard, ExternalLink, ShieldCheck } from 'lucide-react';

import { apiRequest } from '@/services/api';
import { useSnackbarStore } from '@/stores/useSnackbarStore';

type Usage = {
  backtests_this_month: number;
  monthly_backtest_limit: number;
  strategy_count: number;
  max_strategies: number;
  active_tasks: number;
  max_concurrent_tasks: number;
};

type UserInfo = { subscription_tier?: string; username?: string; email?: string };
type BillingResponse = { url?: string; portal_url?: string; checkout_url?: string };

const defaultUsage: Usage = {
  backtests_this_month: 0,
  monthly_backtest_limit: 50,
  strategy_count: 0,
  max_strategies: 20,
  active_tasks: 0,
  max_concurrent_tasks: 3,
};

const TIER_LABEL: Record<string, string> = {
  free: '体验版',
  pro: '信号版',
  enterprise: '专业版',
};

export default function SubscriptionPage() {
  const router = useRouter();
  const showSnackbar = useSnackbarStore();

  /** 【P0-6】null = 尚未确认，绝不用 'free' 冒充 */
  const [tier, setTier] = useState<string | null>(null);
  const [usage, setUsage] = useState<Usage>(defaultUsage);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [portalLoading, setPortalLoading] = useState(false);

  const loadSubscription = async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const dashboard = await apiRequest.get<{ user?: UserInfo; usage?: Usage }>(
        '/dashboard/stats'
      );
      const resolvedTier = dashboard?.user?.subscription_tier ?? null;
      setTier(resolvedTier);
      setUsage({ ...defaultUsage, ...dashboard?.usage });
      if (dashboard?.user) {
        window.localStorage.setItem('user_info', JSON.stringify(dashboard.user));
      }
    } catch (e: unknown) {
      // 兜底读本地缓存的登录信息（刷新后仍在），但**不谎报档位**
      const storedUser = window.localStorage.getItem('user_info');
      let cachedTier: string | null = null;
      if (storedUser) {
        try {
          const user = JSON.parse(storedUser) as UserInfo;
          cachedTier = user.subscription_tier ?? null;
        } catch {
          cachedTier = null;
        }
      }
      setTier(cachedTier);
      setLoadError(
        e instanceof Error ? e.message : '加载订阅信息失败，请稍后重试'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const token = window.localStorage.getItem('access_token');
    if (!token) {
      router.replace('/login');
      return;
    }
    void loadSubscription();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [router]);

  const openBillingPortal = async () => {
    setPortalLoading(true);
    try {
      const result = await apiRequest.post<BillingResponse>('/billing/portal');
      const portalUrl = result?.portal_url || result?.url;
      if (portalUrl) {
        window.location.assign(portalUrl);
      } else {
        showSnackbar.showSuccess('订阅管理请求已提交。');
      }
    } catch {
      showSnackbar.showError('暂时无法打开订阅管理，请稍后重试。');
    } finally {
      setPortalLoading(false);
    }
  };

  const usagePercent = (value: number, limit: number) =>
    Math.min(100, Math.round((value / Math.max(limit, 1)) * 100));

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 8 }}>
        <CircularProgress />
      </Box>
    );
  }

  const tierLabel = tier ? (TIER_LABEL[tier] ?? tier.toUpperCase()) : '未知';
  const isFree = tier === 'free';

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Box>
        <Typography variant="caption" color="text.secondary">
          ACCOUNT / 订阅管理
        </Typography>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600, mt: 0.5 }}>
          订阅与用量
        </Typography>
        <Typography variant="body2" color="text.secondary">
          查看当前方案、资源使用情况，并管理账单和自动续费。
        </Typography>
      </Box>

      {/* 【P0-6】加载失败：明确告知"无法确认套餐"，不谎报 FREE */}
      {loadError && (
        <Alert
          severity="warning"
          action={
            <Button color="inherit" size="small" onClick={() => void loadSubscription()}>
              重试
            </Button>
          }
        >
          订阅信息加载失败（{loadError}）。套餐档位可能不是最新状态，请重试确认。
        </Alert>
      )}

      <Grid container spacing={2}>
        <Grid size={{ xs: 12, md: 10 }}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5 }}>
                  <CreditCard size={22} aria-hidden style={{ color: '#1976d2' }} />
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      当前套餐
                    </Typography>
                    <Typography variant="h4" component="p" sx={{ fontWeight: 600, mt: 0.25 }}>
                      {tier ? tierLabel : '—'}
                    </Typography>
                  </Box>
                </Box>

                <Chip
                  label={
                    tier === null ? '档位未知' : isFree ? '免费方案' : '订阅有效'
                  }
                  color={tier === null ? 'warning' : isFree ? 'default' : 'primary'}
                  sx={{ width: 'fit-content' }}
                />

                <Typography variant="body2" color="text.secondary">
                  方案权益会在账单周期开始时自动刷新。升级后可立即获得更高资源配额。
                </Typography>

                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  <Button
                    variant="contained"
                    startIcon={<ArrowUpRight size={16} aria-hidden />}
                    onClick={() => router.push('/pricing')}
                    sx={{ minHeight: 44 }}
                  >
                    查看升级方案
                  </Button>
                  {!isFree && tier !== null && (
                    <Button
                      variant="outlined"
                      startIcon={<ExternalLink size={16} aria-hidden />}
                      disabled={portalLoading}
                      onClick={openBillingPortal}
                      sx={{ minHeight: 44 }}
                    >
                      管理账单
                    </Button>
                  )}
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, md: 14 }}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <ShieldCheck size={18} aria-hidden style={{ color: '#16a34a' }} />
                <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
                  本月用量
                </Typography>
              </Box>

              <Grid container spacing={2}>
                {[
                  {
                    label: '回测次数',
                    value: usage.backtests_this_month,
                    limit: usage.monthly_backtest_limit,
                  },
                  {
                    label: '策略数量',
                    value: usage.strategy_count,
                    limit: usage.max_strategies,
                  },
                  {
                    label: '并发任务',
                    value: usage.active_tasks,
                    limit: usage.max_concurrent_tasks,
                  },
                ].map(item => (
                  <Grid size={{ xs: 12, sm: 4 }} key={item.label}>
                    <Typography variant="body2" color="text.secondary">
                      {item.label}
                    </Typography>
                    <Typography variant="h5" component="p" sx={{ fontWeight: 600 }}>
                      {item.value}
                      <Typography
                        component="span"
                        variant="body2"
                        color="text.secondary"
                        sx={{ ml: 0.5 }}
                      >
                        / {item.limit}
                      </Typography>
                    </Typography>
                  </Grid>
                ))}
              </Grid>

              <Divider sx={{ my: 2 }} />

              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Box>
                  <Typography variant="body2" sx={{ mb: 0.5 }}>
                    回测配额
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={usagePercent(
                      usage.backtests_this_month,
                      usage.monthly_backtest_limit
                    )}
                    sx={{ height: 6, borderRadius: 3 }}
                  />
                </Box>
                <Box>
                  <Typography variant="body2" sx={{ mb: 0.5 }}>
                    策略配额
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={usagePercent(usage.strategy_count, usage.max_strategies)}
                    color={
                      usagePercent(usage.strategy_count, usage.max_strategies) > 90
                        ? 'error'
                        : 'primary'
                    }
                    sx={{ height: 6, borderRadius: 3 }}
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Card>
        <CardContent>
          <Box sx={{ textAlign: 'center', py: 2 }}>
            <Typography variant="body2" color="text.secondary">
              {isFree
                ? '当前为免费方案，暂无账单记录'
                : '账单记录将由支付服务同步'}
            </Typography>
            <Button
              onClick={openBillingPortal}
              disabled={isFree}
              sx={{ mt: 1, minHeight: 44 }}
            >
              打开账单门户
            </Button>
          </Box>
        </CardContent>
      </Card>

      {!isFree && tier !== null && (
        <Alert
          severity="warning"
          action={
            <Button color="error" variant="outlined" onClick={openBillingPortal}>
              取消订阅
            </Button>
          }
        >
          <Typography variant="body1" sx={{ fontWeight: 600 }}>
            取消订阅
          </Typography>
          取消操作将在支付门户中完成，当前计费周期结束前仍可继续使用已购买权益。
        </Alert>
      )}
    </Box>
  );
}
