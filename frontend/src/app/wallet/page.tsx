'use client';

/**
 * 钱包中心
 *
 * 【P0 · landing】本页原有 4 个阻断级问题，全部已修：
 *  1. 「充值」按钮是空操作 —— 旧实现靠 `document.getElementById('deposit-form')?.click()`
 *     去点一个 `display:none` 的 antd Form 提交按钮，用户看不到任何反馈。
 *     现在改为受控 Dialog + 真实表单，金额输入 → 校验 → 下单 → 展示收款指引。
 *  2. 充值金额校验永远不通过 —— 旧 rules 同时要求 `type:'number'` 而 Input 传字符串，
 *     任何金额都提交不了。现在在提交前显式 parse + 范围校验，错误就地显示。
 *  3. 4 个接口因响应封装不匹配全部失败 —— 走统一 apiRequest（后端已统一信封）。
 *  4. 余额恒显示 0.00 —— 余额卡片区分「加载失败」与「真的是 0」。
 *
 * 【P1 · design】已从 antd 迁移到 MUI，token 走 src/theme/tokens.ts。
 * 【P1 · mobile】触摸目标 ≥44px；表格用 ResponsiveTable 支持横向滚动。
 */

import { useCallback, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  Grid,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material';
import {
  CreditCard,
  History,
  RefreshCw,
  TrendingDown,
  Wallet as WalletIcon,
} from 'lucide-react';

import { apiRequest } from '@/services/api';
import { ResponsiveTable } from '@/components/mobile/ResponsiveTable';
import { useSnackbarStore } from '@/stores/useSnackbarStore';

type Balance = {
  balance_cents: number;
  balance_yuan: number;
  credited_cents: number;
  total_spent_cents: number;
  updated_at: string;
};

type BillingRecord = {
  id: string;
  amount_cents: number;
  charge_status: string;
  payment_method?: string;
  remark?: string;
  charged_at?: string;
  created_at: string;
};

type UsageEvent = {
  event_type: string;
  count: number;
  total_cost_cents: number;
  total_cost_yuan: number;
};

type UsageSummary = {
  month: number;
  year: number;
  total_spent_cents: number;
  total_spent_yuan: number;
  events: Record<string, UsageEvent>;
};

type PaymentConfig = {
  method_name: string;
  payee_account: string;
  qr_image_url: string;
  instructions: string;
  is_enabled: boolean;
  min_deposit_cents?: number;
  max_deposit_cents?: number;
};

type PaymentOrder = {
  order_no: string;
  amount_cents: number;
  status: string;
  created_at: string;
};

type DepositResult = {
  order_no?: string;
  order_status?: string;
  amount_cents?: number;
  payee_account?: string;
  qr_image_url?: string;
  instructions?: string;
};

type Tone = 'default' | 'success' | 'warning' | 'error';

/** 订单状态 → 展示文案 + 语义色 */
const ORDER_STATUS_META: Record<string, { text: string; tone: Tone }> = {
  pending: { text: '待确认到账', tone: 'warning' },
  paid: { text: '已到账', tone: 'success' },
  cancelled: { text: '已取消', tone: 'default' },
  refunded: { text: '已退款', tone: 'error' },
};

/** 账单状态 → 展示文案 */
const BILLING_STATUS_META: Record<string, { text: string; tone: Tone }> = {
  charged: { text: '已扣费', tone: 'success' },
  pending: { text: '待缴', tone: 'warning' },
  failed: { text: '失败', tone: 'error' },
  refunded: { text: '已退款', tone: 'default' },
  credited: { text: '充值', tone: 'success' },
};

export default function WalletPage() {
  const router = useRouter();
  const showSnackbar = useSnackbarStore();

  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [balance, setBalance] = useState<Balance | null>(null);
  const [usageSummary, setUsageSummary] = useState<UsageSummary | null>(null);
  const [billingHistory, setBillingHistory] = useState<BillingRecord[]>([]);
  const [depositLoading, setDepositLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // 充值订单制（2026-10-25）：下单 → 展示收款指引 → 等待管理员确认到账
  const [payConfig, setPayConfig] = useState<PaymentConfig | null>(null);
  const [orders, setOrders] = useState<PaymentOrder[]>([]);
  const [pendingOrder, setPendingOrder] = useState<PaymentOrder | null>(null);
  const [orderModalOpen, setOrderModalOpen] = useState(false);

  // 充值表单（P0：改为真实受控表单，不再是隐藏的 antd Form）
  const [depositDialogOpen, setDepositDialogOpen] = useState(false);
  const [depositAmount, setDepositAmount] = useState('');
  const [depositError, setDepositError] = useState<string | null>(null);

  const minYuan = (payConfig?.min_deposit_cents ?? 1000) / 100;
  const maxYuan = (payConfig?.max_deposit_cents ?? 500000) / 100;

  const loadOrders = useCallback(async () => {
    try {
      const res = await apiRequest.get<{ orders: PaymentOrder[] }>('/commerce/payment/orders');
      setOrders(res?.orders ?? []);
    } catch {
      /* 订单列表失败不阻塞钱包主流程 */
    }
  }, []);

  const loadPayConfig = useCallback(async () => {
    try {
      const cfg = await apiRequest.get<PaymentConfig>('/commerce/payment/config');
      setPayConfig(cfg);
    } catch {
      /* 收款配置拿不到时降级为「请联系客服」 */
    }
  }, []);

  const loadData = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const [balanceRes, usageRes, historyRes] = await Promise.all([
        apiRequest.get<Balance>('/commerce/balance'),
        apiRequest.get<UsageSummary>('/commerce/usage/summary'),
        apiRequest.get<BillingRecord[]>('/commerce/billing/history?limit=20'),
      ]);
      setBalance(balanceRes);
      setUsageSummary(usageRes);
      setBillingHistory(Array.isArray(historyRes) ? historyRes : []);
      await Promise.all([loadOrders(), loadPayConfig()]);
    } catch (e: unknown) {
      // 【P0-1】加载失败要让用户看见，不能静默呈现 0.00 元
      const msg = e instanceof Error ? e.message : '加载钱包数据失败';
      setLoadError(msg);
      showSnackbar.showError(msg);
    } finally {
      setLoading(false);
    }
  }, [loadOrders, loadPayConfig, showSnackbar]);

  useEffect(() => {
    const token = window.localStorage.getItem('access_token');
    if (!token) {
      router.replace('/login');
      return;
    }
    void loadData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [router]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  /**
   * 提交充值订单
   *
   * 【P0-2】旧实现的校验规则是 `type:'number'`，而 antd Input 传的是字符串，
   * 导致任何金额都过不了校验。这里显式 parse + 数值范围校验。
   */
  const handleDeposit = async () => {
    setDepositError(null);

    const amount = Number.parseFloat(depositAmount);
    if (!Number.isFinite(amount) || amount <= 0) {
      setDepositError('请输入有效的充值金额');
      return;
    }
    if (amount < minYuan) {
      setDepositError(`充值金额不能低于 ¥${minYuan}`);
      return;
    }
    if (amount > maxYuan) {
      setDepositError(`充值金额不能高于 ¥${maxYuan}`);
      return;
    }

    setDepositLoading(true);
    try {
      const res = await apiRequest.post<DepositResult>('/commerce/deposit', {
        amount_cents: Math.round(amount * 100),
        remark: '用户充值',
      });
      setPendingOrder({
        order_no: res.order_no || '',
        amount_cents: res.amount_cents ?? Math.round(amount * 100),
        status: res.order_status || 'pending',
        created_at: new Date().toISOString(),
      });
      setOrderModalOpen(true);
      setDepositDialogOpen(false);
      setDepositAmount('');
      showSnackbar.showSuccess('订单已创建，请按指引付款后等待确认到账');
      await loadOrders();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : '下单失败';
      setDepositError(msg);
      showSnackbar.showError(msg);
    } finally {
      setDepositLoading(false);
    }
  };

  const formatEventName = (eventType: string) => {
    const names: Record<string, string> = {
      backtest_basic: '基础回测',
      backtest_realtime: '实时回测',
      backtest_advanced: '高级回测',
      api_call: 'API 调用',
      data_download: '数据下载',
      realtime_quote: '实时行情',
      model_training: '模型训练',
      optimization: '参数优化',
    };
    return names[eventType] || eventType;
  };

  if (loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 384,
          gap: 2,
        }}
      >
        <CircularProgress />
        <Typography variant="body2" color="text.secondary">
          加载钱包数据中...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
          钱包中心
        </Typography>
        <Button
          startIcon={<RefreshCw size={16} aria-hidden />}
          onClick={handleRefresh}
          disabled={refreshing}
          variant="outlined"
          sx={{ minHeight: 44 }}
        >
          刷新
        </Button>
      </Box>

      {/* 【P0-1】加载失败：显式错误条 + 重试，不显示假的 0.00 余额 */}
      {loadError && (
        <Alert
          severity="error"
          action={
            <Button color="inherit" size="small" onClick={handleRefresh}>
              重试
            </Button>
          }
        >
          钱包数据加载失败：{loadError}
        </Alert>
      )}

      {/* 余额概览 */}
      <Grid container spacing={2}>
        <Grid size={{ xs: 12, sm: 4 }}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                当前余额
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1, mt: 0.5 }}>
                <WalletIcon size={20} aria-hidden />
                <Typography variant="h4" component="p" sx={{ fontWeight: 600 }}>
                  {loadError ? '—' : `¥${(balance?.balance_yuan ?? 0).toFixed(2)}`}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={Math.min(100, (balance?.balance_cents ?? 0) / 100)}
                sx={{ mt: 1.5, height: 6, borderRadius: 3 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, sm: 4 }}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                本月累计支出
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1, mt: 0.5 }}>
                <TrendingDown size={20} aria-hidden />
                <Typography variant="h4" component="p" sx={{ fontWeight: 600 }}>
                  {loadError ? '—' : `¥${(usageSummary?.total_spent_yuan ?? 0).toFixed(2)}`}
                </Typography>
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                赠送额：{((balance?.credited_cents ?? 0) / 100).toFixed(2)} 元
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, sm: 4 }}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              <Typography variant="body2" color="text.secondary">
                账户状态
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                <Box
                  aria-hidden
                  sx={{
                    width: 10,
                    height: 10,
                    borderRadius: '50%',
                    bgcolor:
                      loadError || !balance || balance.balance_cents <= 0
                        ? 'statusColors.error.main'
                        : 'statusColors.success.main',
                  }}
                />
                <Typography variant="h4" component="p" sx={{ fontWeight: 600 }}>
                  {loadError
                    ? '—'
                    : balance && balance.balance_cents > 0
                      ? '可用'
                      : '余额不足'}
                </Typography>
              </Box>
              {/* 【P0-2】充值按钮现在打开真实表单对话框，不再是空操作 */}
              <Button
                variant="contained"
                fullWidth
                startIcon={<CreditCard size={16} aria-hidden />}
                onClick={() => {
                  setDepositError(null);
                  setDepositDialogOpen(true);
                }}
                sx={{ mt: 'auto', minHeight: 44 }}
              >
                充值
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* 用量统计 */}
      <Card>
        <CardContent>
          <Typography variant="h6" component="h2" sx={{ fontWeight: 600, mb: 2 }}>
            本月用量统计
          </Typography>
          {usageSummary && Object.keys(usageSummary.events ?? {}).length > 0 ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              {Object.entries(usageSummary.events).map(([type, data]) => (
                <Box
                  key={type}
                  sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1.5 }}
                >
                  <Box
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      mb: 1,
                    }}
                  >
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {formatEventName(type)}
                    </Typography>
                    <Chip
                      size="small"
                      label={`${data.count} 次 / ¥${data.total_cost_yuan.toFixed(2)}`}
                    />
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min(100, (data.count / 1000) * 100)}
                    sx={{ height: 6, borderRadius: 3 }}
                  />
                </Box>
              ))}
            </Box>
          ) : (
            <Typography variant="body2" color="text.secondary">
              本月暂无用量记录
            </Typography>
          )}
        </CardContent>
      </Card>

      {/* 账单历史 */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <History size={18} aria-hidden />
            <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
              账单记录
            </Typography>
          </Box>
          {billingHistory.length > 0 ? (
            <ResponsiveTable minWidth={640}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>时间</TableCell>
                    <TableCell>类型</TableCell>
                    <TableCell>方式</TableCell>
                    <TableCell align="right">金额</TableCell>
                    <TableCell>状态</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {billingHistory.map(record => {
                    const meta = BILLING_STATUS_META[record.charge_status];
                    return (
                      <TableRow key={record.id}>
                        <TableCell>
                          <Typography variant="caption">
                            {record.charged_at || record.created_at
                              ? new Date(record.charged_at || record.created_at).toLocaleString('zh-CN')
                              : '—'}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="caption">{record.remark || '—'}</Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="caption">{record.payment_method || '—'}</Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="caption">
                            ¥{(record.amount_cents / 100).toFixed(2)}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip
                            size="small"
                            label={meta?.text ?? record.charge_status}
                            color={
                              meta?.tone === 'default'
                                ? 'default'
                                : meta?.tone === 'success'
                                  ? 'success'
                                  : meta?.tone === 'warning'
                                    ? 'warning'
                                    : meta?.tone === 'error'
                                      ? 'error'
                                      : 'default'
                            }
                          />
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </ResponsiveTable>
          ) : (
            <Typography variant="body2" color="text.secondary">
              暂无账单记录
            </Typography>
          )}
        </CardContent>
      </Card>

      {/* 充值订单记录 */}
      {orders.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" component="h2" sx={{ fontWeight: 600, mb: 2 }}>
              充值订单
            </Typography>
            <ResponsiveTable minWidth={560}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>订单号</TableCell>
                    <TableCell align="right">金额</TableCell>
                    <TableCell align="right">下单时间</TableCell>
                    <TableCell>状态</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {orders.map(order => {
                    const meta = ORDER_STATUS_META[order.status];
                    return (
                      <TableRow key={order.order_no}>
                        <TableCell>
                          <Typography variant="caption">{order.order_no}</Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="caption">
                            ¥{(order.amount_cents / 100).toFixed(2)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="caption">
                            {order.created_at
                              ? new Date(order.created_at).toLocaleString('zh-CN')
                              : '—'}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip
                            size="small"
                            label={meta?.text ?? order.status}
                            color={
                              meta?.tone === 'success'
                                ? 'success'
                                : meta?.tone === 'warning'
                                  ? 'warning'
                                  : meta?.tone === 'error'
                                    ? 'error'
                                    : 'default'
                            }
                          />
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </ResponsiveTable>
          </CardContent>
        </Card>
      )}

      {/* 充值表单对话框（P0-2：真实可见的表单，校验就地提示） */}
      <Dialog
        open={depositDialogOpen}
        onClose={() => setDepositDialogOpen(false)}
        fullWidth
        maxWidth="xs"
        aria-labelledby="deposit-dialog-title"
      >
        <DialogTitle id="deposit-dialog-title">充值</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 1 }}>
            <Typography variant="body2" color="text.secondary">
              可充值范围 ¥{minYuan} ~ ¥{maxYuan}。下单后请按收款指引付款，管理员确认到账后余额自动更新。
            </Typography>
            <TextField
              label="充值金额（元）"
              type="number"
              value={depositAmount}
              onChange={e => setDepositAmount(e.target.value)}
              error={Boolean(depositError)}
              helperText={depositError ?? ' '}
              slotProps={{
                htmlInput: {
                  min: minYuan,
                  max: maxYuan,
                  step: 10,
                  // 【P1 · mobile】iOS 聚焦 <16px 会自动放大页面
                  style: { fontSize: 16 },
                },
              }}
              autoFocus
              fullWidth
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDepositDialogOpen(false)} sx={{ minHeight: 44 }}>
            取消
          </Button>
          <Button
            variant="contained"
            onClick={handleDeposit}
            disabled={depositLoading}
            startIcon={
              depositLoading ? (
                <CircularProgress size={16} />
              ) : (
                <CreditCard size={16} aria-hidden />
              )
            }
            sx={{ minHeight: 44 }}
          >
            创建充值订单
          </Button>
        </DialogActions>
      </Dialog>

      {/* 订单详情弹窗：展示收款指引 */}
      <Dialog
        open={orderModalOpen}
        onClose={() => setOrderModalOpen(false)}
        fullWidth
        maxWidth="sm"
        aria-labelledby="order-dialog-title"
      >
        <DialogTitle id="order-dialog-title">充值订单</DialogTitle>
        <DialogContent>
          {pendingOrder && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Alert severity={pendingOrder.status === 'paid' ? 'success' : 'info'}>
                {pendingOrder.status === 'paid'
                  ? '该订单已确认到账'
                  : '请按下方方式付款，客服确认后余额自动到账'}
              </Alert>

              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {[
                  { label: '订单号', value: pendingOrder.order_no },
                  {
                    label: '充值金额',
                    value: `¥${(pendingOrder.amount_cents / 100).toFixed(2)}`,
                  },
                  {
                    label: '状态',
                    value: ORDER_STATUS_META[pendingOrder.status]?.text ?? pendingOrder.status,
                  },
                  { label: '收款方式', value: payConfig?.method_name || '请联系客服' },
                  { label: '收款账号', value: payConfig?.payee_account || '—' },
                ].map(row => (
                  <Box
                    key={row.label}
                    sx={{ display: 'flex', gap: 1, borderBottom: 1, borderColor: 'divider', pb: 1 }}
                  >
                    <Typography variant="body2" color="text.secondary" sx={{ minWidth: 84 }}>
                      {row.label}
                    </Typography>
                    <Typography variant="body2" sx={{ wordBreak: 'break-all' }}>
                      {row.value}
                    </Typography>
                  </Box>
                ))}
              </Box>

              {payConfig?.qr_image_url && (
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                    扫码支付
                  </Typography>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={payConfig.qr_image_url}
                    alt="收款码"
                    style={{ maxWidth: 220, width: '100%', borderRadius: 4 }}
                  />
                </Box>
              )}

              {payConfig?.instructions && (
                <Alert severity="warning">{payConfig.instructions}</Alert>
              )}

              <Divider />
              <Button
                variant="contained"
                onClick={async () => {
                  await loadOrders();
                  const found = orders.find(o => o.order_no === pendingOrder?.order_no);
                  if (found?.status === 'paid') {
                    showSnackbar.showSuccess('已确认到账，余额已更新');
                    setOrderModalOpen(false);
                    await loadData();
                  } else {
                    showSnackbar.showInfo('尚未确认到账，请稍后刷新或联系客服');
                  }
                }}
                sx={{ minHeight: 44 }}
              >
                刷新状态
              </Button>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOrderModalOpen(false)} sx={{ minHeight: 44 }}>
            关闭
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
