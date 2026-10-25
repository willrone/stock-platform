'use client';

/**
 * 管理员 · 充值订单管理
 *
 * 2026-10-25 新增。背景：充值改为两段式订单制（用户下单 → 管理员确认到账
 * → 余额入账）后，管理员需要一个界面来确认到账 —— 这是真实收款链路
 * 缺失的最后一环。
 *
 * 配套后端接口：
 *   GET  /api/v1/admin/payment/orders?status=pending
 *   POST /api/v1/admin/payment/orders/{order_no}/confirm
 *   POST /api/v1/admin/payment/orders/{order_no}/cancel
 *   GET  /api/v1/commerce/payment/config
 *   POST /api/v1/admin/payment/config
 */

import React, { useCallback, useEffect, useState } from 'react';
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
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material';
import { RefreshCw, CheckCircle2, XCircle } from 'lucide-react';
import { apiRequest } from '@/services/api';

type Order = {
  id: string;
  order_no: string;
  user_id: string;
  amount_cents: number;
  status: 'pending' | 'paid' | 'cancelled' | 'refunded';
  payment_method: string;
  user_remark?: string | null;
  admin_remark?: string | null;
  created_at: string;
  confirmed_at?: string | null;
};

type PayConfig = {
  method_name: string;
  payee_account: string;
  qr_image_url: string;
  instructions: string;
  is_enabled: boolean;
};

const STATUS_META: Record<string, { label: string; color: 'warning' | 'success' | 'default' | 'error' }> = {
  pending: { label: '待确认', color: 'warning' },
  paid: { label: '已到账', color: 'success' },
  cancelled: { label: '已取消', color: 'default' },
  refunded: { label: '已退款', color: 'error' },
};

export default function AdminPaymentsPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [orders, setOrders] = useState<Order[]>([]);
  const [config, setConfig] = useState<PayConfig | null>(null);
  const [filter, setFilter] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [busyOrder, setBusyOrder] = useState<string>('');
  const [savingConfig, setSavingConfig] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const q = filter ? `?status=${filter}&limit=200` : '?limit=200';
      const res = await apiRequest.get<{ orders: Order[] }>(`/admin/payment/orders${q}`);
      setOrders(res.orders || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载订单失败');
    } finally {
      setLoading(false);
    }
  }, [filter]);

  const loadConfig = useCallback(async () => {
    try {
      const cfg = await apiRequest.get<PayConfig>('/commerce/payment/config');
      setConfig(cfg);
    } catch {
      /* 忽略：配置读不到时表单留空 */
    }
  }, []);

  useEffect(() => {
    const token = window.localStorage.getItem('access_token');
    if (!token) {
      router.replace('/login');
      return;
    }
    load();
    loadConfig();
  }, [router, load, loadConfig]);

  const confirm = async (orderNo: string) => {
    setBusyOrder(orderNo);
    setError(null);
    try {
      await apiRequest.post(`/admin/payment/orders/${orderNo}/confirm?remark=后台确认到账`);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : '确认失败');
    } finally {
      setBusyOrder('');
    }
  };

  const cancel = async (orderNo: string) => {
    setBusyOrder(orderNo);
    setError(null);
    try {
      await apiRequest.post(`/admin/payment/orders/${orderNo}/cancel?remark=管理员取消`);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : '取消失败');
    } finally {
      setBusyOrder('');
    }
  };

  const saveConfig = async () => {
    if (!config) return;
    setSavingConfig(true);
    setError(null);
    try {
      await apiRequest.post('/admin/payment/config', {
        method_name: config.method_name || '',
        payee_account: config.payee_account || '',
        qr_image_url: config.qr_image_url || '',
        instructions: config.instructions || '',
        is_enabled: Boolean(config.is_enabled),
      });
      await loadConfig();
    } catch (e) {
      setError(e instanceof Error ? e.message : '保存失败');
    } finally {
      setSavingConfig(false);
    }
  };

  const pendingCount = orders.filter((o) => o.status === 'pending').length;
  const pendingAmount = orders
    .filter((o) => o.status === 'pending')
    .reduce((s, o) => s + o.amount_cents, 0);

  return (
    <Box sx={{ p: 3 }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
        <Typography variant="h5">充值订单管理</Typography>
        <Button startIcon={<RefreshCw size={16} />} onClick={load} disabled={loading}>
          刷新
        </Button>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
        <Card sx={{ minWidth: 180 }}>
          <CardContent>
            <Typography variant="caption" color="text.secondary">
              待确认订单
            </Typography>
            <Typography variant="h5">{pendingCount}</Typography>
          </CardContent>
        </Card>
        <Card sx={{ minWidth: 180 }}>
          <CardContent>
            <Typography variant="caption" color="text.secondary">
              待确认金额
            </Typography>
            <Typography variant="h5">¥{(pendingAmount / 100).toFixed(2)}</Typography>
          </CardContent>
        </Card>
      </Stack>

      <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
        {['', 'pending', 'paid', 'cancelled'].map((s) => (
          <Chip
            key={s || 'all'}
            label={s === '' ? '全部' : STATUS_META[s]?.label || s}
            color={filter === s ? 'primary' : 'default'}
            variant={filter === s ? 'filled' : 'outlined'}
            onClick={() => setFilter(s)}
            sx={{ cursor: 'pointer' }}
          />
        ))}
      </Stack>

      <Card>
        <CardContent>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          ) : orders.length === 0 ? (
            <Typography color="text.secondary">暂无充值订单</Typography>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>订单号</TableCell>
                  <TableCell align="right">金额</TableCell>
                  <TableCell align="right">下单时间</TableCell>
                  <TableCell>状态</TableCell>
                  <TableCell>备注</TableCell>
                  <TableCell align="right">操作</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {orders.map((o) => (
                  <TableRow key={o.id}>
                    <TableCell sx={{ fontFamily: 'monospace' }}>{o.order_no}</TableCell>
                    <TableCell align="right">¥{(o.amount_cents / 100).toFixed(2)}</TableCell>
                    <TableCell align="right" sx={{ whiteSpace: 'nowrap' }}>
                      {o.created_at ? new Date(o.created_at).toLocaleString('zh-CN') : '—'}
                    </TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        color={STATUS_META[o.status]?.color || 'default'}
                        label={STATUS_META[o.status]?.label || o.status}
                      />
                    </TableCell>
                    <TableCell sx={{ fontSize: 12, color: 'text.secondary' }}>
                      {o.user_remark || o.admin_remark || '—'}
                    </TableCell>
                    <TableCell align="right">
                      {o.status === 'pending' ? (
                        <Stack direction="row" spacing={1} justifyContent="flex-end">
                          <Button
                            size="small"
                            variant="contained"
                            color="success"
                            startIcon={<CheckCircle2 size={16} />}
                            disabled={busyOrder === o.order_no}
                            onClick={() => confirm(o.order_no)}
                          >
                            确认到账
                          </Button>
                          <Button
                            size="small"
                            color="error"
                            startIcon={<XCircle size={16} />}
                            disabled={busyOrder === o.order_no}
                            onClick={() => cancel(o.order_no)}
                          >
                            取消
                          </Button>
                        </Stack>
                      ) : (
                        <Typography variant="caption" color="text.secondary">
                          {o.confirmed_at
                            ? new Date(o.confirmed_at).toLocaleString('zh-CN')
                            : '—'}
                        </Typography>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Divider sx={{ my: 3 }} />

      <Typography variant="h6" sx={{ mb: 1 }}>
        收款方式配置
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        没有接入自动支付渠道时，用户在充值页看到这里的收款码 / 账号，
        线下转账后把订单号报给你，你在上面点「确认到账」。
      </Typography>
      {config && (
        <Stack spacing={2} sx={{ maxWidth: 560 }}>
          <TextField
            label="收款方式名称"
            value={config.method_name}
            onChange={(e) => setConfig({ ...config, method_name: e.target.value })}
            placeholder="例如：微信转账"
            size="small"
          />
          <TextField
            label="收款账号"
            value={config.payee_account}
            onChange={(e) => setConfig({ ...config, payee_account: e.target.value })}
            placeholder="微信号 / 支付宝账号"
            size="small"
          />
          <TextField
            label="收款码图片 URL"
            value={config.qr_image_url}
            onChange={(e) => setConfig({ ...config, qr_image_url: e.target.value })}
            placeholder="/images/pay-qr.png 或外链"
            size="small"
          />
          <TextField
            label="支付说明"
            value={config.instructions}
            onChange={(e) => setConfig({ ...config, instructions: e.target.value })}
            multiline
            minRows={2}
            size="small"
          />
          <Stack direction="row" spacing={2} alignItems="center">
            <Button
              variant="contained"
              onClick={saveConfig}
              disabled={savingConfig}
            >
              保存收款配置
            </Button>
            <Button
              size="small"
              onClick={() => setConfig({ ...config, is_enabled: !config.is_enabled })}
            >
              {config.is_enabled ? '当前：已开启' : '当前：已关闭'}
            </Button>
          </Stack>
        </Stack>
      )}
    </Box>
  );
}
