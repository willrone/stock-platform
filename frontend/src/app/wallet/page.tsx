'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Alert, Button, Card, Col, Form, Input, message, Progress, Row, Statistic, Tag, Typography } from 'antd';
import { RefreshCw, CreditCard, Wallet, TrendingDown, History } from 'lucide-react';
import { apiRequest } from '@/services/api';

const { Title, Text, Paragraph } = Typography;

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

export default function WalletPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [balance, setBalance] = useState<Balance | null>(null);
  const [usageSummary, setUsageSummary] = useState<UsageSummary | null>(null);
  const [billingHistory, setBillingHistory] = useState<BillingRecord[]>([]);
  const [depositLoading, setDepositLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    const token = window.localStorage.getItem('access_token');
    if (!token) {
      router.replace('/login');
      return;
    }
    loadData();
  }, [router]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [balanceRes, usageRes, historyRes] = await Promise.all([
        apiRequest.get<Balance>('/commerce/balance'),
        apiRequest.get<UsageSummary>('/commerce/usage/summary'),
        apiRequest.get<BillingRecord[]>('/commerce/billing/history?limit=20'),
      ]);
      setBalance(balanceRes);
      setUsageSummary(usageRes);
      setBillingHistory(historyRes);
    } catch (e) {
      message.error('加载数据失败');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  const handleDeposit = async (values: { amount: number }) => {
    setDepositLoading(true);
    try {
      await apiRequest.post('/commerce/deposit', {
        amount_cents: values.amount * 100,
        remark: '用户充值',
      });
      message.success('充值成功！');
      await loadData();
    } catch (e) {
      message.error('充值失败');
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

  const getStatusTag = (status: string) => {
    const colors: Record<string, 'default' | 'processing' | 'success' | 'warning' | 'error'> = {
      charged: 'success',
      pending: 'warning',
      failed: 'error',
      refunded: 'default',
      credited: 'processing',
    };
    const labels: Record<string, string> = {
      charged: '已扣费',
      pending: '待缴',
      failed: '失败',
      refunded: '已退款',
      credited: '充值',
    };
    return <Tag color={colors[status]}>{labels[status] || status}</Tag>;
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">加载钱包数据中...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-4">
      <div className="flex justify-between items-center mb-6">
        <Title level={2}>钱包中心</Title>
        <Button
          icon={<RefreshCw size={14} />}
          onClick={handleRefresh}
          loading={refreshing}
          ghost
        >
          刷新
        </Button>
      </div>

      {/* 余额概览 */}
      <Row gutter={[24, 16]} className="mb-6">
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="当前余额"
              value={balance?.balance_yuan || 0}
              precision={2}
              suffix="元"
              prefix={<Wallet size={20} />}
            />
            <Progress
              percent={balance ? (balance.balance_cents / 100) / 100 * 100 : 0}
              size="small"
              style={{ marginTop: 8 }}
              strokeColor="#1890ff"
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="本月累计支出"
              value={usageSummary?.total_spent_yuan || 0}
              precision={2}
              suffix="元"
              prefix={<TrendingDown size={20} color="#52c41a" />}
            />
            <Text type="secondary" className="text-xs block mt-2">
              赠送额：{(balance?.credited_cents || 0) / 100} 元
            </Text>
          </Card>
        </Col>
        
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="账户状态"
              value={balance && balance.balance_cents > 0 ? '可用' : '余额不足'}
              prefix={
                <div
                  className={`w-2 h-2 rounded-full mr-2 ${
                    balance && balance.balance_cents > 0 ? 'bg-green-500' : 'bg-red-500'
                  }`}
                />
              }
            />
            <Button
              type="primary"
              block
              icon={<CreditCard size={16} />}
              onClick={() => document.getElementById('deposit-form')?.click()}
              style={{ marginTop: 12 }}
            >
              充值
            </Button>
          </Card>
        </Col>
      </Row>

      {/* 用量统计 */}
      <Card title="本月用量统计" className="mb-6">
        {usageSummary ? (
          <div className="space-y-4">
            {Object.entries(usageSummary.events).map(([type, data]) => (
              <div key={type} className="border rounded p-3">
                <div className="flex justify-between items-center mb-2">
                  <Text strong>{formatEventName(type)}</Text>
                  <Tag>{data.count} 次 / ¥{data.total_cost_yuan.toFixed(2)}</Tag>
                </div>
                <Progress
                  percent={Math.min(100, (data.count / 1000) * 100)}
                  size="small"
                  strokeColor="#52c41a"
                />
              </div>
            ))}
          </div>
        ) : (
          <Text type="secondary">暂无用量数据</Text>
        )}
      </Card>

      {/* 账单历史 */}
      <Card title="账单历史" extra={<Button type="link" onClick={() => router.push('/commerce/billing/history')}>查看全部</Button>}>
        {billingHistory.length > 0 ? (
          <div className="space-y-3">
            {billingHistory.slice(0, 5).map((record) => (
              <div key={record.id} className="border rounded p-3 flex justify-between items-center">
                <div>
                  <Text strong>{record.amount_cents / 100} 元</Text>
                  <br />
                  <Text type="secondary" className="text-xs">
                    {record.remark || '账单'} • {new Date(record.created_at).toLocaleDateString()}
                  </Text>
                </div>
                {getStatusTag(record.charge_status)}
              </div>
            ))}
            {billingHistory.length > 5 && (
              <Button type="link" onClick={() => router.push('/commerce/billing/history')}>
                查看更多
              </Button>
            )}
          </div>
        ) : (
          <Text type="secondary">暂无账单记录</Text>
        )}
      </Card>

      {/* 充值弹窗 */}
      <Form id="deposit-form" onFinish={handleDeposit} layout="vertical" style={{ display: 'none' }}>
        <Form.Item
          name="amount"
          label="充值金额（元）"
          rules={[{ required: true, message: '请输入充值金额' }]}
        >
          <Input type="number" min={10} step={10} placeholder="例如：100" addonAfter="元" />
        </Form.Item>
        <Form.Item>
          <Button type="primary" htmlType="submit" loading={depositLoading} icon={<CreditCard size={16} />}>
            确认充值
          </Button>
        </Form.Item>
      </Form>
    </div>
  );
}