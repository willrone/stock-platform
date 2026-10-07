'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Alert,
  Button,
  Card,
  Col,
  Divider,
  Empty,
  Progress,
  Row,
  Space,
  Statistic,
  Tag,
  Typography,
  message,
} from 'antd';
import { ArrowUpRight, CreditCard, ExternalLink, ShieldCheck } from 'lucide-react';
import { apiRequest } from '@/services/api';

const { Title, Paragraph, Text } = Typography;

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

export default function SubscriptionPage() {
  const router = useRouter();
  const [messageApi, contextHolder] = message.useMessage();
  const [tier, setTier] = useState('free');
  const [usage, setUsage] = useState<Usage>(defaultUsage);
  const [loading, setLoading] = useState(true);
  const [portalLoading, setPortalLoading] = useState(false);

  useEffect(() => {
    const token = window.localStorage.getItem('access_token');
    if (!token) {
      router.replace('/login');
      return;
    }

    const loadSubscription = async () => {
      try {
        const dashboard = await apiRequest.get<{ user?: UserInfo; usage?: Usage }>(
          '/dashboard/stats'
        );
        setTier(dashboard?.user?.subscription_tier || 'free');
        setUsage({ ...defaultUsage, ...dashboard?.usage });
        if (dashboard?.user) {
          window.localStorage.setItem('user_info', JSON.stringify(dashboard.user));
        }
      } catch {
        const storedUser = window.localStorage.getItem('user_info');
        if (storedUser) {
          try {
            const user = JSON.parse(storedUser) as UserInfo;
            setTier(user.subscription_tier || 'free');
          } catch {
            setTier('free');
          }
        }
      } finally {
        setLoading(false);
      }
    };

    void loadSubscription();
  }, [router]);

  const openBillingPortal = async () => {
    setPortalLoading(true);
    try {
      const result = await apiRequest.post<BillingResponse>('/billing/portal');
      const portalUrl = result?.portal_url || result?.url;
      if (portalUrl) {
        window.location.assign(portalUrl);
      } else {
        messageApi.success('订阅管理请求已提交。');
      }
    } catch {
      messageApi.error('暂时无法打开订阅管理，请稍后重试。');
    } finally {
      setPortalLoading(false);
    }
  };

  const usagePercent = (value: number, limit: number) =>
    Math.min(100, Math.round((value / Math.max(limit, 1)) * 100));

  if (loading) {
    return <div style={{ padding: 40, textAlign: 'center' }}>正在加载订阅信息…</div>;
  }

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto', padding: '12px 0 48px' }}>
      {contextHolder}
      <Space direction="vertical" size={4} style={{ marginBottom: 24 }}>
        <Text type="secondary">ACCOUNT / 订阅管理</Text>
        <Title level={1} style={{ margin: 0 }}>
          订阅与用量
        </Title>
        <Paragraph type="secondary" style={{ margin: 0 }}>
          查看当前方案、资源使用情况，并管理账单和自动续费。
        </Paragraph>
      </Space>

      <Row gutter={[16, 16]}>
        <Col xs={24} md={10}>
          <Card style={{ height: '100%' }}>
            <Space direction="vertical" size={18} style={{ width: '100%' }}>
              <Space align="start">
                <CreditCard size={22} color="#1976d2" />
                <div>
                  <Text type="secondary">当前套餐</Text>
                  <Title level={2} style={{ margin: '3px 0 0' }}>
                    {tier.toUpperCase()}
                  </Title>
                </div>
              </Space>
              <Tag color={tier === 'free' ? 'default' : 'blue'} style={{ width: 'fit-content' }}>
                {tier === 'free' ? '免费方案' : '订阅有效'}
              </Tag>
              <Text type="secondary">
                方案权益会在账单周期开始时自动刷新。升级后可立即获得更高资源配额。
              </Text>
              <Space wrap>
                <Button
                  type="primary"
                  icon={<ArrowUpRight size={16} />}
                  onClick={() => router.push('/pricing')}
                >
                  查看升级方案
                </Button>
                {tier !== 'free' && (
                  <Button
                    icon={<ExternalLink size={16} />}
                    loading={portalLoading}
                    onClick={openBillingPortal}
                  >
                    管理账单
                  </Button>
                )}
              </Space>
            </Space>
          </Card>
        </Col>
        <Col xs={24} md={14}>
          <Card
            title={
              <Space>
                <ShieldCheck size={18} color="#16a34a" />
                本月用量
              </Space>
            }
          >
            <Row gutter={[16, 24]}>
              <Col xs={24} sm={8}>
                <Statistic
                  title="回测次数"
                  value={usage.backtests_this_month}
                  suffix={`/ ${usage.monthly_backtest_limit}`}
                />
              </Col>
              <Col xs={24} sm={8}>
                <Statistic
                  title="策略数量"
                  value={usage.strategy_count}
                  suffix={`/ ${usage.max_strategies}`}
                />
              </Col>
              <Col xs={24} sm={8}>
                <Statistic
                  title="并发任务"
                  value={usage.active_tasks}
                  suffix={`/ ${usage.max_concurrent_tasks}`}
                />
              </Col>
            </Row>
            <Divider />
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <div>
                <Text>回测配额</Text>
                <Progress
                  percent={usagePercent(usage.backtests_this_month, usage.monthly_backtest_limit)}
                />
              </div>
              <div>
                <Text>策略配额</Text>
                <Progress
                  percent={usagePercent(usage.strategy_count, usage.max_strategies)}
                  status={
                    usagePercent(usage.strategy_count, usage.max_strategies) > 90
                      ? 'exception'
                      : 'normal'
                  }
                />
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      <Card style={{ marginTop: 16 }}>
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            tier === 'free' ? '当前为免费方案，暂无账单记录' : '账单记录将由支付服务同步'
          }
        >
          <Button type="link" onClick={openBillingPortal} disabled={tier === 'free'}>
            打开账单门户
          </Button>
        </Empty>
      </Card>

      {tier !== 'free' && (
        <Alert
          type="warning"
          showIcon
          style={{ marginTop: 16 }}
          message="取消订阅"
          description="取消操作将在支付门户中完成，当前计费周期结束前仍可继续使用已购买权益。"
          action={
            <Button danger onClick={openBillingPortal}>
              取消订阅
            </Button>
          }
        />
      )}
    </div>
  );
}
