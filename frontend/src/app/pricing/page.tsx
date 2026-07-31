'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Alert, Button, Card, Col, Divider, List, Row, Space, Tag, Typography, message } from 'antd';
import { Check, Crown, Building2, Sparkles } from 'lucide-react';
import { apiRequest } from '@/services/api';

const { Title, Paragraph, Text } = Typography;

type Plan = {
  id: 'free' | 'pro' | 'enterprise';
  name: string;
  price: string;
  description: string;
  accent: string;
  icon: typeof Crown;
  features: string[];
  limits: string[];
};

type CheckoutResponse = { checkout_url?: string; url?: string };

const plans: Plan[] = [
  {
    id: 'free',
    name: 'Free',
    price: '¥0 / 月',
    description: '适合初次探索量化研究的个人用户。',
    accent: '#64748b',
    icon: Sparkles,
    features: ['基础行情与因子分析', '标准回测报告', '社区策略模板'],
    limits: ['每月 50 次回测', '最多 20 个策略', '最多 3 个并发任务'],
  },
  {
    id: 'pro',
    name: 'Pro',
    price: '¥99 / 月',
    description: '为持续迭代策略的研究者提供更高配额。',
    accent: '#1976d2',
    icon: Crown,
    features: ['包含 Free 全部功能', '高级模型与参数优化', '优先任务队列与导出'],
    limits: ['每月 1,000 次回测', '最多 200 个策略', '最多 10 个并发任务'],
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    price: '联系销售',
    description: '为团队研究、权限和私有化部署而设计。',
    accent: '#0f766e',
    icon: Building2,
    features: ['包含 Pro 全部功能', '团队协作与权限管理', '专属支持与部署方案'],
    limits: ['按团队规模定制', '策略数量不限', '并发资源按需配置'],
  },
];

export default function PricingPage() {
  const router = useRouter();
  const [messageApi, contextHolder] = message.useMessage();
  const [currentTier, setCurrentTier] = useState<Plan['id'] | null>(null);
  const [loggedIn, setLoggedIn] = useState(false);
  const [loadingPlan, setLoadingPlan] = useState<Plan['id'] | null>(null);

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
      messageApi.info('Enterprise 套餐将由销售团队为你定制方案。');
      return;
    }

    setLoadingPlan(plan.id);
    try {
      const result = await apiRequest.post<CheckoutResponse>('/billing/checkout', {
        plan_id: plan.id,
        interval: 'monthly',
      });
      const checkoutUrl = result?.checkout_url || result?.url;
      if (checkoutUrl) {
        window.location.assign(checkoutUrl);
      } else {
        messageApi.success('结算会话已创建，请稍候刷新订阅状态。');
      }
    } catch {
      messageApi.error('暂时无法创建结算会话，请稍后重试。');
    } finally {
      setLoadingPlan(null);
    }
  };

  return (
    <div style={{ maxWidth: 1180, margin: '0 auto', padding: '12px 0 48px' }}>
      {contextHolder}
      <Space direction="vertical" size={4} style={{ marginBottom: 28 }}>
        <Text type="secondary">PLANS / 订阅方案</Text>
        <Title level={1} style={{ margin: 0 }}>选择适合你的研究节奏</Title>
        <Paragraph type="secondary" style={{ maxWidth: 660, margin: 0 }}>
          从小规模策略验证到团队级研究协作，按需升级资源配额。所有方案都包含安全的数据访问和可复现的回测报告。
        </Paragraph>
      </Space>

      {loggedIn && currentTier && (
        <Alert
          showIcon
          type="info"
          message={<>当前套餐：<strong>{currentTier.toUpperCase()}</strong></>}
          action={<Button type="link" onClick={() => router.push('/settings/subscription')}>管理订阅</Button>}
          style={{ marginBottom: 20 }}
        />
      )}

      <Row gutter={[16, 16]} align="stretch">
        {plans.map(plan => {
          const Icon = plan.icon;
          const isCurrent = currentTier === plan.id;
          return (
            <Col xs={24} md={8} key={plan.id} style={{ display: 'flex' }}>
              <Card
                title={<Space><Icon size={19} color={plan.accent} /><span>{plan.name}</span>{plan.id === 'pro' && <Tag color="blue">推荐</Tag>}</Space>}
                style={{ width: '100%', borderTop: `3px solid ${plan.accent}` }}
                styles={{ body: { display: 'flex', flexDirection: 'column', height: '100%' } }}
              >
                <Title level={2} style={{ marginTop: 0 }}>{plan.price}</Title>
                <Paragraph type="secondary" style={{ minHeight: 48 }}>{plan.description}</Paragraph>
                <Divider style={{ margin: '8px 0 16px' }} />
                <Space direction="vertical" size={10} style={{ flex: 1 }}>
                  {plan.features.map(feature => <Text key={feature}><Check size={15} color="#16a34a" style={{ verticalAlign: 'text-bottom', marginRight: 8 }} />{feature}</Text>)}
                </Space>
                <Divider style={{ margin: '20px 0 14px' }} />
                <Space direction="vertical" size={6} style={{ marginBottom: 18 }}>
                  {plan.limits.map(limit => <Text type="secondary" key={limit}>{limit}</Text>)}
                </Space>
                <Button
                  type={plan.id === 'pro' ? 'primary' : 'default'}
                  block
                  disabled={isCurrent}
                  loading={loadingPlan === plan.id}
                  onClick={() => choosePlan(plan)}
                >
                  {isCurrent ? '当前套餐' : loggedIn ? (plan.id === 'enterprise' ? '联系销售' : '升级方案') : '开始使用'}
                </Button>
              </Card>
            </Col>
          );
        })}
      </Row>
    </div>
  );
}
