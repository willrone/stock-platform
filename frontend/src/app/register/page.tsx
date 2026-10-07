'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Alert, Button, Card, Form, Input, Space, Typography } from 'antd';
import { ArrowLeft, UserPlus } from 'lucide-react';
import { useAppStore } from '@/stores/useAppStore';

const { Title, Paragraph, Text } = Typography;
const apiBase = process.env.NEXT_PUBLIC_API_URL || '';
const authBase = apiBase.endsWith('/api') ? `${apiBase}/v1/auth` : `${apiBase}/api/v1/auth`;

type RegisterValues = {
  email: string;
  username: string;
  password: string;
  confirmPassword: string;
};

type AuthResponse = {
  access_token: string;
  refresh_token?: string;
  user: Record<string, unknown>;
};

export default function RegisterPage() {
  const router = useRouter();
  const setUser = useAppStore(state => state.setUser);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async (values: RegisterValues) => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${authBase}/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: values.email,
          username: values.username,
          password: values.password,
        }),
      });
      const data = (await response.json()) as AuthResponse & { detail?: string };
      if (!response.ok) {
        throw new Error(data.detail || '注册失败，请检查输入信息');
      }
      localStorage.setItem('access_token', data.access_token);
      if (data.refresh_token) localStorage.setItem('refresh_token', data.refresh_token);
      localStorage.setItem('user_info', JSON.stringify(data.user));
      setUser(data.user as unknown as Parameters<typeof setUser>[0]);
      router.push('/dashboard');
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : '注册失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 460, margin: '28px auto', padding: '0 12px' }}>
      <Card>
        <Space direction="vertical" size={4} style={{ width: '100%', marginBottom: 24 }}>
          <Link href="/login">
            <ArrowLeft size={15} style={{ verticalAlign: 'text-bottom' }} /> 返回登录
          </Link>
          <Title level={2} style={{ margin: '12px 0 0' }}>
            <UserPlus size={23} style={{ marginRight: 8, verticalAlign: '-3px' }} />
            创建账号
          </Title>
          <Paragraph type="secondary" style={{ margin: 0 }}>
            注册后即可开始创建策略和运行回测。
          </Paragraph>
        </Space>
        {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }} />}
        <Form layout="vertical" onFinish={submit} requiredMark={false}>
          <Form.Item
            name="email"
            label="邮箱"
            rules={[{ required: true, type: 'email', message: '请输入有效邮箱' }]}
          >
            <Input placeholder="you@example.com" autoComplete="email" />
          </Form.Item>
          <Form.Item
            name="username"
            label="用户名"
            rules={[{ required: true, min: 2, max: 50, message: '用户名长度为 2–50 个字符' }]}
          >
            <Input placeholder="你的研究昵称" autoComplete="username" />
          </Form.Item>
          <Form.Item
            name="password"
            label="密码"
            rules={[{ required: true, min: 6, message: '密码至少 6 位' }]}
          >
            <Input.Password placeholder="至少 6 位" autoComplete="new-password" />
          </Form.Item>
          <Form.Item
            name="confirmPassword"
            label="确认密码"
            dependencies={['password']}
            rules={[
              { required: true, message: '请再次输入密码' },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  return !value || getFieldValue('password') === value
                    ? Promise.resolve()
                    : Promise.reject(new Error('两次输入的密码不一致'));
                },
              }),
            ]}
          >
            <Input.Password placeholder="再次输入密码" autoComplete="new-password" />
          </Form.Item>
          <Button type="primary" htmlType="submit" block loading={loading}>
            创建账号
          </Button>
        </Form>
        <Text type="secondary" style={{ display: 'block', textAlign: 'center', marginTop: 18 }}>
          已有账号？ <Link href="/login">立即登录</Link>
        </Text>
      </Card>
    </div>
  );
}
