'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { Alert, Button, Card, Form, Input, Typography } from 'antd';
import { ArrowLeft, KeyRound, CheckCircle } from 'lucide-react';

const { Title, Text } = Typography;
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function ResetPasswordPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams.get('token');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [requestLoading, setRequestLoading] = useState(false);
  const [requestSent, setRequestSent] = useState(false);
  const [requestError, setRequestError] = useState('');

  // 如果 URL 中有 token，显示重置密码表单；否则显示"发送重置邮件"表单
  const isResetMode = !!token;

  const handleRequestReset = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setRequestLoading(true);
    setRequestError('');
    const form = new FormData(e.currentTarget);
    try {
      const res = await fetch(`${API_BASE}/api/v1/auth/forgot-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: form.get('email') }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || '发送失败');
      }
      setRequestSent(true);
    } catch (err: any) {
      setRequestError(err.message || '发送失败，请稍后重试');
    } finally {
      setRequestLoading(false);
    }
  };

  const handleResetPassword = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    const form = new FormData(e.currentTarget);
    const password = form.get('password') as string;
    const confirmPassword = form.get('confirmPassword') as string;

    if (password !== confirmPassword) {
      setError('两次输入的密码不一致');
      setLoading(false);
      return;
    }

    if (password.length < 6) {
      setError('密码至少 6 位');
      setLoading(false);
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/api/v1/auth/reset-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, password }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || '重置失败');
      }
      setSuccess(true);
      setTimeout(() => router.push('/login'), 3000);
    } catch (err: any) {
      setError(err.message || '重置失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: '#f0f2f5',
      padding: '20px',
    }}>
      <Card style={{ width: '100%', maxWidth: 420, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
        <div style={{ textAlign: 'center', marginBottom: 24 }}>
          <KeyRound size={48} style={{ color: '#1890ff', marginBottom: 12 }} />
          <Title level={3} style={{ marginBottom: 4 }}>
            {isResetMode ? '重置密码' : '忘记密码'}
          </Title>
          <Text type="secondary">
            {isResetMode ? '请输入您的新密码' : '输入您的邮箱，我们将发送重置链接'}
          </Text>
        </div>

        {success ? (
          <div style={{ textAlign: 'center', padding: '20px 0' }}>
            <CheckCircle size={64} style={{ color: '#52c41a', marginBottom: 16 }} />
            <Title level={4} style={{ color: '#52c41a' }}>密码重置成功！</Title>
            <Text type="secondary">正在跳转到登录页面...</Text>
          </div>
        ) : isResetMode ? (
          <Form onFinish={handleResetPassword} layout="vertical">
            {error && <Alert message={error} type="error" showIcon style={{ marginBottom: 16 }} />}
            <Form.Item name="password" label="新密码" rules={[{ required: true, message: '请输入新密码' }, { min: 6, message: '密码至少 6 位' }]}>
              <Input.Password placeholder="至少 6 位" />
            </Form.Item>
            <Form.Item name="confirmPassword" label="确认密码" rules={[{ required: true, message: '请再次输入密码' }]}>
              <Input.Password placeholder="再次输入新密码" />
            </Form.Item>
            <Button type="primary" htmlType="submit" block loading={loading}>
              重置密码
            </Button>
          </Form>
        ) : requestSent ? (
          <Alert
            message="重置链接已发送"
            description="请查收您的邮箱，点击链接重置密码。如果没有收到邮件，请检查垃圾邮件文件夹。"
            type="success"
            showIcon
            style={{ marginBottom: 16 }}
          />
        ) : (
          <Form onFinish={handleRequestReset} layout="vertical">
            {requestError && <Alert message={requestError} type="error" showIcon style={{ marginBottom: 16 }} />}
            <Form.Item name="email" label="邮箱" rules={[{ required: true, type: 'email', message: '请输入有效邮箱' }]}>
              <Input placeholder="your@email.com" />
            </Form.Item>
            <Button type="primary" htmlType="submit" block loading={requestLoading}>
              发送重置链接
            </Button>
          </Form>
        )}

        <div style={{ textAlign: 'center', marginTop: 24 }}>
          <Link href="/login" style={{ color: '#1890ff' }}>
            <ArrowLeft size={14} style={{ marginRight: 4 }} />
            返回登录
          </Link>
        </div>
      </Card>
    </div>
  );
}
