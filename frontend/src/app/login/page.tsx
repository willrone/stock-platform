'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAppStore } from '@/stores/useAppStore';
import {
  Box, Card, TextField, Button, Typography, Tabs, Tab, Alert,
  CircularProgress, Stack
} from '@mui/material';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function LoginPage() {
  const router = useRouter();
  const setUser = useAppStore((s) => s.setUser);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState(0);
  const [error, setError] = useState('');

  const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    const form = new FormData(e.currentTarget);
    try {
      const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: form.get('email'),
          password: form.get('password'),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '登录失败');
      localStorage.setItem('access_token', data.access_token);
      if (data.refresh_token) {
        localStorage.setItem('refresh_token', data.refresh_token);
      }
      localStorage.setItem('user_info', JSON.stringify(data.user));
      setUser(data.user);
      router.push('/dashboard');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    const form = new FormData(e.currentTarget);
    try {
      const res = await fetch(`${API_BASE}/api/v1/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: form.get('email'),
          username: form.get('username'),
          password: form.get('password'),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '注册失败');
      localStorage.setItem('access_token', data.access_token);
      if (data.refresh_token) {
        localStorage.setItem('refresh_token', data.refresh_token);
      }
      localStorage.setItem('user_info', JSON.stringify(data.user));
      setUser(data.user);
      router.push('/dashboard');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      bgcolor: '#f0f2f5',
    }}>
      <Card sx={{ width: 420, p: 4, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
        <Typography variant="h5" textAlign="center" gutterBottom>
          📈 量化研究平台
        </Typography>
        <Typography variant="body2" textAlign="center" color="text.secondary" sx={{ mb: 3 }}>
          AI 驱动的量化回测与因子分析工具
        </Typography>

        <Tabs value={tab} onChange={(_, v) => { setTab(v); setError(''); }} centered sx={{ mb: 3 }}>
          <Tab label="登录" />
          <Tab label="注册" />
        </Tabs>

        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

        {tab === 0 ? (
          <Box component="form" onSubmit={handleLogin}>
            <Stack spacing={2.5}>
              <TextField name="email" label="邮箱" type="email" required fullWidth size="small" />
              <TextField name="password" label="密码" type="password" required fullWidth size="small" />
              <Button type="submit" variant="contained" fullWidth disabled={loading}>
                {loading ? <CircularProgress size={24} /> : '登录'}
              </Button>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 2 }}>
                <Link href="/reset-password" style={{ color: '#1976d2', fontSize: 14 }}>
                  忘记密码？
                </Link>
                <Link href="/register" style={{ color: '#1976d2', fontSize: 14 }}>
                  注册新账号
                </Link>
              </Box>
            </Stack>
          </Box>
        ) : (
          <Box component="form" onSubmit={handleRegister}>
            <Stack spacing={2.5}>
              <TextField name="email" label="邮箱" type="email" required fullWidth size="small" />
              <TextField name="username" label="用户名" required fullWidth size="small" />
              <TextField name="password" label="密码" type="password" required fullWidth size="small" inputProps={{ minLength: 6 }} />
              <Button type="submit" variant="contained" fullWidth disabled={loading}>
                {loading ? <CircularProgress size={24} /> : '注册'}
              </Button>
            </Stack>
          </Box>
        )}
      </Card>
    </Box>
  );
}
