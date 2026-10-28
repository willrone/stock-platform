'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { normalizeUser, useAppStore } from '@/stores/useAppStore';
import { apiRequest } from '@/services/api';
import {
  Box,
  Card,
  TextField,
  Button,
  Typography,
  Tabs,
  Tab,
  Alert,
  CircularProgress,
  Stack,
} from '@mui/material';

// 用相对路径，走 Next.js rewrites 代理到后端（服务端代理，远程客户端可用）
// 【P0 · state】所有请求统一走 apiRequest（内部用相对路径 /api/v1 前缀），
// 这里不再需要单独的 API_BASE 常量。

/** 登录/注册响应（apiRequest 已归一信封，这里直接取 payload） */
type AuthResponse = {
  access_token?: string;
  refresh_token?: string;
  user?: Record<string, unknown>;
};

export default function LoginPage() {
  const router = useRouter();
  const setUser = useAppStore(s => s.setUser);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState(0);
  const [error, setError] = useState('');

  const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    const form = new FormData(e.currentTarget);
    try {
      // 【P0 · state】走统一 apiRequest：后端 /auth/login 返回裸体或信封都能读
      const data = await apiRequest.post<AuthResponse>('/auth/login', {
        email: String(form.get('email') ?? ''),
        password: String(form.get('password') ?? ''),
      });
      if (!data?.access_token) {
        throw new Error('登录响应缺少 access_token');
      }
      localStorage.setItem('access_token', data.access_token);
      if (data.refresh_token) {
        localStorage.setItem('refresh_token', data.refresh_token);
      }
      if (data.user) localStorage.setItem('user_info', JSON.stringify(data.user));
      // 统一归一：subscription_tier / is_admin 由 normalizeUser 处理，
      // 不再手写映射（手写版本漏字段会让管理员入口不显示）
      const normalized = normalizeUser(data.user ?? null);
      if (normalized) setUser(normalized);
      router.push('/dashboard');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '登录失败');
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
      const data = await apiRequest.post<AuthResponse>('/auth/register', {
        email: String(form.get('email') ?? ''),
        username: String(form.get('username') ?? ''),
        password: String(form.get('password') ?? ''),
      });
      if (!data?.access_token) {
        throw new Error('注册响应缺少 access_token');
      }
      localStorage.setItem('access_token', data.access_token);
      if (data.refresh_token) {
        localStorage.setItem('refresh_token', data.refresh_token);
      }
      if (data.user) localStorage.setItem('user_info', JSON.stringify(data.user));
      const normalized = normalizeUser(data.user ?? null);
      if (normalized) setUser(normalized);
      router.push('/dashboard');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '注册失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: '#f0f2f5',
      }}
    >
      <Card sx={{ width: 420, p: 4, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
        <Typography variant="h5" textAlign="center" gutterBottom>
          📈 量化研究平台
        </Typography>
        <Typography variant="body2" textAlign="center" color="text.secondary" sx={{ mb: 3 }}>
          AI 驱动的量化回测与因子分析工具
        </Typography>

        <Tabs
          value={tab}
          onChange={(_, v) => {
            setTab(v);
            setError('');
          }}
          centered
          sx={{ mb: 3 }}
        >
          <Tab label="登录" />
          <Tab label="注册" />
        </Tabs>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {tab === 0 ? (
          <Box component="form" onSubmit={handleLogin}>
            <Stack spacing={2.5}>
              <TextField name="email" label="邮箱" type="email" required fullWidth size="small" />
              <TextField
                name="password"
                label="密码"
                type="password"
                required
                fullWidth
                size="small"
              />
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
              <TextField
                name="password"
                label="密码"
                type="password"
                required
                fullWidth
                size="small"
                inputProps={{ minLength: 6 }}
              />
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
