'use client';

/**
 * 注册
 *
 * 【P0 · state】统一走 apiRequest（自动兼容 `{success,data}` 信封与裸 payload），
 * 不再用裸 fetch —— 后端 /auth/register 返回形态变化时裸 fetch 会静默拿不到 token。
 *
 * 【P0 · state】用户字段归一走 store 的 normalizeUser，删掉 `as unknown as`，
 * 保证 subscription_tier / is_admin 在注册后就是对的（否则管理员入口不显示）。
 *
 * 【P1 · design】从 antd 迁移到 MUI。
 */

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  TextField,
  Typography,
} from '@mui/material';
import { ArrowLeft, UserPlus } from 'lucide-react';

import { apiRequest } from '@/services/api';
import { getAnonId } from '@/services/funnel';
import { normalizeUser, useAppStore } from '@/stores/useAppStore';

type AuthResponse = {
  access_token?: string;
  refresh_token?: string;
  user?: Record<string, unknown>;
};

export default function RegisterPage() {
  const router = useRouter();
  const setUser = useAppStore(state => state.setUser);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // 受控表单值
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  const submit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError('');

    if (!email.trim()) {
      setError('请输入邮箱');
      return;
    }
    if (username.trim().length < 2 || username.trim().length > 50) {
      setError('用户名长度为 2–50 个字符');
      return;
    }
    if (password.length < 6) {
      setError('密码至少 6 位');
      return;
    }
    if (password !== confirmPassword) {
      setError('两次输入的密码不一致');
      return;
    }

    setLoading(true);
    try {
      const data = await apiRequest.post<AuthResponse>('/auth/register', {
        email: email.trim(),
        username: username.trim(),
        password,
        // 漏斗埋点：带上匿名设备标识，后端才能把"注册前看了价格页"
        // 这类匿名行为归并到注册后的账号上。
        anon_id: getAnonId(),
      });

      if (!data?.access_token) {
        throw new Error('注册响应缺少 access_token');
      }

      localStorage.setItem('access_token', data.access_token);
      if (data.refresh_token) {
        localStorage.setItem('refresh_token', data.refresh_token);
      }
      if (data.user) {
        localStorage.setItem('user_info', JSON.stringify(data.user));
      }

      // 统一归一：保证 subscription_tier / is_admin 正确
      const normalized = normalizeUser(data.user ?? null);
      if (normalized) {
        setUser(normalized);
      }

      router.push('/dashboard');
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : '注册失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 460, mx: 'auto', pt: 3.5, px: 1.5 }}>
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mb: 3 }}>
            <Link
              href="/login"
              style={{
                color: 'inherit',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 0.5,
                minHeight: 44,
              }}
            >
              <ArrowLeft size={15} aria-hidden /> 返回登录
            </Link>
            <Typography
              variant="h4"
              component="h1"
              sx={{ mt: 1, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}
            >
              <UserPlus size={23} aria-hidden />
              创建账号
            </Typography>
            <Typography variant="body2" color="text.secondary">
              注册后即可开始创建策略和运行回测。
            </Typography>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Box
            component="form"
            onSubmit={submit}
            noValidate
            sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}
          >
            <TextField
              label="邮箱"
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="you@example.com"
              autoComplete="email"
              required
              fullWidth
              slotProps={{ htmlInput: { style: { fontSize: 16 } } }}
            />
            <TextField
              label="用户名"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="你的研究昵称"
              autoComplete="username"
              required
              fullWidth
              slotProps={{ htmlInput: { minLength: 2, maxLength: 50, style: { fontSize: 16 } } }}
            />
            <TextField
              label="密码"
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="至少 6 位"
              autoComplete="new-password"
              required
              fullWidth
              slotProps={{ htmlInput: { minLength: 6, style: { fontSize: 16 } } }}
            />
            <TextField
              label="确认密码"
              type="password"
              value={confirmPassword}
              onChange={e => setConfirmPassword(e.target.value)}
              placeholder="再次输入密码"
              autoComplete="new-password"
              required
              fullWidth
              slotProps={{ htmlInput: { minLength: 6, style: { fontSize: 16 } } }}
            />
            <Button
              type="submit"
              variant="contained"
              fullWidth
              disabled={loading}
              startIcon={loading ? <CircularProgress size={16} /> : undefined}
              sx={{ minHeight: 44 }}
            >
              创建账号
            </Button>
          </Box>

          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ display: 'block', textAlign: 'center', mt: 2.25 }}
          >
            已有账号？ <Link href="/login">立即登录</Link>
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
}
