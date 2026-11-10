'use client';

/**
 * 忘记密码 / 重置密码
 *
 * 【P0 · landing】旧实现两个模式完全不可用：`e.preventDefault is not a function`。
 * 根因：antd 的 `Form.onFinish` 回调收到的是**表单值对象**，不是 React 事件，
 * 所以 `e.preventDefault()` 直接抛 TypeError，用户点了没反应。
 *
 * 本页已迁移到 MUI，改用受控 state + 真实 <form> onSubmit，事件语义正确。
 *
 * 【P1 · design】从 antd 迁移到 MUI，token 走 src/theme/tokens.ts。
 * 【P1 · mobile】触摸目标 ≥44px；输入框 font-size ≥16px（iOS 聚焦不缩放）。
 */

import { useState } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
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
import { ArrowLeft, CheckCircle, KeyRound } from 'lucide-react';

import { apiRequest } from '@/services/api';

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

  // 受控表单值
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  // 如果 URL 中有 token，显示重置密码表单；否则显示"发送重置邮件"表单
  const isResetMode = Boolean(token);

  const handleRequestReset = async (e: React.FormEvent<HTMLFormElement>) => {
    // 这里是真正的 React 事件，preventDefault 存在
    e.preventDefault();
    if (!email.trim()) {
      setRequestError('请输入邮箱');
      return;
    }
    setRequestLoading(true);
    setRequestError('');
    try {
      await apiRequest.post('/auth/forgot-password', { email: email.trim() });
      setRequestSent(true);
    } catch (err: unknown) {
      setRequestError(err instanceof Error ? err.message : '发送失败，请稍后重试');
    } finally {
      setRequestLoading(false);
    }
  };

  const handleResetPassword = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError('');

    if (password !== confirmPassword) {
      setError('两次输入的密码不一致');
      return;
    }
    if (password.length < 6) {
      setError('密码至少 6 位');
      return;
    }

    setLoading(true);
    try {
      await apiRequest.post('/auth/reset-password', { token, password });
      setSuccess(true);
      setTimeout(() => router.push('/login'), 3000);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '重置失败，请稍后重试');
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
        bgcolor: 'background.default',
        p: 2.5,
      }}
    >
      <Card sx={{ width: '100%', maxWidth: 420 }}>
        <CardContent>
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <KeyRound size={44} aria-hidden style={{ color: 'primary.main' }} />
            <Typography variant="h5" component="h1" sx={{ mt: 1.5, mb: 0.5, fontWeight: 600 }}>
              {isResetMode ? '重置密码' : '忘记密码'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {isResetMode ? '请输入您的新密码' : '输入您的邮箱，我们将发送重置链接'}
            </Typography>
          </Box>

          {success ? (
            <Box sx={{ textAlign: 'center', py: 3 }}>
              <CheckCircle size={56} aria-hidden style={{ color: 'statusColors.success.main' }} />
              <Typography variant="h6" component="h2" sx={{ mt: 1.5, fontWeight: 600 }}>
                密码重置成功！
              </Typography>
              <Typography variant="body2" color="text.secondary">
                正在跳转到登录页面...
              </Typography>
            </Box>
          ) : isResetMode ? (
            <Box
              component="form"
              onSubmit={handleResetPassword}
              noValidate
              sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}
            >
              {error && <Alert severity="error">{error}</Alert>}
              <TextField
                label="新密码"
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="至少 6 位"
                required
                fullWidth
                slotProps={{ htmlInput: { minLength: 6, style: { fontSize: 16 } } }}
              />
              <TextField
                label="确认密码"
                type="password"
                value={confirmPassword}
                onChange={e => setConfirmPassword(e.target.value)}
                placeholder="再次输入新密码"
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
                重置密码
              </Button>
            </Box>
          ) : requestSent ? (
            <Alert severity="success" sx={{ mb: 2 }}>
              重置链接已发送。请查收您的邮箱，点击链接重置密码。如果没有收到邮件，请检查垃圾邮件文件夹。
            </Alert>
          ) : (
            <Box
              component="form"
              onSubmit={handleRequestReset}
              noValidate
              sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}
            >
              {requestError && <Alert severity="error">{requestError}</Alert>}
              <TextField
                label="邮箱"
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="your@email.com"
                required
                fullWidth
                slotProps={{ htmlInput: { style: { fontSize: 16 } } }}
              />
              <Button
                type="submit"
                variant="contained"
                fullWidth
                disabled={requestLoading}
                startIcon={requestLoading ? <CircularProgress size={16} /> : undefined}
                sx={{ minHeight: 44 }}
              >
                发送重置链接
              </Button>
            </Box>
          )}

          <Box sx={{ textAlign: 'center', mt: 3 }}>
            <Link
              href="/login"
              style={{
                color: 'inherit',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 4,
                minHeight: 44,
              }}
            >
              <ArrowLeft size={14} aria-hidden />
              返回登录
            </Link>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
