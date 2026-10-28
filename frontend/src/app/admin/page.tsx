'use client';

/**
 * 管理后台索引页
 *
 * 【P1 · layout】此前 /admin 下只有 /admin/payments，管理员直接访问 /admin 会 404。
 * 本页汇总所有管理入口，作为管理侧的唯一索引。
 */

import React from 'react';
import { useRouter } from 'next/navigation';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
} from '@mui/material';
import { ChevronRight, CreditCard, ShieldCheck } from 'lucide-react';

import { useAppStore } from '@/stores/useAppStore';

export default function AdminIndexPage() {
  const router = useRouter();
  const user = useAppStore(state => state.user);
  const isAdmin = Boolean(user?.is_admin);

  const entries = [
    {
      key: '/admin/payments',
      label: '充值订单',
      description: '审核用户充值订单、确认到账并升级套餐',
      icon: CreditCard,
      adminOnly: true,
    },
  ];

  const visible = entries.filter(entry => !entry.adminOnly || isAdmin);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Box>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
          管理后台
        </Typography>
        <Typography variant="body2" color="text.secondary">
          平台管理入口
        </Typography>
      </Box>

      {!isAdmin && (
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
              <ShieldCheck size={20} aria-hidden />
              <Typography variant="body2" color="text.secondary">
                当前账号不是管理员，仅能看到公开管理入口。
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}

      {visible.map(entry => {
        const Icon = entry.icon;
        return (
          <Card key={entry.key}>
            <CardContent>
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 2,
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, minWidth: 0 }}>
                  <Icon size={22} aria-hidden />
                  <Box sx={{ minWidth: 0 }}>
                    <Typography variant="body1" sx={{ fontWeight: 600 }}>
                      {entry.label}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {entry.description}
                    </Typography>
                  </Box>
                </Box>
                <Button
                  variant="outlined"
                  endIcon={<ChevronRight size={16} aria-hidden />}
                  onClick={() => router.push(entry.key)}
                  sx={{ flexShrink: 0 }}
                >
                  进入
                </Button>
              </Box>
            </CardContent>
          </Card>
        );
      })}
    </Box>
  );
}
