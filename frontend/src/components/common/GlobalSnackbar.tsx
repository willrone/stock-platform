'use client';

/**
 * 全局 Snackbar 组件
 *
 * 在应用底部居中显示通知消息，支持：
 * - 自动消失
 * - 多条消息堆叠
 * - 不同类型（成功/错误/警告/信息）
 * - 手动关闭
 */

import React from 'react';
import { Snackbar, Alert, Box, Typography, IconButton } from '@mui/material';
import { X } from 'lucide-react';
import { useSnackbarStore } from '../../stores/useSnackbarStore';

export const GlobalSnackbar: React.FC = () => {
  const { messages, removeMessage } = useSnackbarStore();
  // 只显示最新的一条消息，避免遮挡
  const latestMessage = messages.length > 0 ? messages[messages.length - 1] : null;
  // 其他消息排队等待
  const queuedMessageCount = messages.length - 1;

  const handleClose = (_event?: React.SyntheticEvent | Event, reason?: string) => {
    if (reason === 'clickaway') {
      return;
    }
    if (latestMessage) {
      removeMessage(latestMessage.id);
    }
  };

  return (
    <Box
      sx={{
        position: 'fixed',
        bottom: 24,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 10000,
        display: 'flex',
        flexDirection: 'column',
        gap: 1,
        pointerEvents: 'none',
      }}
    >
      {messages.map((message, index) => {
        const isTop = index === messages.length - 1;
        return (
          <Snackbar
            key={message.id}
            open={true}
            autoHideDuration={message.duration || 4000}
            onClose={(event, reason) => {
              if (reason === 'clickaway') return;
              removeMessage(message.id);
            }}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
            sx={{
              position: 'relative',
              transform: isTop ? 'none' : `scale(${1 - (messages.length - 1 - index) * 0.05})`,
              opacity: isTop ? 1 : 0.6,
              pointerEvents: isTop ? 'auto' : 'none',
              mb: isTop ? 0 : -4,
            }}
          >
            <Alert
              severity={message.severity}
              variant="filled"
              sx={{
                minWidth: 320,
                maxWidth: 600,
                boxShadow: 3,
                borderRadius: 2,
                '& .MuiAlert-message': {
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                },
              }}
              action={
                <IconButton size="small" color="inherit" onClick={() => removeMessage(message.id)}>
                  <X size={16} />
                </IconButton>
              }
            >
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {message.message}
              </Typography>
            </Alert>
          </Snackbar>
        );
      })}
    </Box>
  );
};
