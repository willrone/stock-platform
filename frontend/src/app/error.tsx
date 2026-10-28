'use client';

import { useEffect } from 'react';
import { Alert, Box, Button, Typography } from '@mui/material';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('页面错误:', error);
  }, [error]);

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '60vh',
        p: 4,
        textAlign: 'center',
      }}
    >
      <Typography variant="h5" sx={{ fontWeight: 600, color: 'error.main', mb: 2 }}>
        出错了
      </Typography>
      <Alert severity="error" sx={{ maxWidth: 480, textAlign: 'left' }}>
        {error.message || '页面加载时遇到错误'}
        {error.digest ? `（错误编号：${error.digest}）` : ''}
      </Alert>
      <Button
        variant="contained"
        onClick={reset}
        sx={{
          mt: 3,
          '&:focus-visible': {
            outline: '2px solid primary.main',
            outlineOffset: 2,
          },
        }}
      >
        重试
      </Button>
    </Box>
  );
}
