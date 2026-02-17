'use client';

import React from 'react';
import { Card, CardContent, Box, Typography, Chip } from '@mui/material';
import { AlertTriangle, Clock } from 'lucide-react';

interface ErrorStat {
  error_type: string;
  count: number;
  last_occurrence: string;
  sample_message: string;
}

interface MobileErrorCardProps {
  error: ErrorStat;
}

export const MobileErrorCard: React.FC<MobileErrorCardProps> = ({ error }) => {
  const getSeverityColor = (type: string): 'error' | 'warning' | 'info' => {
    if (type.toLowerCase().includes('critical') || type.toLowerCase().includes('error')) {
      return 'error';
    }
    if (type.toLowerCase().includes('warning')) {
      return 'warning';
    }
    return 'info';
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);

    if (diffMins < 60) {
      return `${diffMins}分钟前`;
    } else if (diffHours < 24) {
      return `${diffHours}小时前`;
    } else {
      return date.toLocaleDateString('zh-CN');
    }
  };

  return (
    <Card
      sx={{
        mb: 1.5,
        borderRadius: 2,
        boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
        borderLeft: 4,
        borderLeftColor: getSeverityColor(error.error_type) + '.main',
      }}
    >
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        {/* 标题行 */}
        <Box
          sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}
        >
          <Box sx={{ flex: 1, pr: 1 }}>
            <Typography variant="body2" fontWeight={600} sx={{ fontSize: '0.9rem', mb: 0.5 }}>
              {error.error_type}
            </Typography>
            {error.sample_message && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                {error.sample_message.length > 60
                  ? error.sample_message.substring(0, 60) + '...'
                  : error.sample_message}
              </Typography>
            )}
          </Box>
          <Chip
            label={error.count}
            color={getSeverityColor(error.error_type)}
            size="small"
            sx={{
              fontWeight: 600,
              minWidth: 40,
            }}
          />
        </Box>

        {/* 时间信息 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Clock size={14} color="#666" />
          <Typography variant="caption" color="text.secondary">
            最后发生: {formatDate(error.last_occurrence)}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};
