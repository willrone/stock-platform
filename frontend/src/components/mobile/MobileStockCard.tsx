'use client';

import React from 'react';
import { 
  Card, 
  CardContent, 
  Box, 
  Typography, 
  Chip,
} from '@mui/material';
import { TrendingUp, Calendar, Database } from 'lucide-react';

interface RemoteStock {
  ts_code: string;
  name?: string;
  data_range?: {
    start_date: string;
    end_date: string;
    total_days?: number;
  };
  last_update?: string;
  status?: string;
}

interface LocalStock {
  ts_code: string;
  name?: string;
  data_range?: {
    start_date: string;
    end_date: string;
    total_days?: number;
  };
  file_count?: number;
  total_size?: number;
  record_count?: number;
}

interface MobileStockCardProps {
  stock: RemoteStock | LocalStock;
  type: 'remote' | 'local';
}

export const MobileStockCard: React.FC<MobileStockCardProps> = ({ stock, type }) => {
  const isRemote = type === 'remote';
  const remoteStock = isRemote ? (stock as RemoteStock) : undefined;
  const localStock = !isRemote ? (stock as LocalStock) : undefined;

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return '-';
    return dateStr.replace(/(\d{4})(\d{2})(\d{2})/, '$1-$2-$3');
  };

  const formatSize = (bytes?: number) => {
    if (!bytes) return '-';
    const mb = bytes / (1024 * 1024);
    return mb >= 1 ? `${mb.toFixed(2)} MB` : `${(bytes / 1024).toFixed(2)} KB`;
  };

  return (
    <Card 
      sx={{ 
        mb: 1.5, 
        borderRadius: 2,
        boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
      }}
    >
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        {/* 标题行 */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Box>
            <Typography variant="body1" fontWeight={600} sx={{ fontSize: '0.95rem' }}>
              {stock.ts_code}
            </Typography>
            {stock.name && (
              <Typography variant="caption" color="text.secondary">
                {stock.name}
              </Typography>
            )}
          </Box>
          {remoteStock?.status && (
            <Chip 
              label={remoteStock.status} 
              size="small"
              color={remoteStock.status === 'active' ? 'success' : 'default'}
              sx={{ fontSize: '0.7rem', height: 20 }}
            />
          )}
        </Box>

        {/* 数据范围 */}
        {stock.data_range && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
            <Calendar size={14} color="#666" />
            <Typography variant="caption" color="text.secondary">
              {formatDate(stock.data_range.start_date)} ~ {formatDate(stock.data_range.end_date)}
            </Typography>
            {stock.data_range.total_days && (
              <Typography variant="caption" color="primary.main" fontWeight={600}>
                ({stock.data_range.total_days}天)
              </Typography>
            )}
          </Box>
        )}

        {/* 本地存储信息 */}
        {localStock && (
          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
            {localStock.file_count != null && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Database size={14} color="#666" />
                <Typography variant="caption" color="text.secondary">
                  {localStock.file_count} 文件
                </Typography>
              </Box>
            )}
            {localStock.total_size != null && (
              <Typography variant="caption" color="text.secondary">
                {formatSize(localStock.total_size)}
              </Typography>
            )}
            {localStock.record_count != null && (
              <Typography variant="caption" color="text.secondary">
                {localStock.record_count.toLocaleString()} 条记录
              </Typography>
            )}
          </Box>
        )}

        {/* 远端更新时间 */}
        {remoteStock?.last_update && (
          <Typography variant="caption" color="text.secondary">
            更新: {formatDate(remoteStock.last_update)}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};
