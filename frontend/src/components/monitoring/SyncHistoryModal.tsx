/**
 * 同步历史模态框组件
 * 
 * 显示数据同步的历史记录，包括：
 * - 历史同步任务列表
 * - 同步结果统计
 * - 详细的成功/失败信息
 * - 重试功能
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Button,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Tooltip,
  Pagination,
  Box,
  Typography,
  IconButton,
  CircularProgress,
} from '@mui/material';
import {
  History,
  CheckCircle,
  XCircle,
  Clock,
  RotateCcw,
  Eye,
  RefreshCw,
} from 'lucide-react';
import { DataService } from '../../services/dataService';

interface SyncHistoryEntry {
  sync_id: string;
  request: {
    stock_codes: string[];
    start_date: string | null;
    end_date: string | null;
    force_update: boolean;
    sync_mode: string;
    max_concurrent: number;
    retry_count: number;
  };
  result: {
    success: boolean;
    total_stocks: number;
    success_count: number;
    failure_count: number;
    total_records: number;
    message: string;
  };
  created_at: string;
}

interface SyncHistoryData {
  history: SyncHistoryEntry[];
  total: number;
  limit: number;
}

interface SyncHistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SyncHistoryModal({ isOpen, onClose }: SyncHistoryModalProps) {
  const [historyData, setHistoryData] = useState<SyncHistoryData | null>(null);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [selectedEntry, setSelectedEntry] = useState<SyncHistoryEntry | null>(null);

  const loadHistory = async () => {
    try {
      setLoading(true);
      const data = await DataService.getSyncHistory(50);
      setHistoryData(data);
    } catch (error) {
      console.error('获取同步历史失败:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadHistory();
    }
  }, [isOpen]);

  const handleRetry = async (syncId: string) => {
    try {
      await DataService.retrySyncFailed(syncId);
      // 重新加载历史记录
      await loadHistory();
    } catch (error) {
      console.error('重试同步失败:', error);
    }
  };

  const formatDuration = (startTime: string, endTime?: string) => {
    const start = new Date(startTime);
    const end = endTime ? new Date(endTime) : new Date();
    const duration = Math.floor((end.getTime() - start.getTime()) / 1000);
    
    if (duration < 60) {
      return `${duration}秒`;
    } else if (duration < 3600) {
      return `${Math.floor(duration / 60)}分钟`;
    } else {
      return `${Math.floor(duration / 3600)}小时`;
    }
  };

  const getStatusColor = (success: boolean): "success" | "error" => {
    return success ? 'success' : 'error';
  };

  const getStatusIcon = (success: boolean) => {
    return success ? (
      <CheckCircle size={16} />
    ) : (
      <XCircle size={16} />
    );
  };

  const getSyncModeText = (mode: string) => {
    return mode === 'incremental' ? '增量' : '全量';
  };

  // 分页处理
  const itemsPerPage = 10;
  const totalPages = historyData ? Math.ceil(historyData.history.length / itemsPerPage) : 1;
  const startIndex = (page - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentItems = historyData?.history.slice(startIndex, endIndex) || [];

  return (
    <>
      <Dialog 
        open={isOpen} 
        onClose={onClose}
        maxWidth="xl"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <History size={20} color="#1976d2" />
                <span>同步历史记录</span>
              </Box>
              <IconButton
                size="small"
                onClick={loadHistory}
                disabled={loading}
              >
                <RefreshCw size={16} />
              </IconButton>
            </Box>
            {historyData && (
              <Typography variant="caption" color="text.secondary">
                共 {historyData.total} 条记录
              </Typography>
            )}
          </Box>
        </DialogTitle>
        
        <DialogContent>
          {loading && !historyData ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <CircularProgress size={32} sx={{ mb: 2 }} />
              <Typography variant="body2" color="text.secondary">
                加载同步历史中...
              </Typography>
            </Box>
          ) : historyData && historyData.history.length > 0 ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ overflowX: 'auto' }}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>同步ID</TableCell>
                      <TableCell>股票数量</TableCell>
                      <TableCell>同步模式</TableCell>
                      <TableCell>结果</TableCell>
                      <TableCell>记录数</TableCell>
                      <TableCell>创建时间</TableCell>
                      <TableCell>操作</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {currentItems.map((entry) => (
                      <TableRow key={entry.sync_id} hover>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                            {entry.sync_id.substring(0, 8)}...
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            <strong>{entry.request.stock_codes.length}</strong>
                            <Typography component="span" variant="body2" color="text.secondary" sx={{ ml: 0.5 }}>
                              只股票
                            </Typography>
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={getSyncModeText(entry.request.sync_mode)}
                            color={entry.request.sync_mode === 'incremental' ? 'primary' : 'secondary'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Chip
                              label={entry.result.success ? '成功' : '失败'}
                              color={getStatusColor(entry.result.success)}
                              size="small"
                              icon={getStatusIcon(entry.result.success)}
                            />
                            <Typography variant="caption" color="text.secondary">
                              {entry.result.success_count}/{entry.result.total_stocks}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            {entry.result.total_records.toLocaleString()}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Box>
                            <Typography variant="body2">
                              {new Date(entry.created_at).toLocaleDateString()}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {new Date(entry.created_at).toLocaleTimeString()}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Tooltip title="查看详情">
                              <IconButton
                                size="small"
                                onClick={() => setSelectedEntry(entry)}
                              >
                                <Eye size={16} />
                              </IconButton>
                            </Tooltip>
                            {entry.result.failure_count > 0 && (
                              <Tooltip title="重试失败项">
                                <IconButton
                                  size="small"
                                  color="warning"
                                  onClick={() => handleRetry(entry.sync_id)}
                                >
                                  <RotateCcw size={16} />
                                </IconButton>
                              </Tooltip>
                            )}
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Box>

              {totalPages > 1 && (
                <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                  <Pagination
                    count={totalPages}
                    page={page}
                    onChange={(e, newPage) => setPage(newPage)}
                    color="primary"
                  />
                </Box>
              )}
            </Box>
          ) : (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <History size={48} color="#ccc" style={{ margin: '0 auto 16px' }} />
              <Typography variant="body2" color="text.secondary">
                暂无同步历史记录
              </Typography>
            </Box>
          )}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={onClose}>关闭</Button>
        </DialogActions>
      </Dialog>

      {/* 详情模态框 */}
      {selectedEntry && (
        <Dialog
          open={!!selectedEntry}
          onClose={() => setSelectedEntry(null)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>
            同步详情 - {selectedEntry.sync_id.substring(0, 12)}...
          </DialogTitle>
          <DialogContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {/* 基本信息 */}
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    同步模式
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {getSyncModeText(selectedEntry.request.sync_mode)}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    强制更新
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {selectedEntry.request.force_update ? '是' : '否'}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    最大并发
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {selectedEntry.request.max_concurrent}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    重试次数
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {selectedEntry.request.retry_count}
                  </Typography>
                </Box>
              </Box>

              {/* 结果统计 */}
              <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: 600, mb: 2 }}>
                  同步结果
                </Typography>
                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 2, textAlign: 'center' }}>
                  <Box>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {selectedEntry.result.total_stocks}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      总数
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.main' }}>
                      {selectedEntry.result.success_count}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      成功
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: 'error.main' }}>
                      {selectedEntry.result.failure_count}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      失败
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: 'primary.main' }}>
                      {selectedEntry.result.total_records.toLocaleString()}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      记录数
                    </Typography>
                  </Box>
                </Box>
              </Box>

              {/* 股票列表 */}
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                  同步股票列表
                </Typography>
                <Box sx={{ maxHeight: 128, overflowY: 'auto', p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selectedEntry.request.stock_codes.map((code) => (
                      <Chip key={code} label={code} size="small" />
                    ))}
                  </Box>
                </Box>
              </Box>

              {/* 消息 */}
              {selectedEntry.result.message && (
                <Box>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                    结果消息
                  </Typography>
                  <Typography variant="body2" sx={{ p: 1.5, bgcolor: 'grey.50', borderRadius: 1 }}>
                    {selectedEntry.result.message}
                  </Typography>
                </Box>
              )}
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setSelectedEntry(null)}>关闭</Button>
            {selectedEntry.result.failure_count > 0 && (
              <Button
                color="warning"
                variant="outlined"
                startIcon={<RotateCcw size={16} />}
                onClick={() => {
                  handleRetry(selectedEntry.sync_id);
                  setSelectedEntry(null);
                }}
              >
                重试失败项
              </Button>
            )}
          </DialogActions>
        </Dialog>
      )}
    </>
  );
}
