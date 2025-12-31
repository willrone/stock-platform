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
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Tooltip,
  Pagination,
} from '@heroui/react';
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

  const getStatusColor = (success: boolean) => {
    return success ? 'success' : 'danger';
  };

  const getStatusIcon = (success: boolean) => {
    return success ? (
      <CheckCircle className="w-4 h-4" />
    ) : (
      <XCircle className="w-4 h-4" />
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
    <Modal 
      isOpen={isOpen} 
      onClose={onClose}
      size="5xl"
      scrollBehavior="inside"
    >
      <ModalContent>
        {(onModalClose) => (
          <>
            <ModalHeader className="flex flex-col gap-1">
              <div className="flex items-center justify-between w-full">
                <div className="flex items-center space-x-2">
                  <History className="w-5 h-5 text-primary" />
                  <span>同步历史记录</span>
                </div>
                <Button
                  isIconOnly
                  variant="light"
                  size="sm"
                  onPress={loadHistory}
                  isLoading={loading}
                >
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>
              {historyData && (
                <p className="text-sm text-default-500 font-normal">
                  共 {historyData.total} 条记录
                </p>
              )}
            </ModalHeader>
            
            <ModalBody>
              {loading && !historyData ? (
                <div className="text-center py-8">
                  <RefreshCw className="w-8 h-8 text-primary mx-auto mb-4 animate-spin" />
                  <p className="text-default-500">加载同步历史中...</p>
                </div>
              ) : historyData && historyData.history.length > 0 ? (
                <div className="space-y-4">
                  <Table aria-label="同步历史记录">
                    <TableHeader>
                      <TableColumn>同步ID</TableColumn>
                      <TableColumn>股票数量</TableColumn>
                      <TableColumn>同步模式</TableColumn>
                      <TableColumn>结果</TableColumn>
                      <TableColumn>记录数</TableColumn>
                      <TableColumn>创建时间</TableColumn>
                      <TableColumn>操作</TableColumn>
                    </TableHeader>
                    <TableBody>
                      {currentItems.map((entry) => (
                        <TableRow key={entry.sync_id}>
                          <TableCell>
                            <div className="font-mono text-sm">
                              {entry.sync_id.substring(0, 8)}...
                            </div>
                          </TableCell>
                          <TableCell>
                            <div className="text-sm">
                              <span className="font-medium">{entry.request.stock_codes.length}</span>
                              <span className="text-default-500 ml-1">只股票</span>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Chip
                              size="sm"
                              variant="flat"
                              color={entry.request.sync_mode === 'incremental' ? 'primary' : 'secondary'}
                            >
                              {getSyncModeText(entry.request.sync_mode)}
                            </Chip>
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center space-x-2">
                              <Chip
                                size="sm"
                                variant="flat"
                                color={getStatusColor(entry.result.success)}
                                startContent={getStatusIcon(entry.result.success)}
                              >
                                {entry.result.success ? '成功' : '失败'}
                              </Chip>
                              <span className="text-xs text-default-500">
                                {entry.result.success_count}/{entry.result.total_stocks}
                              </span>
                            </div>
                          </TableCell>
                          <TableCell>
                            <span className="text-sm font-medium">
                              {entry.result.total_records.toLocaleString()}
                            </span>
                          </TableCell>
                          <TableCell>
                            <div className="text-sm">
                              <div>{new Date(entry.created_at).toLocaleDateString()}</div>
                              <div className="text-xs text-default-500">
                                {new Date(entry.created_at).toLocaleTimeString()}
                              </div>
                            </div>
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center space-x-1">
                              <Tooltip content="查看详情">
                                <Button
                                  isIconOnly
                                  variant="light"
                                  size="sm"
                                  onPress={() => setSelectedEntry(entry)}
                                >
                                  <Eye className="w-4 h-4" />
                                </Button>
                              </Tooltip>
                              {entry.result.failure_count > 0 && (
                                <Tooltip content="重试失败项">
                                  <Button
                                    isIconOnly
                                    variant="light"
                                    size="sm"
                                    color="warning"
                                    onPress={() => handleRetry(entry.sync_id)}
                                  >
                                    <RotateCcw className="w-4 h-4" />
                                  </Button>
                                </Tooltip>
                              )}
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>

                  {totalPages > 1 && (
                    <div className="flex justify-center">
                      <Pagination
                        total={totalPages}
                        page={page}
                        onChange={setPage}
                        showControls
                        showShadow
                        color="primary"
                      />
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8">
                  <History className="w-12 h-12 text-default-300 mx-auto mb-4" />
                  <p className="text-default-500">暂无同步历史记录</p>
                </div>
              )}

              {/* 详情模态框 */}
              {selectedEntry && (
                <Modal
                  isOpen={!!selectedEntry}
                  onClose={() => setSelectedEntry(null)}
                  size="2xl"
                >
                  <ModalContent>
                    <ModalHeader>
                      同步详情 - {selectedEntry.sync_id.substring(0, 12)}...
                    </ModalHeader>
                    <ModalBody>
                      <div className="space-y-4">
                        {/* 基本信息 */}
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <p className="text-sm text-default-500">同步模式</p>
                            <p className="font-medium">{getSyncModeText(selectedEntry.request.sync_mode)}</p>
                          </div>
                          <div>
                            <p className="text-sm text-default-500">强制更新</p>
                            <p className="font-medium">{selectedEntry.request.force_update ? '是' : '否'}</p>
                          </div>
                          <div>
                            <p className="text-sm text-default-500">最大并发</p>
                            <p className="font-medium">{selectedEntry.request.max_concurrent}</p>
                          </div>
                          <div>
                            <p className="text-sm text-default-500">重试次数</p>
                            <p className="font-medium">{selectedEntry.request.retry_count}</p>
                          </div>
                        </div>

                        {/* 结果统计 */}
                        <div className="p-4 bg-default-50 rounded-lg">
                          <h4 className="font-medium mb-3">同步结果</h4>
                          <div className="grid grid-cols-4 gap-4 text-center">
                            <div>
                              <p className="text-lg font-bold">{selectedEntry.result.total_stocks}</p>
                              <p className="text-xs text-default-500">总数</p>
                            </div>
                            <div>
                              <p className="text-lg font-bold text-success">{selectedEntry.result.success_count}</p>
                              <p className="text-xs text-default-500">成功</p>
                            </div>
                            <div>
                              <p className="text-lg font-bold text-danger">{selectedEntry.result.failure_count}</p>
                              <p className="text-xs text-default-500">失败</p>
                            </div>
                            <div>
                              <p className="text-lg font-bold text-primary">{selectedEntry.result.total_records.toLocaleString()}</p>
                              <p className="text-xs text-default-500">记录数</p>
                            </div>
                          </div>
                        </div>

                        {/* 股票列表 */}
                        <div>
                          <h4 className="font-medium mb-2">同步股票列表</h4>
                          <div className="max-h-32 overflow-y-auto p-3 bg-default-50 rounded-lg">
                            <div className="flex flex-wrap gap-1">
                              {selectedEntry.request.stock_codes.map((code) => (
                                <Chip key={code} size="sm" variant="flat">
                                  {code}
                                </Chip>
                              ))}
                            </div>
                          </div>
                        </div>

                        {/* 消息 */}
                        {selectedEntry.result.message && (
                          <div>
                            <h4 className="font-medium mb-2">结果消息</h4>
                            <p className="text-sm p-3 bg-default-50 rounded-lg">
                              {selectedEntry.result.message}
                            </p>
                          </div>
                        )}
                      </div>
                    </ModalBody>
                    <ModalFooter>
                      <Button variant="light" onPress={() => setSelectedEntry(null)}>
                        关闭
                      </Button>
                      {selectedEntry.result.failure_count > 0 && (
                        <Button
                          color="warning"
                          startContent={<RotateCcw className="w-4 h-4" />}
                          onPress={() => {
                            handleRetry(selectedEntry.sync_id);
                            setSelectedEntry(null);
                          }}
                        >
                          重试失败项
                        </Button>
                      )}
                    </ModalFooter>
                  </ModalContent>
                </Modal>
              )}
            </ModalBody>
            
            <ModalFooter>
              <Button variant="light" onPress={onModalClose}>
                关闭
              </Button>
            </ModalFooter>
          </>
        )}
      </ModalContent>
    </Modal>
  );
}