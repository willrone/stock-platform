/**
 * 任务基本信息组件
 */

import React, { useState } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Box,
  Typography,
  Chip,
  Divider,
  IconButton,
} from '@mui/material';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { Task } from '@/stores/useTaskStore';

interface TaskInfoProps {
  task: Task;
}

export function TaskInfo({ task }: TaskInfoProps) {
  const [selectedStocksPage, setSelectedStocksPage] = useState(1);
  const STOCKS_PER_PAGE = 12;

  return (
    <Card>
      <CardHeader title="任务信息" />
      <CardContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
            {task.task_type === 'hyperparameter_optimization' ? (
              <Box>
                <Typography variant="caption" color="text.secondary">
                  已完成轮次
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                  {task.optimization_info?.completed_trials ?? 0} /{' '}
                  {task.optimization_info?.n_trials ?? 0}
                </Typography>
              </Box>
            ) : (
              <>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    模型
                  </Typography>
                  <Chip label={task.model_id} color="secondary" size="small" sx={{ mt: 0.5 }} />
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    股票数量
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                    {task.stock_codes.length}
                  </Typography>
                </Box>
              </>
            )}
            <Box>
              <Typography variant="caption" color="text.secondary">
                创建时间
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                {new Date(task.created_at).toLocaleString()}
              </Typography>
            </Box>
            {task.completed_at && (
              <Box>
                <Typography variant="caption" color="text.secondary">
                  完成时间
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                  {new Date(task.completed_at).toLocaleString()}
                </Typography>
              </Box>
            )}
          </Box>

          <Divider />

          <Box>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ mb: 1, display: 'block' }}
            >
              选择的股票
            </Typography>
            <Box
              sx={{
                height: 200,
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: 1,
                p: 1.5,
              }}
            >
              {task.stock_codes && task.stock_codes.length > 0 ? (
                <>
                  <Box
                    sx={{
                      flex: 1,
                      overflowY: 'auto',
                      display: 'flex',
                      flexWrap: 'wrap',
                      gap: 1,
                      alignContent: 'flex-start',
                      pb: 1,
                    }}
                  >
                    {(() => {
                      const startIndex = (selectedStocksPage - 1) * STOCKS_PER_PAGE;
                      const endIndex = startIndex + STOCKS_PER_PAGE;
                      const currentStocks = task.stock_codes.slice(startIndex, endIndex);

                      return currentStocks.map(code => <Chip key={code} label={code} size="small" />);
                    })()}
                  </Box>

                  {(() => {
                    const totalPages = Math.ceil(task.stock_codes.length / STOCKS_PER_PAGE);

                    if (totalPages > 1) {
                      return (
                        <Box
                          sx={{
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            gap: 1,
                            pt: 1,
                            borderTop: '1px solid',
                            borderColor: 'divider',
                          }}
                        >
                          <IconButton
                            size="small"
                            disabled={selectedStocksPage === 1}
                            onClick={() => setSelectedStocksPage(prev => Math.max(1, prev - 1))}
                          >
                            <ChevronLeft size={16} />
                          </IconButton>

                          <Typography variant="caption" color="text.secondary">
                            第 {selectedStocksPage} / {totalPages} 页
                          </Typography>

                          <IconButton
                            size="small"
                            disabled={selectedStocksPage >= totalPages}
                            onClick={() =>
                              setSelectedStocksPage(prev => Math.min(totalPages, prev + 1))
                            }
                          >
                            <ChevronRight size={16} />
                          </IconButton>
                        </Box>
                      );
                    }
                    return null;
                  })()}

                  <Box
                    sx={{
                      pt: 1,
                      mt: 1,
                      borderTop: '1px solid',
                      borderColor: 'divider',
                      display: 'flex',
                      justifyContent: 'center',
                    }}
                  >
                    <Typography variant="body2" color="text.secondary">
                      已选择 <strong>{task.stock_codes.length}</strong> 只股票
                    </Typography>
                  </Box>
                </>
              ) : (
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: '100%',
                  }}
                >
                  <Typography variant="body2" color="text.secondary">
                    暂无选择的股票
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}
