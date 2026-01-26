/**
 * 用户工作流程测试组件
 *
 * 提供完整的用户工作流程测试界面，包括：
 * - 测试执行控制
 * - 实时测试进度显示
 * - 测试结果展示
 * - 错误诊断和修复建议
 */

'use client';

import React, { useState, useCallback } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Button,
  Chip,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Box,
  Typography,
  CircularProgress,
  LinearProgress,
} from '@mui/material';
import { ExpandMore } from '@mui/icons-material';
import {
  Play,
  CheckCircle,
  XCircle,
  Clock,
  Download,
  RefreshCw,
  AlertTriangle,
  Info,
} from 'lucide-react';

import {
  integrationTestManager,
  TestSuiteResult,
  TestResult,
  generateTestReport,
} from '../../utils/integrationTest';

interface WorkflowTestProps {
  onTestComplete?: (result: TestSuiteResult) => void;
}

export const WorkflowTest: React.FC<WorkflowTestProps> = ({ onTestComplete }) => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentTest, setCurrentTest] = useState<string>('');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<TestSuiteResult | null>(null);

  // 运行测试
  const runTests = useCallback(async () => {
    setIsRunning(true);
    setCurrentTest('');
    setProgress(0);
    setResult(null);

    try {
      // 模拟测试进度更新
      const testNames = [
        'API连接测试',
        'API基础功能测试',
        'WebSocket连接测试',
        'WebSocket消息传输测试',
        '任务创建流程测试',
        '数据获取流程测试',
        '错误处理机制测试',
        '完整用户工作流程测试',
      ];

      // 运行测试并更新进度
      const testResult = await integrationTestManager.runAllTests();

      setResult(testResult);
      setProgress(100);
      setCurrentTest('测试完成');

      if (onTestComplete) {
        onTestComplete(testResult);
      }
    } catch (error) {
      console.error('测试执行失败:', error);
      setCurrentTest('测试失败');
    } finally {
      setIsRunning(false);
    }
  }, [onTestComplete]);

  // 下载测试报告
  const downloadReport = useCallback(() => {
    if (!result) {
      return;
    }

    const report = generateTestReport(result);
    const blob = new Blob([report], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `integration-test-report-${new Date().toISOString().split('T')[0]}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [result]);

  // 获取测试状态图标
  const getStatusIcon = (success: boolean) => {
    return success ? (
      <CheckCircle size={20} color="#2e7d32" />
    ) : (
      <XCircle size={20} color="#d32f2f" />
    );
  };

  // 获取测试状态颜色
  const getStatusColor = (success: boolean): 'success' | 'error' => {
    return success ? 'success' : 'error';
  };

  // 格式化持续时间
  const formatDuration = (ms: number) => {
    if (ms < 1000) {
      return `${ms}ms`;
    }
    return `${(ms / 1000).toFixed(1)}s`;
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 测试控制面板 */}
      <Card>
        <CardHeader
          title={
            <Box>
              <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
                前后端集成测试
              </Typography>
              <Typography variant="body2" color="text.secondary">
                测试前后端集成功能和完整用户工作流程
              </Typography>
            </Box>
          }
          action={
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="contained"
                color="primary"
                startIcon={<Play size={16} />}
                onClick={runTests}
                disabled={isRunning}
              >
                {isRunning ? '测试中...' : '开始测试'}
              </Button>
              {result && (
                <Button
                  variant="outlined"
                  startIcon={<Download size={16} />}
                  onClick={downloadReport}
                >
                  下载报告
                </Button>
              )}
            </Box>
          }
        />

        {isRunning && (
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    测试进度
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {progress}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={progress}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              {currentTest && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2">{currentTest}</Typography>
                </Box>
              )}
            </Box>
          </CardContent>
        )}
      </Card>

      {/* 测试结果概览 */}
      {result && (
        <Card>
          <CardHeader
            title={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {getStatusIcon(result.success)}
                <Typography variant="h6" component="h4" sx={{ fontWeight: 600 }}>
                  测试结果 {result.success ? '通过' : '失败'}
                </Typography>
              </Box>
            }
          />
          <CardContent>
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', md: 'repeat(4, 1fr)' },
                gap: 2,
              }}
            >
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" sx={{ fontWeight: 600 }}>
                  {result.totalTests}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  总测试数
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                  {result.passedTests}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  通过
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" sx={{ fontWeight: 600, color: 'error.main' }}>
                  {result.failedTests}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  失败
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" sx={{ fontWeight: 600, color: 'primary.main' }}>
                  {formatDuration(result.duration)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  总耗时
                </Typography>
              </Box>
            </Box>

            <Divider sx={{ my: 2 }} />

            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  成功率:
                </Typography>
                <Chip
                  label={`${((result.passedTests / result.totalTests) * 100).toFixed(1)}%`}
                  color={result.success ? 'success' : 'error'}
                  size="small"
                />
              </Box>
              <Typography variant="body2" color="text.secondary">
                测试时间: {new Date().toLocaleString()}
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* 详细测试结果 */}
      {result && (
        <Card>
          <CardHeader title="详细测试结果" />
          <CardContent>
            <Box>
              {result.results.map((testResult, index) => (
                <Accordion key={index}>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        width: '100%',
                        pr: 2,
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {getStatusIcon(testResult.success)}
                        <Typography variant="body2">{testResult.name}</Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Chip
                          label={testResult.success ? '通过' : '失败'}
                          color={getStatusColor(testResult.success)}
                          size="small"
                        />
                        <Typography variant="caption" color="text.secondary">
                          {formatDuration(testResult.duration)}
                        </Typography>
                      </Box>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      <Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                          <Info size={16} color="#1976d2" />
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            测试消息
                          </Typography>
                        </Box>
                        <Typography variant="body2">{testResult.message}</Typography>
                      </Box>

                      {testResult.details && (
                        <Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                            <Clock size={16} color="#666" />
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              详细信息
                            </Typography>
                          </Box>
                          <Box
                            component="pre"
                            sx={{
                              bgcolor: 'grey.900',
                              color: '#4ade80',
                              p: 2,
                              borderRadius: 1,
                              overflowX: 'auto',
                              fontFamily: 'monospace',
                              fontSize: '0.875rem',
                            }}
                          >
                            {JSON.stringify(testResult.details, null, 2)}
                          </Box>
                        </Box>
                      )}

                      {!testResult.success && (
                        <Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                            <AlertTriangle size={16} color="#ed6c02" />
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              修复建议
                            </Typography>
                          </Box>
                          <Box component="ul" sx={{ pl: 2, m: 0 }}>
                            {getFixSuggestions(testResult.name).map((suggestion, idx) => (
                              <Typography key={idx} component="li" variant="body2">
                                {suggestion}
                              </Typography>
                            ))}
                          </Box>
                        </Box>
                      )}
                    </Box>
                  </AccordionDetails>
                </Accordion>
              ))}
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

// 获取修复建议
function getFixSuggestions(testName: string): string[] {
  const suggestions: Record<string, string[]> = {
    API连接测试: [
      '检查后端服务是否正常运行',
      '确认API_URL环境变量配置正确',
      '检查网络连接和防火墙设置',
      '验证CORS配置是否正确',
    ],
    WebSocket连接测试: [
      '检查WebSocket服务是否启用',
      '确认WS_URL环境变量配置正确',
      '检查浏览器WebSocket支持',
      '验证代理服务器WebSocket配置',
    ],
    任务创建流程测试: [
      '检查任务管理服务是否正常',
      '验证数据库连接状态',
      '确认模型服务可用性',
      '检查任务创建权限',
    ],
    数据获取流程测试: [
      '检查数据服务连接状态',
      '验证远端数据服务可用性',
      '确认Parquet文件访问权限',
      '检查数据同步配置',
    ],
  };

  return (
    suggestions[testName] || [
      '检查相关服务状态',
      '查看详细错误日志',
      '验证配置参数',
      '联系技术支持',
    ]
  );
}
