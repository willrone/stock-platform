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
  CardBody,
  CardHeader,
  Button,
  Progress,
  Chip,
  Divider,
  Accordion,
  AccordionItem,
  Code,
  Spinner,
} from '@heroui/react';
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

export const WorkflowTest: React.FC<WorkflowTestProps> = ({
  onTestComplete,
}) => {
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
    if (!result) return;

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
      <CheckCircle className="w-5 h-5 text-success" />
    ) : (
      <XCircle className="w-5 h-5 text-danger" />
    );
  };

  // 获取测试状态颜色
  const getStatusColor = (success: boolean) => {
    return success ? 'success' : 'danger';
  };

  // 格式化持续时间
  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  return (
    <div className="space-y-6">
      {/* 测试控制面板 */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between w-full">
            <div>
              <h3 className="text-lg font-semibold">前后端集成测试</h3>
              <p className="text-sm text-default-600">
                测试前后端集成功能和完整用户工作流程
              </p>
            </div>
            <div className="flex gap-2">
              <Button
                color="primary"
                startContent={<Play className="w-4 h-4" />}
                onPress={runTests}
                isDisabled={isRunning}
                isLoading={isRunning}
              >
                {isRunning ? '测试中...' : '开始测试'}
              </Button>
              {result && (
                <Button
                  variant="light"
                  startContent={<Download className="w-4 h-4" />}
                  onPress={downloadReport}
                >
                  下载报告
                </Button>
              )}
            </div>
          </div>
        </CardHeader>

        {isRunning && (
          <CardBody>
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">测试进度</span>
                  <span className="text-sm text-default-600">{progress}%</span>
                </div>
                <Progress value={progress} color="primary" />
              </div>
              
              {currentTest && (
                <div className="flex items-center gap-2">
                  <Spinner size="sm" />
                  <span className="text-sm">{currentTest}</span>
                </div>
              )}
            </div>
          </CardBody>
        )}
      </Card>

      {/* 测试结果概览 */}
      {result && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              {getStatusIcon(result.success)}
              <h4 className="text-lg font-semibold">
                测试结果 {result.success ? '通过' : '失败'}
              </h4>
            </div>
          </CardHeader>
          <CardBody>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-default-900">
                  {result.totalTests}
                </div>
                <div className="text-sm text-default-600">总测试数</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-success">
                  {result.passedTests}
                </div>
                <div className="text-sm text-default-600">通过</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-danger">
                  {result.failedTests}
                </div>
                <div className="text-sm text-default-600">失败</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-primary">
                  {formatDuration(result.duration)}
                </div>
                <div className="text-sm text-default-600">总耗时</div>
              </div>
            </div>

            <Divider className="my-4" />

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">成功率:</span>
                <Chip
                  color={result.success ? 'success' : 'danger'}
                  variant="flat"
                  size="sm"
                >
                  {((result.passedTests / result.totalTests) * 100).toFixed(1)}%
                </Chip>
              </div>
              <div className="text-sm text-default-600">
                测试时间: {new Date().toLocaleString()}
              </div>
            </div>
          </CardBody>
        </Card>
      )}

      {/* 详细测试结果 */}
      {result && (
        <Card>
          <CardHeader>
            <h4 className="text-lg font-semibold">详细测试结果</h4>
          </CardHeader>
          <CardBody>
            <Accordion variant="splitted">
              {result.results.map((testResult, index) => (
                <AccordionItem
                  key={index}
                  title={
                    <div className="flex items-center justify-between w-full">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(testResult.success)}
                        <span>{testResult.name}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Chip
                          color={getStatusColor(testResult.success)}
                          variant="flat"
                          size="sm"
                        >
                          {testResult.success ? '通过' : '失败'}
                        </Chip>
                        <span className="text-sm text-default-600">
                          {formatDuration(testResult.duration)}
                        </span>
                      </div>
                    </div>
                  }
                >
                  <div className="space-y-4">
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <Info className="w-4 h-4 text-primary" />
                        <span className="font-medium">测试消息</span>
                      </div>
                      <p className="text-sm text-default-700">
                        {testResult.message}
                      </p>
                    </div>

                    {testResult.details && (
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <Clock className="w-4 h-4 text-default-500" />
                          <span className="font-medium">详细信息</span>
                        </div>
                        <Code className="block w-full">
                          {JSON.stringify(testResult.details, null, 2)}
                        </Code>
                      </div>
                    )}

                    {!testResult.success && (
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <AlertTriangle className="w-4 h-4 text-warning" />
                          <span className="font-medium">修复建议</span>
                        </div>
                        <div className="text-sm text-default-700 space-y-1">
                          {getFixSuggestions(testResult.name)}
                        </div>
                      </div>
                    )}
                  </div>
                </AccordionItem>
              ))}
            </Accordion>
          </CardBody>
        </Card>
      )}
    </div>
  );
};

// 获取修复建议
function getFixSuggestions(testName: string): React.ReactNode {
  const suggestions: Record<string, string[]> = {
    'API连接测试': [
      '检查后端服务是否正常运行',
      '确认API_URL环境变量配置正确',
      '检查网络连接和防火墙设置',
      '验证CORS配置是否正确',
    ],
    'WebSocket连接测试': [
      '检查WebSocket服务是否启用',
      '确认WS_URL环境变量配置正确',
      '检查浏览器WebSocket支持',
      '验证代理服务器WebSocket配置',
    ],
    '任务创建流程测试': [
      '检查任务管理服务是否正常',
      '验证数据库连接状态',
      '确认模型服务可用性',
      '检查任务创建权限',
    ],
    '数据获取流程测试': [
      '检查数据服务连接状态',
      '验证远端数据服务可用性',
      '确认Parquet文件访问权限',
      '检查数据同步配置',
    ],
  };

  const testSuggestions = suggestions[testName] || [
    '检查相关服务状态',
    '查看详细错误日志',
    '验证配置参数',
    '联系技术支持',
  ];

  return (
    <ul className="list-disc list-inside space-y-1">
      {testSuggestions.map((suggestion, index) => (
        <li key={index}>{suggestion}</li>
      ))}
    </ul>
  );
}