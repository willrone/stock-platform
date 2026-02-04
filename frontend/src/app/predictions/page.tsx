/**
 * 预测分析页面
 *
 * 提供股票预测功能，包括：
 * - 选择模型和股票进行预测
 * - 查看预测结果和置信度
 * - 风险评估展示
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Button,
  Box,
  Typography,
  TextField,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Alert,
  Slider,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Brain,
  RefreshCw,
  Info,
  AlertTriangle,
  CheckCircle,
} from 'lucide-react';
import { DataService, PredictionRequest, PredictionResponse } from '../../services/dataService';
import { useDataStore, Model } from '../../stores/useDataStore';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';

interface PredictionResult {
  stock_code: string;
  predicted_direction: number;
  predicted_return: number;
  predicted_price?: number;
  confidence_score: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  risk_assessment: {
    value_at_risk?: number;
    volatility?: number;
    max_drawdown?: number;
  };
}

export default function PredictionsPage() {
  const { models, setModels } = useDataStore();
  const [loading, setLoading] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [stockCodes, setStockCodes] = useState<string>('');
  const [horizon, setHorizon] = useState<'intraday' | 'short_term' | 'medium_term'>('short_term');
  const [confidenceLevel, setConfidenceLevel] = useState<number>(0.95);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  // 加载模型列表
  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      const data = await DataService.getModels();
      setModels(data.models || data);
    } catch (error) {
      console.error('加载模型列表失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 执行预测
  const handlePredict = async () => {
    if (!selectedModel) {
      setError('请选择预测模型');
      return;
    }

    const codes = stockCodes
      .split(/[,，\s]+/)
      .map(code => code.trim())
      .filter(code => code.length > 0);

    if (codes.length === 0) {
      setError('请输入至少一个股票代码');
      return;
    }

    setError(null);
    setPredicting(true);

    try {
      const request: PredictionRequest = {
        stock_codes: codes,
        model_id: selectedModel,
        horizon: horizon,
        confidence_level: confidenceLevel,
      };

      const response = await DataService.createPrediction(request);

      // 处理响应数据
      if (response && (response as any).predictions) {
        setPredictions((response as any).predictions);
      } else if (Array.isArray(response)) {
        setPredictions(response as unknown as PredictionResult[]);
      }
    } catch (err: any) {
      setError(err.message || '预测失败，请稍后重试');
      console.error('预测失败:', err);
    } finally {
      setPredicting(false);
    }
  };

  // 获取方向图标和颜色
  const getDirectionDisplay = (direction: number) => {
    if (direction > 0) {
      return {
        icon: <TrendingUp size={20} />,
        color: '#4caf50',
        text: '看涨',
      };
    } else if (direction < 0) {
      return {
        icon: <TrendingDown size={20} />,
        color: '#f44336',
        text: '看跌',
      };
    }
    return {
      icon: <TrendingUp size={20} style={{ opacity: 0.5 }} />,
      color: '#9e9e9e',
      text: '中性',
    };
  };

  // 获取置信度颜色
  const getConfidenceColor = (score: number) => {
    if (score >= 0.8) return '#4caf50';
    if (score >= 0.6) return '#ff9800';
    return '#f44336';
  };

  // 格式化百分比
  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  if (loading) {
    return <LoadingSpinner message="加载中..." />;
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Brain size={32} />
        预测分析
      </Typography>

      {/* 预测配置卡片 */}
      <Card sx={{ mb: 3 }}>
        <CardHeader title="预测配置" />
        <CardContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* 模型选择 */}
            <FormControl fullWidth>
              <InputLabel>选择预测模型</InputLabel>
              <Select
                value={selectedModel}
                label="选择预测模型"
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                {models.map((model: Model) => (
                  <MenuItem key={model.model_id} value={model.model_id}>
                    {model.model_name} ({model.model_type})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* 股票代码输入 */}
            <TextField
              fullWidth
              label="股票代码"
              placeholder="输入股票代码，多个用逗号分隔，如: 000001.SZ, 600000.SH"
              value={stockCodes}
              onChange={(e) => setStockCodes(e.target.value)}
              helperText="支持输入多个股票代码，用逗号或空格分隔"
            />

            {/* 预测周期 */}
            <FormControl fullWidth>
              <InputLabel>预测周期</InputLabel>
              <Select
                value={horizon}
                label="预测周期"
                onChange={(e) => setHorizon(e.target.value as any)}
              >
                <MenuItem value="intraday">日内 (Intraday)</MenuItem>
                <MenuItem value="short_term">短期 (1-5天)</MenuItem>
                <MenuItem value="medium_term">中期 (5-20天)</MenuItem>
              </Select>
            </FormControl>

            {/* 置信水平 */}
            <Box>
              <Typography gutterBottom>
                置信水平: {(confidenceLevel * 100).toFixed(0)}%
              </Typography>
              <Slider
                value={confidenceLevel}
                onChange={(_, value) => setConfidenceLevel(value as number)}
                min={0.8}
                max={0.99}
                step={0.01}
                marks={[
                  { value: 0.8, label: '80%' },
                  { value: 0.9, label: '90%' },
                  { value: 0.95, label: '95%' },
                  { value: 0.99, label: '99%' },
                ]}
              />
            </Box>

            {/* 错误提示 */}
            {error && (
              <Alert severity="error" onClose={() => setError(null)}>
                {error}
              </Alert>
            )}

            {/* 预测按钮 */}
            <Button
              variant="contained"
              size="large"
              onClick={handlePredict}
              disabled={predicting || !selectedModel}
              startIcon={predicting ? <CircularProgress size={20} /> : <Brain size={20} />}
            >
              {predicting ? '预测中...' : '开始预测'}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* 预测结果卡片 */}
      {predictions.length > 0 && (
        <Card>
          <CardHeader
            title="预测结果"
            action={
              <Button
                startIcon={<RefreshCw size={16} />}
                onClick={handlePredict}
                disabled={predicting}
              >
                刷新
              </Button>
            }
          />
          <CardContent>
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>股票代码</TableCell>
                    <TableCell align="center">预测方向</TableCell>
                    <TableCell align="right">预测收益率</TableCell>
                    <TableCell align="right">预测价格</TableCell>
                    <TableCell align="center">置信度</TableCell>
                    <TableCell align="center">置信区间</TableCell>
                    <TableCell align="center">风险评估</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {predictions.map((prediction) => {
                    const direction = getDirectionDisplay(prediction.predicted_direction);
                    return (
                      <TableRow key={prediction.stock_code}>
                        <TableCell>
                          <Typography fontWeight="bold">
                            {prediction.stock_code}
                          </Typography>
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            icon={direction.icon}
                            label={direction.text}
                            sx={{
                              backgroundColor: `${direction.color}20`,
                              color: direction.color,
                              fontWeight: 'bold',
                            }}
                          />
                        </TableCell>
                        <TableCell align="right">
                          <Typography
                            color={prediction.predicted_return >= 0 ? 'success.main' : 'error.main'}
                            fontWeight="bold"
                          >
                            {prediction.predicted_return >= 0 ? '+' : ''}
                            {formatPercent(prediction.predicted_return)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          {prediction.predicted_price
                            ? `¥${prediction.predicted_price.toFixed(2)}`
                            : '-'}
                        </TableCell>
                        <TableCell align="center">
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                            <CircularProgress
                              variant="determinate"
                              value={prediction.confidence_score * 100}
                              size={40}
                              sx={{
                                color: getConfidenceColor(prediction.confidence_score),
                              }}
                            />
                            <Typography
                              variant="body2"
                              color={getConfidenceColor(prediction.confidence_score)}
                              fontWeight="bold"
                            >
                              {formatPercent(prediction.confidence_score)}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell align="center">
                          <Typography variant="body2">
                            [{formatPercent(prediction.confidence_interval.lower)},{' '}
                            {formatPercent(prediction.confidence_interval.upper)}]
                          </Typography>
                        </TableCell>
                        <TableCell align="center">
                          <Tooltip
                            title={
                              <Box>
                                <Typography variant="body2">
                                  VaR: {prediction.risk_assessment.value_at_risk
                                    ? formatPercent(prediction.risk_assessment.value_at_risk)
                                    : 'N/A'}
                                </Typography>
                                <Typography variant="body2">
                                  波动率: {prediction.risk_assessment.volatility
                                    ? formatPercent(prediction.risk_assessment.volatility)
                                    : 'N/A'}
                                </Typography>
                                <Typography variant="body2">
                                  最大回撤: {prediction.risk_assessment.max_drawdown
                                    ? formatPercent(prediction.risk_assessment.max_drawdown)
                                    : 'N/A'}
                                </Typography>
                              </Box>
                            }
                          >
                            <IconButton size="small">
                              {prediction.risk_assessment.value_at_risk &&
                              prediction.risk_assessment.value_at_risk > 0.1 ? (
                                <AlertTriangle size={20} color="#f44336" />
                              ) : (
                                <CheckCircle size={20} color="#4caf50" />
                              )}
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>

            {/* 预测说明 */}
            <Alert severity="info" sx={{ mt: 2 }}>
              <Typography variant="body2">
                <strong>说明：</strong>预测结果仅供参考，不构成投资建议。
                置信度表示模型对预测结果的确信程度，置信区间表示预测值可能的波动范围。
                风险评估包括 VaR（风险价值）、波动率和最大回撤等指标。
              </Typography>
            </Alert>
          </CardContent>
        </Card>
      )}

      {/* 空状态 */}
      {predictions.length === 0 && !predicting && (
        <Card>
          <CardContent>
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                py: 6,
                color: 'text.secondary',
              }}
            >
              <Brain size={64} style={{ opacity: 0.3 }} />
              <Typography variant="h6" sx={{ mt: 2 }}>
                暂无预测结果
              </Typography>
              <Typography variant="body2">
                选择模型和股票代码，点击"开始预测"按钮进行预测分析
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
