/**
 * 创建模型表单组件
 */

'use client';

import React from 'react';
import {
  TextField,
  Select,
  MenuItem,
  Checkbox,
  FormControl,
  InputLabel,
  FormHelperText,
  Box,
  Typography,
} from '@mui/material';
import { StockSelector } from '../tasks/StockSelector';
import { FeatureSelector } from './FeatureSelector';

interface CreateModelFormProps {
  formData: {
    model_name: string;
    model_type: string;
    stock_codes: string[];
    start_date: string;
    end_date: string;
    description: string;
    hyperparameters: Record<string, any>;
    enable_hyperparameter_tuning: boolean;
    num_iterations: number;
    selected_features: string[];
    feature_set: string;
    label_type: string;
    binary_threshold: number;
    split_method: string;
    train_end_date: string;
    val_end_date: string;
    // 滚动训练（P2）
    enable_rolling: boolean;
    rolling_window_type: string;
    rolling_step: number;
    rolling_train_window: number;
    rolling_valid_window: number;
    enable_sample_decay: boolean;
    sample_decay_rate: number;
    // CSRankNorm 标签变换
    enable_cs_rank_norm: boolean;
  };
  errors: Record<string, string>;
  useAllFeatures: boolean;
  onFormDataChange: (field: string, value: any) => void;
  onUseAllFeaturesChange: (useAll: boolean) => void;
  onSelectedFeaturesChange: (features: string[]) => void;
}

export function CreateModelForm({
  formData,
  errors,
  useAllFeatures,
  onFormDataChange,
  onUseAllFeaturesChange,
  onSelectedFeaturesChange,
}: CreateModelFormProps) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <TextField
        label="模型名称"
        placeholder="请输入模型名称"
        value={formData.model_name}
        onChange={e => onFormDataChange('model_name', e.target.value)}
        error={!!errors.model_name}
        helperText={errors.model_name}
        required
        fullWidth
      />

      <FormControl fullWidth>
        <InputLabel>模型类型</InputLabel>
        <Select
          value={formData.model_type}
          label="模型类型"
          onChange={e => onFormDataChange('model_type', e.target.value)}
        >
          <MenuItem value="lightgbm">
            <Box>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                LightGBM (推荐)
              </Typography>
              <Typography variant="caption" color="text.secondary">
                推荐：高效的梯度提升模型，适合表格数据
              </Typography>
            </Box>
          </MenuItem>
          <MenuItem value="xgboost">
            <Box>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                XGBoost
              </Typography>
              <Typography variant="caption" color="text.secondary">
                经典的梯度提升模型，性能稳定
              </Typography>
            </Box>
          </MenuItem>
          <MenuItem value="linear_regression">
            <Box>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                线性回归
              </Typography>
              <Typography variant="caption" color="text.secondary">
                简单的线性回归模型，训练快速
              </Typography>
            </Box>
          </MenuItem>
          <MenuItem value="transformer">
            <Box>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                Transformer
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Transformer模型，适合复杂时序模式
              </Typography>
            </Box>
          </MenuItem>
        </Select>
        <FormHelperText>选择要训练的模型类型（基于Qlib框架统一训练）</FormHelperText>
      </FormControl>

      <StockSelector
        value={formData.stock_codes}
        onChange={codes => onFormDataChange('stock_codes', codes)}
      />
      {errors.stock_codes && <FormHelperText error>{errors.stock_codes}</FormHelperText>}

      <Box
        sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}
      >
        <TextField
          type="date"
          label="训练数据开始日期"
          value={formData.start_date}
          onChange={e => onFormDataChange('start_date', e.target.value)}
          error={!!errors.start_date}
          helperText={errors.start_date}
          required
          fullWidth
          InputLabelProps={{ shrink: true }}
        />
        <TextField
          type="date"
          label="训练数据结束日期"
          value={formData.end_date}
          onChange={e => onFormDataChange('end_date', e.target.value)}
          error={!!errors.end_date}
          helperText={errors.end_date}
          required
          fullWidth
          InputLabelProps={{ shrink: true }}
        />
      </Box>

      <TextField
        label="模型描述（可选）"
        placeholder="请输入模型描述"
        value={formData.description}
        onChange={e => onFormDataChange('description', e.target.value)}
        fullWidth
      />

      <TextField
        type="number"
        label="训练迭代次数（Epochs）"
        placeholder="请输入训练迭代次数"
        value={formData.num_iterations}
        onChange={e => {
          const num = parseInt(e.target.value) || 1000;
          onFormDataChange('num_iterations', num);
          onFormDataChange('hyperparameters', {
            ...formData.hyperparameters,
            num_iterations: num,
          });
        }}
        helperText="控制模型训练的迭代次数。Qlib官方基准=1000，建议范围：100-5000"
        inputProps={{ min: 10, max: 5000 }}
        required
        fullWidth
      />

      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <Checkbox
          checked={formData.enable_hyperparameter_tuning}
          onChange={e => onFormDataChange('enable_hyperparameter_tuning', e.target.checked)}
        />
        <Typography variant="body2">启用自动超参数调优</Typography>
      </Box>

      {/* 特征集选择 */}
      <FormControl fullWidth>
        <InputLabel>特征集</InputLabel>
        <Select
          value={formData.feature_set}
          label="特征集"
          onChange={e => onFormDataChange('feature_set', e.target.value)}
        >
          <MenuItem value="alpha158">Alpha158（158个因子，推荐）</MenuItem>
          <MenuItem value="technical_62">手工62特征</MenuItem>
          <MenuItem value="custom">自定义</MenuItem>
        </Select>
        <FormHelperText>选择用于训练的特征集</FormHelperText>
      </FormControl>

      {/* 标签类型选择 */}
      <FormControl fullWidth>
        <InputLabel>标签类型</InputLabel>
        <Select
          value={formData.label_type}
          label="标签类型"
          onChange={e => onFormDataChange('label_type', e.target.value)}
        >
          <MenuItem value="regression">回归（预测收益率）</MenuItem>
          <MenuItem value="binary">二分类（预测涨跌）</MenuItem>
        </Select>
        <FormHelperText>回归适合排序选股，二分类适合涨跌判断</FormHelperText>
      </FormControl>

      {/* 二分类阈值（仅 binary 时显示） */}
      {formData.label_type === 'binary' && (
        <TextField
          type="number"
          label="二分类阈值"
          value={formData.binary_threshold}
          onChange={e => onFormDataChange('binary_threshold', parseFloat(e.target.value) || 0)}
          helperText="收益率高于此阈值标记为1（涨），否则为0（跌）。默认0.0"
          inputProps={{ step: 0.001 }}
          fullWidth
        />
      )}

      {/* CSRankNorm 标签变换 */}
      {formData.label_type === 'regression' && (
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Checkbox
            checked={formData.enable_cs_rank_norm}
            onChange={e => onFormDataChange('enable_cs_rank_norm', e.target.checked)}
          />
          <Box>
            <Typography variant="body2">启用 CSRankNorm 标签变换</Typography>
            <Typography variant="caption" color="text.secondary">
              截面排名标准化：将收益率标签映射为正态分布，提升 RankIC 稳定性
            </Typography>
          </Box>
        </Box>
      )}

      {/* 数据分割方式 */}
      <FormControl fullWidth>
        <InputLabel>数据分割方式</InputLabel>
        <Select
          value={formData.split_method}
          label="数据分割方式"
          onChange={e => onFormDataChange('split_method', e.target.value)}
        >
          <MenuItem value="purged_cv">Purged K-Fold（防信息泄漏，推荐）</MenuItem>
          <MenuItem value="ratio">比例分割（默认8:2）</MenuItem>
          <MenuItem value="hardcut">按日期分割</MenuItem>
        </Select>
        <FormHelperText>
          比例分割自动划分训练/验证/测试集，按日期分割需手动指定截止日期
        </FormHelperText>
      </FormControl>

      {/* 按日期分割时的日期选择器 */}
      {formData.split_method === 'hardcut' && (
        <Box
          sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}
        >
          <TextField
            type="date"
            label="训练截止日期"
            value={formData.train_end_date}
            onChange={e => onFormDataChange('train_end_date', e.target.value)}
            helperText="训练集的结束日期"
            fullWidth
            InputLabelProps={{ shrink: true }}
          />
          <TextField
            type="date"
            label="验证截止日期"
            value={formData.val_end_date}
            onChange={e => onFormDataChange('val_end_date', e.target.value)}
            helperText="验证集的结束日期，之后为测试集"
            fullWidth
            InputLabelProps={{ shrink: true }}
          />
        </Box>
      )}

      {/* 滚动训练配置（P2） */}
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <Checkbox
          checked={formData.enable_rolling}
          onChange={e => onFormDataChange('enable_rolling', e.target.checked)}
        />
        <Typography variant="body2">启用滚动训练（适应市场变化）</Typography>
      </Box>

      {formData.enable_rolling && (
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
            pl: 2,
            borderLeft: '2px solid #1976d2',
          }}
        >
          <FormControl fullWidth>
            <InputLabel>窗口类型</InputLabel>
            <Select
              value={formData.rolling_window_type}
              label="窗口类型"
              onChange={e => onFormDataChange('rolling_window_type', e.target.value)}
            >
              <MenuItem value="sliding">固定窗口滑动（Sliding）</MenuItem>
              <MenuItem value="expanding">扩展窗口（Expanding）</MenuItem>
            </Select>
            <FormHelperText>
              Sliding: 固定大小窗口向前滑动；Expanding: 训练窗口不断扩大
            </FormHelperText>
          </FormControl>

          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' },
              gap: 2,
            }}
          >
            <TextField
              type="number"
              label="滚动步长（交易日）"
              value={formData.rolling_step}
              onChange={e => onFormDataChange('rolling_step', parseInt(e.target.value) || 60)}
              helperText="每隔多少天重新训练，默认60"
              inputProps={{ min: 20, max: 240 }}
              fullWidth
            />
            <TextField
              type="number"
              label="训练窗口（交易日）"
              value={formData.rolling_train_window}
              onChange={e =>
                onFormDataChange('rolling_train_window', parseInt(e.target.value) || 480)
              }
              helperText="训练数据天数，默认480"
              inputProps={{ min: 120, max: 1200 }}
              fullWidth
            />
            <TextField
              type="number"
              label="验证窗口（交易日）"
              value={formData.rolling_valid_window}
              onChange={e =>
                onFormDataChange('rolling_valid_window', parseInt(e.target.value) || 60)
              }
              helperText="验证数据天数，默认60"
              inputProps={{ min: 20, max: 240 }}
              fullWidth
            />
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Checkbox
              checked={formData.enable_sample_decay}
              onChange={e => onFormDataChange('enable_sample_decay', e.target.checked)}
            />
            <Typography variant="body2">样本时间衰减权重</Typography>
          </Box>

          {formData.enable_sample_decay && (
            <TextField
              type="number"
              label="衰减率（每天）"
              value={formData.sample_decay_rate}
              onChange={e =>
                onFormDataChange('sample_decay_rate', parseFloat(e.target.value) || 0.999)
              }
              helperText="近期样本权重更高，默认0.999（每天衰减0.1%）"
              inputProps={{ min: 0.99, max: 1.0, step: 0.001 }}
              fullWidth
            />
          )}
        </Box>
      )}

      {/* 特征选择部分 */}
      <FeatureSelector
        startDate={formData.start_date}
        endDate={formData.end_date}
        selectedFeatures={formData.selected_features}
        onFeaturesChange={onSelectedFeaturesChange}
        useAllFeatures={useAllFeatures}
        onUseAllFeaturesChange={onUseAllFeaturesChange}
      />
    </Box>
  );
}
