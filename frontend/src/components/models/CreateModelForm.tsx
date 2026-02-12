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
          const num = parseInt(e.target.value) || 100;
          onFormDataChange('num_iterations', num);
          onFormDataChange('hyperparameters', {
            ...formData.hyperparameters,
            num_iterations: num,
          });
        }}
        helperText="控制模型训练的迭代次数，LightGBM/XGBoost使用此参数。建议范围：50-1000，默认100"
        inputProps={{ min: 10, max: 1000 }}
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

      {/* 数据分割方式 */}
      <FormControl fullWidth>
        <InputLabel>数据分割方式</InputLabel>
        <Select
          value={formData.split_method}
          label="数据分割方式"
          onChange={e => onFormDataChange('split_method', e.target.value)}
        >
          <MenuItem value="ratio">比例分割（默认8:1:1）</MenuItem>
          <MenuItem value="hardcut">按日期分割</MenuItem>
        </Select>
        <FormHelperText>比例分割自动划分训练/验证/测试集，按日期分割需手动指定截止日期</FormHelperText>
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

      {/* 特征选择部分 */}
      <FeatureSelector
        stockCodes={formData.stock_codes}
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
