/**
 * 创建模型表单组件
 */

'use client';

import React from 'react';
import {
  Input,
  Select,
  SelectItem,
  Checkbox,
} from '@heroui/react';
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
    <div className="space-y-4">
      <Input
        label="模型名称"
        placeholder="请输入模型名称"
        value={formData.model_name}
        onValueChange={(value) => onFormDataChange('model_name', value)}
        isInvalid={!!errors.model_name}
        errorMessage={errors.model_name}
        isRequired
      />

      <Select
        label="模型类型"
        selectedKeys={[formData.model_type]}
        onSelectionChange={(keys) => {
          const type = Array.from(keys)[0] as string;
          onFormDataChange('model_type', type);
        }}
        description="选择要训练的模型类型（基于Qlib框架统一训练）"
      >
        <SelectItem key="lightgbm" description="推荐：高效的梯度提升模型，适合表格数据">
          LightGBM (推荐)
        </SelectItem>
        <SelectItem key="xgboost" description="经典的梯度提升模型，性能稳定">
          XGBoost
        </SelectItem>
        <SelectItem key="linear_regression" description="简单的线性回归模型，训练快速">
          线性回归
        </SelectItem>
        <SelectItem key="transformer" description="Transformer模型，适合复杂时序模式">
          Transformer
        </SelectItem>
      </Select>

      <StockSelector
        value={formData.stock_codes}
        onChange={(codes) => onFormDataChange('stock_codes', codes)}
      />
      {errors.stock_codes && (
        <p className="text-danger text-sm mt-1">{errors.stock_codes}</p>
      )}

      <div className="grid grid-cols-2 gap-4">
        <Input
          type="date"
          label="训练数据开始日期"
          value={formData.start_date}
          onValueChange={(value) => onFormDataChange('start_date', value)}
          isInvalid={!!errors.start_date}
          errorMessage={errors.start_date}
          isRequired
        />
        <Input
          type="date"
          label="训练数据结束日期"
          value={formData.end_date}
          onValueChange={(value) => onFormDataChange('end_date', value)}
          isInvalid={!!errors.end_date}
          errorMessage={errors.end_date}
          isRequired
        />
      </div>

      <Input
        label="模型描述（可选）"
        placeholder="请输入模型描述"
        value={formData.description}
        onValueChange={(value) => onFormDataChange('description', value)}
      />

      <Input
        type="number"
        label="训练迭代次数（Epochs）"
        placeholder="请输入训练迭代次数"
        value={String(formData.num_iterations)}
        onValueChange={(value) => {
          const num = parseInt(value) || 100;
          onFormDataChange('num_iterations', num);
          onFormDataChange('hyperparameters', {
            ...formData.hyperparameters,
            num_iterations: num
          });
        }}
        description="控制模型训练的迭代次数，LightGBM/XGBoost使用此参数。建议范围：50-1000，默认100"
        min={10}
        max={1000}
        isRequired
      />

      <Checkbox
        isSelected={formData.enable_hyperparameter_tuning}
        onValueChange={(checked) => onFormDataChange('enable_hyperparameter_tuning', checked)}
      >
        启用自动超参数调优
      </Checkbox>

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
    </div>
  );
}

