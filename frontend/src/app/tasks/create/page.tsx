<<<<<<< HEAD
/**
 * 任务创建页面
 * 
 * 提供任务创建表单，包括：
 * - 股票选择器
 * - 模型选择
 * - 参数配置
 * - 预测设置
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  Form,
  Input,
  Select,
  Button,
  Space,
  Typography,
  Row,
  Col,
  Divider,
  InputNumber,
  Switch,
  message,
  Tag,
} from 'antd';
import {
  ArrowLeftOutlined,
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import { useDataStore } from '../../../stores/useDataStore';
import { useTaskStore } from '../../../stores/useTaskStore';
import { TaskService, CreateTaskRequest } from '../../../services/taskService';
import { DataService } from '../../../services/dataService';
import { StockSelector } from '../../../components/tasks/StockSelector';
import { LoadingSpinner } from '../../../components/common/LoadingSpinner';

const { Title, Text } = Typography;
const { Option } = Select;
const { TextArea } = Input;



export default function CreateTaskPage() {
  const router = useRouter();
  const [form] = Form.useForm();
  const { models, selectedModel, setModels, setSelectedModel } = useDataStore();
  const { setCreating } = useTaskStore();

  const [loading, setLoading] = useState(false);
  const [selectedStocks, setSelectedStocks] = useState<string[]>([]);

  // 加载模型列表
  useEffect(() => {
    const loadModels = async () => {
      try {
        const result = await DataService.getModels();
        setModels(result.models);
        if (result.models.length > 0 && !selectedModel) {
          setSelectedModel(result.models[0]);
          form.setFieldsValue({ model_id: result.models[0].model_id });
        }
      } catch (error) {
        message.error('加载模型列表失败');
      }
    };

    loadModels();
  }, [setModels, setSelectedModel, selectedModel, form]);


  // 提交表单
  const handleSubmit = async (values: any) => {
    if (selectedStocks.length === 0) {
      message.error('请至少选择一只股票');
      return;
    }

    setLoading(true);
    setCreating(true);

    try {
      const request: CreateTaskRequest = {
        task_name: values.task_name,
        stock_codes: selectedStocks,
        model_id: values.model_id,
        prediction_config: {
          horizon: values.horizon,
          confidence_level: values.confidence_level / 100,
          risk_assessment: values.risk_assessment,
        },
      };

      const task = await TaskService.createTask(request);
      message.success('任务创建成功');
      router.push(`/tasks/${task.task_id}`);
    } catch (error) {
      message.error('创建任务失败');
      console.error('创建任务失败:', error);
    } finally {
      setLoading(false);
      setCreating(false);
    }
  };

  // 返回任务列表
  const handleBack = () => {
    router.push('/tasks');
  };

  return (
    <div>
      {/* 页面标题 */}
      <div style={{ marginBottom: 24 }}>
        <Space>
          <Button
            icon={<ArrowLeftOutlined />}
            onClick={handleBack}
          >
            返回
          </Button>
          <Title level={2} style={{ margin: 0 }}>
            创建预测任务
          </Title>
        </Space>
      </div>

      <Row gutter={24}>
        <Col span={16}>
          <Card title="任务配置">
            <Form
              form={form}
              layout="vertical"
              onFinish={handleSubmit}
              initialValues={{
                horizon: 'short_term',
                confidence_level: 95,
                risk_assessment: true,
              }}
            >
              {/* 基本信息 */}
              <Form.Item
                label="任务名称"
                name="task_name"
                rules={[{ required: true, message: '请输入任务名称' }]}
              >
                <Input placeholder="请输入任务名称" />
              </Form.Item>

              <Form.Item
                label="任务描述"
                name="description"
              >
                <TextArea
                  rows={3}
                  placeholder="请输入任务描述（可选）"
                />
              </Form.Item>

              <Divider>股票选择</Divider>

              {/* 股票选择器 */}
              <Form.Item
                label="选择股票"
                name="stock_codes"
                rules={[{ required: true, message: '请至少选择一只股票' }]}
              >
                <StockSelector
                  value={selectedStocks}
                  onChange={(stocks) => {
                    setSelectedStocks(stocks);
                    form.setFieldsValue({ stock_codes: stocks });
                  }}
                  maxCount={20}
                  placeholder="搜索股票代码或名称"
                />
              </Form.Item>

              <Divider>模型和参数</Divider>

              {/* 模型选择 */}
              <Form.Item
                label="预测模型"
                name="model_id"
                rules={[{ required: true, message: '请选择预测模型' }]}
              >
                <Select
                  placeholder="请选择预测模型"
                  onChange={(value) => {
                    const model = models.find(m => m.model_id === value);
                    setSelectedModel(model || null);
                  }}
                >
                  {models.map(model => (
                    <Option key={model.model_id} value={model.model_id}>
                      <div>
                        <Text strong>{model.model_name}</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          准确率: {(model.accuracy * 100).toFixed(1)}% | 类型: {model.model_type}
                        </Text>
                      </div>
                    </Option>
                  ))}
                </Select>
              </Form.Item>

              {/* 预测参数 */}
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    label="预测时间维度"
                    name="horizon"
                    rules={[{ required: true, message: '请选择预测时间维度' }]}
                  >
                    <Select>
                      <Option value="intraday">日内 (1-4小时)</Option>
                      <Option value="short_term">短期 (1-5天)</Option>
                      <Option value="medium_term">中期 (1-4周)</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="置信水平 (%)"
                    name="confidence_level"
                    rules={[{ required: true, message: '请设置置信水平' }]}
                  >
                    <InputNumber
                      min={80}
                      max={99}
                      step={1}
                      style={{ width: '100%' }}
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Form.Item
                label="风险评估"
                name="risk_assessment"
                valuePropName="checked"
              >
                <Switch />
                <Text type="secondary" style={{ marginLeft: 8 }}>
                  启用风险评估和VaR计算
                </Text>
              </Form.Item>

              {/* 提交按钮 */}
              <Form.Item style={{ marginTop: 32 }}>
                <Space>
                  <Button
                    type="primary"
                    htmlType="submit"
                    loading={loading}
                    size="large"
                  >
                    创建任务
                  </Button>
                  <Button
                    size="large"
                    onClick={handleBack}
                  >
                    取消
                  </Button>
                </Space>
              </Form.Item>
            </Form>
          </Card>
        </Col>

        <Col span={8}>
          {/* 模型信息 */}
          {selectedModel && (
            <Card title="模型信息" style={{ marginBottom: 16 }}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text strong>模型名称:</Text>
                  <br />
                  <Text>{selectedModel.model_name}</Text>
                </div>
                <div>
                  <Text strong>模型类型:</Text>
                  <br />
                  <Tag>{selectedModel.model_type}</Tag>
                </div>
                <div>
                  <Text strong>准确率:</Text>
                  <br />
                  <Text>{(selectedModel.accuracy * 100).toFixed(1)}%</Text>
                </div>
                <div>
                  <Text strong>版本:</Text>
                  <br />
                  <Text>{selectedModel.version}</Text>
                </div>
                <div>
                  <Text strong>创建时间:</Text>
                  <br />
                  <Text type="secondary">
                    {new Date(selectedModel.created_at).toLocaleDateString()}
                  </Text>
                </div>
              </Space>
            </Card>
          )}

          {/* 预测说明 */}
          <Card title="预测说明">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>时间维度说明:</Text>
                <ul style={{ marginTop: 8, paddingLeft: 20 }}>
                  <li><Text type="secondary">日内: 适合短线交易</Text></li>
                  <li><Text type="secondary">短期: 适合波段操作</Text></li>
                  <li><Text type="secondary">中期: 适合趋势投资</Text></li>
                </ul>
              </div>
              <div>
                <Text strong>置信水平:</Text>
                <br />
                <Text type="secondary">
                  置信水平越高，预测区间越宽，但可靠性越高
                </Text>
              </div>
              <div>
                <Text strong>风险评估:</Text>
                <br />
                <Text type="secondary">
                  包含VaR计算和波动率分析，帮助评估投资风险
                </Text>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
=======
/**
 * 任务创建页面
 * 
 * 提供任务创建表单，包括：
 * - 股票选择器
 * - 模型选择
 * - 参数配置
 * - 预测设置
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Button,
  Input,
  Select,
  SelectItem,
  Textarea,
  Switch,
  Slider,
  Chip,
  Divider,
} from '@heroui/react';
import {
  ArrowLeft,
  Settings,
  Target,
  TrendingUp,
  Shield,
  Info,
} from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useDataStore } from '../../../stores/useDataStore';
import { useTaskStore } from '../../../stores/useTaskStore';
import { TaskService, CreateTaskRequest } from '../../../services/taskService';
import { DataService } from '../../../services/dataService';
import { StockSelector } from '../../../components/tasks/StockSelector';
import { LoadingSpinner } from '../../../components/common/LoadingSpinner';



export default function CreateTaskPage() {
  const router = useRouter();
  const { models, selectedModel, setModels, setSelectedModel } = useDataStore();
  const { setCreating } = useTaskStore();

  const [loading, setLoading] = useState(false);
  const [selectedStocks, setSelectedStocks] = useState<string[]>([]);
  const [formData, setFormData] = useState({
    task_name: '',
    description: '',
    model_id: '',
    horizon: 'short_term' as 'intraday' | 'short_term' | 'medium_term',
    confidence_level: 95,
    risk_assessment: true,
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  // 加载模型列表
  useEffect(() => {
    const loadModels = async () => {
      try {
        const result = await DataService.getModels();
        setModels(result.models);
        if (result.models.length > 0 && !selectedModel) {
          setSelectedModel(result.models[0]);
          setFormData(prev => ({ ...prev, model_id: result.models[0].model_id }));
        }
      } catch (error) {
        console.error('加载模型列表失败');
      }
    };

    loadModels();
  }, [setModels, setSelectedModel, selectedModel]);

  // 表单验证
  const validateForm = () => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.task_name.trim()) {
      newErrors.task_name = '请输入任务名称';
    }
    
    if (selectedStocks.length === 0) {
      newErrors.stock_codes = '请至少选择一只股票';
    }
    
    if (!formData.model_id) {
      newErrors.model_id = '请选择预测模型';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // 提交表单
  const handleSubmit = async () => {
    if (!validateForm()) {
      return;
    }

    setLoading(true);
    setCreating(true);

    try {
      const request: CreateTaskRequest = {
        task_name: formData.task_name,
        stock_codes: selectedStocks,
        model_id: formData.model_id,
        prediction_config: {
          horizon: formData.horizon,
          confidence_level: formData.confidence_level / 100,
          risk_assessment: formData.risk_assessment,
        },
      };

      const task = await TaskService.createTask(request);
      console.log('任务创建成功');
      router.push(`/tasks/${task.task_id}`);
    } catch (error) {
      console.error('创建任务失败:', error);
    } finally {
      setLoading(false);
      setCreating(false);
    }
  };

  // 返回任务列表
  const handleBack = () => {
    router.push('/tasks');
  };

  // 更新表单数据
  const updateFormData = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    // 清除对应字段的错误
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div className="flex items-center space-x-4">
        <Button
          isIconOnly
          variant="light"
          onPress={handleBack}
        >
          <ArrowLeft className="w-4 h-4" />
        </Button>
        <div>
          <h1 className="text-2xl font-bold">创建预测任务</h1>
          <p className="text-default-500">配置股票预测任务参数</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 主要配置区域 */}
        <div className="lg:col-span-2 space-y-6">
          {/* 基本信息 */}
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-2">
                <Settings className="w-5 h-5" />
                <h3 className="text-lg font-semibold">基本信息</h3>
              </div>
            </CardHeader>
            <CardBody className="space-y-4">
              <Input
                label="任务名称"
                placeholder="请输入任务名称"
                value={formData.task_name}
                onValueChange={(value) => updateFormData('task_name', value)}
                isInvalid={!!errors.task_name}
                errorMessage={errors.task_name}
                isRequired
              />
              
              <Textarea
                label="任务描述"
                placeholder="请输入任务描述（可选）"
                value={formData.description}
                onValueChange={(value) => updateFormData('description', value)}
                minRows={3}
              />
            </CardBody>
          </Card>

          {/* 股票选择 */}
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-2">
                <Target className="w-5 h-5" />
                <h3 className="text-lg font-semibold">股票选择</h3>
              </div>
            </CardHeader>
            <CardBody>
              <div className="space-y-4">
                <StockSelector
                  value={selectedStocks}
                  onChange={(stocks) => {
                    setSelectedStocks(stocks);
                    if (errors.stock_codes) {
                      setErrors(prev => ({ ...prev, stock_codes: '' }));
                    }
                  }}
                  maxCount={20}
                  placeholder="搜索股票代码或名称"
                />
                {errors.stock_codes && (
                  <p className="text-danger text-sm">{errors.stock_codes}</p>
                )}
                {selectedStocks.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {selectedStocks.map(stock => (
                      <Chip
                        key={stock}
                        onClose={() => {
                          setSelectedStocks(prev => prev.filter(s => s !== stock));
                        }}
                        variant="flat"
                      >
                        {stock}
                      </Chip>
                    ))}
                  </div>
                )}
              </div>
            </CardBody>
          </Card>

          {/* 模型和参数 */}
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-2">
                <TrendingUp className="w-5 h-5" />
                <h3 className="text-lg font-semibold">模型和参数</h3>
              </div>
            </CardHeader>
            <CardBody className="space-y-6">
              <Select
                label="预测模型"
                placeholder="请选择预测模型"
                selectedKeys={formData.model_id ? [formData.model_id] : []}
                onSelectionChange={(keys) => {
                  const modelId = Array.from(keys)[0] as string;
                  updateFormData('model_id', modelId);
                  const model = models.find(m => m.model_id === modelId);
                  setSelectedModel(model || null);
                }}
                isInvalid={!!errors.model_id}
                errorMessage={errors.model_id}
                isRequired
              >
                {models.map(model => (
                  <SelectItem key={model.model_id}>
                    <div className="flex flex-col">
                      <span className="font-medium">{model.model_name}</span>
                      <span className="text-xs text-default-500">
                        准确率: {(model.accuracy * 100).toFixed(1)}% | 类型: {model.model_type}
                      </span>
                    </div>
                  </SelectItem>
                ))}
              </Select>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Select
                  label="预测时间维度"
                  selectedKeys={[formData.horizon]}
                  onSelectionChange={(keys) => {
                    const horizon = Array.from(keys)[0] as string;
                    updateFormData('horizon', horizon);
                  }}
                >
                  <SelectItem key="intraday">
                    日内 (1-4小时)
                  </SelectItem>
                  <SelectItem key="short_term">
                    短期 (1-5天)
                  </SelectItem>
                  <SelectItem key="medium_term">
                    中期 (1-4周)
                  </SelectItem>
                </Select>

                <div className="space-y-2">
                  <label className="text-sm font-medium">置信水平: {formData.confidence_level}%</label>
                  <Slider
                    value={formData.confidence_level}
                    onChange={(value) => updateFormData('confidence_level', Array.isArray(value) ? value[0] : value)}
                    minValue={80}
                    maxValue={99}
                    step={1}
                    className="w-full"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Shield className="w-5 h-5 text-primary" />
                  <div>
                    <p className="font-medium">风险评估</p>
                    <p className="text-sm text-default-500">启用风险评估和VaR计算</p>
                  </div>
                </div>
                <Switch
                  isSelected={formData.risk_assessment}
                  onValueChange={(value) => updateFormData('risk_assessment', value)}
                />
              </div>
            </CardBody>
          </Card>

          {/* 提交按钮 */}
          <div className="flex space-x-4">
            <Button
              color="primary"
              size="lg"
              onPress={handleSubmit}
              isLoading={loading}
              className="flex-1"
            >
              创建任务
            </Button>
            <Button
              variant="light"
              size="lg"
              onPress={handleBack}
            >
              取消
            </Button>
          </div>
        </div>

        {/* 侧边栏信息 */}
        <div className="space-y-6">
          {/* 模型信息 */}
          {selectedModel && (
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">模型信息</h3>
              </CardHeader>
              <CardBody className="space-y-4">
                <div>
                  <p className="text-sm text-default-500">模型名称</p>
                  <p className="font-medium">{selectedModel.model_name}</p>
                </div>
                <div>
                  <p className="text-sm text-default-500">模型类型</p>
                  <Chip variant="flat" color="secondary">{selectedModel.model_type}</Chip>
                </div>
                <div>
                  <p className="text-sm text-default-500">准确率</p>
                  <p className="font-medium">{(selectedModel.accuracy * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-sm text-default-500">版本</p>
                  <p className="font-medium">{selectedModel.version}</p>
                </div>
                <div>
                  <p className="text-sm text-default-500">创建时间</p>
                  <p className="text-default-600">
                    {new Date(selectedModel.created_at).toLocaleDateString()}
                  </p>
                </div>
              </CardBody>
            </Card>
          )}

          {/* 预测说明 */}
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-2">
                <Info className="w-5 h-5" />
                <h3 className="text-lg font-semibold">预测说明</h3>
              </div>
            </CardHeader>
            <CardBody className="space-y-4">
              <div>
                <p className="font-medium mb-2">时间维度说明:</p>
                <ul className="space-y-1 text-sm text-default-600">
                  <li>• 日内: 适合短线交易</li>
                  <li>• 短期: 适合波段操作</li>
                  <li>• 中期: 适合趋势投资</li>
                </ul>
              </div>
              <Divider />
              <div>
                <p className="font-medium mb-2">置信水平:</p>
                <p className="text-sm text-default-600">
                  置信水平越高，预测区间越宽，但可靠性越高
                </p>
              </div>
              <Divider />
              <div>
                <p className="font-medium mb-2">风险评估:</p>
                <p className="text-sm text-default-600">
                  包含VaR计算和波动率分析，帮助评估投资风险
                </p>
              </div>
            </CardBody>
          </Card>
        </div>
      </div>
    </div>
  );
>>>>>>> a6754c6 (feat(platform): Complete stock prediction platform with deployment and monitoring)
}