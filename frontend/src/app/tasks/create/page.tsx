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
}