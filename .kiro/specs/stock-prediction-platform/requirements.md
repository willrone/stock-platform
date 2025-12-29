# 需求文档

## 介绍

股票预测平台是一个综合性的金融分析系统，通过从Tushare获取实时股票数据，结合技术指标计算和机器学习模型，为用户提供股票投资决策支持。平台包含完整的前端用户界面和后端数据处理服务。

## 术语表

- **Stock_Data_Service**: 负责集成现有back_test_data_service（192.168.3.62）并管理本地Parquet文件的服务
- **Prediction_Engine**: 基于任务管理的股票预测引擎，支持任务创建、进度跟踪和结果管理
- **Technical_Indicator_Calculator**: 计算各种技术指标（如MA、RSI、MACD等）的计算器
- **Model_Training_Service**: 负责训练和管理机器学习模型的服务
- **Web_Frontend**: 基于现代前端框架和开源金融组件构建的用户交互界面
- **API_Gateway**: 后端API网关，处理前后端通信
- **Database**: 统一使用Parquet格式存储时序数据，SQLite存储任务信息、预测结果和系统配置
- **User_Management_System**: 用户认证和权限管理系统（暂不实现）

## 需求

### 需求 1: 数据获取与集成

**用户故事:** 作为系统集成者，我希望能够集成现有的Tushare数据服务并统一使用Parquet格式，以便为预测分析提供高效的数据基础。

#### 验收标准

1. WHEN 系统启动时，THE Stock_Data_Service SHALL 连接到现有的back_test_data_service（192.168.3.62）
2. WHEN 请求股票数据时，THE Stock_Data_Service SHALL 首先检查本地Parquet文件是否存在，如不存在则从远端服务获取
3. WHEN 从远端获取数据时，THE Stock_Data_Service SHALL 将数据以Parquet格式保存到本地存储
4. WHEN 数据服务不可用时，THE Stock_Data_Service SHALL 使用本地Parquet文件并记录服务状态
5. THE Stock_Data_Service SHALL 支持增量更新，仅获取缺失的时间段数据

### 需求 2: 技术指标计算

**用户故事:** 作为量化分析师，我希望系统能够计算各种技术指标，以便进行技术分析和模型特征工程。

#### 验收标准

1. WHEN 提供股票价格数据时，THE Technical_Indicator_Calculator SHALL 计算移动平均线（MA5、MA10、MA20、MA60）
2. WHEN 提供价格和成交量数据时，THE Technical_Indicator_Calculator SHALL 计算RSI、MACD、布林带指标
3. WHEN 计算技术指标时，THE Technical_Indicator_Calculator SHALL 验证输入数据的完整性和有效性
4. WHEN 指标计算完成时，THE Technical_Indicator_Calculator SHALL 返回结构化的指标数据
5. THE Technical_Indicator_Calculator SHALL 支持批量计算多只股票的技术指标

### 需求 3: 机器学习模型训练

**用户故事:** 作为数据科学家，我希望能够训练基于现代深度学习架构的股票预测模型，以便提供高精度的预测服务。

#### 验收标准

1. WHEN 提供训练数据时，THE Model_Training_Service SHALL 预处理数据并构建多模态特征集（价格序列、技术指标、基本面数据）
2. WHEN 开始训练时，THE Model_Training_Service SHALL 支持现代算法（Transformer、TimesNet、PatchTST、Informer、以及传统的LSTM、XGBoost作为基线）
3. WHEN 模型训练完成时，THE Model_Training_Service SHALL 评估模型性能（准确率、夏普比率、最大回撤）并保存最佳模型
4. WHEN 模型验证时，THE Model_Training_Service SHALL 使用时间序列交叉验证确保模型在不同市场环境下的泛化能力
5. THE Model_Training_Service SHALL 支持模型集成（ensemble）、在线学习和增量训练功能

### 需求 4: 预测任务管理和执行

**用户故事:** 作为投资者，我希望通过任务式的方式管理股票预测，以便跟踪预测进度并查看历史预测结果。

#### 验收标准

1. WHEN 用户创建预测任务时，THE Prediction_Engine SHALL 保存任务信息（股票列表、预测参数、创建时间）并分配唯一任务ID
2. WHEN 执行预测任务时，THE Prediction_Engine SHALL 实时更新任务进度状态（待执行、进行中、已完成、失败）
3. WHEN 预测任务完成时，THE Prediction_Engine SHALL 保存每只股票的详细预测结果（盈利概率、置信区间、风险评估）
4. WHEN 用户查看任务结果时，THE Prediction_Engine SHALL 展示每只股票的交易记录、技术指标计算和回测性能指标
5. THE Prediction_Engine SHALL 支持历史任务查询、任务结果对比和任务重新执行功能

### 需求 5: Web前端界面

**用户故事:** 作为普通用户，我希望通过现代化的网页界面管理预测任务、查看预测结果和管理数据服务，以便轻松使用平台功能。

#### 验收标准

1. WHEN 用户访问平台时，THE Web_Frontend SHALL 基于现代React/Vue框架显示任务管理界面，包括创建新任务和查看任务列表功能
2. WHEN 用户创建预测任务时，THE Web_Frontend SHALL 提供股票选择器、预测参数配置和任务信息输入界面
3. WHEN 显示任务进度时，THE Web_Frontend SHALL 实时更新任务状态并显示进度条和当前处理的股票信息
4. WHEN 用户查看任务结果时，THE Web_Frontend SHALL 展示每只股票的详细信息（价格走势图、技术指标、预测结果、交易记录、回测指标）
5. WHEN 用户访问数据管理页面时，THE Web_Frontend SHALL 显示远端数据服务状态、本地Parquet文件列表和数据同步控制功能

### 需求 6: API网关和后端服务

**用户故事:** 作为前端开发者，我希望有统一的API接口访问后端服务，以便构建稳定的用户界面。

#### 验收标准

1. WHEN 前端发送请求时，THE API_Gateway SHALL 验证请求格式并处理基本的输入验证
2. WHEN 处理API请求时，THE API_Gateway SHALL 路由请求到相应的后端服务
3. WHEN 返回响应时，THE API_Gateway SHALL 统一响应格式并处理错误
4. WHEN 系统负载较高时，THE API_Gateway SHALL 实施基本的限流策略
5. THE API_Gateway SHALL 提供完整的API文档和版本管理

### 需求 7: 数据存储和管理

**用户故事:** 作为系统架构师，我希望使用统一的Parquet格式进行数据存储，以便简化架构并提高数据处理效率。

#### 验收标准

1. WHEN 存储股票数据时，THE Database SHALL 统一使用Parquet格式存储所有时间序列数据（价格、成交量、技术指标）
2. WHEN 存储任务和系统数据时，THE Database SHALL 使用轻量级SQLite数据库存储预测任务信息、任务结果、模型元数据和系统配置
3. WHEN 进行回测任务时，THE Database SHALL 首先检查本地Parquet文件，缺失数据时自动从远端服务获取
4. WHEN 管理数据文件时，THE Database SHALL 支持按股票代码和时间范围组织Parquet文件的目录结构
5. THE Database SHALL 实施简单的数据清理策略，定期清理过期的临时文件和日志，但保留历史任务记录