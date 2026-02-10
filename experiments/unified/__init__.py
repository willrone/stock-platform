"""
统一 ML 训练 Pipeline

合并独立训练脚本（62 个手工特征）和 Qlib 引擎（数据预处理流程）

模块：
  - config: 训练配置数据类
  - constants: 常量定义
  - data_loader: 数据加载与 Embargo 分割
  - features: 62 个技术因子计算
  - fundamental_factors: 13 个基本面因子（PE/PB/ROE 等）
  - cross_validation: Purged Group Time Series Split
  - trainer: LightGBM/XGBoost 训练
  - stacking: Stacking 集成（Ridge meta-learner）
  - evaluation: 回测评估
  - model_io: 模型保存/加载
"""
