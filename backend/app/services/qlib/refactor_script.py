#!/usr/bin/env python3
"""
重构脚本：将 unified_qlib_training_engine.py 拆分为多个模块
"""

import os
import re
from pathlib import Path

# 读取原始文件
source_file = Path('unified_qlib_training_engine.py')
with open(source_file, 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')

# 创建输出目录
output_dir = Path('training')
output_dir.mkdir(exist_ok=True)

print(f"开始重构 {source_file.name} ({len(lines)} 行)")
print(f"输出目录: {output_dir}")

# ============================================================================
# 1. 提取导入语句和 QLIB_AVAILABLE 检查
# ============================================================================
imports_section = '\n'.join(lines[0:75])

# ============================================================================
# 2. 创建 qlib_check.py - Qlib可用性检查
# ============================================================================
qlib_check_content = '''"""
Qlib可用性检查模块
"""

from loguru import logger

# 检测Qlib可用性
try:
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.filter import ExpressionDFilter, NameDFilter
    from qlib.utils import init_instance_by_config

    QLIB_AVAILABLE = True
    logger.info("Qlib库已成功导入")
except ImportError as e:
    error_msg = str(e)
    missing_module = None

    # 检测缺失的模块
    if "setuptools_scm" in error_msg:
        missing_module = "setuptools_scm"
    elif "ruamel" in error_msg or "ruamel.yaml" in error_msg:
        missing_module = "ruamel.yaml"
    elif "cvxpy" in error_msg:
        missing_module = "cvxpy"
    elif "lightgbm" in error_msg:
        missing_module = "lightgbm"

    if missing_module:
        logger.warning(
            f"Qlib缺少依赖 {missing_module}。导入错误: {e}\\n"
            f"解决方法: pip install {missing_module}\\n"
            f"如果还有其他依赖缺失，请运行修复脚本: ./fix_qlib_dependencies.sh\\n"
            f"或手动安装所有依赖: pip install setuptools_scm cvxpy dill fire gym jupyter lightgbm matplotlib mlflow nbconvert pymongo python-redis-lock redis 'ruamel.yaml>=0.17.38'\\n"
            f"详细说明: 查看 backend/QLIB_INSTALLATION.md"
        )
    else:
        logger.warning(
            f"Qlib未安装或缺少依赖。导入错误: {e}\\n"
            f"安装方法: pip install git+https://github.com/microsoft/qlib.git\\n"
            f"或使用 Gitee 镜像: pip install git+https://gitee.com/mirrors/qlib.git\\n"
            f"如果已安装但缺少依赖，请运行: ./fix_qlib_dependencies.sh\\n"
            f"详细说明: 查看 backend/QLIB_INSTALLATION.md"
        )
    QLIB_AVAILABLE = False
except Exception as e:
    logger.error(f"Qlib导入时发生未知错误: {e}")
    QLIB_AVAILABLE = False
'''

with open(output_dir / 'qlib_check.py', 'w', encoding='utf-8') as f:
    f.write(qlib_check_content)
print("✓ 创建 qlib_check.py")

# ============================================================================
# 3. 创建 dataset_preparation.py - 数据准备和处理
# ============================================================================
# 提取 _process_stock_data 和 _prepare_training_datasets 方法
dataset_prep_start = 354  # _process_stock_data 开始
dataset_prep_end = 1091   # _train_qlib_model 之前

dataset_prep_lines = lines[dataset_prep_start:dataset_prep_end]

# 找到方法定义并调整缩进
dataset_prep_content = '''"""
数据集准备模块

包含数据处理、特征工程和数据集分割功能
"""

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from .config import QlibTrainingConfig
from .data_preprocessing import OutlierHandler, RobustFeatureScaler
from .qlib_check import QLIB_AVAILABLE


def process_stock_data(
    stock_data: pd.DataFrame, stock_code: str, prediction_horizon: int = 5
) -> pd.DataFrame:
    """处理单个股票的数据，包括特征计算和标签生成"""
    try:
        # 复制数据以避免修改原始数据
        processed_data = stock_data.copy()

        # 计算基本特征
        if "$close" in processed_data.columns:
            close = processed_data["$close"]
            # 计算收益率
            processed_data["RET1"] = close.pct_change(1)
            processed_data["RET5"] = close.pct_change(5)
            processed_data["RET20"] = close.pct_change(20)

            # 计算移动平均线
            processed_data["MA5"] = close.rolling(5).mean()
            processed_data["MA20"] = close.rolling(20).mean()

            # 计算标准差
            processed_data["STD5"] = close.rolling(5).std()
            processed_data["STD20"] = close.rolling(20).std()

        if "$volume" in processed_data.columns:
            volume = processed_data["$volume"]
            processed_data["VOL1"] = volume.pct_change(1)
            processed_data["VOL5"] = volume.pct_change(5)

        # 生成标签 - 修复：使用prediction_horizon参数计算未来N天收益率
        if "$close" in processed_data.columns:
            # 正确计算未来N天收益率作为标签
            current_price = processed_data["$close"]
            if isinstance(processed_data.index, pd.MultiIndex):
                # 按股票分组，计算未来N天的价格
                future_price = (
                    processed_data.groupby(level=0)["$close"]
                    .shift(-prediction_horizon)
                )
            else:
                # 直接计算未来N天的价格
                future_price = processed_data["$close"].shift(-prediction_horizon)

            # 计算收益率：(未来价格 - 当前价格) / 当前价格
            label_values = (future_price - current_price) / current_price

            if isinstance(label_values, pd.Series):
                processed_data["label"] = label_values.fillna(0)
            else:
                processed_data["label"] = pd.Series(
                    label_values.iloc[:, 0].values
                    if hasattr(label_values, "iloc")
                    else label_values,
                    index=processed_data.index,
                ).fillna(0)
            
            logger.debug(
                f"股票 {stock_code} 标签创建完成，预测周期={prediction_horizon}天，"
                f"标签范围=[{processed_data['label'].min():.6f}, {processed_data['label'].max():.6f}]"
            )

        # 填充缺失值
        processed_data = processed_data.fillna(0)

        return processed_data
    except Exception as e:
        logger.error(f"处理股票 {stock_code} 数据时发生错误: {e}")
        return stock_data


async def prepare_training_datasets(
    dataset: pd.DataFrame,
    validation_split: float,
    config: QlibTrainingConfig = None,
) -> Tuple[Any, Any]:
    """准备训练和验证数据集，返回qlib DatasetH对象"""
    if not QLIB_AVAILABLE:
        error_msg = (
            "Qlib不可用，无法准备数据集。\\n"
            "请安装Qlib库：\\n"
            "  pip install git+https://github.com/microsoft/qlib.git\\n"
            "或者查看安装文档：backend/QLIB_INSTALLATION.md"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 确定数据索引类型
    if isinstance(dataset.index, pd.MultiIndex) and dataset.index.nlevels == 2:
        # MultiIndex: (stock_code, date)
        logger.info("使用MultiIndex数据结构，按股票分组并行处理")

        # 按股票分割数据
        stock_groups = {}
        stock_codes = dataset.index.get_level_values(0).unique()

        for stock_code in stock_codes:
            try:
                stock_data = dataset.xs(stock_code, level=0, drop_level=False)
                if not stock_data.empty:
                    stock_groups[stock_code] = stock_data
            except KeyError:
                logger.warning(f"股票 {stock_code} 不在数据中")
                continue

        # 使用多进程并行处理
        processed_stocks = []

        max_workers = min(mp.cpu_count(), 8)
        # 获取prediction_horizon参数
        prediction_horizon = config.prediction_horizon if config else 5
        
        # 临时禁用多进程以避免 pickle 序列化问题
        # 单进程处理
        logger.info("使用单进程处理数据")
        for stock_code, stock_data in stock_groups.items():
            processed_data = process_stock_data(stock_data, stock_code, prediction_horizon)
            if not processed_data.empty:
                processed_stocks.append(processed_data)
                logger.debug(f"完成股票 {stock_code} 的数据处理")

        # 合并处理后的数据
        if processed_stocks:
            dataset = pd.concat(processed_stocks)
            logger.info(f"数据处理完成，合并后数据形状: {dataset.shape}")
        else:
            logger.warning("没有处理任何股票数据")

    # 按时间分割数据（时间序列数据不能随机分割）
    if isinstance(dataset.index, pd.MultiIndex):
        # 获取所有日期
        dates = dataset.index.get_level_values(1).unique().sort_values()
    else:
        dates = dataset.index.unique().sort_values()

    split_idx = int(len(dates) * (1 - validation_split))
    train_dates = dates[:split_idx]
    val_dates = dates[split_idx:]

    if isinstance(dataset.index, pd.MultiIndex):
        train_data = dataset[dataset.index.get_level_values(1).isin(train_dates)]
        val_data = dataset[dataset.index.get_level_values(1).isin(val_dates)]
    else:
        train_data = dataset[dataset.index.isin(train_dates)]
        val_data = dataset[dataset.index.isin(val_dates)]

    # 异常值处理（在标签创建后、特征标准化前）
    outlier_handler = OutlierHandler(method="winsorize", lower_percentile=0.01, upper_percentile=0.99)
    if "label" in train_data.columns:
        logger.info("开始处理标签异常值")
        train_data = outlier_handler.handle_label_outliers(train_data, label_col="label")
        if val_data is not None and "label" in val_data.columns:
            val_data = outlier_handler.handle_label_outliers(val_data, label_col="label")
        logger.info("标签异常值处理完成")

    # 特征标准化（时间序列安全）
    feature_scaler = RobustFeatureScaler()
    # 获取特征列（排除标签列）
    feature_cols = [
        col for col in train_data.columns if col != "label"
    ]
    
    if feature_cols:
        logger.info(f"开始特征标准化，特征列数: {len(feature_cols)}")
        # 在训练集上拟合并转换
        train_data = feature_scaler.fit_transform(train_data, feature_cols)
        # 在验证集上只转换（使用训练集的统计量）
        if val_data is not None and len(val_data) > 0:
            val_data = feature_scaler.transform(val_data, feature_cols)
        logger.info("特征标准化完成")
    else:
        logger.warning("未找到特征列，跳过特征标准化")

    # 导入 DatasetAdapter（延迟导入避免循环依赖）
    from .dataset_adapter import DataFrameDatasetAdapter
    
    # 创建DatasetH适配器
    train_dataset = DataFrameDatasetAdapter(train_data, val_data, config.prediction_horizon if config else 5)
    val_dataset = train_dataset  # 适配器内部已包含验证集
    
    return train_dataset, val_dataset
'''

with open(output_dir / 'dataset_preparation.py', 'w', encoding='utf-8') as f:
    f.write(dataset_prep_content)
print("✓ 创建 dataset_preparation.py")

print(f"\n重构完成！创建了 {len(list(output_dir.glob('*.py')))} 个模块文件")
EOFPYTHON
