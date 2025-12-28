#!/usr/bin/env python3
"""
初始化股票列表文件
从Tushare获取股票列表并保存到Parquet文件
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_service.fetcher import DataFetcher
from data_service.config import Config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    try:
        logger.info("=" * 60)
        logger.info("开始初始化股票列表文件")
        logger.info("=" * 60)
        
        # 验证配置
        if not Config.validate():
            logger.error("配置验证失败，请检查TUSHARE_TOKEN是否设置")
            return 1
        
        # 创建数据获取服务
        fetcher = DataFetcher()
        
        # 获取并保存股票列表
        logger.info("正在从Tushare获取股票列表...")
        if fetcher.fetch_and_save_stock_list():
            logger.info("✅ 股票列表初始化成功！")
            
            # 验证文件是否存在
            stock_list = fetcher.dao.get_stock_list()
            if stock_list:
                logger.info(f"✅ 验证成功：已保存 {len(stock_list)} 只股票")
                return 0
            else:
                logger.warning("⚠️  股票列表文件已创建，但读取为空")
                return 1
        else:
            logger.error("❌ 股票列表初始化失败")
            return 1
            
    except Exception as e:
        logger.error(f"初始化股票列表失败: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

