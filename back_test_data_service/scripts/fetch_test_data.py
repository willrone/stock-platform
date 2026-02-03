#!/usr/bin/env python3
"""
获取测试数据：500只股票，3年数据
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

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
        logger.info("开始获取测试数据：500只股票，3年数据")
        logger.info("=" * 60)
        
        # 验证配置
        if not Config.validate():
            logger.error("配置验证失败")
            return 1
        
        # 创建数据获取服务
        fetcher = DataFetcher()
        
        # 获取股票列表
        stock_list = fetcher.dao.get_stock_list()
        if not stock_list:
            logger.error("股票列表为空，请先运行 init_stock_list.py")
            return 1
        
        logger.info(f"股票列表总数: {len(stock_list)}")
        
        # 取前500只股票
        test_stocks = stock_list[:500]
        logger.info(f"选择前 {len(test_stocks)} 只股票进行测试")
        
        # 计算日期范围（3年）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)
        
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        logger.info(f"日期范围: {start_date_str} - {end_date_str}")
        
        # 批量获取数据
        success_count = 0
        fail_count = 0
        
        for i, stock_info in enumerate(test_stocks, 1):
            ts_code = stock_info['ts_code']
            name = stock_info.get('name', 'Unknown')
            
            logger.info(f"[{i}/{len(test_stocks)}] 获取 {ts_code} ({name}) 的数据...")
            
            try:
                result = fetcher.fetch_and_save_stock_data(
                    ts_code=ts_code,
                    start_date=start_date_str,
                    end_date=end_date_str
                )
                
                # result: True=成功, False=失败, None=非交易日
                if result is True:
                    success_count += 1
                    logger.info(f"  ✅ 成功获取并保存数据")
                elif result is False:
                    fail_count += 1
                    logger.warning(f"  ❌ 获取失败")
                else:  # None
                    logger.info(f"  ⚠️  无数据（可能是非交易日）")
                    
            except Exception as e:
                fail_count += 1
                logger.error(f"  ❌ 获取失败: {e}")
            
            # 每10只股票输出一次进度
            if i % 10 == 0:
                logger.info(f"进度: {i}/{len(test_stocks)} ({i*100//len(test_stocks)}%) - 成功: {success_count}, 失败: {fail_count}")
        
        logger.info("=" * 60)
        logger.info(f"数据获取完成！")
        logger.info(f"  成功: {success_count} 只股票")
        logger.info(f"  失败: {fail_count} 只股票")
        logger.info("=" * 60)
        
        return 0 if fail_count == 0 else 1
            
    except Exception as e:
        logger.error(f"获取测试数据失败: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
