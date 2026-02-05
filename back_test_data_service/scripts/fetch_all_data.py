#!/usr/bin/env python3
"""
获取全量股票数据：所有股票，3年数据
预计耗时：5478只股票 × 2-3秒/只 ≈ 3-4小时
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import time

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
        logger.info("开始获取全量股票数据")
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
        
        total_stocks = len(stock_list)
        logger.info(f"股票列表总数: {total_stocks}")
        
        # 检查已有数据，跳过已获取的股票
        existing_files = set()
        data_dir = project_root / ".." / "data" / "parquet" / "stock_data"
        if data_dir.exists():
            for f in data_dir.glob("*.parquet"):
                # 文件名格式: 000001.SZ.parquet -> 000001.SZ
                ts_code = f.stem
                existing_files.add(ts_code)
        
        logger.info(f"已有数据文件: {len(existing_files)} 只股票")
        
        # 过滤出需要获取的股票
        stocks_to_fetch = [s for s in stock_list if s['ts_code'] not in existing_files]
        logger.info(f"需要获取: {len(stocks_to_fetch)} 只股票")
        
        if not stocks_to_fetch:
            logger.info("所有股票数据已获取完成！")
            return 0
        
        # 计算日期范围（3年）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)
        
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        logger.info(f"日期范围: {start_date_str} - {end_date_str}")
        
        # 批量获取数据
        success_count = 0
        fail_count = 0
        start_time = time.time()
        
        for i, stock_info in enumerate(stocks_to_fetch, 1):
            ts_code = stock_info['ts_code']
            name = stock_info.get('name', 'Unknown')
            
            try:
                result = fetcher.fetch_and_save_stock_data(
                    ts_code=ts_code,
                    start_date=start_date_str,
                    end_date=end_date_str
                )
                
                if result is True:
                    success_count += 1
                elif result is False:
                    fail_count += 1
                    logger.warning(f"[{i}/{len(stocks_to_fetch)}] ❌ {ts_code} ({name}) 获取失败")
                    
            except Exception as e:
                fail_count += 1
                logger.error(f"[{i}/{len(stocks_to_fetch)}] ❌ {ts_code} ({name}) 异常: {e}")
            
            # 每100只股票输出一次进度
            if i % 100 == 0:
                elapsed = time.time() - start_time
                speed = i / elapsed
                remaining = (len(stocks_to_fetch) - i) / speed
                logger.info(
                    f"进度: {i}/{len(stocks_to_fetch)} ({i*100//len(stocks_to_fetch)}%) | "
                    f"成功: {success_count} | 失败: {fail_count} | "
                    f"速度: {speed:.1f}只/秒 | 剩余: {remaining/60:.0f}分钟"
                )
        
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"数据获取完成！")
        logger.info(f"  成功: {success_count} 只股票")
        logger.info(f"  失败: {fail_count} 只股票")
        logger.info(f"  总耗时: {elapsed/60:.1f} 分钟")
        logger.info("=" * 60)
        
        return 0 if fail_count == 0 else 1
            
    except Exception as e:
        logger.error(f"获取数据失败: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
