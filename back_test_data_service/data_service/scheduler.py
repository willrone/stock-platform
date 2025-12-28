"""
定时任务调度服务
定期从Tushare获取数据并保存到Parquet文件
"""
import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from .fetcher import DataFetcher

logger = logging.getLogger(__name__)


class DataScheduler:
    """数据调度服务"""
    
    def __init__(self):
        """初始化调度服务"""
        self.fetcher = DataFetcher()
        self.scheduler = BlockingScheduler()
        logger.info("数据调度服务初始化成功")
    
    def update_stock_list(self):
        """更新股票列表"""
        try:
            logger.info("开始更新股票列表")
            success = self.fetcher.fetch_and_save_stock_list()
            if success:
                logger.info("股票列表更新成功")
            else:
                logger.error("股票列表更新失败")
        except Exception as e:
            logger.error(f"更新股票列表异常: {e}")
    
    def update_all_stocks_daily(self):
        """每日更新所有股票的最新数据"""
        try:
            logger.info("开始每日更新股票数据")
            
            # 获取股票列表
            stock_list = self.fetcher.dao.get_stock_list()
            if not stock_list:
                logger.warning("股票列表为空，先更新股票列表")
                self.update_stock_list()
                stock_list = self.fetcher.dao.get_stock_list()
            
            success_count = 0
            failed_count = 0
            
            for stock in stock_list:
                ts_code = stock['ts_code']
                try:
                    if self.fetcher.update_stock_data_incremental(ts_code):
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"更新股票数据失败: {ts_code}, 错误: {e}")
                    failed_count += 1
            
            logger.info(f"每日更新完成: 成功={success_count}, 失败={failed_count}, 总计={len(stock_list)}")
            
        except Exception as e:
            logger.error(f"每日更新异常: {e}")
    
    def update_all_stocks_full(self):
        """全量更新所有股票的历史数据"""
        try:
            logger.info("开始全量更新股票数据")
            
            # 获取股票列表
            stock_list = self.fetcher.dao.get_stock_list()
            if not stock_list:
                logger.warning("股票列表为空，先更新股票列表")
                self.update_stock_list()
                stock_list = self.fetcher.dao.get_stock_list()
            
            from .config import Config
            start_date = Config.DEFAULT_START_DATE
            end_date = datetime.now().strftime('%Y%m%d')
            
            success_count = 0
            failed_count = 0
            
            for stock in stock_list:
                ts_code = stock['ts_code']
                try:
                    if self.fetcher.fetch_and_save_stock_data(ts_code, start_date, end_date):
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"全量更新股票数据失败: {ts_code}, 错误: {e}")
                    failed_count += 1
            
            logger.info(f"全量更新完成: 成功={success_count}, 失败={failed_count}, 总计={len(stock_list)}")
            
        except Exception as e:
            logger.error(f"全量更新异常: {e}")
    
    def start(self):
        """启动调度服务"""
        try:
            # 每日凌晨2点更新股票列表（每周一次）
            self.scheduler.add_job(
                self.update_stock_list,
                trigger=CronTrigger(day_of_week='mon', hour=2, minute=0),
                id='update_stock_list',
                name='更新股票列表',
                replace_existing=True
            )
            
            # 每个交易日收盘后（18:00）更新所有股票的最新数据
            self.scheduler.add_job(
                self.update_all_stocks_daily,
                trigger=CronTrigger(hour=18, minute=0),
                id='update_daily_data',
                name='每日更新股票数据',
                replace_existing=True
            )
            
            # 每周日凌晨3点全量更新一次（可选，根据需求调整）
            self.scheduler.add_job(
                self.update_all_stocks_full,
                trigger=CronTrigger(day_of_week='sun', hour=3, minute=0),
                id='update_full_data',
                name='全量更新股票数据',
                replace_existing=True
            )
            
            logger.info("调度服务启动成功")
            logger.info("已添加以下定时任务:")
            logger.info("  - 每周一凌晨2点: 更新股票列表")
            logger.info("  - 每天18:00: 更新所有股票的最新数据")
            logger.info("  - 每周日凌晨3点: 全量更新股票数据")
            
            # 启动时立即执行一次
            logger.info("执行初始任务...")
            self.update_stock_list()
            
            # 启动调度器
            self.scheduler.start()
            
        except KeyboardInterrupt:
            logger.info("收到停止信号，正在关闭调度服务...")
            self.stop()
        except Exception as e:
            logger.error(f"调度服务异常: {e}")
            self.stop()
    
    def stop(self):
        """停止调度服务"""
        try:
            self.scheduler.shutdown()
            self.fetcher.close()
            logger.info("调度服务已停止")
        except Exception as e:
            logger.error(f"停止调度服务异常: {e}")

