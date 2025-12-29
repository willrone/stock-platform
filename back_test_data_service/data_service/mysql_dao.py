"""
数据访问层
支持MySQL和SQLite（自动降级）
负责股票数据的存储和查询
"""
import logging
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime
import os
import threading
from .config import Config

logger = logging.getLogger(__name__)

# 尝试导入MySQL驱动，如果失败则使用SQLite
try:
    import pymysql
    from pymysql import cursors
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logger.warning("PyMySQL不可用，将使用SQLite作为降级选项")

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False


class MySQLDAO:
    """MySQL数据访问对象（支持多进程/多线程）"""
    
    def __init__(self):
        """初始化MySQL连接"""
        self._connection_lock = threading.Lock()  # 连接操作锁
        self._local = threading.local()  # 线程本地存储
        self._connect_params = {
            'host': Config.MYSQL_HOST,
            'port': Config.MYSQL_PORT,
            'user': Config.MYSQL_USER,
            'password': Config.MYSQL_PASSWORD,
            'database': Config.MYSQL_DATABASE,
            'charset': 'utf8mb4',
            'cursorclass': cursors.DictCursor,
            'autocommit': False
        }
        # 确保数据库和表存在
        self._ensure_database()
        self._create_tables()
    
    def _get_connection(self):
        """获取当前线程的数据库连接（线程本地存储）"""
        with self._connection_lock:
            if not hasattr(self._local, 'connection') or self._local.connection is None:
                self._local.connection = self._create_connection()
            return self._local.connection
    
    def _create_connection(self):
        """创建新的数据库连接"""
        try:
            connection = pymysql.connect(**self._connect_params)
            logger.debug(f"创建新的MySQL连接: {Config.MYSQL_HOST}:{Config.MYSQL_PORT}/{Config.MYSQL_DATABASE}")
            return connection
        except Exception as e:
            logger.error(f"创建MySQL连接失败: {e}")
            raise
    
    def _ensure_database(self):
        """确保数据库存在"""
        try:
            # 尝试连接到数据库
            connection = pymysql.connect(
                host=Config.MYSQL_HOST,
                port=Config.MYSQL_PORT,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                charset='utf8mb4',
                autocommit=True
            )
            try:
                with connection.cursor() as cursor:
                    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{Config.MYSQL_DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                logger.info(f"数据库 {Config.MYSQL_DATABASE} 已确保存在")
            finally:
                connection.close()
        except Exception as e:
            logger.warning(f"确保数据库存在时出错: {e}")
    
    def _connect(self):
        """建立MySQL连接（已废弃，保留以兼容旧代码）"""
        # 这个方法现在只是获取连接，实际连接是延迟创建的
        return self._get_connection()
    
    def _create_tables(self):
        """创建数据表（如果不存在）"""
        try:
            # 确保连接有效（会自动重连）
            connection = self._ensure_connection()
            if connection is None:
                raise RuntimeError("无法获取有效的数据库连接")
            with connection.cursor() as cursor:
                # 股票数据表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stock_data (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        ts_code VARCHAR(20) NOT NULL COMMENT '股票代码',
                        trade_date DATE NOT NULL COMMENT '交易日期',
                        open DECIMAL(10, 2) NOT NULL COMMENT '开盘价',
                        high DECIMAL(10, 2) NOT NULL COMMENT '最高价',
                        low DECIMAL(10, 2) NOT NULL COMMENT '最低价',
                        close DECIMAL(10, 2) NOT NULL COMMENT '收盘价',
                        volume BIGINT NOT NULL COMMENT '成交量',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                        UNIQUE KEY uk_stock_date (ts_code, trade_date),
                        INDEX idx_ts_code (ts_code),
                        INDEX idx_trade_date (trade_date)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票日线数据表'
                """)
                
                # 股票列表表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stock_list (
                        ts_code VARCHAR(20) PRIMARY KEY COMMENT '股票代码',
                        name VARCHAR(100) NOT NULL COMMENT '股票名称',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票列表表'
                """)
                
                connection.commit()
                logger.info("数据表创建成功")
        except Exception as e:
            logger.error(f"创建数据表失败: {e}")
            try:
                connection.rollback()
            except:
                pass
            raise
    
    def save_stock_data(self, ts_code: str, df: pd.DataFrame) -> int:
        """
        保存股票数据到MySQL
        
        Args:
            ts_code: 股票代码
            df: 股票数据DataFrame，必须包含date, open, high, low, close, volume列
            
        Returns:
            保存的记录数
        """
        if df is None or df.empty:
            logger.warning(f"股票数据为空: {ts_code}")
            return 0
        
        try:
            # 确保连接有效（会自动重连）
            connection = self._ensure_connection()
            if connection is None:
                raise RuntimeError("无法获取有效的数据库连接")
            # 确保date列是datetime类型
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df['date'] = pd.to_datetime(df['date'])
            
            # 准备数据
            records = []
            for _, row in df.iterrows():
                records.append((
                    ts_code,
                    row['date'].strftime('%Y-%m-%d'),
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume'])
                ))
            
            # 批量插入（使用INSERT ... ON DUPLICATE KEY UPDATE实现upsert）
            with connection.cursor() as cursor:
                sql = """
                    INSERT INTO stock_data (ts_code, trade_date, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        open = VALUES(open),
                        high = VALUES(high),
                        low = VALUES(low),
                        close = VALUES(close),
                        volume = VALUES(volume),
                        updated_at = CURRENT_TIMESTAMP
                """
                cursor.executemany(sql, records)
                connection.commit()
                
                # 返回实际处理的记录数（包括插入和更新）
                saved_count = len(records)
                logger.info(f"保存股票数据成功: {ts_code}, 记录数: {saved_count}")
                return saved_count
                
        except Exception as e:
            logger.error(f"保存股票数据失败: {ts_code}, 错误: {e}")
            try:
                connection.rollback()
            except:
                pass
            raise
    
    def get_stock_data(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        从MySQL获取股票数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)
            
        Returns:
            股票数据DataFrame，如果不存在返回None
        """
        try:
            # 确保连接有效（会自动重连）
            connection = self._ensure_connection()
            if connection is None:
                logger.error("无法获取有效的数据库连接")
                return None
            
            # 转换日期格式
            start_date_formatted = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
            end_date_formatted = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
            
            with connection.cursor() as cursor:
                sql = """
                    SELECT trade_date, open, high, low, close, volume
                    FROM stock_data
                    WHERE ts_code = %s AND trade_date >= %s AND trade_date <= %s
                    ORDER BY trade_date ASC
                """
                cursor.execute(sql, (ts_code, start_date_formatted, end_date_formatted))
                rows = cursor.fetchall()
                
                if not rows:
                    logger.debug(f"未找到股票数据: {ts_code} ({start_date} - {end_date})")
                    return None
                
                # 转换为DataFrame
                df = pd.DataFrame(rows)
                df.rename(columns={'trade_date': 'date'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                logger.debug(f"从MySQL获取股票数据成功: {ts_code}, 记录数: {len(df)}")
                return df
                
        except Exception as e:
            logger.error(f"获取股票数据失败: {ts_code}, 错误: {e}")
            return None
    
    def save_stock_list(self, stock_list: List[Dict[str, str]]):
        """
        保存股票列表
        
        Args:
            stock_list: 股票列表，每个元素包含ts_code和name
        """
        try:
            # 确保连接有效（会自动重连）
            connection = self._ensure_connection()
            if connection is None:
                raise RuntimeError("无法获取有效的数据库连接")
            with connection.cursor() as cursor:
                sql = """
                    INSERT INTO stock_list (ts_code, name)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE
                        name = VALUES(name),
                        updated_at = CURRENT_TIMESTAMP
                """
                records = [(item['ts_code'], item['name']) for item in stock_list]
                cursor.executemany(sql, records)
                connection.commit()
                
                logger.info(f"保存股票列表成功: {len(records)} 条记录")
        except Exception as e:
            logger.error(f"保存股票列表失败: {e}")
            try:
                connection.rollback()
            except:
                pass
            raise
    
    def get_stock_list(self) -> List[Dict[str, str]]:
        """获取股票列表"""
        try:
            # 确保连接有效（会自动重连）
            connection = self._ensure_connection()
            if connection is None:
                logger.error("无法获取有效的数据库连接")
                return []
            
            with connection.cursor() as cursor:
                cursor.execute("SELECT ts_code, name FROM stock_list ORDER BY ts_code")
                rows = cursor.fetchall()
                return [{'ts_code': row['ts_code'], 'name': row['name']} for row in rows]
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def get_latest_date(self, ts_code: str) -> Optional[str]:
        """
        获取股票的最新数据日期

        Args:
            ts_code: 股票代码

        Returns:
            最新日期 (YYYYMMDD格式)，如果不存在返回None
        """
        try:
            # 确保连接有效（会自动重连）
            connection = self._ensure_connection()
            if connection is None:
                logger.error("无法获取有效的数据库连接")
                return None
            
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT MAX(trade_date) as latest_date
                    FROM stock_data
                    WHERE ts_code = %s
                """, (ts_code,))
                row = cursor.fetchone()

                if row and row['latest_date']:
                    return row['latest_date'].strftime('%Y%m%d')
                return None
        except Exception as e:
            logger.error(f"获取最新日期失败: {ts_code}, 错误: {e}")
            return None

    def is_connection_alive(self, connection=None) -> bool:
        """检查数据库连接是否有效"""
        if connection is None:
            if not hasattr(self._local, 'connection') or self._local.connection is None:
                return False
            connection = self._local.connection
        
        # 检查连接对象是否有效
        if connection is None:
            return False
        
        try:
            # 使用 ping 检查连接，但不自动重连
            connection.ping(reconnect=False)
            return True
        except (AttributeError, TypeError):
            # 连接对象无效
            return False
        except Exception as e:
            # 其他异常（如网络错误、连接已关闭等）
            logger.debug(f"数据库连接检查失败: {e}")
            return False
    
    def _ensure_connection(self, connection=None):
        """确保数据库连接有效，如果失效则重连"""
        with self._connection_lock:
            # 如果传入的连接无效，获取当前线程的连接
            if connection is None:
                connection = self._get_connection()
            
            # 检查连接是否有效
            if not self.is_connection_alive(connection):
                logger.warning("数据库连接失效，尝试重新连接...")
                try:
                    # 关闭旧连接（如果存在）
                    old_connection = None
                    if hasattr(self._local, 'connection'):
                        old_connection = self._local.connection
                        self._local.connection = None
                    
                    # 关闭旧连接
                    if old_connection:
                        try:
                            old_connection.close()
                        except:
                            pass
                    
                    # 创建新连接
                    self._local.connection = self._create_connection()
                    logger.info("数据库重连成功")
                except Exception as e:
                    logger.error(f"数据库重连失败: {e}")
                    self._local.connection = None
                    raise
            
            # 返回最新的连接（可能已经重新创建）
            return self._local.connection
    
    def fetch_one(self, query: str, params=None):
        """执行查询并返回第一行（兼容SQLite接口）"""
        try:
            # 确保连接有效（会自动重连）
            connection = self._ensure_connection()
            
            if connection is None:
                logger.error("无法获取有效的数据库连接")
                return None
            
            with connection.cursor() as cursor:
                cursor.execute(query, params or ())
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"查询失败: {e}, 查询语句: {query[:100]}...")
            # 尝试重新连接并重试一次
            try:
                with self._connection_lock:
                    if hasattr(self._local, 'connection'):
                        self._local.connection = None
                connection = self._ensure_connection()
                if connection:
                    with connection.cursor() as cursor:
                        cursor.execute(query, params or ())
                        row = cursor.fetchone()
                        return dict(row) if row else None
            except:
                pass
            return None
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, 'connection') and self._local.connection:
            try:
                self._local.connection.close()
                logger.debug("MySQL连接已关闭")
            except:
                pass
            finally:
                self._local.connection = None
    
    def __del__(self):
        """析构函数"""
        try:
            self.close()
        except:
            pass


class SQLiteDAO:
    """SQLite数据访问对象（MySQL降级选项）"""

    def __init__(self):
        """初始化SQLite连接"""
        self.db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'stock_data.db')
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.connection = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """建立SQLite连接"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            logger.info(f"SQLite连接成功: {self.db_path}")
        except Exception as e:
            logger.error(f"SQLite连接失败: {e}")
            raise

    def _create_tables(self):
        """创建数据表（如果不存在）"""
        try:
            # 股票数据表
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_code TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ts_code, trade_date)
                )
            """)

            # 股票列表表
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS stock_list (
                    ts_code TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建索引
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_ts_code ON stock_data(ts_code)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_trade_date ON stock_data(trade_date)")

            self.connection.commit()
            logger.info("SQLite数据表创建成功")
        except Exception as e:
            logger.error(f"创建SQLite数据表失败: {e}")
            raise

    def save_stock_data(self, ts_code: str, df: pd.DataFrame) -> int:
        """保存股票数据到SQLite"""
        if df is None or df.empty:
            logger.warning(f"股票数据为空: {ts_code}")
            return 0

        try:
            # 确保date列是datetime类型
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df['date'] = pd.to_datetime(df['date'])

            # 准备数据
            records = []
            for _, row in df.iterrows():
                records.append((
                    ts_code,
                    row['date'].strftime('%Y-%m-%d'),
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume'])
                ))

            # 使用INSERT OR REPLACE实现upsert
            sql = """
                INSERT OR REPLACE INTO stock_data
                (ts_code, trade_date, open, high, low, close, volume, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """

            cursor = self.connection.executemany(sql, records)
            self.connection.commit()

            saved_count = cursor.rowcount
            logger.info(f"保存股票数据成功: {ts_code}, 记录数: {saved_count}")
            return saved_count

        except Exception as e:
            logger.error(f"保存股票数据失败: {ts_code}, 错误: {e}")
            raise

    def get_stock_data(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从SQLite获取股票数据"""
        try:
            # 转换日期格式
            start_date_formatted = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
            end_date_formatted = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

            cursor = self.connection.execute("""
                SELECT trade_date, open, high, low, close, volume
                FROM stock_data
                WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date ASC
            """, (ts_code, start_date_formatted, end_date_formatted))

            rows = cursor.fetchall()

            if not rows:
                logger.debug(f"未找到股票数据: {ts_code} ({start_date} - {end_date})")
                return None

            # 转换为DataFrame
            df = pd.DataFrame([dict(row) for row in rows])
            df.rename(columns={'trade_date': 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            logger.debug(f"从SQLite获取股票数据成功: {ts_code}, 记录数: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"获取股票数据失败: {ts_code}, 错误: {e}")
            return None

    def save_stock_list(self, stock_list: List[Dict[str, str]]):
        """保存股票列表"""
        try:
            sql = """
                INSERT OR REPLACE INTO stock_list
                (ts_code, name, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """
            records = [(item['ts_code'], item['name']) for item in stock_list]
            self.connection.executemany(sql, records)
            self.connection.commit()

            logger.info(f"保存股票列表成功: {len(records)} 条记录")
        except Exception as e:
            logger.error(f"保存股票列表失败: {e}")
            raise

    def get_stock_list(self) -> List[Dict[str, str]]:
        """获取股票列表"""
        try:
            cursor = self.connection.execute("SELECT ts_code, name FROM stock_list ORDER BY ts_code")
            rows = cursor.fetchall()
            return [{'ts_code': row['ts_code'], 'name': row['name']} for row in rows]
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []

    def get_latest_date(self, ts_code: str) -> Optional[str]:
        """获取股票的最新数据日期"""
        try:
            cursor = self.connection.execute("""
                SELECT MAX(trade_date) as latest_date
                FROM stock_data
                WHERE ts_code = ?
            """, (ts_code,))
            row = cursor.fetchone()

            if row and row['latest_date']:
                # 转换回YYYYMMDD格式
                date_obj = datetime.strptime(row['latest_date'], '%Y-%m-%d')
                return date_obj.strftime('%Y%m%d')
            return None
        except Exception as e:
            logger.error(f"获取最新日期失败: {ts_code}, 错误: {e}")
            return None

    def fetch_one(self, query: str, params=None):
        """执行查询并返回第一行（兼容MySQL接口）"""
        try:
            cursor = self.connection.execute(query, params or ())
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return None

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("SQLite连接已关闭")

    def __del__(self):
        """析构函数"""
        self.close()


# 创建DAO实例的工厂函数
def create_dao():
    """创建合适的DAO实例（MySQL优先，SQLite降级）"""
    if MYSQL_AVAILABLE:
        try:
            return MySQLDAO()
        except Exception as e:
            logger.warning(f"MySQL连接失败，降级到SQLite: {e}")

    if SQLITE_AVAILABLE:
        logger.info("使用SQLite作为数据存储")
        return SQLiteDAO()
    else:
        raise RuntimeError("没有可用的数据库后端（MySQL和SQLite都不可用）")
