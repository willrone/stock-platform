"""
数据状态API服务
提供数据服务器的状态查询接口
"""
import logging
from flask import Flask, Blueprint, jsonify, render_template
from typing import Dict, List
import os
from pathlib import Path
from .parquet_dao import create_dao

logger = logging.getLogger(__name__)

# 创建Flask应用和蓝图
app = Flask(__name__,
            template_folder='../templates',
            static_folder='../static')
data_bp = Blueprint('data', __name__)

def create_app():
    """创建Flask应用"""
    app.register_blueprint(data_bp, url_prefix='/api/data')
    return app

# 初始化Parquet DAO
dao = None

def init_dao():
    """初始化DAO（Parquet）"""
    global dao
    if dao is None:
        try:
            dao = create_dao()
            logger.info("DAO初始化成功")
        except Exception as e:
            logger.error(f"DAO初始化失败: {e}")
            dao = None

@data_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    init_dao()
    # 服务本身是健康的，即使数据存储不可用，服务也可以运行
    # 数据存储状态只是作为信息返回
    storage_type = type(dao).__name__ if dao else None
    storage_type_str = "Parquet" if storage_type == "ParquetDAO" else "None"

    status = {
        'status': 'healthy',  # 服务本身是健康的
        'service': 'data_service',
        'storage_available': dao is not None,
        'storage_type': storage_type_str,
        'message': f'服务运行正常 (存储: {storage_type_str})' if dao else '服务运行正常，但数据存储不可用'
    }
    return jsonify(status), 200  # 始终返回200，服务已启动

@data_bp.route('/stock_data_status', methods=['GET'])
def get_stock_data_status():
    """
    获取每只股票的数据状态

    返回格式:
    {
        "total_stocks": 100,
        "stocks": [
            {
                "ts_code": "000001.SZ",
                "name": "平安银行",
                "data_range": {
                    "start_date": "2020-03-01",
                    "end_date": "2024-12-01",
                    "total_days": 1234
                },
                "last_update": "2024-12-01",
                "status": "complete"  // complete, incomplete, missing
            }
        ]
    }
    """
    init_dao()
    if not dao:
        return jsonify({'error': '数据存储初始化失败'}), 500

    try:
        # 获取股票列表
        stock_list = dao.get_stock_list()
        if not stock_list:
            return jsonify({
                'total_stocks': 0,
                'stocks': []
            })

        stock_data_status = []
        total_stocks = len(stock_list)

        for stock in stock_list:
            ts_code = stock['ts_code']
            name = stock['name']

            # 获取股票数据状态
            status_info = get_single_stock_status(ts_code)
            if status_info:
                stock_data_status.append({
                    'ts_code': ts_code,
                    'name': name,
                    **status_info
                })

        return jsonify({
            'total_stocks': total_stocks,
            'stocks': stock_data_status
        })

    except Exception as e:
        logger.error(f"获取股票数据状态失败: {e}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500

@data_bp.route('/stock_data_status/<ts_code>', methods=['GET'])
def get_single_stock_status_api(ts_code):
    """获取单个股票的数据状态"""
    init_dao()
    if not dao:
        return jsonify({'error': '数据存储初始化失败'}), 500

    try:
        status_info = get_single_stock_status(ts_code)
        if not status_info:
            return jsonify({'error': f'未找到股票 {ts_code} 的数据'}), 404

        # 获取股票名称
        stock_list = dao.get_stock_list()
        stock_name = next((s['name'] for s in stock_list if s['ts_code'] == ts_code), '未知')

        return jsonify({
            'ts_code': ts_code,
            'name': stock_name,
            **status_info
        })

    except Exception as e:
        logger.error(f"获取股票 {ts_code} 数据状态失败: {e}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500

def get_single_stock_status(ts_code: str) -> Dict:
    """
    获取单个股票的数据状态

    Args:
        ts_code: 股票代码

    Returns:
        股票数据状态信息，如果不存在返回None
    """
    if not dao:
        return None

    try:
        # 从Parquet文件获取数据范围
        data_range_info = dao.get_stock_data_range(ts_code)

        if not data_range_info:
            return {
                'data_range': None,
                'last_update': None,
                'status': 'missing'
            }

        start_date = data_range_info['start_date']
        end_date = data_range_info['end_date']
        total_records = data_range_info['total_records']
        last_update = data_range_info['last_update']

        # 计算预期的数据天数（工作日）
        if start_date and end_date:
            from datetime import datetime
            start_dt = datetime.strptime(str(start_date), '%Y-%m-%d')
            end_dt = datetime.strptime(str(end_date), '%Y-%m-%d')

            # 计算工作日数量（简单估算，排除周末）
            total_days = (end_dt - start_dt).days + 1
            expected_workdays = total_days * 5 // 7  # 简单的工作日估算

            # 判断数据完整性
            if total_records >= expected_workdays * 0.8:  # 80%以上为完整
                status = 'complete'
            else:
                status = 'incomplete'

            data_range = {
                'start_date': str(start_date),
                'end_date': str(end_date),
                'total_days': total_records
            }
        else:
            status = 'incomplete'
            data_range = None

        return {
            'data_range': data_range,
            'last_update': str(last_update) if last_update else None,
            'status': status
        }

    except Exception as e:
        logger.error(f"获取股票 {ts_code} 状态失败: {e}")
        return None

@data_bp.route('/data_summary', methods=['GET'])
def get_data_summary():
    """
    获取数据汇总统计

    返回格式:
    {
        "total_stocks": 100,
        "complete_stocks": 85,
        "incomplete_stocks": 10,
        "missing_stocks": 5,
        "total_records": 100000,
        "last_update": "2024-12-01 10:00:00"
    }
    """
    global dao  # 必须在函数开始处声明
    
    init_dao()
    if not dao:
        logger.error("数据存储初始化失败，无法获取数据汇总")
        return jsonify({'error': '数据存储初始化失败'}), 500

    try:
        # 从Parquet文件获取汇总信息
        stock_list = dao.get_stock_list()
        total_stocks = len(stock_list)
        
        # 统计总记录数
        total_records = 0
        last_update = None
        
        for stock in stock_list:
            ts_code = stock['ts_code']
            data_range_info = dao.get_stock_data_range(ts_code)
            if data_range_info:
                total_records += data_range_info.get('total_records', 0)
                # 获取最新的更新时间
                stock_last_update = data_range_info.get('last_update')
                if stock_last_update:
                    if last_update is None or stock_last_update > last_update:
                        last_update = stock_last_update

        logger.debug(f"数据汇总查询成功: total_stocks={total_stocks}, total_records={total_records}")

        # 简化统计：只返回基本信息，避免遍历所有股票导致超时
        # TODO: 可以考虑添加缓存或异步统计来提供完整的状态信息
        return jsonify({
            'total_stocks': total_stocks,
            'complete_stocks': 0,  # 暂时设为0，需要异步统计
            'incomplete_stocks': 0,  # 暂时设为0，需要异步统计
            'missing_stocks': 0,  # 暂时设为0，需要异步统计
            'total_records': total_records,
            'last_update': str(last_update) if last_update else None,
            'note': '详细统计信息将在后续版本中提供'
        })

    except Exception as e:
        logger.error(f"获取数据汇总失败: {e}", exc_info=True)
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500

@data_bp.route('/logs/<log_type>', methods=['GET'])
def get_logs(log_type):
    """获取日志内容"""
    init_dao()
    if not dao:
        return jsonify({'error': '数据存储初始化失败'}), 500

    try:
        logs_dir = Path(__file__).parent.parent / "logs"
        log_files = {
            'api': logs_dir / "data_api.log",
            'service': logs_dir / "data_service.log",
            'error': logs_dir / "data_api_error.log"
        }

        if log_type not in log_files:
            return jsonify({'error': '无效的日志类型'}), 400

        log_file = log_files[log_type]
        if not log_file.exists():
            return jsonify({'content': '日志文件不存在', 'lines': 0})

        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.count('\n') + 1

        return jsonify({
            'content': content,
            'lines': lines,
            'file': log_file.name
        })

    except Exception as e:
        logger.error(f"获取日志失败: {e}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500

@data_bp.route('/manual_fetch', methods=['POST'])
def manual_data_fetch():
    """手动触发数据获取"""
    init_dao()
    if not dao:
        return jsonify({'error': '数据存储初始化失败'}), 500

    try:
        from .fetcher import DataFetcher
        import threading

        def fetch_data_background():
            """后台执行数据获取"""
            try:
                logger.info("开始手动数据获取...")
                fetcher = DataFetcher()

                # 更新股票列表
                logger.info("更新股票列表...")
                fetcher.fetch_and_save_stock_list()

                # 获取前10只股票的数据作为示例
                stock_list = fetcher.fetch_stock_list()
                logger.info(f"将获取前{len(stock_list)}只股票的数据")

                success_count = 0
                for stock in stock_list:
                    ts_code = stock['ts_code']
                    try:
                        # 使用配置的默认开始日期，而不是最近30天
                        from datetime import datetime
                        from .config import Config
                        end_date = datetime.now().strftime('%Y%m%d')
                        start_date = Config.DEFAULT_START_DATE  # 使用配置的默认开始日期（20200301）

                        logger.info(f"开始获取股票数据: {ts_code} ({start_date} - {end_date})")
                        if fetcher.fetch_and_save_stock_data(ts_code, start_date, end_date):
                            success_count += 1
                            logger.info(f"成功获取股票数据: {ts_code}")
                        else:
                            logger.error(f"获取股票数据失败: {ts_code}")
                    except Exception as e:
                        logger.error(f"获取股票数据异常: {ts_code}, {e}")

                logger.info(f"手动数据获取完成: 成功={success_count}")

            except Exception as e:
                logger.error(f"手动数据获取异常: {e}")

        # 在后台线程中执行数据获取
        thread = threading.Thread(target=fetch_data_background, daemon=True)
        thread.start()

        return jsonify({
            'message': '已开始手动数据获取，请查看日志了解进度',
            'status': 'running'
        })

    except Exception as e:
        logger.error(f"启动手动数据获取失败: {e}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500

@data_bp.route('/stock/<ts_code>/daily', methods=['GET'])
def get_stock_daily_data(ts_code):
    """
    获取股票日线数据
    
    Args:
        ts_code: 股票代码
        
    Query Parameters:
        start_date: 开始日期 (YYYY-MM-DD格式)
        end_date: 结束日期 (YYYY-MM-DD格式)
    
    Returns:
        {
            "success": true,
            "data": [
                {
                    "date": "2024-01-01",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.5,
                    "close": 10.5,
                    "volume": 1000000
                }
            ]
        }
    """
    init_dao()
    if not dao:
        return jsonify({'success': False, 'error': '数据存储初始化失败'}), 500

    try:
        from flask import request
        
        # 获取查询参数
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if not start_date or not end_date:
            return jsonify({
                'success': False, 
                'error': '缺少必需参数: start_date, end_date'
            }), 400
        
        # 转换日期格式 (YYYY-MM-DD -> YYYYMMDD)
        try:
            start_date_formatted = start_date.replace('-', '')
            end_date_formatted = end_date.replace('-', '')
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'日期格式错误，应为YYYY-MM-DD: {e}'
            }), 400
        
        # 从Parquet文件获取数据
        df = dao.get_stock_data(ts_code, start_date_formatted, end_date_formatted)
        
        if df is None or df.empty:
            return jsonify({
                'success': False,
                'error': f'未找到股票 {ts_code} 在 {start_date} 至 {end_date} 期间的数据'
            }), 404
        
        # 转换为API响应格式
        data_list = []
        for date, row in df.iterrows():
            data_list.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            })
        
        return jsonify({
            'success': True,
            'data': data_list,
            'total_records': len(data_list),
            'stock_code': ts_code,
            'start_date': start_date,
            'end_date': end_date
        })
        
    except Exception as e:
        logger.error(f"获取股票数据失败: {ts_code}, 错误: {e}")
        return jsonify({
            'success': False, 
            'error': f'服务器内部错误: {str(e)}'
        }), 500


@data_bp.route('/manual_sync', methods=['POST'])
def manual_sync():
    """手动触发数据同步（已废弃，Parquet存储无需同步）"""
    return jsonify({
        'message': 'Parquet存储无需同步操作',
        'status': 'not_needed'
    }), 200

@app.route('/', methods=['GET'])
def index():
    """数据服务前端界面"""
    return render_template('index.html')
