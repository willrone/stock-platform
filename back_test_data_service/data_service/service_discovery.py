"""
服务发现模块
提供局域网内自动发现数据服务和应用服务的功能
"""
import socket
import threading
import time
import json
import logging
from typing import Optional, Dict, Callable
import struct

logger = logging.getLogger(__name__)

class ServiceDiscovery:
    """服务发现类"""

    # 服务类型定义
    SERVICE_TYPE_DATA = "stock_data_service"
    SERVICE_TYPE_APP = "stock_app_service"

    # 广播端口
    DISCOVERY_PORT = 9999
    BROADCAST_INTERVAL = 30  # 广播间隔（秒）

    def __init__(self, service_type: str, service_port: int,
                 service_name: str = None, metadata: Dict = None):
        """
        初始化服务发现

        Args:
            service_type: 服务类型 ('stock_data_service' 或 'stock_app_service')
            service_port: 服务端口
            service_name: 服务名称（可选）
            metadata: 额外元数据（可选）
        """
        self.service_type = service_type
        self.service_port = service_port
        self.service_name = service_name or socket.gethostname()
        self.metadata = metadata or {}

        # 网络配置
        self.local_ip = self._get_local_ip()
        self.broadcast_socket = None
        self.listen_socket = None

        # 服务列表
        self.discovered_services = {}  # {service_id: service_info}

        # 线程控制
        self.running = False
        self.broadcast_thread = None
        self.listen_thread = None

        # 回调函数
        self.on_service_discovered = None  # 服务发现回调
        self.on_service_lost = None        # 服务丢失回调

        logger.info(f"服务发现初始化: {service_type} at {self.local_ip}:{service_port}")

    def _get_local_ip(self) -> str:
        """获取本地IP地址"""
        try:
            # 创建一个socket连接到外部地址来获取本地IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # 连接到Google DNS
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            # 回退到localhost
            return "127.0.0.1"

    def start(self):
        """启动服务发现"""
        if self.running:
            return

        self.running = True
        logger.info("启动服务发现...")

        # 启动监听线程
        self.listen_thread = threading.Thread(target=self._listen_for_services, daemon=True)
        self.listen_thread.start()

        # 启动广播线程
        self.broadcast_thread = threading.Thread(target=self._broadcast_service, daemon=True)
        self.broadcast_thread.start()

        # 发送初始广播
        self._send_broadcast()

        logger.info("服务发现已启动")

    def stop(self):
        """停止服务发现"""
        if not self.running:
            return

        self.running = False
        logger.info("停止服务发现...")

        # 关闭socket
        if self.broadcast_socket:
            try:
                self.broadcast_socket.close()
            except:
                pass

        if self.listen_socket:
            try:
                self.listen_socket.close()
            except:
                pass

        # 等待线程结束
        if self.broadcast_thread and self.broadcast_thread.is_alive():
            self.broadcast_thread.join(timeout=1)

        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=1)

        logger.info("服务发现已停止")

    def _broadcast_service(self):
        """广播本服务信息"""
        while self.running:
            try:
                self._send_broadcast()
                time.sleep(self.BROADCAST_INTERVAL)
            except Exception as e:
                logger.warning(f"广播服务信息失败: {e}")
                time.sleep(5)

    def _send_broadcast(self):
        """发送广播消息"""
        try:
            # 创建UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(1)

            # 构建服务信息
            service_info = {
                'service_type': self.service_type,
                'service_name': self.service_name,
                'ip': self.local_ip,
                'port': self.service_port,
                'timestamp': time.time(),
                'metadata': self.metadata
            }

            # 序列化消息
            message = json.dumps(service_info).encode('utf-8')

            # 发送广播
            broadcast_addr = '<broadcast>'
            sock.sendto(message, (broadcast_addr, self.DISCOVERY_PORT))

            sock.close()
            logger.debug(f"已广播服务信息: {service_info}")

        except Exception as e:
            logger.warning(f"发送广播失败: {e}")

    def _listen_for_services(self):
        """监听其他服务的广播"""
        while self.running:
            try:
                # 创建监听socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('', self.DISCOVERY_PORT))
                sock.settimeout(1)

                logger.info(f"开始监听服务广播 (端口: {self.DISCOVERY_PORT})")

                while self.running:
                    try:
                        data, addr = sock.recvfrom(4096)
                        self._handle_broadcast(data, addr)
                    except socket.timeout:
                        continue
                    except Exception as e:
                        logger.warning(f"接收广播消息失败: {e}")
                        break

                sock.close()

            except Exception as e:
                logger.error(f"监听服务广播失败: {e}")
                time.sleep(5)

    def _handle_broadcast(self, data: bytes, addr: tuple):
        """处理接收到的广播消息"""
        try:
            # 解析消息
            message = json.loads(data.decode('utf-8'))

            # 验证消息格式
            if not all(key in message for key in ['service_type', 'service_name', 'ip', 'port']):
                return

            # 忽略自己的广播
            if message['ip'] == self.local_ip and message['port'] == self.service_port:
                return

            service_id = f"{message['service_type']}:{message['ip']}:{message['port']}"

            # 检查是否是感兴趣的服务类型
            if self.service_type == self.SERVICE_TYPE_DATA:
                # 数据服务监听应用服务
                if message['service_type'] != self.SERVICE_TYPE_APP:
                    return
            elif self.service_type == self.SERVICE_TYPE_APP:
                # 应用服务监听数据服务
                if message['service_type'] != self.SERVICE_TYPE_DATA:
                    return

            # 更新服务列表
            old_service = self.discovered_services.get(service_id)
            self.discovered_services[service_id] = {
                **message,
                'last_seen': time.time(),
                'addr': addr
            }

            # 触发发现回调
            if not old_service and self.on_service_discovered:
                self.on_service_discovered(service_id, self.discovered_services[service_id])

            logger.debug(f"发现服务: {service_id} -> {message['ip']}:{message['port']}")

        except json.JSONDecodeError:
            logger.warning("接收到无效的广播消息")
        except Exception as e:
            logger.warning(f"处理广播消息失败: {e}")

    def get_discovered_services(self, service_type: str = None) -> Dict:
        """
        获取发现的服务列表

        Args:
            service_type: 过滤的服务类型，不指定则返回所有

        Returns:
            服务字典 {service_id: service_info}
        """
        if service_type:
            return {k: v for k, v in self.discovered_services.items()
                   if v['service_type'] == service_type}
        return self.discovered_services.copy()

    def get_best_service(self, service_type: str) -> Optional[Dict]:
        """
        获取最佳的服务（最近发现的）

        Args:
            service_type: 服务类型

        Returns:
            最佳的服务信息，如果没有找到返回None
        """
        services = self.get_discovered_services(service_type)
        if not services:
            return None

        # 返回最近发现的服务
        return max(services.values(), key=lambda x: x.get('last_seen', 0))

    def cleanup_expired_services(self, timeout: int = 120):
        """
        清理过期的服务

        Args:
            timeout: 超时时间（秒）
        """
        current_time = time.time()
        expired_services = []

        for service_id, service_info in self.discovered_services.items():
            if current_time - service_info.get('last_seen', 0) > timeout:
                expired_services.append(service_id)

        for service_id in expired_services:
            service_info = self.discovered_services.pop(service_id)

            # 触发丢失回调
            if self.on_service_lost:
                self.on_service_lost(service_id, service_info)

            logger.info(f"服务已过期并移除: {service_id}")

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


# 全局服务发现实例
_data_service_discovery = None
_app_service_discovery = None

def get_data_service_discovery() -> ServiceDiscovery:
    """获取数据服务发现实例"""
    global _data_service_discovery
    if _data_service_discovery is None:
        _data_service_discovery = ServiceDiscovery(
            service_type=ServiceDiscovery.SERVICE_TYPE_DATA,
            service_port=5002,  # 数据API服务端口
            service_name="StockDataService"
        )
    return _data_service_discovery

def get_app_service_discovery() -> ServiceDiscovery:
    """获取应用服务发现实例"""
    global _app_service_discovery
    if _app_service_discovery is None:
        _app_service_discovery = ServiceDiscovery(
            service_type=ServiceDiscovery.SERVICE_TYPE_APP,
            service_port=5001,  # 应用服务端口
            service_name="StockAppService"
        )
    return _app_service_discovery


def auto_discover_services(max_wait: int = 30) -> Dict:
    """
    自动发现服务并返回连接配置

    Args:
        max_wait: 最大等待时间（秒）

    Returns:
        发现的服务配置
    """
    config = {
        'data_service': None,
        'app_service': None
    }

    # 创建服务发现实例
    with ServiceDiscovery(ServiceDiscovery.SERVICE_TYPE_DATA, 5002) as data_discovery:
        with ServiceDiscovery(ServiceDiscovery.SERVICE_TYPE_APP, 5001) as app_discovery:

            # 等待服务发现
            start_time = time.time()
            while time.time() - start_time < max_wait:
                # 检查是否发现了数据服务
                if not config['data_service']:
                    data_service = data_discovery.get_best_service(ServiceDiscovery.SERVICE_TYPE_DATA)
                    if data_service:
                        config['data_service'] = {
                            'host': data_service['ip'],
                            'port': data_service['port'],
                            'redis_port': 6379,  # 假设Redis端口
                            'mysql_port': 3306   # 假设MySQL端口
                        }
                        logger.info(f"发现数据服务: {data_service['ip']}:{data_service['port']}")

                # 检查是否发现了应用服务
                if not config['app_service']:
                    app_service = app_discovery.get_best_service(ServiceDiscovery.SERVICE_TYPE_APP)
                    if app_service:
                        config['app_service'] = {
                            'host': app_service['ip'],
                            'port': app_service['port']
                        }
                        logger.info(f"发现应用服务: {app_service['ip']}:{app_service['port']}")

                # 如果都找到了，提前退出
                if config['data_service'] and config['app_service']:
                    break

                time.sleep(1)

                # 定期清理过期服务
                data_discovery.cleanup_expired_services()
                app_discovery.cleanup_expired_services()

    return config


if __name__ == "__main__":
    # 测试服务发现
    logging.basicConfig(level=logging.INFO)

    print("开始服务发现测试...")

    with ServiceDiscovery(ServiceDiscovery.SERVICE_TYPE_DATA, 5002, "TestDataService") as discovery:
        time.sleep(5)  # 等待广播

        print("发现的服务:")
        services = discovery.get_discovered_services()
        for service_id, service_info in services.items():
            print(f"  {service_id}: {service_info}")

        best_service = discovery.get_best_service(ServiceDiscovery.SERVICE_TYPE_APP)
        if best_service:
            print(f"最佳应用服务: {best_service['ip']}:{best_service['port']}")
        else:
            print("未发现应用服务")

    print("服务发现测试完成")
