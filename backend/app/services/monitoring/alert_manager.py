"""
告警和通知机制
实现性能下降告警，支持邮件和WebSocket通知
"""

import json
import smtplib
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from app.services.monitoring.drift_detector import DriftReport, DriftSeverity
from app.services.monitoring.performance_monitor import (
    Alert,
)


class NotificationChannel(Enum):
    """通知渠道"""

    EMAIL = "email"
    WEBSOCKET = "websocket"
    WEBHOOK = "webhook"
    SMS = "sms"
    SLACK = "slack"


class NotificationStatus(Enum):
    """通知状态"""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    DELIVERED = "delivered"


@dataclass
class NotificationConfig:
    """通知配置"""

    channel: NotificationChannel
    enabled: bool = True
    # 邮件配置
    smtp_server: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    # WebSocket配置
    websocket_endpoints: List[str] = field(default_factory=list)
    # Webhook配置
    webhook_url: Optional[str] = None
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    # 通用配置
    rate_limit_minutes: int = 5  # 限流时间（分钟）
    max_notifications_per_hour: int = 10  # 每小时最大通知数

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel.value,
            "enabled": self.enabled,
            "smtp_server": self.smtp_server,
            "smtp_port": self.smtp_port,
            "smtp_username": self.smtp_username,
            "email_recipients": self.email_recipients,
            "websocket_endpoints": self.websocket_endpoints,
            "webhook_url": self.webhook_url,
            "webhook_headers": self.webhook_headers,
            "rate_limit_minutes": self.rate_limit_minutes,
            "max_notifications_per_hour": self.max_notifications_per_hour,
        }


@dataclass
class NotificationRecord:
    """通知记录"""

    notification_id: str
    alert_id: str
    channel: NotificationChannel
    recipient: str
    subject: str
    content: str
    status: NotificationStatus
    created_at: datetime
    sent_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notification_id": self.notification_id,
            "alert_id": self.alert_id,
            "channel": self.channel.value,
            "recipient": self.recipient,
            "subject": self.subject,
            "content": self.content,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }


class EmailNotifier:
    """邮件通知器"""

    def __init__(self, config: NotificationConfig):
        self.config = config

    async def send_notification(
        self, subject: str, content: str, recipients: List[str]
    ) -> bool:
        """发送邮件通知"""
        if not self.config.enabled or not recipients:
            return False

        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg["From"] = self.config.smtp_username
            msg["Subject"] = subject

            # 添加HTML内容
            html_content = self._format_html_content(content)
            msg.attach(MIMEText(html_content, "html"))

            # 发送邮件
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)

                for recipient in recipients:
                    msg["To"] = recipient
                    server.send_message(msg)
                    del msg["To"]

            logger.info(f"邮件通知已发送给 {len(recipients)} 个收件人")
            return True

        except Exception as e:
            logger.error(f"发送邮件通知失败: {e}")
            return False

    def _format_html_content(self, content: str) -> str:
        """格式化HTML内容"""
        html_template = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert-header {{ background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; }}
                .alert-content {{ margin: 20px 0; }}
                .alert-footer {{ color: #6c757d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h2>🚨 MLOps系统告警</h2>
            </div>
            <div class="alert-content">
                <pre>{content}</pre>
            </div>
            <div class="alert-footer">
                <p>此邮件由MLOps监控系统自动发送，请勿回复。</p>
                <p>发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        return html_template


class WebSocketNotifier:
    """WebSocket通知器"""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self.connections: Set[Any] = set()  # WebSocket连接集合

    def add_connection(self, websocket):
        """添加WebSocket连接"""
        self.connections.add(websocket)
        logger.info(f"添加WebSocket连接，当前连接数: {len(self.connections)}")

    def remove_connection(self, websocket):
        """移除WebSocket连接"""
        self.connections.discard(websocket)
        logger.info(f"移除WebSocket连接，当前连接数: {len(self.connections)}")

    async def send_notification(self, subject: str, content: str) -> bool:
        """发送WebSocket通知"""
        if not self.config.enabled or not self.connections:
            return False

        try:
            notification_data = {
                "type": "alert",
                "subject": subject,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }

            message = json.dumps(notification_data)

            # 发送给所有连接的客户端
            disconnected = set()
            for websocket in self.connections:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.warning(f"WebSocket发送失败: {e}")
                    disconnected.add(websocket)

            # 清理断开的连接
            for websocket in disconnected:
                self.connections.discard(websocket)

            logger.info(
                f"WebSocket通知已发送给 {len(self.connections) - len(disconnected)} 个客户端"
            )
            return True

        except Exception as e:
            logger.error(f"发送WebSocket通知失败: {e}")
            return False


class WebhookNotifier:
    """Webhook通知器"""

    def __init__(self, config: NotificationConfig):
        self.config = config

    async def send_notification(self, subject: str, content: str) -> bool:
        """发送Webhook通知"""
        if not self.config.enabled or not self.config.webhook_url:
            return False

        try:
            import aiohttp

            payload = {
                "subject": subject,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "source": "mlops-monitoring",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                    headers=self.config.webhook_headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook通知发送成功: {self.config.webhook_url}")
                        return True
                    else:
                        logger.error(f"Webhook通知发送失败，状态码: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"发送Webhook通知失败: {e}")
            return False


class RateLimiter:
    """限流器"""

    def __init__(self):
        self.notification_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.lock = threading.Lock()

    def can_send_notification(self, key: str, config: NotificationConfig) -> bool:
        """检查是否可以发送通知"""
        with self.lock:
            now = datetime.now()
            history = self.notification_history[key]

            # 清理过期记录
            cutoff_time = now - timedelta(hours=1)
            while history and history[0] < cutoff_time:
                history.popleft()

            # 检查每小时限制
            if len(history) >= config.max_notifications_per_hour:
                return False

            # 检查限流间隔
            if history:
                last_notification = history[-1]
                if now - last_notification < timedelta(
                    minutes=config.rate_limit_minutes
                ):
                    return False

            # 记录本次通知
            history.append(now)
            return True


class AlertNotificationManager:
    """告警通知管理器"""

    def __init__(self):
        self.notification_configs: Dict[NotificationChannel, NotificationConfig] = {}
        self.notifiers: Dict[NotificationChannel, Any] = {}
        self.rate_limiter = RateLimiter()
        self.notification_records: List[NotificationRecord] = []
        self.max_records = 10000
        self.lock = threading.Lock()

        # 初始化默认配置
        self._init_default_configs()

        logger.info("告警通知管理器初始化完成")

    def _init_default_configs(self):
        """初始化默认配置"""
        # 邮件配置
        email_config = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            enabled=False,  # 默认禁用，需要用户配置
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            rate_limit_minutes=10,
            max_notifications_per_hour=5,
        )
        self.notification_configs[NotificationChannel.EMAIL] = email_config

        # WebSocket配置
        websocket_config = NotificationConfig(
            channel=NotificationChannel.WEBSOCKET,
            enabled=True,
            rate_limit_minutes=1,
            max_notifications_per_hour=30,
        )
        self.notification_configs[NotificationChannel.WEBSOCKET] = websocket_config
        self.notifiers[NotificationChannel.WEBSOCKET] = WebSocketNotifier(
            websocket_config
        )

        # Webhook配置
        webhook_config = NotificationConfig(
            channel=NotificationChannel.WEBHOOK,
            enabled=False,
            rate_limit_minutes=5,
            max_notifications_per_hour=10,
        )
        self.notification_configs[NotificationChannel.WEBHOOK] = webhook_config

    def update_config(self, channel: NotificationChannel, config: NotificationConfig):
        """更新通知配置"""
        self.notification_configs[channel] = config

        # 重新初始化通知器
        if channel == NotificationChannel.EMAIL and config.enabled:
            self.notifiers[channel] = EmailNotifier(config)
        elif channel == NotificationChannel.WEBSOCKET:
            self.notifiers[channel] = WebSocketNotifier(config)
        elif channel == NotificationChannel.WEBHOOK and config.enabled:
            self.notifiers[channel] = WebhookNotifier(config)

        logger.info(f"更新通知配置: {channel.value}")

    def get_config(self, channel: NotificationChannel) -> Optional[NotificationConfig]:
        """获取通知配置"""
        return self.notification_configs.get(channel)

    async def send_alert_notification(self, alert: Alert):
        """发送告警通知"""
        subject = f"[{alert.level.value.upper()}] {alert.rule_name}"
        content = self._format_alert_content(alert)

        await self._send_notification("alert", alert.alert_id, subject, content)

    async def send_drift_notification(self, drift_report: DriftReport):
        """发送漂移检测通知"""
        if drift_report.overall_severity in [
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        ]:
            subject = f"[数据漂移] {drift_report.model_id} - {drift_report.overall_severity.value.upper()}"
            content = self._format_drift_content(drift_report)

            await self._send_notification(
                "drift", drift_report.report_id, subject, content
            )

    async def send_custom_notification(
        self, subject: str, content: str, notification_type: str = "custom"
    ):
        """发送自定义通知"""
        notification_id = str(uuid.uuid4())
        await self._send_notification(
            notification_type, notification_id, subject, content
        )

    async def _send_notification(
        self, notification_type: str, source_id: str, subject: str, content: str
    ):
        """发送通知到所有启用的渠道"""
        for channel, config in self.notification_configs.items():
            if not config.enabled:
                continue

            # 检查限流
            rate_limit_key = f"{notification_type}_{channel.value}"
            if not self.rate_limiter.can_send_notification(rate_limit_key, config):
                logger.warning(f"通知被限流: {channel.value}")
                continue

            # 发送通知
            await self._send_to_channel(channel, source_id, subject, content)

    async def _send_to_channel(
        self, channel: NotificationChannel, source_id: str, subject: str, content: str
    ):
        """发送通知到指定渠道"""
        if channel not in self.notifiers:
            logger.warning(f"通知器未初始化: {channel.value}")
            return

        notifier = self.notifiers[channel]
        config = self.notification_configs[channel]

        # 创建通知记录
        notification_record = NotificationRecord(
            notification_id=str(uuid.uuid4()),
            alert_id=source_id,
            channel=channel,
            recipient="",  # 将在发送时填充
            subject=subject,
            content=content,
            status=NotificationStatus.PENDING,
            created_at=datetime.now(),
        )

        try:
            success = False

            if channel == NotificationChannel.EMAIL:
                recipients = config.email_recipients
                if recipients:
                    notification_record.recipient = ", ".join(recipients)
                    success = await notifier.send_notification(
                        subject, content, recipients
                    )

            elif channel == NotificationChannel.WEBSOCKET:
                notification_record.recipient = "websocket_clients"
                success = await notifier.send_notification(subject, content)

            elif channel == NotificationChannel.WEBHOOK:
                notification_record.recipient = config.webhook_url or ""
                success = await notifier.send_notification(subject, content)

            # 更新通知状态
            if success:
                notification_record.status = NotificationStatus.SENT
                notification_record.sent_at = datetime.now()
            else:
                notification_record.status = NotificationStatus.FAILED
                notification_record.error_message = "发送失败"

        except Exception as e:
            notification_record.status = NotificationStatus.FAILED
            notification_record.error_message = str(e)
            logger.error(f"发送通知失败 {channel.value}: {e}")

        # 存储通知记录
        with self.lock:
            self.notification_records.append(notification_record)
            if len(self.notification_records) > self.max_records:
                self.notification_records = self.notification_records[
                    -self.max_records :
                ]

    def _format_alert_content(self, alert: Alert) -> str:
        """格式化告警内容"""
        content = """
告警详情:
- 告警ID: {alert.alert_id}
- 规则名称: {alert.rule_name}
- 告警级别: {alert.level.value.upper()}
- 模型: {alert.model_id} (版本: {alert.model_version})
- 指标值: {alert.metric_value:.2f}
- 阈值: {alert.threshold}
- 触发时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- 消息: {alert.message}

建议操作:
- 检查模型性能和资源使用情况
- 查看相关日志和监控指标
- 如有必要，考虑重启或回滚模型
        """
        return content.strip()

    def _format_drift_content(self, drift_report: DriftReport) -> str:
        """格式化漂移检测内容"""
        content = """
数据漂移检测报告:
- 报告ID: {drift_report.report_id}
- 模型: {drift_report.model_id} (版本: {drift_report.model_version})
- 总体漂移分数: {drift_report.overall_drift_score:.3f}
- 严重程度: {drift_report.overall_severity.value.upper()}
- 检测时间: {drift_report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

检测结果摘要:
- 检查特征数: {drift_report.summary.get('total_features_checked', 0)}
- 发现漂移特征数: {drift_report.summary.get('features_with_drift', 0)}

建议操作:
        """

        for recommendation in drift_report.recommendations:
            content += f"- {recommendation}\n"

        return content.strip()

    def get_notification_history(
        self,
        channel: Optional[NotificationChannel] = None,
        status: Optional[NotificationStatus] = None,
        limit: int = 100,
    ) -> List[NotificationRecord]:
        """获取通知历史"""
        with self.lock:
            records = self.notification_records

            if channel:
                records = [r for r in records if r.channel == channel]

            if status:
                records = [r for r in records if r.status == status]

            return records[-limit:]

    def get_notification_stats(self) -> Dict[str, Any]:
        """获取通知统计"""
        with self.lock:
            records = self.notification_records

            if not records:
                return {}

            # 按渠道统计
            channel_stats = defaultdict(lambda: {"sent": 0, "failed": 0, "total": 0})

            # 按状态统计
            status_stats = defaultdict(int)

            # 最近24小时统计
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_records = [r for r in records if r.created_at >= cutoff_time]

            for record in records:
                channel_stats[record.channel.value]["total"] += 1
                if record.status == NotificationStatus.SENT:
                    channel_stats[record.channel.value]["sent"] += 1
                elif record.status == NotificationStatus.FAILED:
                    channel_stats[record.channel.value]["failed"] += 1

                status_stats[record.status.value] += 1

            return {
                "total_notifications": len(records),
                "recent_24h_notifications": len(recent_records),
                "channel_stats": dict(channel_stats),
                "status_stats": dict(status_stats),
                "success_rate": (
                    status_stats.get("sent", 0) / len(records) if records else 0
                ),
            }

    def add_websocket_connection(self, websocket):
        """添加WebSocket连接"""
        if NotificationChannel.WEBSOCKET in self.notifiers:
            self.notifiers[NotificationChannel.WEBSOCKET].add_connection(websocket)

    def remove_websocket_connection(self, websocket):
        """移除WebSocket连接"""
        if NotificationChannel.WEBSOCKET in self.notifiers:
            self.notifiers[NotificationChannel.WEBSOCKET].remove_connection(websocket)


# 全局告警通知管理器实例
alert_notification_manager = AlertNotificationManager()
