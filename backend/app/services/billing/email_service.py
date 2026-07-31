"""SMTP 邮件服务。

支持纯文本和 HTML 邮件发送；未配置 SMTP 时降级为日志输出，
不会阻断业务流程。
"""

from typing import Optional
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

from loguru import logger

from app.core.config import settings


def _smtp_configured() -> bool:
    """检查 SMTP 是否已配置。"""
    return bool(settings.SMTP_HOST and settings.SMTP_USERNAME and settings.SMTP_PASSWORD)


def send_email(
    to_email: str,
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
    from_email: Optional[str] = None,
    from_name: Optional[str] = None,
) -> bool:
    """发送邮件。

    返回 True 表示发送成功或 SMTP 未配置时的降级成功（日志记录）。
    返回 False 表示发送失败。
    """
    sender = from_email or settings.SMTP_FROM_EMAIL or "noreply@stock-prediction.com"
    sender_name = from_name or settings.SMTP_FROM_NAME or "股票预测平台"

    if not _smtp_configured():
        logger.warning(
            "SMTP 未配置，邮件降级为日志输出: to={}, subject={}",
            to_email,
            subject,
        )
        logger.info("邮件内容（纯文本）:\n{}", body_text)
        if body_html:
            logger.debug("邮件内容（HTML）:\n{}", body_html)
        return True

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{sender_name} <{sender}>"
    msg["To"] = to_email

    # 纯文本部分
    msg.attach(MIMEText(body_text, "plain", "utf-8"))

    # HTML 部分（如有）
    if body_html:
        msg.attach(MIMEText(body_html, "html", "utf-8"))

    try:
        if settings.SMTP_USE_TLS:
            server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT or 587)
            server.starttls()
        elif settings.SMTP_USE_SSL:
            server = smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT or 465)
        else:
            server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT or 25)

        server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
        server.sendmail(sender, [to_email], msg.as_string())
        server.quit()
        logger.info("邮件发送成功: to={}, subject={}", to_email, subject)
        return True
    except Exception as exc:
        logger.opt(exception=True).error(
            "邮件发送失败: to={}, subject={}, error={}", to_email, subject, exc
        )
        return False


def send_password_reset_email(
    to_email: str,
    reset_token: str,
    base_url: str = "http://localhost:13000",
) -> bool:
    """发送密码重置邮件。"""
    reset_url = f"{base_url}/reset-password?token={reset_token}"
    subject = "【股票预测平台】密码重置"

    body_text = (
        f"您好，\n\n"
        f"您请求了密码重置。请点击以下链接重置密码（1小时内有效）：\n\n"
        f"{reset_url}\n\n"
        f"如果您没有请求重置密码，请忽略此邮件。\n\n"
        f"—— 股票预测平台"
    )

    body_html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 480px; margin: 0 auto; padding: 24px;">
        <h2 style="color: #333;">密码重置</h2>
        <p style="color: #555; line-height: 1.6;">
            您好，<br><br>
            您请求了密码重置。请点击下方按钮重置密码（1小时内有效）：
        </p>
        <a href="{reset_url}"
           style="display: inline-block; padding: 12px 24px; background-color: #4F46E5;
                  color: #fff; text-decoration: none; border-radius: 6px; font-weight: 500;
                  margin: 16px 0;">
            重置密码
        </a>
        <p style="color: #999; font-size: 13px; margin-top: 24px;">
            如果按钮无法点击，请复制以下链接到浏览器地址栏：<br>
            <a href="{reset_url}" style="color: #4F46E5;">{reset_url}</a>
        </p>
        <p style="color: #999; font-size: 13px; border-top: 1px solid #eee; padding-top: 16px; margin-top: 24px;">
            如果您没有请求重置密码，请忽略此邮件。
        </p>
    </div>
    """

    return send_email(to_email, subject, body_text, body_html)


def send_welcome_email(to_email: str, username: str) -> bool:
    """发送注册欢迎邮件。"""
    subject = "【股票预测平台】欢迎加入"

    body_text = (
        f"您好 {username}，\n\n"
        f"欢迎注册股票预测平台！您已获得免费版套餐：\n"
        f"  · 每月 50 次回测\n"
        f"  · 最多 20 个策略\n"
        f"  · 同时运行 3 个任务\n\n"
        f"如需更多功能，请升级到专业版。\n\n"
        f"—— 股票预测平台"
    )

    body_html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 480px; margin: 0 auto; padding: 24px;">
        <h2 style="color: #333;">🎉 欢迎加入！</h2>
        <p style="color: #555; line-height: 1.6;">
            您好 <strong>{username}</strong>，<br><br>
            欢迎注册股票预测平台！您已获得<strong>免费版</strong>套餐：
        </p>
        <ul style="color: #555; line-height: 1.8; padding-left: 20px;">
            <li>每月 50 次回测</li>
            <li>最多 20 个策略</li>
            <li>同时运行 3 个任务</li>
        </ul>
        <p style="color: #555; line-height: 1.6;">
            如需更多功能，可随时在
            <a href="http://localhost:13000/pricing" style="color: #4F46E5;">定价页</a>
            升级到专业版。
        </p>
    </div>
    """

    return send_email(to_email, subject, body_text, body_html)
