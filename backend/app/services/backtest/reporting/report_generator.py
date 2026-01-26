"""
回测报告生成器 - 生成PDF和Excel格式的回测报告
"""

import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.error_handler import ErrorSeverity, TaskError


class BacktestReportGenerator:
    """回测报告生成器"""

    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        self.reports_dir = os.path.join(self.temp_dir, "backtest_reports")
        os.makedirs(self.reports_dir, exist_ok=True)

    async def generate_pdf_report(
        self,
        backtest_result: Dict[str, Any],
        include_charts: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
    ) -> str:
        """生成PDF报告"""
        try:
            # 生成报告文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = backtest_result.get("strategy_name", "backtest")
            filename = f"{strategy_name}_report_{timestamp}.pdf"
            filepath = os.path.join(self.reports_dir, filename)

            # 使用reportlab生成PDF
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4, letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )

            # 创建PDF文档
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # 标题
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=24,
                spaceAfter=30,
                alignment=1,  # 居中
            )
            story.append(Paragraph("回测结果报告", title_style))
            story.append(Spacer(1, 20))

            # 基本信息
            basic_info = [
                ["策略名称", backtest_result.get("strategy_name", "")],
                ["股票代码", ", ".join(backtest_result.get("stock_codes", []))],
                [
                    "回测期间",
                    f"{backtest_result.get('start_date', '')} 至 {backtest_result.get('end_date', '')}",
                ],
                ["初始资金", f"¥{backtest_result.get('initial_cash', 0):,.2f}"],
                ["最终价值", f"¥{backtest_result.get('final_value', 0):,.2f}"],
                ["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ]

            basic_table = Table(basic_info, colWidths=[2 * inch, 4 * inch])
            basic_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("BACKGROUND", (1, 0), (1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )

            story.append(Paragraph("基本信息", styles["Heading2"]))
            story.append(basic_table)
            story.append(Spacer(1, 20))

            # 绩效指标
            performance_data = [
                ["指标", "数值"],
                ["总收益率", f"{backtest_result.get('total_return', 0) * 100:.2f}%"],
                ["年化收益率", f"{backtest_result.get('annualized_return', 0) * 100:.2f}%"],
                ["波动率", f"{backtest_result.get('volatility', 0) * 100:.2f}%"],
                ["夏普比率", f"{backtest_result.get('sharpe_ratio', 0):.3f}"],
                ["最大回撤", f"{backtest_result.get('max_drawdown', 0) * 100:.2f}%"],
                ["胜率", f"{backtest_result.get('win_rate', 0) * 100:.2f}%"],
                ["盈亏比", f"{backtest_result.get('profit_factor', 0):.3f}"],
                ["总交易次数", str(backtest_result.get("total_trades", 0))],
                ["盈利交易", str(backtest_result.get("winning_trades", 0))],
                ["亏损交易", str(backtest_result.get("losing_trades", 0))],
            ]

            performance_table = Table(performance_data, colWidths=[3 * inch, 3 * inch])
            performance_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )

            story.append(Paragraph("绩效指标", styles["Heading2"]))
            story.append(performance_table)
            story.append(Spacer(1, 20))

            # 交易记录摘要（如果包含）
            if not include_tables or "trade_summary" in include_tables:
                trade_history = backtest_result.get("trade_history", [])
                if trade_history:
                    story.append(Paragraph("交易记录摘要", styles["Heading2"]))

                    # 只显示前10条交易记录
                    trade_data = [["股票代码", "操作", "数量", "价格", "时间", "盈亏"]]
                    for trade in trade_history[:10]:
                        trade_data.append(
                            [
                                trade.get("stock_code", ""),
                                trade.get("action", ""),
                                str(trade.get("quantity", 0)),
                                f"¥{trade.get('price', 0):.2f}",
                                trade.get("timestamp", "")[:10],  # 只显示日期
                                f"¥{trade.get('pnl', 0):.2f}",
                            ]
                        )

                    trade_table = Table(
                        trade_data,
                        colWidths=[
                            1 * inch,
                            0.8 * inch,
                            0.8 * inch,
                            1 * inch,
                            1.2 * inch,
                            1 * inch,
                        ],
                    )
                    trade_table.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, -1), 8),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            ]
                        )
                    )

                    story.append(trade_table)
                    if len(trade_history) > 10:
                        story.append(
                            Paragraph(
                                f"注：仅显示前10条交易记录，总共{len(trade_history)}条",
                                styles["Normal"],
                            )
                        )
                    story.append(Spacer(1, 20))

            # 生成PDF
            doc.build(story)

            logger.info(f"PDF报告生成成功: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"生成PDF报告失败: {e}", exc_info=True)
            raise TaskError(
                message=f"生成PDF报告失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                original_exception=e,
            )

    async def generate_excel_report(
        self, backtest_result: Dict[str, Any], include_raw_data: bool = False
    ) -> str:
        """生成Excel报告"""
        try:
            # 生成报告文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = backtest_result.get("strategy_name", "backtest")
            filename = f"{strategy_name}_data_{timestamp}.xlsx"
            filepath = os.path.join(self.reports_dir, filename)

            # 创建Excel写入器
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                # 1. 基本信息工作表
                basic_info_data = {
                    "项目": ["策略名称", "股票代码", "回测开始日期", "回测结束日期", "初始资金", "最终价值", "生成时间"],
                    "数值": [
                        backtest_result.get("strategy_name", ""),
                        ", ".join(backtest_result.get("stock_codes", [])),
                        backtest_result.get("start_date", ""),
                        backtest_result.get("end_date", ""),
                        backtest_result.get("initial_cash", 0),
                        backtest_result.get("final_value", 0),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ],
                }
                basic_df = pd.DataFrame(basic_info_data)
                basic_df.to_excel(writer, sheet_name="基本信息", index=False)

                # 2. 绩效指标工作表
                performance_data = {
                    "指标": [
                        "总收益率",
                        "年化收益率",
                        "波动率",
                        "夏普比率",
                        "最大回撤",
                        "胜率",
                        "盈亏比",
                        "总交易次数",
                        "盈利交易",
                        "亏损交易",
                    ],
                    "数值": [
                        backtest_result.get("total_return", 0),
                        backtest_result.get("annualized_return", 0),
                        backtest_result.get("volatility", 0),
                        backtest_result.get("sharpe_ratio", 0),
                        backtest_result.get("max_drawdown", 0),
                        backtest_result.get("win_rate", 0),
                        backtest_result.get("profit_factor", 0),
                        backtest_result.get("total_trades", 0),
                        backtest_result.get("winning_trades", 0),
                        backtest_result.get("losing_trades", 0),
                    ],
                    "格式化显示": [
                        f"{backtest_result.get('total_return', 0) * 100:.2f}%",
                        f"{backtest_result.get('annualized_return', 0) * 100:.2f}%",
                        f"{backtest_result.get('volatility', 0) * 100:.2f}%",
                        f"{backtest_result.get('sharpe_ratio', 0):.3f}",
                        f"{backtest_result.get('max_drawdown', 0) * 100:.2f}%",
                        f"{backtest_result.get('win_rate', 0) * 100:.2f}%",
                        f"{backtest_result.get('profit_factor', 0):.3f}",
                        str(backtest_result.get("total_trades", 0)),
                        str(backtest_result.get("winning_trades", 0)),
                        str(backtest_result.get("losing_trades", 0)),
                    ],
                }
                performance_df = pd.DataFrame(performance_data)
                performance_df.to_excel(writer, sheet_name="绩效指标", index=False)

                # 3. 交易记录工作表
                trade_history = backtest_result.get("trade_history", [])
                if trade_history:
                    trade_df = pd.DataFrame(trade_history)
                    # 重命名列为中文
                    column_mapping = {
                        "stock_code": "股票代码",
                        "action": "操作",
                        "quantity": "数量",
                        "price": "价格",
                        "timestamp": "时间",
                        "commission": "手续费",
                        "pnl": "盈亏",
                    }
                    trade_df = trade_df.rename(columns=column_mapping)
                    trade_df.to_excel(writer, sheet_name="交易记录", index=False)

                # 4. 组合历史工作表
                portfolio_history = backtest_result.get("portfolio_history", [])
                if portfolio_history:
                    portfolio_df = pd.DataFrame(portfolio_history)
                    # 重命名列为中文
                    column_mapping = {
                        "date": "日期",
                        "portfolio_value": "组合价值",
                        "cash": "现金",
                        "positions_count": "持仓数量",
                    }
                    # 只重命名存在的列
                    existing_columns = {
                        k: v
                        for k, v in column_mapping.items()
                        if k in portfolio_df.columns
                    }
                    portfolio_df = portfolio_df.rename(columns=existing_columns)

                    # 计算累积收益率
                    if "组合价值" in portfolio_df.columns:
                        initial_value = portfolio_df["组合价值"].iloc[0]
                        portfolio_df["累积收益率"] = (
                            portfolio_df["组合价值"] / initial_value - 1
                        ) * 100

                    portfolio_df.to_excel(writer, sheet_name="组合历史", index=False)

                # 5. 如果包含原始数据，添加配置信息
                if include_raw_data:
                    config_data = backtest_result.get("backtest_config", {})
                    if config_data:
                        config_items = []
                        for key, value in config_data.items():
                            config_items.append({"配置项": key, "数值": str(value)})

                        if config_items:
                            config_df = pd.DataFrame(config_items)
                            config_df.to_excel(writer, sheet_name="配置信息", index=False)

            logger.info(f"Excel报告生成成功: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"生成Excel报告失败: {e}", exc_info=True)
            raise TaskError(
                message=f"生成Excel报告失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                original_exception=e,
            )
