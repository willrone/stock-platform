"""
回测对比分析器 - 用于对比多个回测结果
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from app.core.error_handler import TaskError, ErrorSeverity


class BacktestComparisonAnalyzer:
    """回测对比分析器"""
    
    async def analyze_comparison(
        self, 
        comparison_results: List[Dict[str, Any]], 
        comparison_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """分析多个回测结果的对比"""
        try:
            if len(comparison_results) < 2:
                raise ValueError("至少需要2个回测结果进行对比")
            
            # 默认对比指标
            if not comparison_metrics:
                comparison_metrics = [
                    'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
                    'max_drawdown', 'win_rate', 'profit_factor'
                ]
            
            # 提取对比数据
            comparison_table = await self._build_comparison_table(comparison_results, comparison_metrics)
            
            # 计算相关性分析
            correlation_analysis = await self._calculate_correlation_analysis(comparison_results)
            
            # 风险收益散点图数据
            risk_return_scatter = await self._generate_risk_return_scatter(comparison_results)
            
            # 收益曲线对比数据
            equity_curves_comparison = await self._generate_equity_curves_comparison(comparison_results)
            
            # 排名分析
            ranking_analysis = await self._generate_ranking_analysis(comparison_results, comparison_metrics)
            
            return {
                "comparison_table": comparison_table,
                "correlation_analysis": correlation_analysis,
                "risk_return_scatter": risk_return_scatter,
                "equity_curves_comparison": equity_curves_comparison,
                "ranking_analysis": ranking_analysis,
                "summary": {
                    "total_strategies": len(comparison_results),
                    "comparison_metrics": comparison_metrics,
                    "best_performer": ranking_analysis.get("best_performer", {}),
                    "most_stable": ranking_analysis.get("most_stable", {})
                }
            }
            
        except Exception as e:
            logger.error(f"回测对比分析失败: {e}", exc_info=True)
            raise TaskError(
                message=f"回测对比分析失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                original_exception=e
            )
    
    async def _build_comparison_table(
        self, 
        comparison_results: List[Dict[str, Any]], 
        comparison_metrics: List[str]
    ) -> List[Dict[str, Any]]:
        """构建对比表格"""
        comparison_table = []
        
        for result in comparison_results:
            task_data = result["result"]
            row = {
                "task_id": result["task_id"],
                "task_name": result["task_name"],
                "strategy_name": task_data.get("strategy_name", ""),
            }
            
            # 添加对比指标
            for metric in comparison_metrics:
                value = task_data.get(metric, 0)
                
                # 格式化数值显示
                if metric in ['total_return', 'annualized_return', 'volatility', 'max_drawdown']:
                    row[metric] = f"{float(value) * 100:.2f}%"
                    row[f"{metric}_raw"] = float(value)
                elif metric in ['sharpe_ratio', 'win_rate', 'profit_factor']:
                    row[metric] = f"{float(value):.3f}"
                    row[f"{metric}_raw"] = float(value)
                else:
                    row[metric] = value
                    row[f"{metric}_raw"] = value
            
            comparison_table.append(row)
        
        return comparison_table
    
    async def _calculate_correlation_analysis(self, comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算策略间相关性分析"""
        try:
            # 提取所有策略的收益率序列
            strategy_returns = {}
            
            for result in comparison_results:
                task_id = result["task_id"]
                task_name = result["task_name"]
                portfolio_history = result["result"].get("portfolio_history", [])
                
                if portfolio_history:
                    df = pd.DataFrame(portfolio_history)
                    if 'date' in df.columns and 'portfolio_value' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        returns = df['portfolio_value'].pct_change().dropna()
                        strategy_returns[f"{task_name} ({task_id})"] = returns
            
            if len(strategy_returns) < 2:
                return {"correlation_matrix": [], "average_correlation": 0}
            
            # 对齐时间序列
            aligned_returns = pd.DataFrame(strategy_returns)
            aligned_returns = aligned_returns.dropna()
            
            if aligned_returns.empty:
                return {"correlation_matrix": [], "average_correlation": 0}
            
            # 计算相关性矩阵
            correlation_matrix = aligned_returns.corr()
            
            # 转换为前端可用的格式
            correlation_data = []
            strategy_names = list(correlation_matrix.columns)
            
            for i, strategy1 in enumerate(strategy_names):
                for j, strategy2 in enumerate(strategy_names):
                    correlation_data.append({
                        "strategy1": strategy1,
                        "strategy2": strategy2,
                        "correlation": float(correlation_matrix.iloc[i, j])
                    })
            
            # 计算平均相关性（排除对角线）
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            avg_correlation = correlation_matrix.values[mask].mean()
            
            return {
                "correlation_matrix": correlation_data,
                "strategy_names": strategy_names,
                "average_correlation": float(avg_correlation)
            }
            
        except Exception as e:
            logger.error(f"计算相关性分析失败: {e}", exc_info=True)
            return {"correlation_matrix": [], "average_correlation": 0}
    
    async def _generate_risk_return_scatter(self, comparison_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成风险收益散点图数据"""
        scatter_data = []
        
        for result in comparison_results:
            task_data = result["result"]
            
            scatter_point = {
                "task_id": result["task_id"],
                "task_name": result["task_name"],
                "strategy_name": task_data.get("strategy_name", ""),
                "return": float(task_data.get("annualized_return", 0)) * 100,  # 转换为百分比
                "risk": float(task_data.get("volatility", 0)) * 100,  # 转换为百分比
                "sharpe_ratio": float(task_data.get("sharpe_ratio", 0)),
                "max_drawdown": float(task_data.get("max_drawdown", 0)) * 100  # 转换为百分比
            }
            
            scatter_data.append(scatter_point)
        
        return scatter_data
    
    async def _generate_equity_curves_comparison(self, comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成收益曲线对比数据"""
        equity_curves = []
        
        for result in comparison_results:
            task_data = result["result"]
            portfolio_history = task_data.get("portfolio_history", [])
            
            if not portfolio_history:
                continue
            
            # 转换为DataFrame
            df = pd.DataFrame(portfolio_history)
            if 'date' in df.columns and 'portfolio_value' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                
                # 计算累积收益率
                initial_value = df['portfolio_value'].iloc[0]
                df['cumulative_return'] = (df['portfolio_value'] / initial_value - 1) * 100
                
                # 构建曲线数据
                curve_data = []
                for _, row in df.iterrows():
                    curve_data.append({
                        'date': row['date'].isoformat(),
                        'cumulative_return': float(row['cumulative_return'])
                    })
                
                equity_curves.append({
                    "task_id": result["task_id"],
                    "task_name": result["task_name"],
                    "strategy_name": task_data.get("strategy_name", ""),
                    "curve_data": curve_data
                })
        
        return {
            "equity_curves": equity_curves,
            "total_strategies": len(equity_curves)
        }
    
    async def _generate_ranking_analysis(
        self, 
        comparison_results: List[Dict[str, Any]], 
        comparison_metrics: List[str]
    ) -> Dict[str, Any]:
        """生成排名分析"""
        rankings = {}
        
        # 为每个指标生成排名
        for metric in comparison_metrics:
            metric_values = []
            
            for result in comparison_results:
                task_data = result["result"]
                value = float(task_data.get(metric, 0))
                
                metric_values.append({
                    "task_id": result["task_id"],
                    "task_name": result["task_name"],
                    "strategy_name": task_data.get("strategy_name", ""),
                    "value": value
                })
            
            # 根据指标类型决定排序方向
            reverse_sort = metric not in ['volatility', 'max_drawdown']  # 波动率和最大回撤越小越好
            metric_values.sort(key=lambda x: x["value"], reverse=reverse_sort)
            
            # 添加排名
            for i, item in enumerate(metric_values):
                item["rank"] = i + 1
            
            rankings[metric] = metric_values
        
        # 找出最佳表现者（基于夏普比率）
        best_performer = {}
        if 'sharpe_ratio' in rankings and rankings['sharpe_ratio']:
            best_performer = rankings['sharpe_ratio'][0]
        
        # 找出最稳定的策略（基于波动率）
        most_stable = {}
        if 'volatility' in rankings and rankings['volatility']:
            most_stable = rankings['volatility'][0]  # 波动率最小的
        
        # 计算综合得分（简单平均排名）
        composite_scores = {}
        for result in comparison_results:
            task_id = result["task_id"]
            total_rank = 0
            valid_metrics = 0
            
            for metric in comparison_metrics:
                if metric in rankings:
                    for item in rankings[metric]:
                        if item["task_id"] == task_id:
                            total_rank += item["rank"]
                            valid_metrics += 1
                            break
            
            if valid_metrics > 0:
                avg_rank = total_rank / valid_metrics
                composite_scores[task_id] = {
                    "task_id": task_id,
                    "task_name": result["task_name"],
                    "strategy_name": result["result"].get("strategy_name", ""),
                    "composite_score": avg_rank
                }
        
        # 按综合得分排序
        composite_ranking = sorted(composite_scores.values(), key=lambda x: x["composite_score"])
        
        return {
            "metric_rankings": rankings,
            "best_performer": best_performer,
            "most_stable": most_stable,
            "composite_ranking": composite_ranking
        }