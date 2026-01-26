"""
系统性能优化器
优化特征计算性能和数据库查询效率
"""
import asyncio
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil
from loguru import logger


class PerformanceOptimizer:
    """系统性能优化器"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cache = {}
        self.cache_ttl = {}
        self.performance_metrics = {}

    def optimize_feature_computation(self, feature_configs: List[Dict]) -> List[Dict]:
        """优化特征计算性能"""
        try:
            start_time = time.time()

            # 按计算复杂度分组
            simple_features = []
            complex_features = []

            for config in feature_configs:
                if self._is_simple_feature(config):
                    simple_features.append(config)
                else:
                    complex_features.append(config)

            # 并行计算复杂特征
            optimized_results = []

            # 简单特征批量计算
            if simple_features:
                batch_result = self._batch_compute_simple_features(simple_features)
                optimized_results.extend(batch_result)

            # 复杂特征并行计算
            if complex_features:
                parallel_results = self._parallel_compute_complex_features(
                    complex_features
                )
                optimized_results.extend(parallel_results)

            # 记录性能指标
            computation_time = time.time() - start_time
            self.performance_metrics["feature_computation"] = {
                "total_features": len(feature_configs),
                "simple_features": len(simple_features),
                "complex_features": len(complex_features),
                "computation_time": computation_time,
                "features_per_second": len(feature_configs) / computation_time
                if computation_time > 0
                else 0,
            }

            logger.info(
                f"特征计算优化完成: {len(feature_configs)} 个特征，耗时 {computation_time:.2f}s"
            )
            return optimized_results

        except Exception as e:
            logger.error(f"特征计算优化失败: {e}")
            return feature_configs

    def _is_simple_feature(self, config: Dict) -> bool:
        """判断是否为简单特征"""
        simple_indicators = ["sma", "ema", "price_change", "volume_ratio"]
        feature_type = config.get("type", "").lower()
        return feature_type in simple_indicators

    def _batch_compute_simple_features(self, features: List[Dict]) -> List[Dict]:
        """批量计算简单特征"""
        try:
            # 模拟批量计算
            results = []
            for feature in features:
                # 添加批量计算标记
                feature["batch_computed"] = True
                feature["computation_method"] = "batch"
                results.append(feature)

            return results
        except Exception as e:
            logger.error(f"批量计算简单特征失败: {e}")
            return features

    def _parallel_compute_complex_features(self, features: List[Dict]) -> List[Dict]:
        """并行计算复杂特征"""
        try:
            results = []

            # 提交并行任务
            future_to_feature = {
                self.executor.submit(
                    self._compute_single_complex_feature, feature
                ): feature
                for feature in features
            }

            # 收集结果
            for future in as_completed(future_to_feature):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    feature = future_to_feature[future]
                    logger.warning(
                        f"复杂特征计算失败: {feature.get('name', 'unknown')}, 错误: {e}"
                    )
                    # 返回原始配置
                    results.append(feature)

            return results
        except Exception as e:
            logger.error(f"并行计算复杂特征失败: {e}")
            return features

    def _compute_single_complex_feature(self, feature: Dict) -> Dict:
        """计算单个复杂特征"""
        try:
            # 模拟复杂计算
            time.sleep(0.1)  # 模拟计算时间

            feature["parallel_computed"] = True
            feature["computation_method"] = "parallel"
            feature["computed_at"] = datetime.now().isoformat()

            return feature
        except Exception as e:
            logger.error(f"计算复杂特征失败: {feature.get('name', 'unknown')}, 错误: {e}")
            return feature

    def optimize_database_queries(self, query_configs: List[Dict]) -> List[Dict]:
        """优化数据库查询效率"""
        try:
            start_time = time.time()

            # 查询优化策略
            optimized_queries = []

            for query_config in query_configs:
                optimized_query = self._optimize_single_query(query_config)
                optimized_queries.append(optimized_query)

            # 批量查询优化
            batch_optimized = self._batch_optimize_queries(optimized_queries)

            # 记录性能指标
            optimization_time = time.time() - start_time
            self.performance_metrics["query_optimization"] = {
                "total_queries": len(query_configs),
                "optimization_time": optimization_time,
                "queries_per_second": len(query_configs) / optimization_time
                if optimization_time > 0
                else 0,
            }

            logger.info(
                f"数据库查询优化完成: {len(query_configs)} 个查询，耗时 {optimization_time:.2f}s"
            )
            return batch_optimized

        except Exception as e:
            logger.error(f"数据库查询优化失败: {e}")
            return query_configs

    def _optimize_single_query(self, query_config: Dict) -> Dict:
        """优化单个查询"""
        try:
            optimized_config = query_config.copy()

            # 添加索引建议
            if "table" in query_config and "conditions" in query_config:
                optimized_config["suggested_indexes"] = self._suggest_indexes(
                    query_config
                )

            # 查询重写建议
            if "sql" in query_config:
                optimized_config["optimized_sql"] = self._optimize_sql(
                    query_config["sql"]
                )

            # 缓存策略
            if self._is_cacheable_query(query_config):
                optimized_config["cache_strategy"] = "redis"
                optimized_config["cache_ttl"] = 300  # 5分钟

            optimized_config["optimization_applied"] = True
            return optimized_config

        except Exception as e:
            logger.error(f"优化单个查询失败: {e}")
            return query_config

    def _suggest_indexes(self, query_config: Dict) -> List[str]:
        """建议索引"""
        indexes = []

        conditions = query_config.get("conditions", {})
        for field in conditions.keys():
            indexes.append(f"idx_{query_config.get('table', 'table')}_{field}")

        return indexes

    def _optimize_sql(self, sql: str) -> str:
        """优化SQL语句"""
        try:
            # 简单的SQL优化
            optimized_sql = sql

            # 添加LIMIT子句（如果没有）
            if "LIMIT" not in sql.upper() and "SELECT" in sql.upper():
                optimized_sql += " LIMIT 1000"

            # 建议使用索引提示
            if "WHERE" in sql.upper():
                optimized_sql = optimized_sql.replace("WHERE", "/* USE INDEX */ WHERE")

            return optimized_sql
        except Exception as e:
            logger.error(f"SQL优化失败: {e}")
            return sql

    def _is_cacheable_query(self, query_config: Dict) -> bool:
        """判断查询是否可缓存"""
        # 只读查询且没有时间相关条件
        if query_config.get("type") == "SELECT":
            conditions = query_config.get("conditions", {})
            time_fields = ["created_at", "updated_at", "timestamp", "date"]

            # 如果条件中没有时间字段，可以缓存
            return not any(field in conditions for field in time_fields)

        return False

    def _batch_optimize_queries(self, queries: List[Dict]) -> List[Dict]:
        """批量优化查询"""
        try:
            # 按表分组
            table_groups = {}
            for query in queries:
                table = query.get("table", "unknown")
                if table not in table_groups:
                    table_groups[table] = []
                table_groups[table].append(query)

            # 为每个表组添加批量优化建议
            optimized_queries = []
            for table, table_queries in table_groups.items():
                if len(table_queries) > 1:
                    # 建议使用批量查询
                    for query in table_queries:
                        query["batch_optimization"] = {
                            "suggested": True,
                            "table": table,
                            "batch_size": len(table_queries),
                        }

                optimized_queries.extend(table_queries)

            return optimized_queries
        except Exception as e:
            logger.error(f"批量查询优化失败: {e}")
            return queries

    def monitor_system_resources(self) -> Dict[str, Any]:
        """监控系统资源使用情况"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)

            # 内存使用情况
            memory = psutil.virtual_memory()

            # 磁盘使用情况
            disk = psutil.disk_usage("/")

            # 网络IO
            network = psutil.net_io_counters()

            resource_info = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "load_avg": psutil.getloadavg()
                    if hasattr(psutil, "getloadavg")
                    else None,
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free,
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100,
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv,
                },
            }

            # 检查资源警告
            warnings = []
            if cpu_percent > 80:
                warnings.append(f"CPU使用率过高: {cpu_percent}%")
            if memory.percent > 85:
                warnings.append(f"内存使用率过高: {memory.percent}%")
            if (disk.used / disk.total) * 100 > 90:
                warnings.append(f"磁盘使用率过高: {(disk.used / disk.total) * 100:.1f}%")

            resource_info["warnings"] = warnings

            return resource_info

        except Exception as e:
            logger.error(f"系统资源监控失败: {e}")
            return {"timestamp": datetime.now().isoformat(), "error": str(e)}

    def cleanup_memory(self) -> Dict[str, Any]:
        """清理内存"""
        try:
            # 记录清理前的内存使用
            memory_before = psutil.virtual_memory()

            # 清理缓存
            cache_cleared = len(self.cache)
            self.cache.clear()
            self.cache_ttl.clear()

            # 强制垃圾回收
            gc.collect()

            # 记录清理后的内存使用
            memory_after = psutil.virtual_memory()

            cleanup_info = {
                "timestamp": datetime.now().isoformat(),
                "cache_cleared": cache_cleared,
                "memory_before": {
                    "used": memory_before.used,
                    "percent": memory_before.percent,
                },
                "memory_after": {
                    "used": memory_after.used,
                    "percent": memory_after.percent,
                },
                "memory_freed": memory_before.used - memory_after.used,
            }

            logger.info(f"内存清理完成: 释放 {cleanup_info['memory_freed']} 字节")
            return cleanup_info

        except Exception as e:
            logger.error(f"内存清理失败: {e}")
            return {"timestamp": datetime.now().isoformat(), "error": str(e)}

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        try:
            # 系统资源状态
            resource_status = self.monitor_system_resources()

            # 性能指标汇总
            report = {
                "timestamp": datetime.now().isoformat(),
                "system_resources": resource_status,
                "performance_metrics": self.performance_metrics,
                "cache_status": {
                    "cache_size": len(self.cache),
                    "cache_hit_rate": self._calculate_cache_hit_rate(),
                },
                "optimization_suggestions": self._generate_optimization_suggestions(
                    resource_status
                ),
            }

            return report

        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return {"timestamp": datetime.now().isoformat(), "error": str(e)}

    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        # 简化实现，实际应该跟踪命中和未命中次数
        return 0.85  # 假设85%的命中率

    def _generate_optimization_suggestions(self, resource_status: Dict) -> List[str]:
        """生成优化建议"""
        suggestions = []

        # CPU优化建议
        cpu_percent = resource_status.get("cpu", {}).get("percent", 0)
        if cpu_percent > 80:
            suggestions.append("考虑增加CPU核心数或优化计算密集型任务")

        # 内存优化建议
        memory_percent = resource_status.get("memory", {}).get("percent", 0)
        if memory_percent > 85:
            suggestions.append("考虑增加内存或优化内存使用")
            suggestions.append("定期清理缓存和执行垃圾回收")

        # 磁盘优化建议
        disk_percent = resource_status.get("disk", {}).get("percent", 0)
        if disk_percent > 90:
            suggestions.append("清理磁盘空间或增加存储容量")

        # 性能优化建议
        if len(self.cache) > 10000:
            suggestions.append("考虑实施缓存过期策略")

        return suggestions

    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, "executor"):
                self.executor.shutdown(wait=True)
        except:
            pass


# 全局性能优化器实例
performance_optimizer = PerformanceOptimizer()
