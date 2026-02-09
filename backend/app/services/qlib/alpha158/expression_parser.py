"""
Qlib 表达式解析器

将 Qlib Alpha158 因子表达式转换为 pandas 操作。
"""

import re
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class QlibExpressionParser:
    """
    Qlib 表达式解析器

    将 Qlib Alpha158 因子表达式（如 Mean($close, 5)）转换为 pandas 操作。
    支持 Alpha158 中使用的所�� Qlib 函数，包括嵌套函数和复杂表达式。
    """

    def __init__(self):
        self._detailed_error_count = 0

    def evaluate(
        self, data: pd.DataFrame, expression: str
    ) -> Optional[pd.Series]:
        """
        评估 Qlib 表达式

        Args:
            data: 包含 OHLCV 数据的 DataFrame
            expression: Qlib 表达式字符串

        Returns:
            计算结果的 Series，如果失败返回 None
        """
        try:
            # 创建一个数据副本用于计算
            calc_data = data.copy()

            # 标准化列名：确保基础列有 $ 前缀
            column_mapping = {}
            for col in ["open", "high", "low", "close", "volume", "vwap"]:
                if col in calc_data.columns and f"${col}" not in calc_data.columns:
                    column_mapping[col] = f"${col}"
            if column_mapping:
                calc_data = calc_data.rename(columns=column_mapping)

            # 确保 $vwap 列存在
            if "$vwap" not in calc_data.columns:
                if "$close" in calc_data.columns:
                    calc_data["$vwap"] = calc_data["$close"]
                elif "close" in calc_data.columns:
                    calc_data["$vwap"] = calc_data["close"]

            expr = expression

            # 步骤1: 处理 Ref 函数
            expr = self._process_ref_functions(expr)

            # 步骤1.5: 修复嵌套的 calc_data
            expr = self._fix_nested_calc_data(expr)

            # 步骤2: 处理 Log 函数
            expr = self._process_log_functions(expr)

            # 步骤3: 处理 Abs 函数
            expr = self._process_abs_functions(expr)

            # 步骤5: 处理 IdxMax 和 IdxMin 函数
            expr = self._process_idx_functions(expr)

            # 步骤6: 处理单变量滚动函数
            expr = self._process_rolling_functions(expr)

            # 步骤7: 处理 Corr 函数
            expr = self._process_corr_functions(expr)

            # 步骤10: 处理嵌套的 Std 函数
            expr = self._process_nested_std(expr)

            # 步骤11: 处理 Quantile 函数
            expr = self._process_quantile_functions(expr)

            # 步骤12: 处理 Rank 函数
            expr = self._process_rank_functions(expr)

            # 步骤13: 处理 Slope, Rsquare, Resi 函数
            expr = self._process_regression_functions(expr)

            # 步骤14: 替换变量引用
            expr = self._replace_variables(expr)

            # 步骤15: 处理 Mean 和 Std 函数
            expr = self._process_mean_std_functions(expr, calc_data)

            # 定义运行时辅助函数
            runtime_funcs = self._get_runtime_functions(calc_data)

            # 评估表达式
            result = eval(expr, {"np": np, "pd": pd, "calc_data": calc_data, **runtime_funcs})

            # 处理返回值
            if isinstance(result, pd.DataFrame):
                if len(result.columns) == 1:
                    return result.iloc[:, 0]
                else:
                    return result.iloc[:, -1]
            elif isinstance(result, pd.Series):
                return result
            else:
                return pd.Series([result] * len(calc_data), index=calc_data.index)

        except Exception as e:
            if self._detailed_error_count < 5:
                logger.warning(f"表达式评估失败: {expression[:100]}... 错误: {e}")
                self._detailed_error_count += 1
            return None

    def _process_ref_functions(self, expr: str) -> str:
        """处理 Ref 函数"""
        def replace_ref(match):
            var = match.group(1)
            n = int(match.group(2))
            return f"calc_data['${var}'].shift({n})"

        max_iterations = 50
        iteration = 0
        while re.search(r"Ref\(\$(\w+),\s*(\d+)\)", expr) and iteration < max_iterations:
            expr = re.sub(r"Ref\(\$(\w+),\s*(\d+)\)", replace_ref, expr)
            iteration += 1
        return expr

    def _fix_nested_calc_data(self, expr: str) -> str:
        """修复嵌套的 calc_data"""
        var_names = ["close", "open", "high", "low", "volume", "vwap"]
        for var_name in var_names:
            nested_pattern = rf"calc_data\['calc_data\['\${var_name}'\]'\]\.shift\(([^)]+)\)"
            if re.search(nested_pattern, expr):
                expr = re.sub(nested_pattern, rf"calc_data['\${var_name}'].shift(\1)", expr)
            nested_patterns = [
                f"calc_data['calc_data['${var_name}']']",
                f"calc_data['calc_data['${var_name}']'].shift",
            ]
            for nested_pattern in nested_patterns:
                if nested_pattern in expr:
                    expr = expr.replace(nested_pattern, f"calc_data['${var_name}']")
        return expr

    def _process_log_functions(self, expr: str) -> str:
        """处理 Log 函数"""
        def replace_log(match):
            inner = match.group(1)
            if "+" in inner or "-" in inner:
                var_match = re.search(r"\$(\w+)", inner)
                if var_match:
                    var = var_match.group(1)
                    const_match = re.search(r"([+-]\s*(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))", inner)
                    if const_match:
                        const = const_match.group(1).replace(" ", "")
                        return f"np.log(calc_data['${var}']{const})"
            var_match = re.search(r"\$(\w+)", inner)
            if var_match:
                var = var_match.group(1)
                return f"np.log(calc_data['${var}'])"
            return f"np.log({inner})"

        max_iterations = 50
        iteration = 0
        while re.search(r"Log\(([^)]+)\)", expr) and iteration < max_iterations:
            expr = re.sub(r"Log\(([^)]+)\)", replace_log, expr)
            iteration += 1
        return expr

    def _process_abs_functions(self, expr: str) -> str:
        """处理 Abs 函数"""
        def replace_abs(match):
            inner = match.group(1)
            return f"np.abs({inner})"

        max_iterations = 50
        iteration = 0
        while re.search(r"Abs\(([^)]+)\)", expr) and iteration < max_iterations:
            expr = re.sub(r"Abs\(([^)]+)\)", replace_abs, expr)
            iteration += 1
        return expr

    def _process_idx_functions(self, expr: str) -> str:
        """处理 IdxMax 和 IdxMin 函数"""
        def replace_idxmax(match):
            var = match.group(1)
            n = int(match.group(2))
            return f"(calc_data['${var}'].rolling({n}).apply(lambda x: (len(x) - 1 - x.argmax()) / {n} if len(x) == {n} and not np.isnan(x).all() else np.nan, raw=True))"

        def replace_idxmin(match):
            var = match.group(1)
            n = int(match.group(2))
            return f"(calc_data['${var}'].rolling({n}).apply(lambda x: (len(x) - 1 - x.argmin()) / {n} if len(x) == {n} and not np.isnan(x).all() else np.nan, raw=True))"

        max_iterations = 50
        iteration = 0
        while re.search(r"IdxMax\(\$(\w+),\s*(\d+)\)", expr) and iteration < max_iterations:
            expr = re.sub(r"IdxMax\(\$(\w+),\s*(\d+)\)", replace_idxmax, expr)
            iteration += 1

        iteration = 0
        while re.search(r"IdxMin\(\$(\w+),\s*(\d+)\)", expr) and iteration < max_iterations:
            expr = re.sub(r"IdxMin\(\$(\w+),\s*(\d+)\)", replace_idxmin, expr)
            iteration += 1
        return expr

    def _process_rolling_functions(self, expr: str) -> str:
        """处理单变量滚动函数"""
        def replace_max(match):
            var = match.group(1)
            n = int(match.group(2))
            return f"calc_data['${var}'].rolling({n}).max()"

        def replace_min(match):
            var = match.group(1)
            n = int(match.group(2))
            return f"calc_data['${var}'].rolling({n}).min()"

        def replace_mean(match):
            var = match.group(1)
            n = int(match.group(2))
            return f"calc_data['${var}'].rolling({n}).mean()"

        def replace_std(match):
            var = match.group(1)
            n = int(match.group(2))
            return f"calc_data['${var}'].rolling({n}).std()"

        max_iterations = 50
        for pattern, replacer in [
            (r"Max\(\$(\w+),\s*(\d+)\)", replace_max),
            (r"Min\(\$(\w+),\s*(\d+)\)", replace_min),
            (r"Mean\(\$(\w+),\s*(\d+)\)", replace_mean),
            (r"Std\(\$(\w+),\s*(\d+)\)", replace_std),
        ]:
            iteration = 0
            while re.search(pattern, expr) and iteration < max_iterations:
                expr = re.sub(pattern, replacer, expr)
                iteration += 1
        return expr

    def _process_corr_functions(self, expr: str) -> str:
        """处理 Corr 函数"""
        def replace_corr(match):
            var1_expr = match.group(1)
            var2_expr = match.group(2)
            n = int(match.group(3))
            var1_match = re.search(r"\$(\w+)", var1_expr)
            var2_match = re.search(r"\$(\w+)", var2_expr)
            if var1_match and var2_match:
                var1 = var1_match.group(1)
                var2 = var2_match.group(1)
                return f"calc_data['${var1}'].rolling({n}).corr(calc_data['${var2}'])"
            return f"pd.Series({var1_expr}).rolling({n}).corr(pd.Series({var2_expr}))"

        max_iterations = 50
        iteration = 0
        while re.search(r"Corr\(([^,]+),\s*([^,]+),\s*(\d+)\)", expr) and iteration < max_iterations:
            expr = re.sub(r"Corr\(([^,]+),\s*([^,]+),\s*(\d+)\)", replace_corr, expr)
            iteration += 1
        return expr

    def _process_nested_std(self, expr: str) -> str:
        """处理嵌套的 Std 函数"""
        def find_matching_paren(s, start_pos):
            count = 0
            i = start_pos
            while i < len(s):
                if s[i] == "(":
                    count += 1
                elif s[i] == ")":
                    count -= 1
                    if count == 0:
                        return i
                i += 1
            return -1

        max_iterations = 20
        iteration = 0
        prev_expr = ""
        while iteration < max_iterations:
            if expr == prev_expr:
                break
            prev_expr = expr

            std_positions = [m.start() for m in re.finditer(r"Std\(", expr)]
            if not std_positions:
                break

            replaced = False
            for pos in reversed(std_positions):
                end_pos = find_matching_paren(expr, pos + 4)
                if end_pos > 0:
                    comma_pos = expr.rfind(",", pos + 4, end_pos)
                    if comma_pos > 0:
                        inner_expr = expr[pos + 4:comma_pos].strip()
                        n_str = expr[comma_pos + 1:end_pos].strip()
                        try:
                            n = int(n_str)
                            if re.match(r"calc_data\[\'\$\w+\'\]", inner_expr.strip()):
                                continue
                            replacement = f"pd.Series({inner_expr}, index=calc_data.index).rolling({n}).std()"
                            expr = expr[:pos] + replacement + expr[end_pos + 1:]
                            replaced = True
                            break
                        except ValueError:
                            pass

            if not replaced:
                break
            iteration += 1
        return expr

    def _process_quantile_functions(self, expr: str) -> str:
        """处理 Quantile 函数"""
        def replace_quantile(match):
            var = match.group(1)
            n = int(match.group(2))
            q = float(match.group(3))
            return f"calc_data['${var}'].rolling({n}).quantile({q})"

        max_iterations = 50
        iteration = 0
        while re.search(r"Quantile\(\$(\w+),\s*(\d+),\s*([\d.]+)\)", expr) and iteration < max_iterations:
            expr = re.sub(r"Quantile\(\$(\w+),\s*(\d+),\s*([\d.]+)\)", replace_quantile, expr)
            iteration += 1
        return expr

    def _process_rank_functions(self, expr: str) -> str:
        """处理 Rank 函数"""
        def replace_rank(match):
            var = match.group(1)
            n = int(match.group(2))
            return f"calc_data['${var}'].rolling({n}).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == {n} and not pd.Series(x).isna().all() else np.nan, raw=False)"

        max_iterations = 50
        iteration = 0
        while re.search(r"Rank\(\$(\w+),\s*(\d+)\)", expr) and iteration < max_iterations:
            expr = re.sub(r"Rank\(\$(\w+),\s*(\d+)\)", replace_rank, expr)
            iteration += 1
        return expr

    def _process_regression_functions(self, expr: str) -> str:
        """处理 Slope, Rsquare, Resi 函数"""
        def replace_slope(match):
            var = match.group(1)
            n = int(match.group(2))
            return f"calc_data['${var}'].rolling({n}).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == {n} else np.nan, raw=True)"

        def replace_rsquare(match):
            var = match.group(1)
            n = int(match.group(2))
            return f"calc_data['${var}'].rolling({n}).apply(lambda x: 1 - np.var(x.values - np.linspace(x.iloc[0] if len(x) > 0 else 0, x.iloc[-1] if len(x) > 0 else 0, len(x))) / (np.var(x.values) + 1e-8) if len(x) == {n} and np.var(x.values) > 0 else 0, raw=False)"

        def replace_resi(match):
            var = match.group(1)
            n = int(match.group(2))
            return f"calc_data['${var}'].rolling({n}).apply(lambda x: x.iloc[-1] - (np.polyfit(range(len(x)), x.values, 1)[0] * (len(x) - 1) + np.polyfit(range(len(x)), x.values, 1)[1]) if len(x) == {n} else np.nan, raw=False)"

        max_iterations = 50
        for pattern, replacer in [
            (r"Slope\(\$(\w+),\s*(\d+)\)", replace_slope),
            (r"Rsquare\(\$(\w+),\s*(\d+)\)", replace_rsquare),
            (r"Resi\(\$(\w+),\s*(\d+)\)", replace_resi),
        ]:
            iteration = 0
            while re.search(pattern, expr) and iteration < max_iterations:
                expr = re.sub(pattern, replacer, expr)
                iteration += 1
        return expr

    def _replace_variables(self, expr: str) -> str:
        """替换变量引用"""
        var_names = ["close", "open", "high", "low", "volume", "vwap"]
        for var_name in var_names:
            replacement = f"calc_data['${var_name}']"
            if f"${var_name}" in expr:
                # 修复嵌套
                if f"calc_data['calc_data['${var_name}']']" in expr:
                    expr = expr.replace(f"calc_data['calc_data['${var_name}']']", replacement)

                nested_shift_pattern = rf"calc_data\['calc_data\['\${var_name}'\]'\]\.shift\(([^)]+)\)"
                if re.search(nested_shift_pattern, expr):
                    expr = re.sub(nested_shift_pattern, rf"{replacement}.shift(\1)", expr)

                # 替换未处理的变量
                pattern = rf"(?<!calc_data\[\')\${var_name}(?!\'\])"
                max_iterations = 10
                for _ in range(max_iterations):
                    new_expr = re.sub(pattern, replacement, expr)
                    if new_expr == expr:
                        break
                    expr = new_expr
                    # 修复可能产生的嵌套
                    if f"calc_data['calc_data['${var_name}']']" in expr:
                        expr = expr.replace(f"calc_data['calc_data['${var_name}']']", replacement)
        return expr

    def _process_mean_std_functions(self, expr: str, calc_data: pd.DataFrame) -> str:
        """处理 Mean 和 Std 函数"""
        def find_matching_paren(s, start_pos):
            count = 0
            i = start_pos
            while i < len(s):
                if s[i] == "(":
                    count += 1
                elif s[i] == ")":
                    count -= 1
                    if count == 0:
                        return i
                i += 1
            return -1

        # 修复嵌套的 calc_data
        var_names = ["close", "open", "high", "low", "volume", "vwap"]
        for var_name in var_names:
            nested_patterns = [
                f"calc_data['calc_data['${var_name}']']",
                f"calc_data['calc_data['${var_name}']'].shift",
            ]
            for nested_pattern in nested_patterns:
                if nested_pattern in expr:
                    expr = expr.replace(nested_pattern, f"calc_data['${var_name}']")

        # 处理 Mean 函数
        max_iterations = 20
        iteration = 0
        while "Mean(" in expr and iteration < max_iterations:
            mean_positions = [m.start() for m in re.finditer(r"Mean\(", expr)]
            if not mean_positions:
                break

            replaced = False
            for pos in reversed(mean_positions):
                end_pos = find_matching_paren(expr, pos + 5)
                if end_pos > 0:
                    comma_pos = expr.rfind(",", pos + 5, end_pos)
                    if comma_pos > 0:
                        inner_expr = expr[pos + 5:comma_pos].strip()
                        n_str = expr[comma_pos + 1:end_pos].strip()
                        try:
                            n = int(n_str)
                            is_simple = re.match(r"^calc_data\[\'\$\w+\'\]$", inner_expr.strip())
                            has_comparison = any(op in inner_expr for op in [">", "<", ">=", "<=", "==", "!="])
                            if not is_simple or has_comparison:
                                replacement = f"pd.Series({inner_expr}, index=calc_data.index).rolling({n}).mean()"
                                expr = expr[:pos] + replacement + expr[end_pos + 1:]
                                replaced = True
                                break
                        except ValueError:
                            pass

            if not replaced:
                break
            iteration += 1

        # 处理 Std 函数
        iteration = 0
        while "Std(" in expr and iteration < max_iterations:
            std_positions = [m.start() for m in re.finditer(r"Std\(", expr)]
            if not std_positions:
                break

            replaced = False
            for pos in reversed(std_positions):
                end_pos = find_matching_paren(expr, pos + 4)
                if end_pos > 0:
                    comma_pos = expr.rfind(",", pos + 4, end_pos)
                    if comma_pos > 0:
                        inner_expr = expr[pos + 4:comma_pos].strip()
                        n_str = expr[comma_pos + 1:end_pos].strip()
                        try:
                            n = int(n_str)
                            is_simple = re.match(r"^calc_data\[\'\$\w+\'\]$", inner_expr.strip())
                            has_comparison = any(op in inner_expr for op in [">", "<", ">=", "<=", "==", "!="])
                            if not is_simple or has_comparison:
                                replacement = f"pd.Series({inner_expr}, index=calc_data.index).rolling({n}).std()"
                                expr = expr[:pos] + replacement + expr[end_pos + 1:]
                                replaced = True
                                break
                        except ValueError:
                            pass

            if not replaced:
                break
            iteration += 1

        return expr

    def _get_runtime_functions(self, calc_data: pd.DataFrame) -> dict:
        """获取运行时辅助函数"""
        def _as_series(x, ref):
            if isinstance(x, pd.Series):
                return x
            if np.isscalar(x):
                return pd.Series([x] * len(ref), index=ref.index)
            if isinstance(x, np.ndarray):
                return pd.Series(x, index=ref.index)
            return pd.Series(x, index=ref.index)

        def Greater(a, b=0):
            if isinstance(a, pd.Series) or isinstance(b, pd.Series):
                ref = a if isinstance(a, pd.Series) else b
                a_s = _as_series(a, ref)
                b_s = _as_series(b, ref)
                return (a_s > b_s).astype(float)
            return float(a > b) if np.isscalar(a) and np.isscalar(b) else np.maximum(a, b)

        def Less(a, b=0):
            if isinstance(a, pd.Series) or isinstance(b, pd.Series):
                ref = a if isinstance(a, pd.Series) else b
                a_s = _as_series(a, ref)
                b_s = _as_series(b, ref)
                return (a_s < b_s).astype(float)
            return float(a < b) if np.isscalar(a) and np.isscalar(b) else np.minimum(a, b)

        def Sum(x, n):
            if not isinstance(x, pd.Series):
                x_s = _as_series(x, calc_data)
            else:
                x_s = x
            if not x_s.index.equals(calc_data.index):
                x_s = x_s.reindex(calc_data.index)
            if x_s.dtype == bool:
                x_s = x_s.astype(float)
            return x_s.rolling(int(n)).sum()

        def Mean(x, n):
            if not isinstance(x, pd.Series):
                x_s = _as_series(x, calc_data)
            else:
                x_s = x
            if not x_s.index.equals(calc_data.index):
                x_s = x_s.reindex(calc_data.index)
            if x_s.dtype == bool:
                x_s = x_s.astype(float)
            return x_s.rolling(int(n)).mean()

        def Std(x, n):
            if not isinstance(x, pd.Series):
                x_s = _as_series(x, calc_data)
            else:
                x_s = x
            if not x_s.index.equals(calc_data.index):
                x_s = x_s.reindex(calc_data.index)
            if x_s.dtype == bool:
                x_s = x_s.astype(float)
            return x_s.rolling(int(n)).std()

        def Max(a, b):
            if isinstance(a, pd.Series) or isinstance(b, pd.Series):
                ref = a if isinstance(a, pd.Series) else b
                a_s = _as_series(a, ref)
                b_s = _as_series(b, ref)
                return pd.concat([a_s, b_s], axis=1).max(axis=1)
            return max(a, b) if np.isscalar(a) and np.isscalar(b) else np.maximum(a, b)

        def Min(a, b):
            if isinstance(a, pd.Series) or isinstance(b, pd.Series):
                ref = a if isinstance(a, pd.Series) else b
                a_s = _as_series(a, ref)
                b_s = _as_series(b, ref)
                return pd.concat([a_s, b_s], axis=1).min(axis=1)
            return min(a, b) if np.isscalar(a) and np.isscalar(b) else np.minimum(a, b)

        return {
            "Greater": Greater,
            "Less": Less,
            "Sum": Sum,
            "Mean": Mean,
            "Std": Std,
            "Max": Max,
            "Min": Min,
        }
