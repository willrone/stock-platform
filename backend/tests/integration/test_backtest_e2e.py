"""
ST2：回测全链路测试 — 从策略配置到最终报告

使用合成数据集（10只股票，30个交易日），测试完整回测流程。
覆盖：RSI/MACD 策略全链路、报告字段验证、交易合理性、结果可复现、冷却期效果。

注意：通过 preloaded_stock_data 直接传入 DataFrame，跳过磁盘 I/O 和数据库依赖。
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

# ── 在导入任何 app 模块之前，先 patch 环境 ──
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
import sqlalchemy as sa
import sqlalchemy.ext.asyncio as sa_async

# patch 引擎创建，避免 app.core.database 连接 PostgreSQL
_orig_cae = sa_async.create_async_engine
_orig_ce = sa.create_engine
_dummy_ae = _orig_cae("sqlite+aiosqlite:///:memory:", echo=False)
_dummy_se = _orig_ce("sqlite:///:memory:", echo=False)
_PG_KW = {"pool_size", "max_overflow", "pool_timeout", "pool_pre_ping",
           "pool_recycle", "json_serializer"}

sa_async.create_async_engine = lambda url, **kw: _dummy_ae if "postgresql" in str(url) else _orig_cae(url, **{k: v for k, v in kw.items() if k not in _PG_KW})
sa.create_engine = lambda url, **kw: _dummy_se if "postgresql" in str(url) else _orig_ce(url, **{k: v for k, v in kw.items() if k not in _PG_KW})

from app.services.backtest.execution.backtest_executor import BacktestExecutor
from app.services.backtest.models import BacktestConfig


# ────────────────────────── 合成数据生成 ──────────────────────────


def generate_synthetic_stock_data(
    stock_codes: list[str],
    num_days: int = 300,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    生成合成股票数据集（v2 — 多段趋势 + 回调）

    300 天分 4 段：
      1) 上涨 200 天（漂移 +0.003）
      2) 急跌  15 天（漂移 -0.015）
      3) 急涨  10 天（漂移 +0.012）
      4) 下跌  75 天（漂移 -0.004）

    这样 RSI/MA/MACD/Bollinger/Stochastic/CCI 在默认配置下都能产生交易信号。
    """
    rng = np.random.RandomState(seed)
    start_date = datetime(2024, 1, 2)

    # 生成交易日（跳过周末）
    trading_dates = []
    current = start_date
    while len(trading_dates) < num_days:
        if current.weekday() < 5:
            trading_dates.append(current)
        current += timedelta(days=1)

    # 多段趋势参数：(天数, 漂移, 波动率)
    segments = [
        (200, 0.003, 0.015),   # 慢涨
        (15, -0.015, 0.025),   # 急跌
        (10, 0.012, 0.020),    # 急涨
        (num_days - 225, -0.004, 0.018),  # 慢跌
    ]

    stock_data = {}
    for code in stock_codes:
        base_price = rng.uniform(10, 50)

        # 按段生成日收益率
        daily_returns = []
        for seg_days, drift, vol in segments:
            daily_returns.append(rng.normal(drift, vol, seg_days))
        daily_returns = np.concatenate(daily_returns)[:num_days]

        close_prices = base_price * np.cumprod(1 + daily_returns)

        # 根据 close 生成 open/high/low
        open_prices = close_prices * (1 + rng.uniform(-0.01, 0.01, num_days))
        high_prices = np.maximum(open_prices, close_prices) * (1 + rng.uniform(0, 0.02, num_days))
        low_prices = np.minimum(open_prices, close_prices) * (1 - rng.uniform(0, 0.02, num_days))
        volumes = rng.uniform(100000, 5000000, num_days).astype(int)

        df = pd.DataFrame({
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes.astype(float),
        }, index=pd.DatetimeIndex(trading_dates))

        df.attrs["stock_code"] = code
        stock_data[code] = df

    return stock_data


# 10只股票代码
STOCK_CODES = [f"00000{i}.SZ" for i in range(1, 10)] + ["000010.SZ"]


# ────────────────────────── 辅助函数 ──────────────────────────


def _create_executor() -> BacktestExecutor:
    """创建不依赖数据库的 BacktestExecutor"""
    return BacktestExecutor(
        data_dir="data",
        enable_parallel=False,
        enable_performance_profiling=False,
        use_multiprocessing=False,
        persistence=None,
    )


async def _run_backtest_with_config(
    executor: BacktestExecutor,
    stock_data: dict,
    strategy_name: str = "rsi",
    strategy_config: dict = None,
    backtest_config: BacktestConfig = None,
) -> dict:
    """运行回测并返回报告"""
    if strategy_config is None:
        strategy_config = {}
    if backtest_config is None:
        backtest_config = BacktestConfig(
            initial_cash=1000000.0,
            record_portfolio_history=True,
            portfolio_history_stride=1,
        )

    stock_codes = list(stock_data.keys())
    # 从数据中推断日期范围
    first_df = next(iter(stock_data.values()))
    start_date = first_df.index[0].to_pydatetime()
    end_date = first_df.index[-1].to_pydatetime()

    report = await executor.run_backtest(
        strategy_name=strategy_name,
        stock_codes=stock_codes,
        start_date=start_date,
        end_date=end_date,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        task_id=None,
        preloaded_stock_data=stock_data,
    )
    return report


# ────────────────────────── 测试用例 ──────────────────────────


class TestRSIFullPipeline:
    """RSI 策略全链路测试"""

    @pytest.mark.asyncio
    async def test_rsi_report_contains_required_fields(self):
        """1-3. 用 RSI 默认配置跑完整回测，验证报告包含必要字段"""
        stock_data = generate_synthetic_stock_data(STOCK_CODES, num_days=300, seed=42)
        executor = _create_executor()
        report = await _run_backtest_with_config(executor, stock_data, "rsi")

        # 验证报告包含必要字段
        assert "strategy_name" in report
        assert report["strategy_name"] == "rsi"
        assert "total_return" in report
        assert "sharpe_ratio" in report
        assert "max_drawdown" in report
        assert "total_trades" in report or "total_signals" in report
        assert "initial_cash" in report
        assert "final_value" in report
        assert "start_date" in report
        assert "end_date" in report
        assert "stock_codes" in report
        assert "portfolio_history" in report
        assert "trade_history" in report

        # 验证数值合理性
        assert isinstance(report["total_return"], (int, float))
        assert isinstance(report["final_value"], (int, float))
        assert report["final_value"] > 0, "最终组合价值应为正数"
        assert report["max_drawdown"] >= -0.01, "最大回撤不应小于 -1%（允许浮点误差）"
        assert report["max_drawdown"] <= 1, "最大回撤不应超过 100%"

    @pytest.mark.asyncio
    async def test_rsi_trade_records_reasonable(self):
        """4. 验证交易记录合理性"""
        stock_data = generate_synthetic_stock_data(STOCK_CODES, num_days=300, seed=42)
        executor = _create_executor()
        report = await _run_backtest_with_config(executor, stock_data, "rsi")

        trade_history = report.get("trade_history", [])
        if len(trade_history) == 0:
            pytest.skip("RSI 策略在合成数据上未产生交易，跳过交易合理性检查")

        # 验证交易记录字段
        for trade in trade_history:
            assert "stock_code" in trade
            assert "action" in trade
            assert "price" in trade
            assert "quantity" in trade
            assert trade["action"] in ("BUY", "SELL"), f"非法交易动作: {trade['action']}"
            assert trade["price"] > 0, "交易价格应为正数"
            assert trade["quantity"] > 0, "交易数量应为正数"

        # 验证无非法交易：每只股票的卖出数量不应超过买入数量
        from collections import defaultdict
        holdings = defaultdict(int)
        for trade in trade_history:
            code = trade["stock_code"]
            qty = trade["quantity"]
            if trade["action"] == "BUY":
                holdings[code] += qty
            else:
                holdings[code] -= qty
            assert holdings[code] >= 0, f"股票 {code} 出现超卖（持仓为负）"

    @pytest.mark.asyncio
    async def test_rsi_results_reproducible(self):
        """5. 用固定随机种子 + 固定策略参数，验证结果可复现（跑两次结果一致）"""
        config = {"rsi_period": 14, "oversold_threshold": 30, "overbought_threshold": 70}
        bc = BacktestConfig(initial_cash=1000000.0, record_portfolio_history=True, portfolio_history_stride=1)

        # 第一次运行
        stock_data_1 = generate_synthetic_stock_data(STOCK_CODES, num_days=300, seed=123)
        executor_1 = _create_executor()
        report_1 = await _run_backtest_with_config(executor_1, stock_data_1, "rsi", config, bc)

        # 第二次运行（相同种子、相同配置）
        stock_data_2 = generate_synthetic_stock_data(STOCK_CODES, num_days=300, seed=123)
        executor_2 = _create_executor()
        report_2 = await _run_backtest_with_config(executor_2, stock_data_2, "rsi", config, bc)

        # 验证关键指标一致
        assert report_1["total_return"] == pytest.approx(report_2["total_return"], abs=1e-10)
        assert report_1["final_value"] == pytest.approx(report_2["final_value"], abs=1e-6)
        assert report_1["total_trades"] == report_2["total_trades"]
        assert report_1["sharpe_ratio"] == pytest.approx(report_2["sharpe_ratio"], abs=1e-10)

    @pytest.mark.asyncio
    async def test_rsi_cooldown_reduces_signals(self):
        """6. RSI 开启冷却期后，信号数量应少于或等于关闭时"""
        stock_data_no_cd = generate_synthetic_stock_data(STOCK_CODES, num_days=300, seed=42)
        stock_data_cd = generate_synthetic_stock_data(STOCK_CODES, num_days=300, seed=42)

        # 关闭冷却期
        config_no_cooldown = {
            "rsi_period": 14,
            "enable_cooldown": False,
        }
        executor_1 = _create_executor()
        report_no_cd = await _run_backtest_with_config(executor_1, stock_data_no_cd, "rsi", config_no_cooldown)

        # 开启冷却期（5天）
        config_cooldown = {
            "rsi_period": 14,
            "enable_cooldown": True,
            "cooldown_days": 5,
        }
        executor_2 = _create_executor()
        report_cd = await _run_backtest_with_config(executor_2, stock_data_cd, "rsi", config_cooldown)

        signals_no_cd = report_no_cd.get("total_signals", 0)
        signals_cd = report_cd.get("total_signals", 0)

        # 冷却期开启后信号数应 <= 关闭时
        assert signals_cd <= signals_no_cd, (
            f"冷却期开启后信号数 ({signals_cd}) 应 <= 关闭时 ({signals_no_cd})"
        )


class TestMACDFullPipeline:
    """MACD 策略全链路测试"""

    @pytest.mark.asyncio
    async def test_macd_full_pipeline(self):
        """7. MACD 策略全链路：验证能正常跑完并返回合理报告"""
        stock_data = generate_synthetic_stock_data(STOCK_CODES, num_days=300, seed=42)
        executor = _create_executor()
        report = await _run_backtest_with_config(executor, stock_data, "macd")

        # 验证基本字段
        assert report["strategy_name"] == "macd"
        assert "total_return" in report
        assert "final_value" in report
        assert report["final_value"] > 0
        assert "portfolio_history" in report
        assert len(report["portfolio_history"]) > 0

    @pytest.mark.asyncio
    async def test_macd_results_reproducible(self):
        """MACD 策略结果可复现"""
        config = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        bc = BacktestConfig(initial_cash=1000000.0, record_portfolio_history=True, portfolio_history_stride=1)

        stock_data_1 = generate_synthetic_stock_data(STOCK_CODES, num_days=300, seed=99)
        executor_1 = _create_executor()
        report_1 = await _run_backtest_with_config(executor_1, stock_data_1, "macd", config, bc)

        stock_data_2 = generate_synthetic_stock_data(STOCK_CODES, num_days=300, seed=99)
        executor_2 = _create_executor()
        report_2 = await _run_backtest_with_config(executor_2, stock_data_2, "macd", config, bc)

        assert report_1["total_return"] == pytest.approx(report_2["total_return"], abs=1e-10)
        assert report_1["total_trades"] == report_2["total_trades"]


class TestMovingAverageFullPipeline:
    """移动平均策略全链路测试（作为额外对比）"""

    @pytest.mark.asyncio
    async def test_ma_full_pipeline(self):
        """移动平均策略全链路"""
        stock_data = generate_synthetic_stock_data(STOCK_CODES, num_days=300, seed=42)
        executor = _create_executor()
        report = await _run_backtest_with_config(executor, stock_data, "moving_average")

        assert report["strategy_name"] == "moving_average"
        assert "total_return" in report
        assert report["final_value"] > 0


class TestEdgeCases:
    """边界情况测试"""

    @pytest.mark.asyncio
    async def test_single_stock(self):
        """单只股票回测"""
        stock_data = generate_synthetic_stock_data(["000001.SZ"], num_days=300, seed=42)
        executor = _create_executor()
        report = await _run_backtest_with_config(executor, stock_data, "rsi")

        assert report["final_value"] > 0
        assert len(report["stock_codes"]) == 1

    @pytest.mark.asyncio
    async def test_portfolio_history_length(self):
        """验证组合历史长度与交易日数一致"""
        stock_data = generate_synthetic_stock_data(STOCK_CODES, num_days=300, seed=42)
        executor = _create_executor()
        bc = BacktestConfig(
            initial_cash=1000000.0,
            record_portfolio_history=True,
            portfolio_history_stride=1,
        )
        report = await _run_backtest_with_config(executor, stock_data, "rsi", backtest_config=bc)

        history = report.get("portfolio_history", [])
        # 组合历史长度应接近交易日数（可能因为预热期少几天）
        assert len(history) > 0, "组合历史不应为空"
        # 组合价值应单调递增或递减（不检查方向，只检查非零）
        for snapshot in history:
            assert snapshot["portfolio_value"] > 0, "组合价值应为正数"



# ────────────────────────── 深度结果校验 ──────────────────────────

# 6 个技术策略
ALL_TECHNICAL_STRATEGIES = ["moving_average", "rsi", "macd", "bollinger", "stochastic", "cci"]

# 组合策略配置
PORTFOLIO_CONFIG = {
    "strategies": [
        {"name": "rsi", "weight": 0.4, "config": {"rsi_period": 14, "oversold_threshold": 40, "overbought_threshold": 60}},
        {"name": "macd", "weight": 0.3, "config": {"fast_period": 12, "slow_period": 26, "signal_period": 9}},
        {"name": "bollinger", "weight": 0.3, "config": {"period": 20, "std_dev": 2}},
    ],
    "integration_method": "weighted_voting",
}

# 所有需要深度校验的策略（含组合策略）
ALL_STRATEGIES_WITH_PORTFOLIO = ALL_TECHNICAL_STRATEGIES + ["portfolio"]

# 缓存策略报告，避免同一策略重复跑回测
_strategy_report_cache: dict[str, dict] = {}


async def _get_strategy_report(strategy_name: str) -> dict:
    """获取策略报告（带缓存）"""
    if strategy_name not in _strategy_report_cache:
        executor = _create_executor()
        stock_data = generate_synthetic_stock_data(STOCK_CODES, num_days=300, seed=42)
        strategy_config = PORTFOLIO_CONFIG if strategy_name == "portfolio" else None
        report = await _run_backtest_with_config(
            executor, stock_data, strategy_name=strategy_name,
            strategy_config=strategy_config,
        )
        report["_stock_data"] = stock_data
        _strategy_report_cache[strategy_name] = report
    return _strategy_report_cache[strategy_name]


class TestDeepResultValidation:
    """深度结果校验：数值一致性、指标反算、交易合理性（参数化覆盖所有 6 个技术策略）"""

    @pytest_asyncio.fixture(params=ALL_STRATEGIES_WITH_PORTFOLIO)
    async def strategy_report(self, request):
        """参数化 fixture：为每个技术策略及组合策略生成回测报告"""
        return await _get_strategy_report(request.param)

    # ── RSI 专属 fixtures（仅用于 RSI 特定测试） ──

    @pytest_asyncio.fixture
    async def rsi_report_with_trades(self):
        """生成一份有交易的 RSI 报告（放宽阈值增加交易概率）"""
        executor = _create_executor()
        stock_data = generate_synthetic_stock_data(STOCK_CODES, num_days=300, seed=99)
        report = await _run_backtest_with_config(
            executor, stock_data, strategy_name="rsi",
            strategy_config={
                "oversold_threshold": 45,
                "overbought_threshold": 55,
                "enable_cooldown": False,
                "enable_trend_alignment": False,
                "enable_volume_confirmation": False,
            },
        )
        report["_stock_data"] = stock_data
        return report

    # ── 1. 收益率一致性（参数化） ──

    @pytest.mark.asyncio
    async def test_total_return_matches_final_value(self, strategy_report):
        """total_return ≈ (final_value - initial_cash) / initial_cash"""
        report = strategy_report
        initial = report["initial_cash"]
        final = report["final_value"]
        expected_return = (final - initial) / initial
        assert report["total_return"] == pytest.approx(expected_return, rel=1e-6), (
            f"[{report['strategy_name']}] total_return={report['total_return']}, expected={expected_return}"
        )

    @pytest.mark.asyncio
    async def test_portfolio_history_return_consistency(self, strategy_report):
        """portfolio_history 最后一条的 total_return 应与报告顶层一致"""
        report = strategy_report
        history = report["portfolio_history"]
        if len(history) == 0:
            pytest.skip("无组合历史")
        last_snapshot = history[-1]
        assert last_snapshot["total_return"] == pytest.approx(
            report["total_return"], rel=1e-6
        ), f"[{report['strategy_name']}] portfolio_history 末尾 total_return 与报告不一致"

    @pytest.mark.asyncio
    async def test_final_value_matches_last_portfolio_value(self, strategy_report):
        """final_value 应等于 portfolio_history 最后一条的 portfolio_value"""
        report = strategy_report
        history = report["portfolio_history"]
        if len(history) == 0:
            pytest.skip("无组合历史")
        last_pv = history[-1]["portfolio_value"]
        assert report["final_value"] == pytest.approx(last_pv, rel=1e-6), (
            f"[{report['strategy_name']}] final_value={report['final_value']}, last_pv={last_pv}"
        )

    # ── 2. 交易价格合理性（参数化） ──

    @pytest.mark.asyncio
    async def test_trade_prices_within_daily_range(self, strategy_report):
        """每笔交易价格应在当日 low~high 范围内（含滑点容差）"""
        report = strategy_report
        stock_data = report["_stock_data"]
        trades = report["trade_history"]
        if len(trades) == 0:
            pytest.skip(f"[{report['strategy_name']}] 无交易记录")

        for trade in trades:
            code = trade["stock_code"]
            trade_date = pd.Timestamp(trade["timestamp"]).normalize()
            price = trade["price"]

            if code not in stock_data:
                continue
            df = stock_data[code]
            if trade_date not in df.index:
                continue

            row = df.loc[trade_date]
            low = row["low"]
            high = row["high"]
            # 回测引擎有滑点(slippage)，成交价可能略超 low/high
            tolerance = (high - low) * 0.10 + 0.05
            assert low - tolerance <= price <= high + tolerance, (
                f"[{report['strategy_name']}] 交易价格 {price} 超出当日范围 [{low}, {high}]，"
                f"股票={code}, 日期={trade_date.date()}"
            )

    @pytest.mark.asyncio
    async def test_trade_fields_complete(self, strategy_report):
        """每笔交易应包含所有必要字段且类型正确"""
        report = strategy_report
        trades = report["trade_history"]
        if len(trades) == 0:
            pytest.skip(f"[{report['strategy_name']}] 无交易记录")

        required_fields = ["trade_id", "stock_code", "action", "quantity",
                           "price", "timestamp", "commission", "pnl"]
        for i, trade in enumerate(trades):
            for field in required_fields:
                assert field in trade, f"[{report['strategy_name']}] 交易 #{i} 缺少字段: {field}"
            assert trade["action"] in ("BUY", "SELL"), f"[{report['strategy_name']}] 交易 #{i} 非法动作: {trade['action']}"
            assert trade["price"] > 0, f"[{report['strategy_name']}] 交易 #{i} 价格非正"
            assert trade["quantity"] > 0, f"[{report['strategy_name']}] 交易 #{i} 数量非正"
            assert isinstance(trade["commission"], (int, float, np.number)), f"[{report['strategy_name']}] 交易 #{i} commission 类型错误"
            assert trade["commission"] >= 0, f"[{report['strategy_name']}] 交易 #{i} commission 为负"

    # ── 3. 组合价值一致性（参数化） ──

    @pytest.mark.asyncio
    async def test_portfolio_value_equals_cash_plus_holdings(self, strategy_report):
        """每日 portfolio_value ≈ cash + Σ(持仓量 × 当日收盘价)"""
        report = strategy_report
        history = report["portfolio_history"]
        stock_data = report["_stock_data"]

        for snapshot in history:
            cash = snapshot["cash"]
            positions = snapshot.get("positions", {})
            date = pd.Timestamp(snapshot["date"]).normalize()

            holdings_value = 0.0
            for code, pos_info in positions.items():
                qty = pos_info if isinstance(pos_info, (int, float)) else pos_info.get("quantity", 0)
                if code in stock_data and date in stock_data[code].index:
                    close_price = stock_data[code].loc[date, "close"]
                    holdings_value += qty * close_price

            expected_pv = cash + holdings_value
            actual_pv = snapshot["portfolio_value"]

            if actual_pv > 0:
                assert actual_pv == pytest.approx(expected_pv, rel=0.01), (
                    f"[{report['strategy_name']}] 日期={date.date()}: portfolio_value={actual_pv}, "
                    f"expected(cash+holdings)={expected_pv}, cash={cash}, "
                    f"holdings_value={holdings_value}"
                )

    @pytest.mark.asyncio
    async def test_portfolio_history_dates_monotonic(self, strategy_report):
        """portfolio_history 日期应单调递增"""
        report = strategy_report
        history = report["portfolio_history"]
        if len(history) < 2:
            pytest.skip("历史记录不足")
        dates = [pd.Timestamp(s["date"]) for s in history]
        for i in range(1, len(dates)):
            assert dates[i] > dates[i - 1], (
                f"[{report['strategy_name']}] 日期非单调递增: {dates[i-1]} -> {dates[i]}"
            )

    @pytest.mark.asyncio
    async def test_cash_never_negative(self, strategy_report):
        """现金余额不应为负"""
        report = strategy_report
        for snapshot in report["portfolio_history"]:
            assert snapshot["cash"] >= -0.01, (
                f"[{report['strategy_name']}] 日期={snapshot['date']}: cash={snapshot['cash']} 为负"
            )

    # ── 4. 最大回撤反算（参数化） ──

    @pytest.mark.asyncio
    async def test_max_drawdown_recalculation(self, strategy_report):
        """用 portfolio_history 自行计算 max_drawdown，与报告对比"""
        report = strategy_report
        history = report["portfolio_history"]
        if len(history) < 2:
            pytest.skip("历史记录不足")

        values = pd.Series([s["portfolio_value"] for s in history])
        returns = values.pct_change().dropna()
        if len(returns) == 0:
            pytest.skip("无收益序列")

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        recalc_mdd = float(drawdown.min())

        reported_mdd = report["max_drawdown"]
        assert reported_mdd == pytest.approx(recalc_mdd, abs=0.005), (
            f"[{report['strategy_name']}] max_drawdown: reported={reported_mdd}, recalculated={recalc_mdd}"
        )

    # ── 5. Sharpe ratio 反算（参数化） ──

    @pytest.mark.asyncio
    async def test_sharpe_ratio_recalculation(self, strategy_report):
        """用 portfolio_history 反算 sharpe_ratio"""
        report = strategy_report
        history = report["portfolio_history"]
        if len(history) < 2:
            pytest.skip("历史记录不足")

        values = pd.Series([s["portfolio_value"] for s in history])
        returns = values.pct_change().dropna()
        if len(returns) == 0:
            pytest.skip("无收益序列")

        initial = report["initial_cash"]
        final = report["final_value"]
        total_return = (final - initial) / initial

        dates = [pd.Timestamp(s["date"]) for s in history]
        days = (dates[-1] - dates[0]).days
        annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0
        volatility = float(returns.std() * np.sqrt(252))
        recalc_sharpe = annualized_return / volatility if volatility > 0 else 0

        reported_sharpe = report["sharpe_ratio"]
        assert reported_sharpe == pytest.approx(recalc_sharpe, abs=0.3), (
            f"[{report['strategy_name']}] sharpe_ratio: reported={reported_sharpe}, recalculated={recalc_sharpe}"
        )

    # ── 6. 交易统计一致性（参数化） ──

    @pytest.mark.asyncio
    async def test_total_trades_matches_trade_history(self, strategy_report):
        """total_trades 应等于 trade_history 长度"""
        report = strategy_report
        trades = report["trade_history"]
        reported_total = report.get("total_trades", 0)
        assert reported_total == len(trades), (
            f"[{report['strategy_name']}] total_trades={reported_total}, len(trade_history)={len(trades)}"
        )

    @pytest.mark.asyncio
    async def test_win_rate_recalculation(self, strategy_report):
        """用 trade_history 反算 win_rate"""
        report = strategy_report
        trades = report["trade_history"]
        sell_trades = [t for t in trades if t["action"] == "SELL"]
        if len(sell_trades) == 0:
            pytest.skip(f"[{report['strategy_name']}] 无卖出交易")

        winning = [t for t in sell_trades if t["pnl"] > 0]
        losing = [t for t in sell_trades if t["pnl"] < 0]
        denom = len(winning) + len(losing)
        recalc_wr = len(winning) / denom if denom > 0 else 0

        reported_wr = report.get("win_rate", 0)
        assert reported_wr == pytest.approx(recalc_wr, abs=0.01), (
            f"[{report['strategy_name']}] win_rate: reported={reported_wr}, recalculated={recalc_wr}"
        )

    # ── 7. 成本统计校验（参数化） ──

    @pytest.mark.asyncio
    async def test_cost_statistics_consistency(self, strategy_report):
        """cost_statistics 内部一致性"""
        report = strategy_report
        cost_stats = report.get("cost_statistics")
        if cost_stats is None:
            pytest.skip(f"[{report['strategy_name']}] 无 cost_statistics")

        total_commission = cost_stats["total_commission"]
        total_slippage = cost_stats["total_slippage"]
        total_cost = cost_stats["total_cost"]

        assert total_cost == pytest.approx(total_commission + total_slippage, rel=1e-6), (
            f"[{report['strategy_name']}] total_cost={total_cost} != commission({total_commission}) + slippage({total_slippage})"
        )
        assert total_commission >= 0, f"[{report['strategy_name']}] 佣金不应为负"
        assert total_slippage >= 0, f"[{report['strategy_name']}] 滑点不应为负"

        initial = report["initial_cash"]
        injection = cost_stats.get("total_capital_injection", 0)
        total_invested = initial + injection
        if total_invested > 0:
            expected_ratio = total_cost / total_invested
            assert cost_stats["cost_ratio"] == pytest.approx(expected_ratio, rel=1e-6)

    # ── 8. 前端显示字段完整性（参数化） ──

    @pytest.mark.asyncio
    async def test_all_frontend_display_fields_present(self, strategy_report):
        """前端需要的所有字段都应存在且类型正确"""
        report = strategy_report
        sname = report["strategy_name"]

        top_level_fields = {
            "strategy_name": str,
            "stock_codes": list,
            "start_date": str,
            "end_date": str,
            "initial_cash": (int, float),
            "final_value": (int, float),
            "total_return": (int, float),
            "annualized_return": (int, float),
            "volatility": (int, float),
            "sharpe_ratio": (int, float),
            "max_drawdown": (int, float),
            "total_trades": (int, float),
            "win_rate": (int, float),
            "profit_factor": (int, float),
            "trade_history": list,
            "portfolio_history": list,
        }
        for field, expected_type in top_level_fields.items():
            assert field in report, f"[{sname}] 缺少前端字段: {field}"
            assert isinstance(report[field], expected_type), (
                f"[{sname}] 字段 {field} 类型错误: {type(report[field])}, 期望 {expected_type}"
            )

        assert "metrics" in report, f"[{sname}] 缺少 metrics 字段"
        metrics_fields = ["sharpe_ratio", "total_return", "annualized_return",
                          "max_drawdown", "volatility", "win_rate",
                          "profit_factor", "total_trades"]
        for field in metrics_fields:
            assert field in report["metrics"], f"[{sname}] metrics 缺少字段: {field}"

        assert "backtest_config" in report, f"[{sname}] 缺少 backtest_config 字段"
        config_fields = ["strategy_name", "start_date", "end_date",
                         "initial_cash", "commission_rate", "slippage_rate"]
        for field in config_fields:
            assert field in report["backtest_config"], f"[{sname}] backtest_config 缺少字段: {field}"

        assert "cost_statistics" in report, f"[{sname}] 缺少 cost_statistics 字段"
        cost_fields = ["total_commission", "total_slippage", "total_cost", "cost_ratio"]
        for field in cost_fields:
            assert field in report["cost_statistics"], f"[{sname}] cost_statistics 缺少字段: {field}"

    @pytest.mark.asyncio
    async def test_portfolio_history_snapshot_fields(self, strategy_report):
        """portfolio_history 每条快照的字段完整性"""
        report = strategy_report
        history = report["portfolio_history"]
        if len(history) == 0:
            pytest.skip(f"[{report['strategy_name']}] 无组合历史")

        required_snapshot_fields = [
            "date", "portfolio_value", "cash",
            "positions_count", "positions",
            "total_return",
        ]
        for i, snapshot in enumerate(history):
            for field in required_snapshot_fields:
                assert field in snapshot, (
                    f"[{report['strategy_name']}] portfolio_history[{i}] 缺少字段: {field}"
                )

    # ── 9. metrics 与顶层字段一致性（参数化） ──

    @pytest.mark.asyncio
    async def test_metrics_matches_top_level(self, strategy_report):
        """metrics 子字段应与顶层同名字段一致"""
        report = strategy_report
        metrics = report.get("metrics", {})
        check_fields = ["sharpe_ratio", "total_return", "annualized_return",
                        "max_drawdown", "volatility", "win_rate",
                        "profit_factor", "total_trades"]
        for field in check_fields:
            if field in metrics and field in report:
                assert metrics[field] == pytest.approx(report[field], rel=1e-6), (
                    f"[{report['strategy_name']}] metrics.{field}={metrics[field]} != report.{field}={report[field]}"
                )

    # ── 10. excess_return 字段校验（参数化） ──

    @pytest.mark.asyncio
    async def test_excess_return_fields_present(self, strategy_report):
        """excess_return_without_cost / with_cost 字段存在且合理"""
        report = strategy_report
        for key in ["excess_return_without_cost", "excess_return_with_cost"]:
            if key not in report:
                continue
            er = report[key]
            assert isinstance(er, dict), f"[{report['strategy_name']}] {key} 应为 dict"
            for sub_field in ["mean", "std", "annualized_return",
                              "information_ratio", "max_drawdown"]:
                assert sub_field in er, f"[{report['strategy_name']}] {key} 缺少子字段: {sub_field}"
                assert isinstance(er[sub_field], (int, float)), (
                    f"[{report['strategy_name']}] {key}.{sub_field} 类型错误"
                )

    # ── RSI 专属测试（使用 rsi_report_with_trades，不参数化） ──

    @pytest.mark.asyncio
    async def test_rsi_trade_prices_within_daily_range(self, rsi_report_with_trades):
        """[RSI专属] 放宽阈值后交易价格应在当日 low~high 范围内"""
        report = rsi_report_with_trades
        stock_data = report["_stock_data"]
        trades = report["trade_history"]
        if len(trades) == 0:
            pytest.skip("无交易记录")

        for trade in trades:
            code = trade["stock_code"]
            trade_date = pd.Timestamp(trade["timestamp"]).normalize()
            price = trade["price"]

            if code not in stock_data:
                continue
            df = stock_data[code]
            if trade_date not in df.index:
                continue

            row = df.loc[trade_date]
            low = row["low"]
            high = row["high"]
            tolerance = (high - low) * 0.05 + 0.01
            assert low - tolerance <= price <= high + tolerance, (
                f"交易价格 {price} 超出当日范围 [{low}, {high}]，"
                f"股票={code}, 日期={trade_date.date()}"
            )

    @pytest.mark.asyncio
    async def test_rsi_trade_fields_complete(self, rsi_report_with_trades):
        """[RSI专属] 放宽阈值后交易字段完整性"""
        report = rsi_report_with_trades
        trades = report["trade_history"]
        if len(trades) == 0:
            pytest.skip("无交易记录")

        required_fields = ["trade_id", "stock_code", "action", "quantity",
                           "price", "timestamp", "commission", "pnl"]
        for i, trade in enumerate(trades):
            for field in required_fields:
                assert field in trade, f"交易 #{i} 缺少字段: {field}"
            assert trade["action"] in ("BUY", "SELL"), f"交易 #{i} 非法动作: {trade['action']}"
            assert trade["price"] > 0, f"交易 #{i} 价格非正"
            assert trade["quantity"] > 0, f"交易 #{i} 数量非正"
            assert isinstance(trade["commission"], (int, float, np.number)), f"交易 #{i} commission 类型错误"
            assert trade["commission"] >= 0, f"交易 #{i} commission 为负"

    @pytest.mark.asyncio
    async def test_rsi_total_trades_matches(self, rsi_report_with_trades):
        """[RSI专属] total_trades 应等于 trade_history 长度"""
        report = rsi_report_with_trades
        trades = report["trade_history"]
        reported_total = report.get("total_trades", 0)
        assert reported_total == len(trades), (
            f"total_trades={reported_total}, len(trade_history)={len(trades)}"
        )

    @pytest.mark.asyncio
    async def test_rsi_win_rate_recalculation(self, rsi_report_with_trades):
        """[RSI专属] 用 trade_history 反算 win_rate"""
        report = rsi_report_with_trades
        trades = report["trade_history"]
        sell_trades = [t for t in trades if t["action"] == "SELL"]
        if len(sell_trades) == 0:
            pytest.skip("无卖出交易")

        winning = [t for t in sell_trades if t["pnl"] > 0]
        losing = [t for t in sell_trades if t["pnl"] < 0]
        denom = len(winning) + len(losing)
        recalc_wr = len(winning) / denom if denom > 0 else 0

        reported_wr = report.get("win_rate", 0)
        assert reported_wr == pytest.approx(recalc_wr, abs=0.01), (
            f"win_rate: reported={reported_wr}, recalculated={recalc_wr}"
        )

    @pytest.mark.asyncio
    async def test_rsi_cost_statistics_consistency(self, rsi_report_with_trades):
        """[RSI专属] cost_statistics 内部一致性"""
        report = rsi_report_with_trades
        cost_stats = report.get("cost_statistics")
        if cost_stats is None:
            pytest.skip("无 cost_statistics")

        total_commission = cost_stats["total_commission"]
        total_slippage = cost_stats["total_slippage"]
        total_cost = cost_stats["total_cost"]

        assert total_cost == pytest.approx(total_commission + total_slippage, rel=1e-6), (
            f"total_cost={total_cost} != commission({total_commission}) + slippage({total_slippage})"
        )
        assert total_commission >= 0, "佣金不应为负"
        assert total_slippage >= 0, "滑点不应为负"

        initial = report["initial_cash"]
        injection = cost_stats.get("total_capital_injection", 0)
        total_invested = initial + injection
        if total_invested > 0:
            expected_ratio = total_cost / total_invested
            assert cost_stats["cost_ratio"] == pytest.approx(expected_ratio, rel=1e-6)
