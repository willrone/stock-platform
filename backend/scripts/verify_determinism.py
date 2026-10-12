"""Verify determinism fix: parallel path 3 runs identical + matches serial."""
import sys, os, asyncio, hashlib, importlib.util

os.environ["BASELINE_LOG_LEVEL"] = "ERROR"
sys.argv = ["x"]

ROOT = "/Users/rone/projects/stock-platform/backend"
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location("s", "scripts/sharpe_overlay_experiment.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

codes = m.load_universe()[:500]
h = lambda ts: hashlib.md5(str(ts).encode()).hexdigest()[:12]


def mkex(parallel):
    return m.BacktestExecutor(
        data_dir=str(m.settings.DATA_ROOT_PATH),
        enable_parallel=parallel,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )


async def run_once(ex, tag):
    o = m.Overlay()
    m.install(o)
    r = await ex.run_backtest(
        strategy_name="stochastic",
        stock_codes=codes,
        start_date=m.START,
        end_date=m.END,
        strategy_config={},
        backtest_config=m.base_config(),
    )
    pm = o.last_instance
    trades = [(t["timestamp"].date().isoformat(), t["stock_code"], t["action"]) for t in pm.trades]
    final = float(r.get("final_value") or 0)
    print(f"{tag}: ntrades={len(trades)} final={final:.2f} hash={h(trades)}", flush=True)
    return trades, final


async def main():
    results = {}
    ex_p = mkex(True)
    for i in (1, 2, 3):
        t, f = await run_once(ex_p, f"PAR run{i}")
        results[f"par{i}"] = t
    ex_s = mkex(False)
    t, f = await run_once(ex_s, "SER run1")
    results["ser"] = t

    print("PAR1==PAR2:", results["par1"] == results["par2"])
    print("PAR1==PAR3:", results["par1"] == results["par3"])
    print("PAR1==SER :", results["par1"] == results["ser"])


asyncio.run(main())
print("[done]", flush=True)
