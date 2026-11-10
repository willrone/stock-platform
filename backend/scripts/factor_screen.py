#!/usr/bin/env python3
"""因子筛选管线 — IC 排序 + 自动选 topN + 训练回测。

用法:
  python factor_screen.py                    # 默认: top 30/50/80/100, 1天周期, 800只
  python factor_screen.py --topn 50 100      # 只测 top 50 和 100
  python factor_screen.py --horizon 5        # 用 5 天周期
  python factor_screen.py --list             # 只列 IC 排序, 不回测
  python factor_screen.py --stocks 5541      # 用全部股票 (慢)

可反复用: 加新因子/换周期/定期更新, 都跑这个脚本。
"""
import argparse, os, importlib.util, json, sys
# 不许写死家目录：脚本在 backend/scripts/ 下，往上三层就是仓库根
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("PYTHONPATH", "backend")
ds_s = importlib.util.spec_from_file_location("ds", "backend/scripts/daily_signal.py")
ds = importlib.util.module_from_spec(ds_s); ds_s.loader.exec_module(ds)
ml_s = importlib.util.spec_from_file_location("ml", "backend/scripts/multi_line.py")
ml = importlib.util.module_from_spec(ml_s); ml_s.loader.exec_module(ml)
import pandas as pd, numpy as np, lightgbm as lgb

ap = argparse.ArgumentParser(description="因子筛选管线")
ap.add_argument("--topn", type=int, nargs="+", default=[30, 50, 80, 100], help="测哪些 topN")
ap.add_argument("--horizon", type=int, default=1, help="标签周期 (天)")
ap.add_argument("--stocks", type=int, default=800, help="用多少只股票")
ap.add_argument("--start", default="2023-10-16", help="回测起始日")
ap.add_argument("--capital", type=float, default=1_000_000.0, help="初始资金")
ap.add_argument("--list", action="store_true", help="只列 IC 排序, 不回测")
ap.add_argument("--output", default=None, help="结果输出 JSON 路径")
args = ap.parse_args()

codes = ds.list_universe()[:args.stocks]
frames, panel, master = ds.load_data(codes)
names = ml.load_name_map()
close = ds.to_mat(frames, master, "close")
vol = ds.to_mat(frames, master, "volume")
fwd = (close.shift(-args.horizon)/close - 1.0)
start_idx = int(master.searchsorted(pd.Timestamp(args.start)))

# 收集所有因子
all_factors = {}
qlib_names = ["RET1","RET5","RET10","RET20","KMID","KLEN","KMID2","KUP","KUP2","KLOW","KLOW2","KSFT","KSFT2","OPEN0","HIGH0","LOW0","VWAP0",
    "ROC5","ROC10","ROC20","ROC30","ROC60","MA5","MA10","MA20","MA30","MA60","STD5","STD10","STD20","STD30","STD60",
    "BETA5","BETA10","BETA20","BETA30","BETA60","RSQR5","RSQR10","RSQR20","RSQR30","RSQR60","RESI5","RESI10","RESI20","RESI30","RESI60",
    "MAX5","MAX10","MAX20","MAX30","MAX60","MIN5","MIN10","MIN20","MIN30","MIN60","QTLU5","QTLU10","QTLU20","QTLU30","QTLU60",
    "QTLD5","QTLD10","QTLD20","QTLD30","QTLD60","RANK5","RANK10","RANK20","RANK30","RANK60","RSV5","RSV10","RSV20","RSV30","RSV60",
    "IMAX5","IMAX10","IMAX20","IMAX30","IMAX60","IMIN5","IMIN10","IMIN20","IMIN30","IMIN60","IMXD5","IMXD10","IMXD20","IMXD30","IMXD60",
    "CORR5","CORR10","CORR20","CORR30","CORR60","CORD5","CORD10","CORD20","CORD30","CORD60","CNTP5","CNTP10","CNTP20","CNTP30","CNTP60",
    "CNTN5","CNTN10","CNTN20","CNTN30","CNTN60","CNTD5","CNTD10","CNTD20","CNTD30","CNTD60","SUMP5","SUMP10","SUMP20","SUMP30","SUMP60",
    "SUMN5","SUMN10","SUMN20","SUMN30","SUMN60","SUMD5","SUMD10","SUMD20","SUMD30","SUMD60","VMA5","VMA10","VMA20","VMA30","VMA60",
    "VSTD5","VSTD10","VSTD20","VSTD30","VSTD60","WVMA5","WVMA10","WVMA20","WVMA30","WVMA60","VSUMP5","VSUMP10","VSUMP20","VSUMP30","VSUMP60",
    "VSUMN5","VSUMN10","VSUMN20","VSUMN30","VSUMN60","VSUMD5","VSUMD10","VSUMD20","VSUMD30","VSUMD60",
    "ATR14","BOLLINGER_LOWER","BOLLINGER_MIDDLE","BOLLINGER_UPPER","KDJ_D","KDJ_K","MA50","MACD","MACD_HIST","MACD_SIGNAL","RSI14",
    "price_change","price_change_5d","price_change_20d","volume_change","volume_ma_ratio","volatility_5d","volatility_20d","price_position"]
for name in qlib_names:
    try:
        m = ds.to_panel_mat(panel, master, name)
        if np.isfinite(m).sum() > 1000: all_factors[name] = m
    except Exception: pass
r = close.pct_change(fill_method=None); vchg = vol.pct_change(fill_method=None)
with np.errstate(divide="ignore", invalid="ignore"):
    for n in (5,10,20,30,60):
        all_factors[f"CUSTOM_MIN{n}"] = (close.rolling(n).min()/close-1.0).to_numpy()
        all_factors[f"CUSTOM_MAX{n}"] = (close.rolling(n).max()/close-1.0).to_numpy()
        all_factors[f"CUSTOM_STD{n}"] = (close.rolling(n).std()/close).to_numpy()
        all_factors[f"CUSTOM_CORR{n}"] = close.rolling(n).corr(vol).to_numpy()
        all_factors[f"CUSTOM_CORD{n}"] = r.rolling(n).corr(vchg).to_numpy()

print(f"总因子数: {len(all_factors)}, 周期: {args.horizon}天, 股票: {len(codes)}只")

# IC 排序
fwd_mat = fwd.to_numpy()
ic_scores = {}
for name, mat in all_factors.items():
    m = mat if isinstance(mat, np.ndarray) else mat.to_numpy()
    ics = []
    for t in range(start_idx, len(master)-args.horizon):
        a = m[t]; b = fwd_mat[t]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 50: continue
        ics.append(np.corrcoef(np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok])))[0,1])
    if ics: ic_scores[name] = abs(np.mean(ics))
ic_ranked = sorted(ic_scores.items(), key=lambda x:-x[1])
print(f"\nTop 20 因子 (按 |IC| 排序, 周期{args.horizon}天):")
for i,(n,v) in enumerate(ic_ranked[:20]): print(f"  {i+1:2d}. {n:<25s} |IC|={v:.4f}")

if args.list:
    print("\n[done - list only]")
    sys.exit(0)

# 训练+回测
close_mat = close.ffill(axis=0); vol_mat = vol; day_ret_mat = close.pct_change(fill_method=None)
mkt_ret = day_ret_mat.mean(axis=1); realized_vol = mkt_ret.rolling(20).std()*np.sqrt(244)
pos_scale = (0.20/realized_vol).clip(upper=1.0).fillna(1.0).loc[master]
def zscore(mat):
    ok=np.isfinite(mat); cnt=ok.sum(axis=1,keepdims=True)
    mu=np.nanmean(np.where(ok,mat,np.nan),axis=1,keepdims=True); sd=np.nanstd(np.where(ok,mat,np.nan),axis=1,keepdims=True)
    z=(mat-mu)/np.where(sd>1e-12,sd,np.nan); return np.where(cnt>=ml.MIN_CROSS,z,np.nan), ok
n_d,n_s=len(master),len(frames); lo=max(0,start_idx-750)
results = {}
print(f"\n{'topN':<8}{'收益%':>9}{'Sharpe':>9}{'回撤%':>9}{'交易':>7}")
print("-"*45)
for topn in args.topn:
    sel = [n for n,_ in ic_ranked[:topn]]
    fl,pl=[],[]
    for f in sel:
        m = all_factors[f] if isinstance(all_factors[f],np.ndarray) else all_factors[f].to_numpy()
        z,ok = zscore(m); fl.append(z); pl.append(ok.astype(float))
    F=np.stack(fl,axis=2); P=np.stack(pl,axis=2); nf=F.shape[2]
    y = fwd[lo:start_idx].to_numpy().reshape(-1)
    Xtr = np.concatenate([F[lo:start_idx].reshape(-1,nf),P[lo:start_idx].reshape(-1,nf)],axis=1)
    mask = np.isfinite(y) & np.isfinite(Xtr[:,:nf]).all(axis=1)
    mdl = lgb.train({"objective":"regression","num_leaves":31,"max_depth":3,"learning_rate":0.05,"verbose":-1,"num_threads":8},
                    lgb.Dataset(Xtr[mask],label=y[mask]), num_boost_round=100)
    scores = np.full((n_d,n_s),np.nan)
    for t in range(start_idx,n_d):
        Xt = np.concatenate([F[t].reshape(-1,nf),P[t].reshape(-1,nf)],axis=1)
        ok_t = np.isfinite(Xt[:,:nf]).all(axis=1); scores[t]=np.where(ok_t,mdl.predict(Xt),np.nan)
    sdf = pd.DataFrame(scores,index=master,columns=list(frames))
    rr = ml.run_line("gbm", sdf, 60, 5, "daily", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names, topk_scale=pos_scale)
    results[f"top{topn}"] = {"return_pct": rr["return_pct"], "sharpe": rr["sharpe"], "max_drawdown_pct": rr["max_drawdown_pct"], "n_trades": rr["n_trades"], "factors": sel}
    print(f"top{topn:<4}{rr['return_pct']:>9.2f}{rr['sharpe']:>9.3f}{rr['max_drawdown_pct']:>9.2f}{rr['n_trades']:>7}")

if args.output:
    with open(args.output, "w") as f: json.dump({"ic_ranked": ic_ranked, "results": results}, f, ensure_ascii=False, indent=1)
    print(f"\n[结果已写入 {args.output}]")
print("\n[done]")
