# 通达信盘后包数据管线

替代已失效的 Tushare（会员过期）。免费、无 token、无注册。

## 数据源选型（实测结论，2026-10-17/18）

| 源 | 结论 | 实测依据 |
|---|---|---|
| **通达信官网盘后包** | ✅ 主力 | 2.6MB/天，全市场含北交所，8s/天，历史回溯 2021-08-16 |
| **mootdx xdxr** | ✅ 复权事件 | 0.02s/只，与东财分红数据逐条一致 |
| Baostock | ⚠️ 备用 | 唯一给 isST/turn 的免费源，但 7.92s/只 × 5500 = 12 小时，不能并行，后期限流 |
| akshare / efinance（东财） | ❌ 不适合批量 | 短时高频封 IP，连 urllib 直连都 RemoteDisconnected |
| mootdx K线命令 | ❌ 已失效 | 连上服务器返回 0 行（#52，2026-09 起） |
| 新浪 hfq.js 乘法因子 | ❌ 已废弃 | 纯现金分红股上引入 0.2~0.3% 误差 |

### 盘后包 URL

```
https://www.tdx.com.cn/products/data/data/g4day/{YYYYMMDD}.zip
```

- 约 2.6MB/天，含沪深京全部证券（A股/B股/指数/板块指数/回购/基金/可转债）
- 历史回溯 **2021-08-16 起**（2021-01-04 返回 404）
- 北交所 **2022-05-06 起**有文件
- 当日包通常收盘后数小时发布

## 架构

```
back_test_data_service/tdx_pipeline/
├── tdx_parser.py            盘后包下载 + 二进制解析（.cod/.md1）
├── pipeline.py              增量拉取 + 按日期分区 + 断点续传
├── ashare.py                A 股筛选（盘后包含债券等约 5 万条/天）
├── adjust.py                前复权 — 通达信加法模型（mootdx xdxr）
├── factor_store.py          除权除息事件批量存储
├── merge_to_stock_data.py   合并进 data/parquet/stock_data
├── fetch_gap.py             一次性补齐缺口
└── requirements.txt         独立依赖（不污染 backend）

data/tdx_daily/
├── raw/date=YYYY-MM-DD/part.parquet    每日全市场不复权价
├── adjust_events/events.parquet        除权除息事件长表
└── state.json                          断点续传状态
```

### 核心设计：存不复权价 + 事件表，不存前复权价

**已验证**：现有 `data/parquet/stock_data` 就是不复权价
（600000 2026-05-08: stock_data=9.07，通达信不复权=9.07，完全一致）。

存不复权价 + 事件表的好处：增量拉取不会让历史数据漂移。
若存前复权价，每新增一次除权，所有历史前复权价都会整体平移。

### volume 单位换算

- 通达信盘后包：**股**
- 现有 stock_data：**手**（892690.33 = 通达信 89269033 股 / 100）
- `merge_to_stock_data.py` 已做 /100 换算

## 前复权：加法（线性）模型

A 股主流软件（通达信/同花顺/大智慧）与腾讯 qfq 都用**加法模型**，
不是乘法因子。

```
qfq(d) = raw(d) · M(d) − N(d)

M(d) = Π (1 + 送转_e + 配股_e)        对所有 e > d
N(d) = Σ B_e · Π (1 + 送转 + 配股)    对所有 e > d
B_e  = 红利_e − 配股价_e · 配股_e
```

### 为什么不能用乘法因子

实测 000001.SZ（平安银行，纯现金分红）：
- 腾讯隐含 adj 不是常数（0.9752~0.9786），乘法模型无法拟合
- 但 `raw - qfq` 恒等于 0.249，正是 2026-09-24 除权的每股红利
- 乘法因子（新浪 hfq.js）在该股上引入 **0.2~0.3% 误差**

### 验证结果

用平台自己的 stock_data 验证 000001.SZ 六月 14 天：
**全部 0.0000% 相对差**（vs 腾讯 qfq）。

10 只抽样（含 000001/600000/600519/601318/600036/300750/000858/002594/300059/000630）：
9 只 0.0000%，300750 最大 0.0173% —— 该偏差来自**腾讯 qfq 自身**，
经东财分红数据（交易所口径）交叉验证，mootdx xdxr 是对的。

### 必须过滤未来除权

实测 000630 的 xdxr 含 2026-10-26 的未来除权（0.06/股）。
若不排除，所有历史日期的前复权价会被多减 0.06（0.93% 偏差）。
`fetch_events()` 默认 `as_of=today` 过滤。

## 停牌识别

规则：**`volume == 0 且 amount == 0`**

实测已知停牌样本，与 Baostock `tradestatus` 完全一致：

| 日期 | 股票 | 包内表现 |
|---|---|---|
| 2026-06-08 | 600228 *ST天成 | close=6.33（停牌前价），vol=0, amt=0 |
| 2026-06-17 | 600228 | close=8.92，vol=0, amt=0 |
| 2026-05-29 | 000691 亚太实业 | close=0.0，vol=0, amt=0 |

**注意：停牌判定只对 A 股有意义。** 债券/回购/部分基金不每日成交，
`volume==0` 是常态。实测 2026-06-08 全市场 40490 条 volume==0，
其中绝大多数是国债/政策性金融债，A 股真停牌只有 20 只。
必须先 `filter_ashare()` 再算停牌率。

## 用法

### 增量拉取（每日收盘后）

```python
from back_test_data_service.tdx_pipeline import run_incremental

# 自动从上次拉取的次日开始
stats = run_incremental()
```

### 加载 + 筛 A 股

```python
from back_test_data_service.tdx_pipeline import load_range, filter_ashare

a = filter_ashare(load_range(start="2026-05-09", end="2026-10-16"))
```

### 前复权

```python
from back_test_data_service.tdx_pipeline import fetch_events, to_qfq

evs = fetch_events("000001.SZ")
qfq = to_qfq(raw_price=10.98, events=evs, date="2026-06-05")  # → 10.371
```

### 批量事件存储

```python
from back_test_data_service.tdx_pipeline.factor_store import refresh_events, get_events

refresh_events()                      # 全量（增量，只拉新股票）
evs = get_events("000001.SZ")         # 从表读
```

### 合并进 stock_data

```bash
# 先 dry-run
env -u PYTHONPATH backend/.venv/bin/python3 \
    back_test_data_service/tdx_pipeline/merge_to_stock_data.py --dry-run

# 实际写盘（自动备份 .bak）
env -u PYTHONPATH backend/.venv/bin/python3 \
    back_test_data_service/tdx_pipeline/merge_to_stock_data.py
```

## 依赖注意事项

**不要**把 mootdx 加进 `backend/requirements*.txt`。

mootdx 曾把 backend 的 httpx 从 0.28.1 降到 0.25.2，而
`backend/app/services/data/simple_data_service.py` 依赖 httpx，
违反 `backend/requirements-minimal.txt` 的 `httpx>=0.27.0`。

实测 mootdx 与 httpx 0.28.1 **兼容**（xdxr 正常），所以装完 mootdx 后
务必确认 httpx 版本没被降：

```bash
env -u PYTHONPATH uv pip install --python backend/.venv/bin/python3 mootdx
env -u PYTHONPATH uv pip install --python backend/.venv/bin/python3 "httpx==0.28.1"
```

## 已完成的补齐（2026-10-18）

- 拉取 **108 个交易日**（2026-05-11 → 2026-10-16），零失败
- 缺失 7 天全是节假日：端午 06-19、国庆中秋 10-01~07、09-25
- A 股 **5589 只**（原 5513 + 新建 78 只新上市）
- 除权事件 **56477 条 / 5137 只**（452 只无事件，多为新上市）
- stock_data 已合并，末日 2026-10-16
- 回测引擎验证：`honest_composite` 横截面打分正常
  （10 只 × 108 天，有效分数 1036，耗时 0.81s）

## 每日维护

收盘后（15:45 之后，包才发布）跑一次：

```bash
cd ~/Projects/stock-platform
env -u PYTHONPATH backend/.venv/bin/python3 -c "
import sys; sys.path.insert(0,'.')
from back_test_data_service.tdx_pipeline import run_incremental
print(run_incremental())
"
env -u PYTHONPATH backend/.venv/bin/python3 \
    back_test_data_service/tdx_pipeline/merge_to_stock_data.py
```

约 10 秒（1 个包 2.6MB + merge）。
