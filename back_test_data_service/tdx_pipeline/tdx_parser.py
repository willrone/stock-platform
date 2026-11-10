"""通达信官网盘后包解析器 — 沪深京全市场日线（不复权价）。

数据源: https://www.tdx.com.cn/products/data/data/g4day/{YYYYMMDD}.zip
- 每个交易日一个 zip，约 2.6MB，含沪深京全部证券
- 免费、无 token、无注册
- 历史回溯: 2021-08-16 起（2021-01-04 已 404）
- 北交所: 2022-05-06 起有文件

包内结构（每市场一对文件）:
- {market}{yyMMdd}.cod  代码表，150 字节/条
- {market}{yyMMdd}.md1  行情块，512 字节/块

.cod 记录布局（150 字节）:
- [0:6]    代码（6 位 ASCII 数字）
- [32:34]  行情块序号（uint16 LE）
- [40:72]  证券名称（GBK）

.md1 行情块布局（512 字节）:
- [4:12]   昨收（double LE）
- [12:20]  开盘（double LE）
- [20:28]  最高（double LE）
- [28:36]  最低（double LE）
- [36:44]  收盘（double LE）
- [56:64]  成交量（uint64 LE，单位：股）
- [72:80]  成交额（double LE，单位：元）

停牌识别: volume == 0 且 amount == 0

布局参考 jing2uo/tdx2db（MIT）的 tdx/merge.go；本模块已用
600519/000001/600000/920992 与腾讯 qfq、Baostock 对拍验证。
"""

from __future__ import annotations

import io
import math
import struct
import zipfile
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import urllib.error
import urllib.request

# 盘后包 URL 模板
TDX_PACKAGE_URL = "https://www.tdx.com.cn/products/data/data/g4day/{ymd}.zip"

# 北交所文件从这天起有（2026-09-20 二分实测）
TDX_BJ_FIRST_DAY = "20220506"

# 各市场有收盘价的最少行数（实测下限，防残缺包）
TDX_MIN_PRICED = {"sh": 10000, "sz": 3000, "bj": 50}


@dataclass
class TdxBar:
    """单只证券单日不复权行情。"""
    date: str          # YYYY-MM-DD
    market: str        # sh / sz / bj
    code: str          # 6 位数字
    name: str          # GBK 证券名称
    prev_close: float  # 昨收
    open: float
    high: float
    low: float
    close: float
    volume: int        # 成交量，单位：股
    amount: float      # 成交额，单位：元

    @property
    def is_suspended(self) -> bool:
        """停牌判定：volume == 0 且 amount == 0。

        实测已知停牌样本：
        - 2026-06-08 sh.600228 *ST天成: close=6.33, vol=0, amt=0
        - 2026-06-17 sh.600228: close=8.92, vol=0, amt=0
        - 2026-05-29 sz.000691 亚太实业: close=0.0, vol=0, amt=0
        """
        return self.volume == 0 and self.amount == 0.0

    @property
    def has_price(self) -> bool:
        """是否有有效收盘价（close > 0 且为有限数）。"""
        return self.close > 0 and math.isfinite(self.close)

    @property
    def ts_code(self) -> str:
        """Tushare 风格代码：600000.SH / 000001.SZ / 920992.BJ"""
        suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}[self.market]
        return f"{self.code}.{suffix}"


def _fetch_package(date_str: str, timeout: int = 90) -> bytes:
    """下载某交易日盘后包。返回 zip bytes。

    Args:
        date_str: YYYY-MM-DD 或 YYYYMMDD

    Raises:
        FileNotFoundError: 404，非交易日/包未发布/早于保留范围
        ConnectionError: 网络错误
        ValueError: 返回内容不是 zip（可能是错误页）
    """
    ymd = date_str.replace("-", "")
    url = TDX_PACKAGE_URL.format(ymd=ymd)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            content = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(
                f"{date_str} 没有通达信盘后包：非交易日、当日包尚未发布"
                f"（通常收盘后数小时），或早于官网保留范围（2021-01-04 已没有）"
            )
        raise ConnectionError(f"下载盘后包失败 HTTP {e.code}: {url}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(f"下载盘后包网络错误: {url}: {e}") from e

    if not content.startswith(b"PK"):
        raise ValueError(f"盘后包不是 zip 文件（可能错误页）: {url}")
    return content


def parse_package(content: bytes, date_str: str) -> List[TdxBar]:
    """解析盘后包，返回全部有价记录。

    Args:
        content: zip bytes
        date_str: YYYY-MM-DD（用于填充 bar.date）

    Returns:
        TdxBar 列表，含沪深京三市场

    Raises:
        RuntimeError: 文件残缺/格式变化（不把部分市场当全市场返回）
        zipfile.BadZipFile: zip 损坏
    """
    ymd = date_str.replace("-", "")
    date_fmt = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    zf = zipfile.ZipFile(io.BytesIO(content))
    names = set(zf.namelist())
    bars: List[TdxBar] = []

    for market in ("sh", "sz", "bj"):
        cod_name = f"{market}{ymd[2:]}.cod"
        md1_name = f"{market}{ymd[2:]}.md1"

        # 北交所 2022-05-06 前无文件，跳过
        if market == "bj" and ymd < TDX_BJ_FIRST_DAY:
            if cod_name not in names and md1_name not in names:
                continue
        if cod_name not in names or md1_name not in names:
            raise RuntimeError(f"盘后包缺少 {cod_name}/{md1_name}，格式可能已变")

        cod = zf.read(cod_name)
        md1 = zf.read(md1_name)

        if len(cod) % 150 or len(md1) % 512:
            raise RuntimeError(f"{market} 代码表或行情块长度不是整块，文件可能被截断")
        if len(cod) // 150 != len(md1) // 512:
            raise RuntimeError(
                f"{market} 代码表 {len(cod)//150} 条、行情块 {len(md1)//512} 块，对不上"
            )

        before = len(bars)
        seen_codes: set = set()
        seen_seqs: set = set()

        for offset in range(0, len(cod), 150):
            record = cod[offset:offset + 150]
            code = record[0:6].rstrip(b"\x00 ").decode("ascii", "replace")
            seq = struct.unpack("<H", record[32:34])[0]

            if not code.isdigit() or len(code) != 6:
                raise RuntimeError(f"{market} 代码表出现非 6 位数字代码 {code!r}，格式可能已变")
            if code in seen_codes or seq in seen_seqs:
                raise RuntimeError(
                    f"{market} 代码表有重复代码/行情块序号（{code!r}, seq={seq}）"
                )
            seen_codes.add(code)
            seen_seqs.add(seq)

            block = md1[seq * 512:(seq + 1) * 512]
            if len(block) != 512:
                raise RuntimeError(f"{market}{code} 行情块越界（seq={seq}）")

            prev_close = struct.unpack("<d", block[4:12])[0]
            open_, high, low, close = struct.unpack("<4d", block[12:44])
            amount = struct.unpack("<d", block[72:80])[0]
            volume = struct.unpack("<Q", block[56:64])[0]

            # NaN 会绕过 close <= 0 被计入下限；实测真实包无非有限值
            if not all(math.isfinite(v) for v in (prev_close, open_, high, low, close, amount)):
                raise RuntimeError(
                    f"{market}{code} 行情块出现非有限数值，文件可能已损坏"
                )

            # close <= 0 = 通达信自编板块指数当日无价，整块为 0，不是证券
            if close <= 0:
                continue

            raw_name = record[40:72].split(b"\x00")[0]
            try:
                name = raw_name.decode("gbk").strip()
            except UnicodeDecodeError as e:
                raise RuntimeError(f"{market}{code} 名称不是 GBK，文件可能已损坏") from e
            if not name:
                raise RuntimeError(f"{market}{code} 有价格却无名称，文件可能已损坏")

            bars.append(TdxBar(
                date=date_fmt,
                market=market,
                code=code,
                name=name,
                prev_close=round(prev_close, 4),
                open=round(open_, 4),
                high=round(high, 4),
                low=round(low, 4),
                close=round(close, 4),
                volume=volume,
                amount=round(amount, 2),
            ))

        # 逐市场核对下限：只查总行数会让沪深撑过门槛、北交所静默缺失
        if len(bars) - before < TDX_MIN_PRICED[market]:
            raise RuntimeError(
                f"{market} 市场只有 {len(bars)-before} 条有价记录"
                f"（实测下限 {TDX_MIN_PRICED[market]}），文件可能残缺或格式已变"
            )

    return bars


def fetch_daily(date_str: str, timeout: int = 90) -> List[TdxBar]:
    """下载并解析某交易日全市场日线。一行入口。"""
    content = _fetch_package(date_str, timeout=timeout)
    return parse_package(content, date_str)


def is_trading_day(date_str: str) -> bool:
    """判断是否交易日（有盘后包 = 交易日）。

    非交易日/包未发布返回 False，不抛异常。
    """
    try:
        _fetch_package(date_str, timeout=30)
        return True
    except FileNotFoundError:
        return False
    except (ConnectionError, ValueError):
        # 网络错误时无法判断，保守返回 False
        return False
