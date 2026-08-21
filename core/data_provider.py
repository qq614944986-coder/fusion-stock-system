# -*- coding: utf-8 -*-
"""L1 数据层：akshare 接口封装 + 本地缓存 + 多源降级 + 配置加载。

降级规则（规格书 §3.1）：任一接口失败 → 返回 None 并记录到 missing 列表，
对应模块输出标注 [数据缺失]，不阻塞其他模块。严禁编造数据。
缓存规则：当日已拉取的数据存 data/cache/YYYYMMDD/，当日重跑不重复请求。

多源架构（防东财 WAF 长期 IP 封禁，实测封禁可持续 48 小时+）：
- 个股日K：新浪主源 → 东财兜底（新浪字段等价且含当日K线，消除最大请求源）
- 全市场快照：腾讯主源（自带主力净流入/近5日涨幅）→ 东财兜底
- 主力净占比：腾讯快照批量一次拉取 → 同花顺全市场资金流（替代逐股东财请求）
- 行业板块/板块资金流/板块指数历史：东财优先（名称体系配套）→ 同花顺兜底
- 指数日K：东财优先（含成交额）→ 新浪兜底（量能走快照聚合兜底）
- 涨停池/炸板池：东财 push2ex 独立域名（实测不受 push2 封禁波及）
- 运行开始 em_ok() 探测东财存活（每实例一次），封禁期全部东财源直接跳过
"""
from __future__ import annotations

import json
import logging
import pickle
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import yaml

log = logging.getLogger("data_provider")

# akshare 懒加载（离线跑测试/回测时不触发网络）
_ak = None


def _akshare():
    global _ak
    if _ak is None:
        # 国内金融数据源（东财/新浪/腾讯/同花顺/乐咕）无需代理；
        # 系统代理（Clash 等）对这些域名的路由不稳（ProxyError/断连），默认绕过。
        # 如确需走代理：设环境变量 FUSION_USE_PROXY=1。
        import os
        if os.environ.get("FUSION_USE_PROXY") != "1":
            os.environ["NO_PROXY"] = "*"
            os.environ["no_proxy"] = "*"
        import akshare as ak
        _ak = ak
    return _ak


def _to_f(v) -> "float | None":
    """宽松转 float（None/异常 → None）。"""
    try:
        f = float(str(v).replace(",", "").replace("%", "").strip())
        return None if f != f else f  # NaN 检查
    except (TypeError, ValueError):
        return None


def _cn_amount_yuan(v) -> "float | None":
    """同花顺金额字符串 → 元：'2.26亿'/'-318.84万'/'-123'。"""
    s = str(v).strip().replace(",", "")
    try:
        if s.endswith("亿"):
            return float(s[:-1]) * 1e8
        if s.endswith("万"):
            return float(s[:-1]) * 1e4
        return float(s)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 配置加载

def load_config(path: Optional[Path] = None) -> dict:
    """加载 config.yaml（第11章模板，参数即规格）。"""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_watchlist(path: Optional[Path] = None) -> dict:
    """加载自选股配置（持仓 + 关注，规格书 §3.2）。"""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "watchlist.yaml"
    if not Path(path).exists():
        return {"positions": [], "watching": []}
    with open(path, "r", encoding="utf-8") as f:
        wl = yaml.safe_load(f) or {}
    wl.setdefault("positions", [])
    wl.setdefault("watching", [])
    return wl


# ---------------------------------------------------------------- 工具

def market_of(code: str) -> str:
    """按代码前缀推断市场（供 stock_individual_fund_flow）。"""
    code = str(code).zfill(6)
    if code.startswith(("6", "9", "5")):
        return "sh"
    if code.startswith(("4", "8")):
        return "bj"
    return "sz"


def index_symbol_map() -> dict:
    """指数温度+宏观大盘所需指数（4.1：上证/创业板/科创50；宏观：另加深成/沪深300）。"""
    return {
        "上证指数": "sh000001",
        "深证成指": "sz399001",
        "创业板指": "sz399006",
        "沪深300": "sh000300",
        "科创50": "sh000688",
    }


def is_main_board(code: str) -> bool:
    """主板判定（用户约束：只做主板 60/00 开头，剔除创业30/科创68/北交8、4）。"""
    code = str(code).zfill(6)
    return code[:2] in ("60", "00")


# ---------------------------------------------------------------- 数据提供者

class DataProvider:
    def __init__(self, cfg: dict, base_dir: Optional[Path] = None, trade_date: Optional[str] = None):
        self.cfg = cfg
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
        self.trade_date = trade_date or datetime.now().strftime("%Y%m%d")
        self.cache_dir = self.base_dir / cfg.get("run", {}).get("data_dir", "data") / "cache" / self.trade_date
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.missing: list[str] = []          # 数据缺失记录（展示层⑥风险提示用）
        self._em_status: Optional[bool] = None  # 东财域名存活状态（每实例探测一次）

    # ---------------- 缓存

    def _cache_path(self, key: str, fmt: str) -> Path:
        safe = re.sub(r"[^\w\-\u4e00-\u9fff]", "_", key)
        return self.cache_dir / f"{safe}.{fmt}"

    def _cached(self, key: str, fmt: str) -> Optional[Any]:
        p = self._cache_path(key, fmt)
        if not p.exists():
            return None
        try:
            if fmt == "pkl":
                return pd.read_pickle(p)
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:  # 缓存损坏视为未命中
            log.warning("缓存读取失败 %s: %s", key, e)
            return None

    def _save(self, key: str, fmt: str, obj: Any) -> None:
        p = self._cache_path(key, fmt)
        try:
            if fmt == "pkl":
                obj.to_pickle(p)
            else:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, default=str)
        except Exception as e:
            log.warning("缓存写入失败 %s: %s", key, e)

    def _fetch(self, key: str, fetch_fn: Callable[[], Any], fmt: str = "pkl") -> Optional[Any]:
        """缓存优先；接口失败 → None + missing 记录（降级，不阻塞）。

        防风控：真实网络请求之间保持最小间隔（默认 0.5 秒），
        失败后间隔 3 秒重试一次（东财 WAF 对高频连接会临时断连）。
        """
        hit = self._cached(key, fmt)
        if hit is not None:
            return hit
        for attempt in (1, 2):
            self._throttle()
            try:
                obj = fetch_fn()
                if obj is None:
                    raise ValueError("接口返回 None")
                if isinstance(obj, pd.DataFrame) and obj.empty:
                    raise ValueError("接口返回空表")
                if isinstance(obj, dict) and not obj:
                    raise ValueError("接口返回空字典")
                self._save(key, fmt, obj)
                return obj
            except Exception as e:
                if attempt == 1:
                    log.warning("第1次失败 %s: %s，3秒后重试", key, type(e).__name__)
                    time.sleep(3)
                    continue
                msg = f"[数据缺失] {key}: {type(e).__name__}: {e}"
                if msg not in self.missing:
                    self.missing.append(msg)
                log.warning(msg)
                return None
        return None

    # ---------------- 请求节流（防数据源风控）

    _MIN_REQUEST_GAP = 0.5   # 两次真实网络请求的最小间隔（秒）
    _last_request_ts: float = 0.0

    @classmethod
    def _throttle(cls) -> None:
        now = time.monotonic()
        wait = cls._MIN_REQUEST_GAP - (now - cls._last_request_ts)
        if wait > 0:
            time.sleep(wait)
        cls._last_request_ts = time.monotonic()

    # ---------------- 东财存活探测（封禁期快速降级）

    def em_ok(self) -> bool:
        """东财 push2/push2his 域名存活探测（每实例一次，结果缓存）。

        东财对高频拉取有 WAF 长期 IP 封禁（实测可持续 48 小时+），封禁期每个
        请求都要耗尽超时+重试才失败（单请求约3.4秒）。运行开始时探测一次：
        封禁期所有东财一级源直接跳过、全量走降级链（新浪/腾讯/同花顺），
        一次运行可省数分钟无效等待。
        注意：涨停池/炸板池走 push2ex 独立域名，不受此探测约束。
        """
        if self._em_status is None:
            try:
                ak = _akshare()
                self._throttle()
                df = ak.stock_board_industry_name_em()
                self._em_status = not (df is None or df.empty)
            except Exception:
                self._em_status = False
                log.warning("东财接口不可用（疑似IP封禁），本次运行全量走降级数据源")
        return self._em_status

    def _record_missing(self, key: str, why: str = "") -> None:
        msg = f"[数据缺失] {key}" + (f": {why}" if why else "")
        if msg not in self.missing:
            self.missing.append(msg)
        log.warning(msg)

    def _fetch_no_missing(self, key: str, fetch_fn: Callable[[], Any], fmt: str = "pkl") -> Optional[Any]:
        """_fetch 的静默变体：失败不记 missing（供降级链第一级使用）。"""
        hit = self._cached(key, fmt)
        if hit is not None:
            return hit
        for attempt in (1, 2):
            self._throttle()
            try:
                obj = fetch_fn()
                if obj is None or (isinstance(obj, pd.DataFrame) and obj.empty):
                    raise ValueError("空数据")
                self._save(key, fmt, obj)
                return obj
            except Exception:
                if attempt == 1:
                    time.sleep(3)
                    continue
                return None
        return None

    # ---------------- 个股日K（前复权）

    def get_stock_daily(self, code: str, days: Optional[int] = None) -> Optional[pd.DataFrame]:
        """前复权日K，标准化列：date/open/close/high/low/volume/amount/turnover/pct_chg。

        降级链（新浪为主源）：新浪（字段等价：换手/成交额齐备、pct_chg 自算、含当日K线）
        → 东财（push2his，仅兜底）。新浪为主源可把单次运行 60-90 次 push2his 请求
        降为 0 —— 这是触发东财 WAF 长期封禁的最大请求源。
        """
        days = days or self.cfg.get("run", {}).get("hist_days", 300)
        code = str(code).zfill(6)

        def _do_em() -> pd.DataFrame:
            ak = _akshare()
            end = datetime.now().strftime("%Y%m%d")
            start = pd.Timestamp(end) - pd.Timedelta(days=int(days * 1.7))
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=start.strftime("%Y%m%d"), end_date=end, adjust="qfq")
            if df is None or df.empty:
                raise ValueError("空日K")
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
                "成交量": "volume", "成交额": "amount", "换手率": "turnover", "涨跌幅": "pct_chg",
            })
            keep = [c for c in ["date", "open", "close", "high", "low", "volume", "amount", "turnover", "pct_chg"] if c in df.columns]
            df = df[keep].copy()
            df["date"] = pd.to_datetime(df["date"])
            for c in df.columns:
                if c != "date":
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.reset_index(drop=True)

        def _do_sina() -> pd.DataFrame:   # 降级：新浪版（换手率/成交额齐备，pct_chg 自算）
            ak = _akshare()
            sym = ("sh" if code.startswith(("6", "9", "5")) else "sz") + code
            df = ak.stock_zh_a_daily(symbol=sym, adjust="qfq")
            if df is None or df.empty:
                raise ValueError("空日K")
            df = df.rename(columns={"date": "date", "开盘": "open"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            # pct_chg 全量自算后再裁剪（避免窗口首条缺值）
            df["pct_chg"] = (df["close"].astype(float).pct_change() * 100.0).fillna(0.0)
            keep = [c for c in ["date", "open", "close", "high", "low", "volume", "amount", "turnover", "pct_chg"] if c in df.columns]
            df = df[keep].copy()
            for c in df.columns:
                if c != "date":
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.tail(int(days)).reset_index(drop=True)

        res = self._fetch_no_missing(f"daily_sina_{code}", _do_sina, "pkl")
        if res is not None:
            return res
        if not self.em_ok():
            self._record_missing(f"daily_{code}", "新浪失败且东财不可用")
            return None
        return self._fetch(f"daily_{code}", _do_em, "pkl")

    # ---------------- 全市场实时快照（收盘后取终值）

    def get_spot(self) -> Optional[pd.DataFrame]:
        """降级链（腾讯为主源）：腾讯（全市场一次拉取，自带主力净流入/近5日涨幅）
        → 东财（push2，仅兜底）。腾讯为主源：一次请求同时供主力资金因子、
        打板R11验证、候选宇宙兜底使用。
        """
        def _do_em() -> pd.DataFrame:
            ak = _akshare()
            df = ak.stock_zh_a_spot_em()
            df = df.rename(columns={
                "代码": "code", "名称": "name", "最新价": "price", "涨跌幅": "pct_chg",
                "成交量": "volume", "成交额": "amount", "换手率": "turnover", "量比": "vr",
                "今开": "open", "最高": "high", "最低": "low", "昨收": "pre_close",
            })
            keep = [c for c in ["code", "name", "price", "pct_chg", "volume", "amount",
                                "turnover", "vr", "open", "high", "low", "pre_close"] if c in df.columns]
            return df[keep].copy()

        def _do_tx() -> pd.DataFrame:   # 降级：腾讯版（code带sh/sz前缀需剥离；turnover单位万元）
            ak = _akshare()
            df = ak.stock_zh_a_spot_tx()
            df = df.rename(columns={
                "code": "_raw_code", "name": "name", "zxj": "price", "zdf": "pct_chg",
                "turnover": "amount", "volume": "volume", "hsl": "turnover", "lb": "vr",
                "zljlr": "main_net", "zdf_d5": "pct5",
            })
            df["code"] = df["_raw_code"].astype(str).str.extract(r"(\d{6})$")
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1e4      # 万元 → 元
            df["main_net"] = pd.to_numeric(df["main_net"], errors="coerce") * 1e4  # 万元 → 元
            keep = [c for c in ["code", "name", "price", "pct_chg", "volume", "amount",
                                "turnover", "vr", "main_net", "pct5"] if c in df.columns]
            return df[keep].copy()

        res = self._fetch_no_missing("spot_tx", _do_tx, "pkl")
        if res is not None:
            return res
        if not self.em_ok():
            self._record_missing("spot", "腾讯失败且东财不可用")
            return None
        return self._fetch("spot", _do_em, "pkl")

    # ---------------- 全市场主力净占比（批量，替代逐股资金流请求）

    def get_main_net_map(self) -> dict:
        """全市场主力净占比 {code: main_net_pct}，一次批量拉取。

        替代逐股东财资金流（原 24 次请求 → 1 次），降级链：
        腾讯快照（zljlr/成交额，复用 get_spot 当日缓存）
        → 同花顺全市场个股资金流（净额/成交额为"亿/万"后缀字符串）。
        两源皆不可用时返回 {}，由调用方决定是否逐股兜底。
        """
        spot = self.get_spot()
        if spot is not None and not spot.empty and {"main_net", "amount"} <= set(spot.columns):
            amt = pd.to_numeric(spot["amount"], errors="coerce")
            mn = pd.to_numeric(spot["main_net"], errors="coerce")
            pct = mn / amt.where(amt > 0) * 100.0
            return {str(c).zfill(6): (None if pd.isna(v) else round(float(v), 2))
                    for c, v in zip(spot["code"], pct)}

        def _do_ths() -> pd.DataFrame:   # 降级：同花顺全市场个股资金流
            ak = _akshare()
            df = ak.stock_fund_flow_individual(symbol="即时")
            df = df.rename(columns={"股票代码": "code", "净额": "_net", "成交额": "_amt"})
            df = df[["code", "_net", "_amt"]].copy()
            df["main_net"] = df["_net"].map(_cn_amount_yuan)
            df["amount"] = df["_amt"].map(_cn_amount_yuan)
            return df[["code", "main_net", "amount"]]

        res = self._fetch("fund_flow_individual_ths", _do_ths, "pkl")
        if res is not None and not res.empty:
            amt = pd.to_numeric(res["amount"], errors="coerce")
            mn = pd.to_numeric(res["main_net"], errors="coerce")
            pct = mn / amt.where(amt > 0) * 100.0
            return {str(c).zfill(6): (None if pd.isna(v) else round(float(v), 2))
                    for c, v in zip(res["code"], pct)}
        return {}

    # ---------------- 市场赚钱效应（涨跌家数）

    def get_market_activity(self) -> Optional[dict]:
        def _do() -> dict:
            ak = _akshare()
            df = ak.stock_market_activity_legu()
            if df is None or (hasattr(df, "empty") and df.empty):
                raise ValueError("空数据")
            d: dict[str, Any] = {}
            if isinstance(df, dict):
                d = {str(k): v for k, v in df.items()}
            else:  # DataFrame: item/value 两列
                cols = [str(c) for c in df.columns]
                kc, vc = cols[0], cols[1]
                for _, row in df.iterrows():
                    d[str(row[kc])] = row[vc]
            out: dict[str, Any] = {}
            for k, v in d.items():
                key = re.sub(r"\s+", "", str(k))
                try:
                    out[key] = float(str(v).replace(",", "").replace("%", "").strip())
                except (TypeError, ValueError):
                    out[key] = v
            return out

        return self._fetch("market_activity", _do, "json")

    # ---------------- 指数日K（东财版，含成交额）

    def get_index_daily(self, symbol: str) -> Optional[pd.DataFrame]:
        """东财指数日K（含 amount 成交额，供两市量能计算）。失败时降级新浪版。"""
        def _do_em() -> pd.DataFrame:
            ak = _akshare()
            df = ak.stock_zh_index_daily_em(symbol=symbol)
            df = df.rename(columns={"date": "date", "open": "open", "close": "close",
                                    "high": "high", "low": "low", "volume": "volume", "amount": "amount"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.tail(10).reset_index(drop=True)
            for c in ["open", "close", "high", "low", "volume", "amount"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            keep = [c for c in ["date", "open", "close", "high", "low", "volume", "amount"] if c in df.columns]
            return df[keep]

        def _do_sina() -> pd.DataFrame:  # 降级：新浪版（无 amount）
            ak = _akshare()
            df = ak.stock_zh_index_daily(symbol=symbol)
            df = df.rename(columns={"date": "date", "open": "open", "close": "close",
                                    "high": "high", "low": "low", "volume": "volume"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.tail(10).reset_index(drop=True)
            for c in ["open", "close", "high", "low", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df

        # 东财优先（含成交额，量能计算必需）→ 新浪版兜底（无成交额，量能走快照聚合）
        if self.em_ok():
            res = self._fetch_no_missing(f"index_em_{symbol}", _do_em, "pkl")
            if res is not None:
                return res
        return self._fetch(f"index_{symbol}", _do_sina, "pkl")

    # ---------------- 外围指数（日经225 / 韩国KOSPI）

    def get_global_indices(self) -> Optional[pd.DataFrame]:
        """全球主要指数快照（东财），提取日经225、韩国KOSPI。失败降级返回 None。

        日经/KOSPI 无可靠替代源（同花顺为新闻流、新浪不支持），东财封禁期
        外围行情标注 [数据缺失]，不阻塞主流程。
        """
        if not self.em_ok():
            self._record_missing("global_indices", "东财不可用（日经/KOSPI无替代源）")
            return None

        def _do() -> pd.DataFrame:
            ak = _akshare()
            df = ak.index_global_spot_em()
            name_col = next((c for c in df.columns if "名称" in str(c)), None)
            price_col = next((c for c in df.columns if "最新" in str(c)), None)
            pct_col = next((c for c in df.columns if "涨跌幅" in str(c)), None)
            if not all([name_col, price_col, pct_col]):
                raise ValueError("外围指数列名不符")
            kw = {"日经": "日经225", "韩": "韩国KOSPI"}
            rows = []
            for _, r in df.iterrows():
                nm = str(r[name_col])
                for k, cn in kw.items():
                    if k in nm:
                        rows.append({"name": cn, "price": _to_f(r[price_col]),
                                     "pct_chg": _to_f(r[pct_col])})
            if not rows:
                raise ValueError("未找到日经/KOSPI")
            return pd.DataFrame(rows)

        return self._fetch("global_indices", _do, "pkl")

    # ---------------- 行业板块列表（名称+当日涨幅+换手率）

    def get_industry_boards(self) -> Optional[pd.DataFrame]:
        """降级链：东财（push2）→ 同花顺（90板块，含上涨/下跌家数、领涨股、净流入；无换手率）。

        注意：同花顺板块名体系与东财不完全一致，与涨停池"所属行业"（东财名）的
        join 可能部分失配 —— 失配方按板块数据缺失降级处理，不阻塞。
        """
        def _do_em() -> pd.DataFrame:
            ak = _akshare()
            df = ak.stock_board_industry_name_em()
            df = df.rename(columns={
                "排名": "rank", "板块名称": "board", "板块代码": "board_code",
                "涨跌幅": "pct_chg", "换手率": "turnover",
                "上涨家数": "up_count", "下跌家数": "down_count", "领涨股票": "leader_stock",
            })
            keep = [c for c in ["rank", "board", "board_code", "pct_chg", "turnover",
                                "up_count", "down_count", "leader_stock"] if c in df.columns]
            return df[keep].copy()

        def _do_ths() -> pd.DataFrame:   # 降级：同花顺（净流入/总成交额单位=亿 → 元）
            ak = _akshare()
            df = ak.stock_board_industry_summary_ths()
            df = df.rename(columns={
                "序号": "rank", "板块": "board", "涨跌幅": "pct_chg", "总成交额": "board_amount",
                "净流入": "main_net_inflow", "上涨家数": "up_count", "下跌家数": "down_count",
                "领涨股": "leader_stock",
            })
            for c in ("main_net_inflow", "board_amount"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce") * 1e8   # 亿 → 元
            keep = [c for c in ["rank", "board", "pct_chg", "up_count", "down_count",
                                "leader_stock", "main_net_inflow", "board_amount"] if c in df.columns]
            return df[keep].copy()

        # 东财优先（板块名与涨停池"所属行业"同体系，join 兼容性最好）；封禁期走同花顺
        if self.em_ok():
            res = self._fetch_no_missing("industry_boards", _do_em, "pkl")
            if res is not None:
                return res
        return self._fetch("industry_boards_ths", _do_ths, "pkl")

    # ---------------- 板块资金流排名（主买占比维度）

    def get_sector_fund_flow(self) -> Optional[pd.DataFrame]:
        """降级链：东财（push2）→ 同花顺（净额单位=亿 → 元；净占比=净额/(流入+流出)自算）。"""
        def _do_em() -> pd.DataFrame:
            ak = _akshare()
            df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
            df = df.rename(columns={
                "名称": "board", "今日涨跌幅": "pct_chg",
                "今日主力净流入-净额": "main_net_inflow", "今日主力净流入-净占比": "main_net_pct",
            })
            keep = [c for c in ["board", "pct_chg", "main_net_inflow", "main_net_pct"] if c in df.columns]
            return df[keep].copy()

        def _do_ths() -> pd.DataFrame:
            ak = _akshare()
            df = ak.stock_fund_flow_industry(symbol="即时")
            df = df.rename(columns={
                "行业": "board", "行业-涨跌幅": "pct_chg",
                "净额": "main_net_inflow", "流入资金": "_in", "流出资金": "_out",
            })
            df["main_net_inflow"] = pd.to_numeric(df["main_net_inflow"], errors="coerce") * 1e8  # 亿 → 元
            gross = pd.to_numeric(df.get("_in"), errors="coerce") + pd.to_numeric(df.get("_out"), errors="coerce")
            df["main_net_pct"] = (pd.to_numeric(df["main_net_inflow"], errors="coerce")
                                  / gross.replace(0, pd.NA) * 100.0).astype(float)
            keep = [c for c in ["board", "pct_chg", "main_net_inflow", "main_net_pct"] if c in df.columns]
            return df[keep].copy()

        if self.em_ok():
            res = self._fetch_no_missing("sector_fund_flow", _do_em, "pkl")
            if res is not None:
                return res
        return self._fetch("sector_fund_flow_ths", _do_ths, "pkl")

    # ---------------- 板块内个股

    def get_board_cons(self, board: str) -> Optional[pd.DataFrame]:
        """板块成分股（东财专属，无替代源）。东财封禁期返回 None，
        候选宇宙由 main.py 走全市场快照兜底。"""
        if not self.em_ok():
            self._record_missing(f"cons_{board}", "东财不可用（成分股无替代源，候选宇宙走快照兜底）")
            return None

        def _do() -> pd.DataFrame:
            ak = _akshare()
            df = ak.stock_board_industry_cons_em(symbol=board)
            df = df.rename(columns={
                "代码": "code", "名称": "name", "最新价": "price", "涨跌幅": "pct_chg",
                "成交量": "volume", "成交额": "amount", "换手率": "turnover",
                "总市值": "mkt_cap", "流通市值": "float_cap",
            })
            keep = [c for c in ["code", "name", "price", "pct_chg", "volume", "amount",
                                "turnover", "mkt_cap", "float_cap"] if c in df.columns]
            return df[keep].copy()

        return self._fetch(f"cons_{board}", _do, "pkl")

    # ---------------- 板块指数历史（量能比 5 日均额用）

    def get_board_hist(self, board: str) -> Optional[pd.DataFrame]:
        """板块指数近10日（量能比 = 当日成交额/5日均额）。

        降级链：东财（名称体系=东财，与东财板块列表配套）
        → 同花顺板块指数（名称体系=同花顺，与同花顺板块列表配套；涨跌幅自算）。
        两套名称体系各自配套：东财板块名查同花顺指数会空返回，自然落链。
        """
        def _do_em() -> pd.DataFrame:
            ak = _akshare()
            df = ak.stock_board_industry_hist_em(symbol=board, period="日K", adjust="")
            df = df.rename(columns={"日期": "date", "收盘": "close", "成交额": "amount", "涨跌幅": "pct_chg"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.tail(10).reset_index(drop=True)
            for c in ["close", "amount", "pct_chg"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[["date", "close", "amount", "pct_chg"]]

        def _do_ths() -> pd.DataFrame:   # 降级：同花顺板块指数（含成交额，元）
            ak = _akshare()
            end = datetime.now()
            df = ak.stock_board_industry_index_ths(symbol=board,
                                                   start_date=(end - pd.Timedelta(days=30)).strftime("%Y%m%d"),
                                                   end_date=end.strftime("%Y%m%d"))
            if df is None or df.empty:
                raise ValueError("空板块历史")
            df = df.rename(columns={"日期": "date", "收盘价": "close", "成交额": "amount"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").tail(10).reset_index(drop=True)
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            df["pct_chg"] = df["close"].pct_change() * 100.0   # 同花顺无涨跌幅列，自算
            return df[["date", "close", "amount", "pct_chg"]]

        if self.em_ok():
            res = self._fetch_no_missing(f"boardhist_{board}", _do_em, "pkl")
            if res is not None:
                return res
        return self._fetch(f"boardhist_ths_{board}", _do_ths, "pkl")

    # ---------------- 个股资金流（主力净流入因子，仅批量源全挂时逐股兜底）

    def get_stock_fund_flow(self, code: str) -> Optional[pd.DataFrame]:
        code = str(code).zfill(6)
        if not self.em_ok():
            self._record_missing(f"fundflow_{code}", "东财不可用（主力资金优先走批量快照源）")
            return None

        def _do() -> pd.DataFrame:
            ak = _akshare()
            df = ak.stock_individual_fund_flow(stock=code, market=market_of(code))
            df = df.rename(columns={"日期": "date", "收盘价": "close", "涨跌幅": "pct_chg",
                                    "主力净流入-净额": "main_net", "主力净流入-净占比": "main_net_pct"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.tail(5).reset_index(drop=True)
            for c in ["close", "main_net", "main_net_pct"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df

        return self._fetch(f"fundflow_{code}", _do, "pkl")

    # ---------------- 涨停股池 / 炸板股池

    def get_zt_pool(self) -> Optional[pd.DataFrame]:
        def _do() -> pd.DataFrame:
            ak = _akshare()
            df = ak.stock_zt_pool_em(date=self.trade_date)
            df = df.rename(columns={
                "代码": "code", "名称": "name", "涨跌幅": "pct_chg", "最新价": "price",
                "成交额": "amount", "流通市值": "float_cap", "总市值": "mkt_cap",
                "换手率": "turnover", "首次封板时间": "first_seal_time", "最后封板时间": "last_seal_time",
                "封板资金": "seal_fund", "炸板次数": "open_times", "涨停统计": "zt_stat",
                "连板数": "lian_ban", "所属行业": "industry",
            })
            keep = [c for c in ["code", "name", "pct_chg", "price", "amount", "float_cap", "mkt_cap",
                                "turnover", "first_seal_time", "last_seal_time", "seal_fund",
                                "open_times", "zt_stat", "lian_ban", "industry"] if c in df.columns]
            return df[keep].copy()

        return self._fetch("zt_pool", _do, "pkl")

    def get_zb_pool(self) -> Optional[pd.DataFrame]:
        def _do() -> pd.DataFrame:
            ak = _akshare()
            df = ak.stock_zt_pool_zbgc_em(date=self.trade_date)
            df = df.rename(columns={"代码": "code", "名称": "name", "涨跌幅": "pct_chg",
                                    "涨停统计": "zt_stat", "炸板次数": "open_times", "所属行业": "industry"})
            keep = [c for c in ["code", "name", "pct_chg", "zt_stat", "open_times", "industry"]
                    if c in df.columns]
            return df[keep].copy()

        return self._fetch("zb_pool", _do, "pkl")
