# -*- coding: utf-8 -*-
"""L1 数据层：akshare 接口封装 + 本地缓存 + 降级处理 + 配置加载。

降级规则（规格书 §3.1）：任一接口失败 → 返回 None 并记录到 missing 列表，
对应模块输出标注 [数据缺失]，不阻塞其他模块。严禁编造数据。
缓存规则：当日已拉取的数据存 data/cache/YYYYMMDD/，当日重跑不重复请求。
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
        import akshare as ak
        _ak = ak
    return _ak


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
    """指数温度所需三大指数（4.1：上证/创业板/科创50）。"""
    return {
        "上证指数": "sh000001",
        "创业板指": "sz399006",
        "科创50": "sh000688",
    }


# ---------------------------------------------------------------- 数据提供者

class DataProvider:
    def __init__(self, cfg: dict, base_dir: Optional[Path] = None, trade_date: Optional[str] = None):
        self.cfg = cfg
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
        self.trade_date = trade_date or datetime.now().strftime("%Y%m%d")
        self.cache_dir = self.base_dir / cfg.get("run", {}).get("data_dir", "data") / "cache" / self.trade_date
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.missing: list[str] = []          # 数据缺失记录（展示层⑥风险提示用）

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

    # ---------------- 个股日K（前复权）

    def get_stock_daily(self, code: str, days: Optional[int] = None) -> Optional[pd.DataFrame]:
        """前复权日K，标准化列：date/open/close/high/low/volume/amount/turnover/pct_chg。"""
        days = days or self.cfg.get("run", {}).get("hist_days", 300)
        code = str(code).zfill(6)

        def _do() -> pd.DataFrame:
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

        return self._fetch(f"daily_{code}", _do, "pkl")

    # ---------------- 全市场实时快照（收盘后取终值）

    def get_spot(self) -> Optional[pd.DataFrame]:
        def _do() -> pd.DataFrame:
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

        return self._fetch("spot", _do, "pkl")

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

    # ---------------- 指数日K

    def get_index_daily(self, symbol: str) -> Optional[pd.DataFrame]:
        def _do() -> pd.DataFrame:
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

        return self._fetch(f"index_{symbol}", _do, "pkl")

    # ---------------- 行业板块列表（名称+当日涨幅+换手率）

    def get_industry_boards(self) -> Optional[pd.DataFrame]:
        def _do() -> pd.DataFrame:
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

        return self._fetch("industry_boards", _do, "pkl")

    # ---------------- 板块资金流排名（主买占比维度）

    def get_sector_fund_flow(self) -> Optional[pd.DataFrame]:
        def _do() -> pd.DataFrame:
            ak = _akshare()
            df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
            df = df.rename(columns={
                "名称": "board", "今日涨跌幅": "pct_chg",
                "今日主力净流入-净额": "main_net_inflow", "今日主力净流入-净占比": "main_net_pct",
            })
            keep = [c for c in ["board", "pct_chg", "main_net_inflow", "main_net_pct"] if c in df.columns]
            return df[keep].copy()

        return self._fetch("sector_fund_flow", _do, "pkl")

    # ---------------- 板块内个股

    def get_board_cons(self, board: str) -> Optional[pd.DataFrame]:
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
        def _do() -> pd.DataFrame:
            ak = _akshare()
            df = ak.stock_board_industry_hist_em(symbol=board, period="日K", adjust="")
            df = df.rename(columns={"日期": "date", "收盘": "close", "成交额": "amount", "涨跌幅": "pct_chg"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.tail(10).reset_index(drop=True)
            for c in ["close", "amount", "pct_chg"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[["date", "close", "amount", "pct_chg"]]

        return self._fetch(f"boardhist_{board}", _do, "pkl")

    # ---------------- 个股资金流（主力净流入因子）

    def get_stock_fund_flow(self, code: str) -> Optional[pd.DataFrame]:
        code = str(code).zfill(6)

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
