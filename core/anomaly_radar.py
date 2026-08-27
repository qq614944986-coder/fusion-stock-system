# -*- coding: utf-8 -*-
"""左轨 · 异动雷达引擎（机会挖掘前置，规格 v1.0 §三）。

只认"反常"，不认"强"。对候选标的检测四类被市场忽略的微弱异常，输出打的标签+数值，
不打总分。判定口径为"是否偏离常态分布"。高价位的强势被视为减分（反追高）。

输入 ds（预聚合数据）：
    lhb_by_code: {code: [row...]}  龙虎榜明细（净买额）
    dzjy_by_code: {code: [row...]} 大宗交易明细（成交价）
    yjyg_by_code: {code: [row...]} 业绩预告（预告类型/变动幅度）
    yjbb_by_code: {code: [row...]} 业绩报表（净利同比）
hist: 前复权日K（date/open/high/low/close/volume/pct_chg）
rrow: 当日行情行（pct_chg/turnover/price，可 None）
返回信号列表 [{kind,name,value,note}]
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _cl_series(hist: pd.DataFrame, col: str):
    return hist[col].astype(float) if hist is not None and len(hist) and col in hist.columns else []  # noqa: E501


def _ma(s, n, i=-1):
    if not len(s):
        return None
    if isinstance(i, int) and i < 0:
        i += len(s)
    if i - n < 0:
        return None
    w = s.iloc[i - n:i]
    if w.empty or w.isna().all():
        return None
    return float(w.mean())


def detect_anomalies(code, name, hist: Optional[pd.DataFrame], rrow: Optional[dict],
                     ds: dict, cfg: dict) -> list[dict]:
    """返回异动信号列表；无则为 []。"""
    if hist is None or len(hist) < 30:
        return []
    signals: list[dict] = []

    close_s = _cl_series(hist, "close")
    vol_s = _cl_series(hist, "volume")
    pct_s = _cl_series(hist, "pct_chg") if "pct_chg" in hist.columns else None
    if not len(close_s):
        return []

    n = len(close_s)
    close = float(close_s.iloc[-1])
    if close <= 0:
        return []

    # ---- 价格位置（估值低位的价格代理：近3年分位）
    hist_len = min(n, 750)                      # 近3年（约750交易日）
    win = close_s.iloc[-hist_len:]
    low3, hi3 = float(win.min()), float(win.max())
    price_pct = ((close - low3) / (hi3 - low3) * 100) if hi3 > low3 else 50.0   # 0-100 高位占比
    # 近60日跌幅
    drop60 = ((float(close_s.iloc[-60]) - close) / float(close_s.iloc[-60]) * 100) if n >= 60 and close_s.iloc[-60] else 0.0
    # 量比（当日/20日均）
    vr = None
    if len(vol_s) >= 21:
        ma20v = float(vol_s.iloc[-21:-1].mean()) if vol_s.iloc[-21:-1].mean() else None
        if ma20v:
            vr = float(vol_s.iloc[-1] / ma20v)
    # 近期未创新低（企稳）
    min5 = float(vol_s.iloc[-5:].min()) if len(vol_s) >= 5 else None

    # ================= 价位类 =================
    pct_now = _num(rrow.get("pct_chg")) if rrow is not None else None
    if pct_now is None and pct_s is not None and len(pct_s):
        pct_now = float(pct_s.iloc[-1])
    # L1a 低位温和放量（先手补涨，非追高）
    if vr is not None and vr >= 1.5 and pct_now is not None and 0.5 <= pct_now <= 5.0 and price_pct < 60:
        signals.append({"kind": "价位", "name": "低位温和放量", "value": f"量比{vr:.1f}·涨{pct_now:+.1f}%·价格分位{price_pct:.0f}%",
                        "note": "放量但未高涨，处于自身价格低位，先手特征"})
    # L1b 深跌到位（超卖 + 已止跌企稳：低分位 且 近5日低点未大幅创新低）
    stopped = False
    if hist is not None and len(hist) >= 20 and "low" in hist.columns:
        low5 = float(hist["low"].astype(float).iloc[-5:].min())
        low20 = float(hist["low"].astype(float).iloc[-20:].min())
        stopped = low20 > 0 and low5 >= low20 * 0.97
    if price_pct < 25 and (stopped or drop60 >= 8):
        signals.append({"kind": "价位", "name": "深跌到位", "value": f"价格分位{price_pct:.0f}%·近60日{drop60:+.1f}%",
                        "note": "情绪超调区，已届止跌位"})
    # 高位放量滞涨（负信号）
    if vr is not None and vr > 1.8 and pct_now is not None and pct_now < 1.0:
        signals.append({"kind": "价位", "name": "放量滞涨", "value": f"量比{vr:.1f}但涨{pct_now:+.1f}%",
                        "note": "量价背离，获利盘出逃，减分"})

    # ================= 资金类 =================
    lhb_rows = ds.get("lhb_by_code", {}).get(code) or []
    if lhb_rows:
        net = _num(lhb_rows[0].get("龙虎榜净买额"))
        if net is not None and net > 0:
            signals.append({"kind": "资金", "name": "龙虎榜机构净买",
                            "value": f"净买{net/1e8:.2f}亿",
                            "note": f"上榜原因：{str(lhb_rows[0].get('上榜原因',''))[:30]}"})
    dzjy_rows = ds.get("dzjy_by_code", {}).get(code) or []
    if dzjy_rows:
        p = _num(dzjy_rows[0].get("成交价"))
        if p is not None and close > 0 and p >= close:
            signals.append({"kind": "资金", "name": "大宗溢价成交", "value": f"成交价{p:.2f}≥收盘{close:.2f}",
                            "note": "大宗优于市价，机构抢筹"})

    # ================= 基本面类 =================
    yjyg_rows = ds.get("yjyg_by_code", {}).get(code) or []
    if yjyg_rows:
        t = str(yjyg_rows[0].get("预告类型", ""))
        if any(k in t for k in ("预增", "略增", "扭亏", "续盈", "减亏")):
            signals.append({"kind": "基本面", "name": f"业绩预告：{t}", "value": f"幅度{str(yjyg_rows[0].get('业绩变动幅度',''))[:20]}",
                            "note": "预告向好，留意是否超一致预期"})
    yjbb_rows = ds.get("yjbb_by_code", {}).get(code) or []
    if yjbb_rows:
        ni = _num(yjbb_rows[0].get("净利润-同比增长"))
        if ni is not None and ni > 0:
            signals.append({"kind": "基本面", "name": "净利同比增长", "value": f"{ni:+.1f}%",
                            "note": "盈利为正增长"})

    # ================= 缩量企稳（左侧布局辅助） =================
    ma20close = _ma(close_s, 20)
    if ma20close and close < ma20close and price_pct < 35 and (vr is None or vr < 0.9):
        signals.append({"kind": "价位", "name": "缩量企稳", "value": f"价在MA20下·量比{vr if vr is not None else 0:.1f}",
                        "note": "卖压出清迹象，可纳入温和左侧布局候选"})

    return signals