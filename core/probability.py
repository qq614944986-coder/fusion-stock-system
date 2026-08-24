# -*- coding: utf-8 -*-
"""个股次日概率引擎（李致远体系 · 用户需求补充）。

用途：三线候选池每只股票输出「次日收红率 / 冲高率 / 平均次日收益 / 涨跌概率倾向」。
方法（用户口径：仅历史相似性，不叠加情绪修正）：
- 回看该股近 N 日（默认120）交易日，取与今日「涨跌幅、BIAS20」相近的历史日 i；
- 统计这些样本的次日表现（相对 i 日收盘价）：次日收红率（clos(次日)>0）、
  冲高率（次日最高≥3%）、平均次日收益；
- 数据诚实纪律：匹配样本 < MIN_SAMPLES 时数值置 None（前端显示"—"），不编造概率。

涨跌概率倾向映射：收红率(rr)：
    rr≥60% → "偏多"；40%≤rr<60% → "震荡"；rr<40% → "偏空"；样本不足 → "—"。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

MIN_SAMPLES = 10          # 样本 <10 不显示统计
SPIKE_PCT = 3.0           # 冲高定义：次日最高较入选收盘 ≥+3%
_PCT_TOL = 3.0            # 涨跌幅匹配容忍度（±百分点）
_BIAS_TOL = 8.0           # BIAS20 匹配容忍度（±百分点）


def _v(df: pd.DataFrame, col: str, i: int) -> Optional[float]:
    if df is None or col not in df.columns or i < 0 or i >= len(df):
        return None
    x = df[col].iloc[i]
    return None if pd.isna(x) else float(x)


def _bias20(df: pd.DataFrame, i: int) -> Optional[float]:
    """第 i 日收盘相对（不含当日）的20日均线的乖离（%）。"""
    if i - 20 < 0:
        return None
    win = df["close"].astype(float).iloc[i - 20:i]
    if len(win) < 20:
        return None
    ma = float(win.mean())
    if not ma:
        return None
    c = float(df["close"].iloc[i])
    return (c - ma) / ma * 100.0


def next_day_probability(hist: Optional[pd.DataFrame], bias_now: Optional[float],
                         pct_now: Optional[float]) -> dict:
    """返回 {red_rate, spike_rate, avg_ret, samples, tendency}（样本不足时为 None）。

    - bias_now：当前 BIAS20（无则略过该维匹配，仅用涨跌幅）；
    - pct_now：当日涨跌幅%。

    red_rate/spike_rate/avg_ret 取整到 1/1/2 位小数；tendency 为"偏多/震荡/偏空/—"。
    """
    if hist is None or len(hist) < 3:
        return _empty()
    close_s = hist["close"].astype(float)
    matches: list[dict] = []
    for i in range(len(hist) - 1):          # i 需存在 i+1（次日）
        pct = _v(hist, "pct_chg", i)
        if pct_now is not None and (pct is None or abs(pct - pct_now) > _PCT_TOL):
            continue
        if bias_now is not None:
            b = _bias20(hist, i)
            if b is None or abs(b - bias_now) > _BIAS_TOL:
                continue
        base = close_s.iloc[i]
        if not base:
            continue
        o = _v(hist, "open", i + 1)
        h = _v(hist, "high", i + 1)
        c = _v(hist, "close", i + 1)
        if None in (o, h, c):
            continue
        matches.append({
            "open_ret": (o - base) / base * 100.0,
            "high_ret": (h - base) / base * 100.0,
            "close_ret": (c - base) / base * 100.0,
        })
    n = len(matches)
    if n < MIN_SAMPLES:
        return {**_empty(), "samples": n}
    red = sum(1 for m in matches if m["close_ret"] > 0)
    spike = sum(1 for m in matches if m["high_ret"] >= SPIKE_PCT)
    rr = red / n * 100.0
    return {
        "red_rate": round(rr, 1),
        "spike_rate": round(spike / n * 100.0, 1),
        "avg_ret": round(sum(m["close_ret"] for m in matches) / n, 2),
        "samples": n,
        "tendency": ("偏多" if rr >= 60 else "震荡" if rr >= 40 else "偏空"),
    }


def _empty() -> dict:
    return {"red_rate": None, "spike_rate": None, "avg_ret": None, "samples": 0,
            "tendency": "—"}