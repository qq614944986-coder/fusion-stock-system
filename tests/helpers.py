# -*- coding: utf-8 -*-
"""测试公共工具：构造合成 OHLCV 日K数据。"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def make_df(closes, vols=None, opens=None, highs=None, lows=None, start="2026-01-01"):
    """按收盘价序列构造日K。默认 open=前收，high=max(o,c)+0.1，low=min(o,c)-0.1。"""
    n = len(closes)
    d0 = datetime.strptime(start, "%Y-%m-%d")
    dates = [d0 + timedelta(days=i) for i in range(n)]
    closes = [float(c) for c in closes]
    if opens is None:
        opens = [closes[0]] + closes[:-1]
    opens = [float(o) for o in opens]
    if highs is None:
        highs = [max(o, c) + 0.1 for o, c in zip(opens, closes)]
    if lows is None:
        lows = [min(o, c) - 0.1 for o, c in zip(opens, closes)]
    if vols is None:
        vols = [10000.0] * n
    pct = [0.0] + [(c / p - 1) * 100 for c, p in zip(closes[1:], closes[:-1])]
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "open": opens, "close": closes,
        "high": [float(h) for h in highs], "low": [float(l) for l in lows],
        "volume": [float(v) for v in vols], "amount": [c * v for c, v in zip(closes, vols)],
        "pct_chg": pct,
    })
