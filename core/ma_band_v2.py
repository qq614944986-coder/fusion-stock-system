# -*- coding: utf-8 -*-
"""老樊引擎 · MABandV2：均线带 + BIAS60 + 趋势判定（规格书 §6.1-6.2）。

历史偏差警示（必须遵守）：
- 短期带 = MA5/8/13，中期带 = MA55/60/65（不是 MA13/21/34 或 MA55/89/144）
- 唯一乖离率基准是 BIAS60（不用 BIAS6/13/34）
- 三均线构成"带"，以带上下沿为突破/跌破依据（不是单条均线支撑）
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class MABandV2:
    def __init__(self, cfg: dict):
        mb = cfg["ma_band"]
        self.short_periods: list[int] = list(mb["short_term"])   # [5, 8, 13]
        self.mid_periods: list[int] = list(mb["mid_term"])       # [55, 60, 65]
        self.bias_period: int = int(mb["bias_ma_period"])        # 60
        self.buy_bias: float = float(mb["buy_bias_threshold"])   # -25
        self.sell_bias: float = float(mb["sell_bias_threshold"]) # 25
        self.tolerance: float = float(mb["band_tolerance_pct"])  # 1.5
        self.vol_ma_period: int = int(mb["vol_ma_period"])       # 20

    # ------------------------------------------------------------ 指标计算

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """在日K上追加均线/带/量比/BIAS60 列。df 需含 date/open/close/high/low/volume。"""
        out = df.copy().reset_index(drop=True)
        close = out["close"].astype(float)
        vol = out["volume"].astype(float)

        for p in self.short_periods:                       # MA5 / MA8 / MA13
            out[f"MA{p}"] = close.rolling(p).mean()
        for p in self.mid_periods:                         # MA55 / MA60 / MA65
            out[f"MA{p}"] = close.rolling(p).mean()

        # 短期带（攻击带）
        s = out[[f"MA{p}" for p in self.short_periods]].astype(float)
        out["short_upper"] = s.max(axis=1)
        out["short_lower"] = s.min(axis=1)
        out["short_middle"] = s.mean(axis=1)
        # 中期带（生命线）
        m = out[[f"MA{p}" for p in self.mid_periods]].astype(float)
        out["mid_upper"] = m.max(axis=1)
        out["mid_lower"] = m.min(axis=1)
        out["mid_middle"] = m.mean(axis=1)

        # 量能
        out["vol_ma20"] = vol.rolling(self.vol_ma_period).mean()
        out["vol_ratio"] = vol / out["vol_ma20"]

        # BIAS60 = (收盘价 - MA60) / MA60 × 100%
        ma60 = out[f"MA{self.bias_period}"].astype(float)
        out["bias60"] = (close - ma60) / ma60 * 100.0
        return out

    # ------------------------------------------------------------ 趋势判定

    @staticmethod
    def _prev(series: pd.Series, i: int) -> Optional[float]:
        j = i - 1
        if j < 0:
            return None
        v = series.iloc[j]
        return None if pd.isna(v) else float(v)

    @staticmethod
    def _at(series: pd.Series, i: int) -> Optional[float]:
        if i < 0 or i >= len(series):
            return None
        v = series.iloc[i]
        return None if pd.isna(v) else float(v)

    def judge(self, b: pd.DataFrame, i: int = -1) -> dict:
        """第 i 行的趋势判定（i=-1 取最后一行）。"""
        if len(b) == 0:
            return self._empty_judgement()
        if i < 0:
            i += len(b)
        if i < 0 or i >= len(b):
            return self._empty_judgement()

        su, sl, sm = (self._at(b[c], i) for c in ("short_upper", "short_lower", "short_middle"))
        mu, ml, mm = (self._at(b[c], i) for c in ("mid_upper", "mid_lower", "mid_middle"))
        close = self._at(b["close"], i)
        bias60 = self._at(b["bias60"], i)
        sm_prev = self._prev(b["short_middle"], i)
        mm_prev = self._prev(b["mid_middle"], i)

        res = {
            "多头排列": False, "空头排列": False,
            "短期带粘合": False, "中期带粘合": False, "粘合信号": False,
            "价格在带内": False,
            "bias60": bias60,
            "bias60_zone": "未知",
            "距中期带上沿pct": None,
            "数据不足": False,
        }

        if any(v is None for v in (su, sl, sm, mu, ml, mm, close)):
            res["数据不足"] = True
            return res

        # 多头排列（三条件须全部满足）
        res["多头排列"] = bool(sl > mu and sm_prev is not None and sm > sm_prev
                               and mm_prev is not None and mm > mm_prev)
        # 空头排列（三条件须全部满足）
        res["空头排列"] = bool(su < ml and sm_prev is not None and sm < sm_prev
                               and mm_prev is not None and mm < mm_prev)
        # 粘合度 = (带上沿-带下沿)/带中轴 ×100% ≤ 1.5%（变盘信号）
        res["短期带粘合"] = bool((su - sl) / sm * 100.0 <= self.tolerance)
        res["中期带粘合"] = bool((mu - ml) / mm * 100.0 <= self.tolerance)
        res["粘合信号"] = res["短期带粘合"] or res["中期带粘合"]
        # 价格在带内：收盘价介于中期带上下沿之间
        res["价格在带内"] = bool(ml <= close <= mu)
        # 距中期带上沿（正=上方，负=下方）
        res["距中期带上沿pct"] = (close - mu) / mu * 100.0

        if bias60 is not None:
            res["bias60_zone"] = self.bias_zone(bias60)
        return res

    def bias_zone(self, bias60: float) -> str:
        """BIAS60 区间：≥+40 严重超买；+25~+40 超买；-25~+25 正常；-40~-25 超卖；≤-40 严重超卖。"""
        if bias60 >= 40:
            return "严重超买"
        if bias60 >= 25:
            return "超买"
        if bias60 > -25:
            return "正常"
        if bias60 > -40:
            return "超卖"
        return "严重超卖"

    @staticmethod
    def _empty_judgement() -> dict:
        return {
            "多头排列": False, "空头排列": False,
            "短期带粘合": False, "中期带粘合": False, "粘合信号": False,
            "价格在带内": False, "bias60": None, "bias60_zone": "未知",
            "距中期带上沿pct": None, "数据不足": True,
        }
