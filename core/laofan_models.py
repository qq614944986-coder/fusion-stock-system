# -*- coding: utf-8 -*-
"""老樊引擎 · 九大交易模型（评分制，规格书 §6.5）。

模型 2/3/4 与买点信号联动：BP1/BP2/BP3 检测结果直接作为对应模型输入。
涨停/跌停判定：主板±10%（检测阈值±9.5%），创业板/科创板±20%（±19.5%），北交所±30%（±29.5%）。
分档量化说明（规格书未给出分档边界处，按保守就近原则实现并在此注明）：
- 缩量缩波·量能萎缩：近5日均量/前20日均量 ≤0.4→30分；≤0.6→20分；≤0.8→10分
- 缩量缩波·波幅收窄：近5日振幅/前5日振幅 ≤0.4→25分；≤0.7→15分
- N式·带附近：收盘介于中期带下沿与带上沿×1.05 之间
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .laofan_signals import LaofanSignalEngine, Signal, StockState

MODEL_META = {
    1: ("极值反转", "买入", 95), 2: ("突破买点模型", "买入", 90), 3: ("回踩买点模型", "买入", 92),
    4: ("乖离买点模型", "买入", 85), 5: ("探路尖兵", "买入", 75), 6: ("N式倍量双涨停", "买入", 88),
    7: ("异动搓揉线", "中性/买", 78), 8: ("缩量缩波", "中性", 80), 9: ("岛型反转", "买/卖", 90),
}


def limit_up_pct(code: str) -> float:
    """按板块返回涨停检测阈值(%)。"""
    code = str(code).zfill(6)
    if code.startswith(("30", "68")):
        return 19.5
    if code.startswith(("4", "8", "92")):
        return 29.5
    return 9.5


class LaofanModels:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.th = cfg["model_thresholds"]

    # ------------------------------------------------------------ 工具

    @staticmethod
    def _v(b: pd.DataFrame, col: str, i: int) -> Optional[float]:
        if i < 0 or i >= len(b):
            return None
        v = b[col].iloc[i]
        return None if pd.isna(v) else float(v)

    def _is_limit_up(self, b: pd.DataFrame, i: int, code: str) -> bool:
        p = self._v(b, "pct_chg", i)
        return p is not None and p >= limit_up_pct(code)

    def _is_limit_down(self, b: pd.DataFrame, i: int, code: str) -> bool:
        p = self._v(b, "pct_chg", i)
        return p is not None and p <= -limit_up_pct(code)

    def _result(self, mid: int, triggered: bool, score: int, details: list, conf: int) -> dict:
        name, direction, base = MODEL_META[mid]
        return {
            "id": mid, "model": name, "direction": direction, "triggered": triggered,
            "score": score, "confidence": conf, "base_confidence": base, "details": details,
        }

    # ------------------------------------------------------------ 总入口

    def evaluate_all(self, bands: pd.DataFrame, state: StockState,
                     signals: Optional[list] = None, i: int = -1, code: str = "") -> list[dict]:
        signals = signals or []
        if len(bands) == 0:
            return []
        if i < 0:
            i += len(bands)
        sig_types = {s.type for s in signals}
        return [
            self.m1_extreme_reversal(bands, i, code),
            self.m2_breakout(bands, sig_types),
            self.m3_pullback(bands, sig_types),
            self.m4_bias_buy(bands, i, sig_types),
            self.m5_pathfinder(bands, i),
            self.m6_n_double_limitup(bands, i, code),
            self.m7_whipsaw(bands, i),
            self.m8_shrink_volatility(bands, i),
            self.m9_island_reversal(bands, i),
        ]

    # ------------------------------------------------------------ 模型1 极值反转

    def m1_extreme_reversal(self, b: pd.DataFrame, i: int, code: str) -> dict:
        score, det = 0, []
        close = self._v(b, "close", i)
        # 近30日内跌停 且 当前/昨日涨停 +30
        ld_idx = None
        for j in range(max(0, i - 30), i + 1):
            if self._is_limit_down(b, j, code):
                ld_idx = j
        lu_today, lu_prev = self._is_limit_up(b, i, code), self._is_limit_up(b, i - 1, code)
        if ld_idx is not None and (lu_today or lu_prev):
            score += 30
            det.append(f"近30日有跌停({str(b['date'].iloc[ld_idx])[:10]})且{'今日' if lu_today else '昨日'}涨停 +30")
            # 涨停量比分档
            lu_i = i if lu_today else i - 1
            vr = self._v(b, "vol_ratio", lu_i)
            if vr is not None:
                if vr >= 1.5:
                    score += 15; det.append(f"涨停量比{vr:.2f}≥1.5 +15")
                elif vr >= 1.2:
                    score += 8; det.append(f"涨停量比{vr:.2f}≥1.2 +8")
            # 跌停后最低价低于跌停收盘价5% +10
            ld_close = self._v(b, "close", ld_idx)
            if ld_close is not None:
                lows = [self._v(b, "low", k) for k in range(ld_idx + 1, i + 1)]
                lows = [x for x in lows if x is not None]
                if lows and min(lows) < ld_close * 0.95:
                    score += 10; det.append("跌停后最低价低于跌停收盘价5% +10")
        ma60 = self._v(b, "MA60", i)
        if close is not None and ma60 is not None and close < ma60:
            score += 10; det.append("股价在60日线下方 +10")
        ml = self._v(b, "mid_lower", i)
        if close is not None and ml is not None and close < ml:
            score += 10; det.append("股价在中期带下沿下方 +10")
        triggered = score >= int(self.th["extreme_reversal"])
        return self._result(1, triggered, score, det, 95 if triggered else 0)

    # ------------------------------------------------------------ 模型2/3 联动模型

    def m2_breakout(self, b: pd.DataFrame, sig_types: set) -> dict:
        trig = "BP2" in sig_types
        det = ["BP2突破买点已触发，联动入场（置信度+5）"] if trig else ["BP2未触发"]
        return self._result(2, trig, 100 if trig else 0, det, 95 if trig else 0)

    def m3_pullback(self, b: pd.DataFrame, sig_types: set) -> dict:
        trig = "BP3" in sig_types
        det = ["BP3回踩买点已触发，联动入场（置信度+5）"] if trig else ["BP3未触发"]
        return self._result(3, trig, 100 if trig else 0, det, 97 if trig else 0)

    # ------------------------------------------------------------ 模型4 乖离买点模型

    def m4_bias_buy(self, b: pd.DataFrame, i: int, sig_types: set) -> dict:
        bias = self._v(b, "bias60", i)
        det, conf, trig = [], 0, False
        if "BP1" in sig_types:
            trig, conf = True, 88
            det = [f"BP1乖离买点已触发（置信度85+3）"]
        elif bias is not None and bias <= -20:
            trig, conf = True, 55
            det = [f"BP1未触发但BIAS60={bias:.1f}%≤-20%（置信度55）"]
        else:
            det = [f"BIAS60={bias:.1f}%" if bias is not None else "BIAS60数据缺失"]
        return self._result(4, trig, 100 if trig else 0, det, conf)

    # ------------------------------------------------------------ 模型5 探路尖兵

    def m5_pathfinder(self, b: pd.DataFrame, i: int) -> dict:
        score, det = 0, []
        close = self._v(b, "close", i)
        mu, ml = self._v(b, "mid_upper", i), self._v(b, "mid_lower", i)
        su = self._v(b, "short_upper", i)
        if None in (close, mu, ml):
            return self._result(5, False, 0, ["中期带数据不足"], 0)
        if ml <= close <= mu:
            score += 25; det.append("股价在中期带内部 +25")
        elif close < ml:
            score += 15; det.append("股价在中期带下方 +15")
        c5ago = self._v(b, "close", i - 5)
        if c5ago:
            gain = (close - c5ago) / c5ago * 100
            if gain > 5:
                score += 20; det.append(f"近5日涨幅{gain:.1f}%>5% +20")
            elif gain > 2:
                score += 10; det.append(f"近5日涨幅{gain:.1f}%>2% +10")
        vr = self._v(b, "vol_ratio", i)
        if vr is not None:
            if vr >= 1.5:
                score += 15; det.append(f"量比{vr:.2f}≥1.5 +15")
            elif vr >= 1.2:
                score += 8; det.append(f"量比{vr:.2f}≥1.2 +8")
        if su is not None and close > su:
            score += 15; det.append("已突破短期带 +15")
        triggered = score >= int(self.th["pathfinder"])
        return self._result(5, triggered, score, det, 75 if triggered else 0)

    # ------------------------------------------------------------ 模型6 N式倍量双涨停

    def m6_n_double_limitup(self, b: pd.DataFrame, i: int, code: str) -> dict:
        score, det = 0, []
        close = self._v(b, "close", i)
        mu, ml = self._v(b, "mid_upper", i), self._v(b, "mid_lower", i)
        if None in (close, mu, ml):
            return self._result(6, False, 0, ["中期带数据不足"], 0)
        # 位置限制：股价不超过中期带上沿130%
        if close > mu * 1.30:
            return self._result(6, False, 0, [f"股价{close:.2f}超过中期带上沿130%（{mu*1.30:.2f}），位置限制排除"], 0)
        # 近20日涨停日列表
        lu_days = [j for j in range(max(0, i - 19), i + 1) if self._is_limit_up(b, j, code)]
        if len(lu_days) >= 2:
            score += 30; det.append(f"近20日{len(lu_days)}个涨停 +30")
            j1, j2 = lu_days[-2], lu_days[-1]           # 最近两个涨停
            gap = j2 - j1
            if gap >= 3:
                score += 15; det.append(f"两涨停间隔{gap}天≥3天 +15")
            else:
                score += 5; det.append(f"两涨停间隔{gap}天<3天 +5")
            vr2 = self._v(b, "vol_ratio", j2)
            if vr2 is not None:
                if vr2 >= 2.0:
                    score += 15; det.append(f"第二涨停量比{vr2:.2f}≥2.0 +15")
                elif vr2 >= 1.5:
                    score += 10; det.append(f"第二涨停量比{vr2:.2f}≥1.5 +10")
            if close < ml:
                score += 20; det.append("股价在中期带下方 +20")
            elif ml <= close <= mu * 1.05:
                score += 10; det.append("股价在中期带附近 +10")
            if j2 == i:
                score += 5; det.append("最后涨停即今日 +5")
        triggered = score >= int(self.th["n_double_limitup"])
        return self._result(6, triggered, score, det, 88 if triggered else 0)

    # ------------------------------------------------------------ 模型7 异动搓揉线

    @staticmethod
    def _candle_parts(o: float, c: float, h: float, l: float) -> Optional[tuple]:
        rng = h - l
        if rng <= 0:
            return None
        upper = (h - max(o, c)) / rng
        lower = (min(o, c) - l) / rng
        body = abs(c - o) / rng
        return upper, lower, body

    def m7_whipsaw(self, b: pd.DataFrame, i: int) -> dict:
        score, det = 0, []
        o1, c1 = self._v(b, "open", i - 1), self._v(b, "close", i - 1)
        h1, l1 = self._v(b, "high", i - 1), self._v(b, "low", i - 1)
        o2, c2 = self._v(b, "open", i), self._v(b, "close", i)
        h2, l2 = self._v(b, "high", i), self._v(b, "low", i)
        if None in (o1, c1, h1, l1, o2, c2, h2, l2):
            return self._result(7, False, 0, ["K线数据不足"], 0)
        p1 = self._candle_parts(o1, c1, h1, l1)
        p2 = self._candle_parts(o2, c2, h2, l2)
        if not p1 or not p2:
            return self._result(7, False, 0, ["K线振幅为零，形态无效"], 0)
        u1, d1, b1 = p1
        u2, d2, b2 = p2
        long_upper = u1 > 0.6 and b1 < 0.3      # 长上影：上影>60% 且 实体<30%
        long_lower = d2 > 0.6 and b2 < 0.3      # 长下影：同标准
        long_lower_then_upper = (d1 > 0.6 and b1 < 0.3) and (u2 > 0.6 and b2 < 0.3)
        if long_upper and long_lower:
            score += 30; det.append("先长上影后长下影（标准搓揉线） +30")
        elif long_lower_then_upper:
            score += 30 + 20; det.append("反向搓揉（先长下影后长上影） +30；反向形态 +20")
        else:
            return self._result(7, False, 0, ["未构成搓揉线形态"], 0)
        mm = self._v(b, "mid_middle", i)
        c2v = self._v(b, "close", i)
        if mm and c2v and abs(c2v - mm) / mm * 100 <= 5:
            score += 15; det.append("出现在中期带±5% +15")
        vr1, vr2 = self._v(b, "vol_ratio", i - 1), self._v(b, "vol_ratio", i)
        if vr1 is not None and vr2 is not None and vr1 < 1.0 and vr2 < 1.0:
            score += 15; det.append(f"搓揉期间缩量(量比{vr1:.2f}/{vr2:.2f}) +15")
        triggered = score >= int(self.th["whipsaw"])
        return self._result(7, triggered, score, det, 78 if triggered else 0)

    # ------------------------------------------------------------ 模型8 缩量缩波

    def m8_shrink_volatility(self, b: pd.DataFrame, i: int) -> dict:
        score, det = 0, []
        if i < 29:
            return self._result(8, False, 0, ["历史数据不足"], 0)
        vols = [self._v(b, "volume", k) for k in range(i - 4, i + 1)]
        vols_prev = [self._v(b, "volume", k) for k in range(i - 24, i - 5 + 1)]
        if all(v is not None for v in vols) and all(v is not None for v in vols_prev):
            r5 = float(np.mean(vols)) / float(np.mean(vols_prev))
            if r5 <= 0.4:
                score += 30; det.append(f"近5日量能萎缩至{r5:.0%} ≤40% +30")
            elif r5 <= 0.6:
                score += 20; det.append(f"近5日量能萎缩至{r5:.0%} ≤60% +20")
            elif r5 <= 0.8:
                score += 10; det.append(f"近5日量能萎缩至{r5:.0%} ≤80% +10")
        a_cur = self._amp(b, i - 4, i)
        a_prev = self._amp(b, i - 9, i - 5)
        if a_cur is not None and a_prev is not None and a_prev > 0:
            ratio = a_cur / a_prev
            if ratio <= 0.4:
                score += 25; det.append(f"波幅收窄至{ratio:.0%} ≤40% +25")
            elif ratio <= 0.7:
                score += 15; det.append(f"波幅收窄至{ratio:.0%} ≤70% +15")
        su, sl, sm = (self._v(b, c, i) for c in ("short_upper", "short_lower", "short_middle"))
        mu, ml, mm = (self._v(b, c, i) for c in ("mid_upper", "mid_lower", "mid_middle"))
        close = self._v(b, "close", i)
        if None not in (mu, ml, mm) and (mu - ml) / mm * 100 <= self.cfg["ma_band"]["band_tolerance_pct"]:
            score += 15; det.append("中期带粘合 +15")
        if None not in (su, sl, sm) and (su - sl) / sm * 100 <= self.cfg["ma_band"]["band_tolerance_pct"]:
            score += 10; det.append("短期带粘合 +10")
        if None not in (ml, mu, close) and ml <= close <= mu:
            score += 10; det.append("股价在带内 +10")
        triggered = score >= int(self.th["shrink_volatility"])
        return self._result(8, triggered, score, det, 80 if triggered else 0)

    def _amp(self, b: pd.DataFrame, j1: int, j2: int) -> Optional[float]:
        """[j1,j2] 区间振幅：(最高-最低)/区间末收盘。"""
        if j1 < 0 or j2 >= len(b):
            return None
        highs = [self._v(b, "high", k) for k in range(j1, j2 + 1)]
        lows = [self._v(b, "low", k) for k in range(j1, j2 + 1)]
        c = self._v(b, "close", j2)
        if any(v is None for v in highs + lows) or c is None or c <= 0:
            return None
        return (max(highs) - min(lows)) / c * 100

    # ------------------------------------------------------------ 模型9 岛型反转

    def m9_island_reversal(self, b: pd.DataFrame, i: int) -> dict:
        score, det = 0, []
        close = self._v(b, "close", i)
        ml = self._v(b, "mid_lower", i)
        mode = None

        def gap_size(j: int, up: bool) -> Optional[float]:
            lo, hi = self._v(b, "low", j), self._v(b, "high", j)
            p_hi, p_lo = self._v(b, "high", j - 1), self._v(b, "low", j - 1)
            if None in (lo, hi, p_hi, p_lo) or p_hi <= 0 or p_lo <= 0:
                return None
            if up and lo > p_hi:
                return (lo - p_hi) / p_hi * 100     # 向上缺口
            if (not up) and hi < p_lo:
                return (p_lo - hi) / p_lo * 100     # 向下缺口
            return None

        # 底部岛型：下跌缺口后3-15天出现上涨缺口
        for jd in range(max(1, i - 15), i + 1):
            d_gap = gap_size(jd, up=False)
            if d_gap is None:
                continue
            for ju in range(jd + 3, min(jd + 16, i + 1)):
                u_gap = gap_size(ju, up=True)
                if u_gap is None:
                    continue
                mode = "底部岛型反转"
                score += 35; det.append(f"下跌缺口({str(b['date'].iloc[jd])[:10]}, {d_gap:.1f}%)后{ju-jd}天出现上涨缺口 +35")
                if d_gap >= 2 and u_gap >= 2:
                    score += 15; det.append("两缺口均≥2% +15")
                elif d_gap >= 1 and u_gap >= 1:
                    score += 8; det.append("两缺口均≥1% +8")
                if close is not None and ml is not None and close < ml:
                    score += 15; det.append("股价在中期带下方 +15")
                vr = self._v(b, "vol_ratio", ju)
                if vr is not None and vr >= 1.5:
                    score += 10; det.append(f"右侧缺口放量(量比{vr:.2f}) +10")
                break
            if mode:
                break

        # 顶部岛型：上涨缺口后3-15天出现下跌缺口
        if not mode:
            for ju in range(max(1, i - 15), i + 1):
                u_gap = gap_size(ju, up=True)
                if u_gap is None:
                    continue
                for jd in range(ju + 3, min(ju + 16, i + 1)):
                    d_gap = gap_size(jd, up=False)
                    if d_gap is None:
                        continue
                    mode = "顶部岛型反转"
                    score += 30; det.append(f"上涨缺口({str(b['date'].iloc[ju])[:10]}, {u_gap:.1f}%)后{jd-ju}天出现下跌缺口 +30")
                    if d_gap >= 2 and u_gap >= 2:
                        score += 10; det.append("两缺口均≥2% +10")
                    break
                if mode:
                    break

        if not mode:
            return self._result(9, False, 0, ["近15日无岛型缺口结构"], 0)
        triggered = score >= int(self.th["island_reversal"])
        return self._result(9, triggered, score, det, 90 if triggered else 0)
