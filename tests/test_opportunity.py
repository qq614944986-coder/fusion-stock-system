# -*- coding: utf-8 -*-
"""双轨机会引擎单元测试（anomaly_radar / expectation_gap / main_wave_upgrade / track_signals / opportunity_card）。"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.anomaly_radar import detect_anomalies
from core.expectation_gap import evaluate as eg_evaluate
from core.main_wave_upgrade import evaluate as mw_evaluate
from core.track_signals import classify, TRACK_STOCKS
from core.opportunity_card import build_card

CFG = {"opportunity": {"universe_limit": 24}}


def _hist_low_volume():
    """低位居多但当日温和放量小阳：价格深跌到低分位，末根放量涨2%。"""
    n = 300
    idx = pd.date_range("2025-06-01", periods=n, freq="B")
    close = np.zeros(n)
    for i in range(n):
        if i < 240:
            close[i] = 100 - (100 - 32) * (i / 239) - np.sin(i / 25) * 2
        else:
            close[i] = 31 + np.sin(i / 8) * 1.2
    df = pd.DataFrame({"date": idx, "close": close})
    df.loc[df.index[-1], "close"] = float(close[-1]) * 1.02
    df["open"] = df["close"].shift(1).fillna(df["close"]).values
    df["high"] = df[["close", "open"]].max(axis=1) * 1.01
    df["low"] = df[["close", "open"]].min(axis=1) * 0.99
    df["volume"] = np.where(df.index >= n - 25, 3e7, 1e7)
    df.loc[df.index[-1], "volume"] = 3e7 * 2.0                    # 末根放量
    df["pct_chg"] = df["close"].pct_change().fillna(0) * 100
    df.loc[df.index[-1], "pct_chg"] = 2.0
    return df.reset_index(drop=True)


def _hist_low_quiet():
    """低位阶段缩量企稳：深跌后低位横盘，且近5日缩量至前20日的一半以下。"""
    n = 300
    idx = pd.date_range("2025-06-01", periods=n, freq="B")
    close = np.zeros(n)
    for i in range(n):
        if i < 240:
            close[i] = 100 - (100 - 32) * (i / 239) - np.sin(i / 25) * 2
        else:
            close[i] = 31 + np.sin(i / 8) * 0.8
    df = pd.DataFrame({"date": idx, "close": close})
    df["open"] = df["close"].shift(1).fillna(df["close"]).values
    df["high"] = df[["close", "open"]].max(axis=1) * 1.005
    df["low"] = df[["close", "open"]].min(axis=1) * 0.995
    df["volume"] = np.where(df.index >= n - 5, 4e6, 2e7)          # 近5日缩量
    df["pct_chg"] = df["close"].pct_change().fillna(0) * 100
    return df.reset_index(drop=True)


def _hist_bull():
    """上升趋势：板块共振、主升跟随样本。"""
    n = 200
    idx = pd.date_range("2025-09-01", periods=n, freq="B")
    base = np.linspace(10, 30, n)
    df = pd.DataFrame({"date": idx, "close": base})
    df["open"] = df["close"].shift(1).fillna(df["close"]).values
    df["high"] = df[["close", "open"]].max(axis=1) * 1.02
    df["low"] = df[["close", "open"]].min(axis=1) * 0.98
    df["volume"] = np.full(n, 2e7)
    df["pct_chg"] = df["close"].pct_change().fillna(0) * 100
    return df.reset_index(drop=True)


class AnomalyRadarTest(unittest.TestCase):
    def test_low_vol_rise_detected(self):
        hist = _hist_low_volume()
        ds = {"lhb_by_code": {}, "dzjy_by_code": {}, "yjyg_by_code": {}, "yjbb_by_code": {}}
        sigs = detect_anomalies("603019", "某股", hist, {"pct_chg": 2.0, "turnover": 8}, ds, CFG)
        names = {s["name"] for s in sigs}
        self.assertIn("低位温和放量", names)     # 深跌后低位 + 温和放量
        self.assertIn("深跌到位", names)         # 价格在低分位


class TrackClassifyTest(unittest.TestCase):
    def test_track_map(self):
        self.assertEqual(classify("600276"), "创新药")
        self.assertEqual(classify("601138"), "AI算力")
        self.assertEqual(classify("002027"), "互联网")
        self.assertEqual(classify("999999"), None)
        for t, stages in TRACK_STOCKS.items():
            total = sum(len(c) for c in stages.values())
            self.assertTrue(total > 0, f"{t} 无主板标的")


class ExpectationGapTest(unittest.TestCase):
    def test_passed_with_guide(self):
        hist = _hist_low_quiet()
        ds = {"code": "600276",
              "yjyg_by_code": {"600276": [{"预告类型": "预增", "业绩变动幅度": "50%"}]},
              "yjbb_by_code": {},
              "lhb_by_code": {"600276": [{"龙虎榜净买额": 5e7}]},
              "resfc_by_code": {}}
        r = eg_evaluate(hist, ds, {"belongs": True, "track": "创新药"}, CFG)
        self.assertTrue(r["anchors"]["est_ok"])         # 低位 + 预告预增
        self.assertTrue(r["anchors"]["trend_ok"])       # 赛道
        self.assertTrue(r["passed"])                    # ≥2 锚
        self.assertTrue(r["mild_left"])                 # 阶段缩量企稳 + 资金回补


class MainWaveTest(unittest.TestCase):
    def test_passes_with_gates(self):
        hist = _hist_bull()
        bctx = {"board": "光模块", "board_pct_chg": 2.0, "board_rank": 5,
                "board_zt_count": 3, "board_alpha": 8.0}
        r = mw_evaluate(hist, {"pct_chg": 3.0, "turnover": 12}, bctx, "AI算力", CFG)
        self.assertTrue(r["passes"])
        self.assertTrue(r["gates"]["板块趋势"])
        self.assertTrue(r["gates"]["产业锚"])


class OpportunityCardTest(unittest.TestCase):
    def test_dual_track_card(self):
        hist_pair = _hist_bull()
        eg = {"mild_left": True, "passed": True, "mild_left_note": "分位30%", 
              "anchors": {"est_ok": True, "forecast_ok": False, "trend_ok": True},
              "evidence": ["分位30%", "赛道"]}
        wave = {"passes": True, "gates": {"板块趋势": True, "量价健康": True, "乖离约束": True,
                                           "板块共振": True, "产业锚": True}, "score": 100}
        card = build_card("600276", "恒瑞医药", "创新药", [{"kind": "资金", "name": "龙虎榜机构净买", "value": "0.5亿", "note": ""}],
                          eg, wave, CFG)
        self.assertEqual(card["exec_kind"], "双轨")
        self.assertIn("温和布局", card["exec_sig"])
        self.assertEqual(card["track"], "创新药")


if __name__ == "__main__":
    unittest.main()