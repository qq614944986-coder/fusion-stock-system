# -*- coding: utf-8 -*-
"""A7 九大交易模型：每个模型至少1组构造数据触发（规格书 §6.5）。

重点：极值反转、N式双涨停、岛型反转的评分累加逐项验证（分数手工验算）。
"""
import unittest

import pandas as pd

from core.data_provider import load_config
from core.laofan_signals import StockState
from core.laofan_models import LaofanModels, limit_up_pct
from core.ma_band_v2 import MABandV2
from tests.helpers import make_df

CFG = load_config()


class MBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = LaofanModels(CFG)
        cls.band_eng = MABandV2(CFG)


class TestM1ExtremeReversal(MBase):
    """M1 极值反转：跌停后涨停+30 + 涨停量比≥1.5+15 + 跌停后新低+10 + 60日线下+10 + 中期带下+10 = 75。"""

    def test_trigger_with_score_accumulation(self):
        # 70日平台100 → 跌停(90)→续跌(85)→涨停(93.5, 量比1.9)
        closes = [100.0] * 70 + [90.0, 85.0, 93.5]
        vols = [10000.0] * 72 + [20000.0]
        b = self.band_eng.compute(make_df(closes, vols=vols))
        r = self.models.m1_extreme_reversal(b, len(b) - 1, "600000")
        self.assertTrue(r["triggered"], msg=str(r["details"]))
        self.assertEqual(r["score"], 30 + 15 + 10 + 10 + 10)
        self.assertEqual(r["confidence"], 95)

    def test_no_limitdown_no_trigger(self):
        b = self.band_eng.compute(make_df([100.0] * 70 + [101.0, 102.0]))
        r = self.models.m1_extreme_reversal(b, len(b) - 1, "600000")
        self.assertFalse(r["triggered"])


class TestM2M3M4Linked(MBase):
    """M2/M3/M4 与买点联动：BP2/BP3/BP1 直接驱动。"""

    def test_m2_bp2_link(self):
        r = self.models.m2_breakout(pd.DataFrame(), {"BP2"})
        self.assertTrue(r["triggered"])
        self.assertEqual(r["confidence"], 95)          # 90+5
        self.assertFalse(self.models.m2_breakout(pd.DataFrame(), set())["triggered"])

    def test_m3_bp3_link(self):
        r = self.models.m3_pullback(pd.DataFrame(), {"BP3"})
        self.assertTrue(r["triggered"])
        self.assertEqual(r["confidence"], 97)          # 92+5

    def test_m4_bp1_and_bias_fallback(self):
        # BP1 联动：置信度 85+3=88
        r = self.models.m4_bias_buy(pd.DataFrame(), 0, {"BP1"})
        self.assertTrue(r["triggered"])
        self.assertEqual(r["confidence"], 88)
        # BP1 未触发但 BIAS60≤-20%：置信度55
        b = self.band_eng.compute(make_df([100.0] * 70 + [78.0]))   # BIAS60=-22%
        r2 = self.models.m4_bias_buy(b, len(b) - 1, set())
        self.assertTrue(r2["triggered"])
        self.assertEqual(r2["confidence"], 55)
        # BIAS60 正常区间：不触发（回归 m4 正常文案分支）
        b3 = self.band_eng.compute(make_df([100.0] * 70 + [99.0]))
        r3 = self.models.m4_bias_buy(b3, len(b3) - 1, set())
        self.assertFalse(r3["triggered"])
        self.assertIn("BIAS60", r3["details"][0])


class TestM5Pathfinder(MBase):
    """M5 探路尖兵：近5日涨幅>5%+20 + 量比≥1.5+15 + 突破短期带+15 = 50≥45。"""

    def test_trigger(self):
        closes = [100.0] * 70 + [101.0, 102.5, 104.0, 105.5, 107.0]   # 近5日+7%
        vols = [10000.0] * 70 + [20000.0] * 5                          # 量比≈1.6
        b = self.band_eng.compute(make_df(closes, vols=vols))
        r = self.models.m5_pathfinder(b, len(b) - 1)
        self.assertTrue(r["triggered"], msg=str(r["details"]))
        self.assertGreaterEqual(r["score"], 45)
        self.assertEqual(r["confidence"], 75)


class TestM6NDoubleLimitup(MBase):
    """M6 N式倍量双涨停：2涨停+30 + 间隔≥3天+15 + 第二板量比≥2.0+15 + 带下方+20 + 今日+5 = 85。"""

    def test_trigger_with_score(self):
        # 平台100 → 深跌至50区域 → 两个涨停(55/50=+10%, 61.6/56=+10%，间隔4日)
        closes = [100.0] * 70 + [55.0, 50.0, 55.0, 54.0, 56.0, 61.6]
        vols = [10000.0] * 70 + [8000.0, 7000.0, 15000.0, 9000.0, 10000.0, 22000.0]
        b = self.band_eng.compute(make_df(closes, vols=vols))
        i = len(b) - 1
        r = self.models.m6_n_double_limitup(b, i, "600000")
        self.assertTrue(r["triggered"], msg=str(r["details"]))
        self.assertEqual(r["score"], 30 + 15 + 15 + 20 + 5)
        self.assertEqual(r["confidence"], 88)

    def test_position_limit_excludes(self):
        # 股价超中期带上沿130% → 直接排除
        b = self.band_eng.compute(make_df([100.0] * 70 + [130.0, 143.0]))
        r = self.models.m6_n_double_limitup(b, len(b) - 1, "600000")
        self.assertFalse(r["triggered"])
        self.assertTrue(any("位置限制" in d for d in r["details"]))


class TestM7Whipsaw(MBase):
    """M7 异动搓揉线：先长上影后长下影+30 + 带附近+15 + 缩量+15 = 60≥40。"""

    def test_trigger(self):
        n = 72
        # 前一日长上影（开100.0收100.3高101.5低99.9：上影75%实体19%）
        # 今日长下影（开100.2收100.0高100.4低98.7：下影76%实体12%），均价位于带附近
        closes = [100.0] * (n - 2) + [100.3, 100.0]
        opens = [100.0] * (n - 2) + [100.0, 100.2]
        highs = [100.1] * (n - 2) + [101.5, 100.4]
        lows = [99.9] * (n - 2) + [99.9, 98.7]
        vols = [10000.0] * (n - 2) + [8000.0, 7000.0]     # 两日量比<1 缩量
        df = make_df(closes, vols=vols, opens=opens, highs=highs, lows=lows)
        b = self.band_eng.compute(df)
        r = self.models.m7_whipsaw(b, len(b) - 1)
        self.assertTrue(r["triggered"], msg=str(r["details"]))
        self.assertGreaterEqual(r["score"], 40)
        self.assertEqual(r["confidence"], 78)


class TestM8ShrinkVolatility(MBase):
    """M8 缩量缩波：量能萎缩+30 + 波幅收窄+25 + 中期带粘合+15 + 短期带粘合+10 ≥45。"""

    def test_trigger(self):
        n = 80
        closes = [100.0] * (n - 5) + [100.5] * 5
        vols = [20000.0] * (n - 5) + [4000.0] * 5          # 近5日/前20日均量≈0.2
        highs = [102.0] * (n - 10) + [103.0] * 5 + [100.6] * 5
        lows = [98.0] * (n - 10) + [97.0] * 5 + [100.0] * 5
        df = make_df(closes, vols=vols, highs=highs, lows=lows)
        b = self.band_eng.compute(df)
        r = self.models.m8_shrink_volatility(b, len(b) - 1)
        self.assertTrue(r["triggered"], msg=str(r["details"]))
        self.assertGreaterEqual(r["score"], 45)
        self.assertEqual(r["confidence"], 80)


class TestM9IslandReversal(MBase):
    """M9 岛型反转（底部）：下跌缺口后3天上涨缺口+35 + 两缺口均≥2%+15 + 带下方+15 + 放量+10 = 75。"""

    def test_bottom_island_trigger(self):
        # 平台100 → 下跌缺口(idx70: 高95.2<前低99.9，缺口4.7%) → 横盘 →
        # 上涨缺口(idx73: 低97.0>前高94.9，缺口2.2%)，收盘97.2<中期带下沿≈99.6
        closes = [100.0] * 70 + [94.0, 94.5, 94.8, 97.2]
        opens = [100.0] * 70 + [94.5, 94.2, 94.6, 97.5]
        highs = [100.1] * 70 + [95.2, 94.9, 95.0, 98.0]
        lows = [99.9] * 70 + [93.8, 94.0, 94.2, 97.0]
        vols = [10000.0] * 73 + [20000.0]                  # 右侧缺口量比≈1.9
        df = make_df(closes, vols=vols, opens=opens, highs=highs, lows=lows)
        b = self.band_eng.compute(df)
        r = self.models.m9_island_reversal(b, len(b) - 1)
        self.assertTrue(r["triggered"], msg=str(r["details"]))
        self.assertEqual(r["score"], 35 + 15 + 15 + 10)
        self.assertEqual(r["confidence"], 90)

    def test_top_island_detected(self):
        # 上涨缺口(idx70: 低104.8>前高100.1) 后3天下跌缺口(idx73: 高99<前低103.5)
        # 顶部计分：+30(结构) +10(两缺口均≥2%) = 40，按 §6.5 阈值45 不触发（顶部满分40<45）
        closes = [100.0] * 70 + [105.0, 104.5, 104.0, 98.0]
        opens = [100.0] * 70 + [105.5, 104.8, 104.2, 97.5]
        highs = [100.1] * 70 + [106.0, 105.3, 104.8, 99.0]
        lows = [99.9] * 70 + [104.8, 104.0, 103.5, 97.2]
        df = make_df(closes, vols=[10000.0] * 74, opens=opens, highs=highs, lows=lows)
        b = self.band_eng.compute(df)
        r = self.models.m9_island_reversal(b, len(b) - 1)
        self.assertTrue(any("上涨缺口" in d and "下跌缺口" in d for d in r["details"]),
                        msg=str(r["details"]))
        self.assertEqual(r["score"], 40)
        self.assertFalse(r["triggered"])

    def test_no_gap_no_trigger(self):
        b = self.band_eng.compute(make_df([100.0] * 75))
        r = self.models.m9_island_reversal(b, len(b) - 1)
        self.assertFalse(r["triggered"])


class TestEvaluateAll(MBase):
    def test_evaluate_all_returns_nine(self):
        b = self.band_eng.compute(make_df([100.0] * 80))
        st = StockState(code="600000", status="EMPTY")
        res = self.models.evaluate_all(b, st, signals=[], i=len(b) - 1, code="600000")
        self.assertEqual(len(res), 9)
        self.assertEqual([r["id"] for r in res], list(range(1, 10)))


class TestLimitUpPct(unittest.TestCase):
    def test_thresholds_by_board(self):
        self.assertEqual(limit_up_pct("600000"), 9.5)    # 主板±10%
        self.assertEqual(limit_up_pct("300750"), 19.5)   # 创业板±20%
        self.assertEqual(limit_up_pct("688981"), 19.5)   # 科创板±20%
        self.assertEqual(limit_up_pct("830799"), 29.5)   # 北交所±30%


if __name__ == "__main__":
    unittest.main()
