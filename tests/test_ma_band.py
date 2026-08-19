# -*- coding: utf-8 -*-
"""A1 均线带计算 / A2 BIAS60：手工构造序列精确验证（规格书 §6.1-6.2）。"""
import unittest

from core.data_provider import load_config
from core.ma_band_v2 import MABandV2
from tests.helpers import make_df

CFG = load_config()


class TestMABandA1(unittest.TestCase):
    """A1：close[i]=100+2i（70日）手工验证 MA5/8/13/55/60/65 与带上下沿/中轴。"""

    @classmethod
    def setUpClass(cls):
        closes = [100 + 2 * i for i in range(70)]
        vols = [10000.0] * 69 + [15000.0]
        cls.b = MABandV2(CFG).compute(make_df(closes, vols=vols))
        cls.i = 69

    def _v(self, col):
        return float(self.b[col].iloc[self.i])

    def test_ma_values_exact(self):
        # 等差数列 MA_n(i) = 100 + 2i - (n-1)，手工可验
        self.assertAlmostEqual(self._v("MA5"), 100 + 138 - 4, places=9)    # 234
        self.assertAlmostEqual(self._v("MA8"), 100 + 138 - 7, places=9)    # 231
        self.assertAlmostEqual(self._v("MA13"), 100 + 138 - 12, places=9)  # 226
        self.assertAlmostEqual(self._v("MA55"), 100 + 138 - 54, places=9)  # 184
        self.assertAlmostEqual(self._v("MA60"), 100 + 138 - 59, places=9)  # 179
        self.assertAlmostEqual(self._v("MA65"), 100 + 138 - 64, places=9)  # 174

    def test_band_bounds_and_middle(self):
        # 短期带：upper=max lower=min middle=mean
        self.assertAlmostEqual(self._v("short_upper"), 234.0, places=9)
        self.assertAlmostEqual(self._v("short_lower"), 226.0, places=9)
        self.assertAlmostEqual(self._v("short_middle"), (234 + 231 + 226) / 3, places=9)
        # 中期带（生命线）：MA55/60/65
        self.assertAlmostEqual(self._v("mid_upper"), 184.0, places=9)
        self.assertAlmostEqual(self._v("mid_lower"), 174.0, places=9)
        self.assertAlmostEqual(self._v("mid_middle"), (184 + 179 + 174) / 3, places=9)

    def test_vol_ratio(self):
        # vols = 10000×69 + 15000 → 近20日均量 = (19×10000+15000)/20 = 10250
        self.assertAlmostEqual(self._v("vol_ma20"), 10250.0, places=9)
        self.assertAlmostEqual(self._v("vol_ratio"), 15000.0 / 10250.0, places=9)

    def test_ma_not_ready_before_window(self):
        # MA65 需65个数据点：索引0-63为NaN，索引64起有效
        self.assertTrue(self.b["MA65"].iloc[64:].notna().all())
        self.assertTrue(self.b["MA65"].iloc[:64].isna().all())


class TestJudgeA1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        closes = [100 + 2 * i for i in range(70)]
        cls.eng = MABandV2(CFG)
        cls.b = cls.eng.compute(make_df(closes))
        cls.j = cls.eng.judge(cls.b, 69)

    def test_bull_trend(self):
        # 短期带下沿226 > 中期带上沿184；双中轴均上行 → 多头排列
        self.assertTrue(self.j["多头排列"])
        self.assertFalse(self.j["空头排列"])

    def test_adhesion_false_price_outside(self):
        # 粘合度 (234-226)/230.33=3.5% 与 (184-174)/179=5.6% 均 >1.5%
        self.assertFalse(self.j["短期带粘合"])
        self.assertFalse(self.j["中期带粘合"])
        self.assertFalse(self.j["价格在带内"])  # 238 不在 [174,184]

    def test_dist_to_mid_upper(self):
        self.assertAlmostEqual(self.j["距中期带上沿pct"], (238 - 184) / 184 * 100, places=6)


class TestBIAS60A2(unittest.TestCase):
    """A2：BIAS60 公式与区间判定。"""

    def test_bias60_formula(self):
        closes = [100.0] * 69 + [74.0]
        b = MABandV2(CFG).compute(make_df(closes))
        ma60 = float(b["MA60"].iloc[69])
        self.assertAlmostEqual(ma60, (59 * 100 + 74) / 60, places=9)
        self.assertAlmostEqual(float(b["bias60"].iloc[69]),
                               (74 - ma60) / ma60 * 100, places=9)

    def test_bias_zones_boundaries(self):
        eng = MABandV2(CFG)
        self.assertEqual(eng.bias_zone(50), "严重超买")
        self.assertEqual(eng.bias_zone(40), "严重超买")
        self.assertEqual(eng.bias_zone(39.9), "超买")
        self.assertEqual(eng.bias_zone(25), "超买")
        self.assertEqual(eng.bias_zone(24.9), "正常")
        self.assertEqual(eng.bias_zone(-24.9), "正常")
        self.assertEqual(eng.bias_zone(-25), "超卖")
        self.assertEqual(eng.bias_zone(-39.9), "超卖")
        self.assertEqual(eng.bias_zone(-40), "严重超卖")


if __name__ == "__main__":
    unittest.main()
