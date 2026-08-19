# -*- coding: utf-8 -*-
"""A8 情绪温度：六因子合成计算正确 + 五区边界（34/47/61/74/75）分类正确（规格书 §4）。"""
import unittest

import pandas as pd

from core.data_provider import load_config
from core.sentiment_engine import compute_sentiment, classify_zone, momentum_score, leader_score

CFG = load_config()


def make_data(rise=2000, fall=2000, flat=200, limit_up=50, limit_down=10,
              index_pct=None, zt_pool=None):
    return {
        "date": "2026-08-18",
        "rise": rise, "fall": fall, "flat": flat,
        "limit_up": limit_up, "limit_down": limit_down,
        "index_pct": index_pct if index_pct is not None else {"上证": 0.5, "创业板": 0.5, "科创50": 0.5},
        "zt_pool": zt_pool,
    }


class TestSixFactors(unittest.TestCase):
    """六因子数值计算（手工可验）+ 加权合成。"""

    @classmethod
    def setUpClass(cls):
        # 全部因子给定：上涨4200/(4200+700+100)=84%；涨跌停 80/(80+20)=80%
        # 指数温度 (1.0均值+5)/10×100=60%；涨停活跃度 4/5=80%（1只一字板排除）；龙头 4板→80
        cls.zt = pd.DataFrame({
            "lian_ban": [4, 2, 1, 1, 1],
            "open_times": [0, 0, 0, 0, 0],
            "first_seal_time": ["09:35", "10:05", "13:30", "14:20", "09:25"],
        })
        cls.res = compute_sentiment(
            make_data(rise=4200, fall=700, flat=100, limit_up=80, limit_down=20,
                      index_pct={"上证": 1.0, "创业板": 1.0, "科创50": 1.0}, zt_pool=cls.zt),
            CFG, prev_base_temp=None,
        )

    def test_factor_values(self):
        f = self.res["factors"]
        self.assertAlmostEqual(f["上涨占比"], 84.0, places=1)
        self.assertAlmostEqual(f["涨跌停比"], 80.0, places=1)
        self.assertAlmostEqual(f["指数温度"], 60.0, places=1)
        self.assertAlmostEqual(f["涨停活跃度"], 80.0, places=1)   # 4/5 非一字未炸板
        self.assertAlmostEqual(f["情绪龙头"], 80.0, places=1)    # 4板：90-(5-4)×10
        self.assertAlmostEqual(f["情绪驱动力"], 50.0, places=1)  # 前日缺失→中性50

    def test_weighted_temperature(self):
        # 前日缺失→驱动力取中性但权重正常参与：
        # 0.2×84 + 0.2×80 + 0.2×60 + 0.15×80 + 0.15×80 + 0.1×50 = 73.8
        self.assertAlmostEqual(self.res["temperature"], 73.8, places=1)

    def test_limit_down_zero_gives_100(self):
        res = compute_sentiment(make_data(limit_up=30, limit_down=0), CFG, None)
        self.assertAlmostEqual(res["factors"]["涨跌停比"], 100.0, places=1)

    def test_index_temp_clipped(self):
        res = compute_sentiment(make_data(index_pct={"上证": 8.0, "创业板": 8.0, "科创50": 8.0}), CFG, None)
        self.assertAlmostEqual(res["factors"]["指数温度"], 100.0, places=1)   # (8+5)/10=130→截断100

    def test_missing_data_degrades_honestly(self):
        res = compute_sentiment({"date": "2026-08-18"}, CFG, None)
        self.assertEqual(res["temperature"], 50.0)
        self.assertTrue(any("数据缺失" in m for m in res["missing"]))


class TestMomentum(unittest.TestCase):
    """情绪驱动力映射五档。"""

    def test_mapping(self):
        self.assertEqual(momentum_score(6.0)[0], 90.0)
        self.assertEqual(momentum_score(3.0)[0], 70.0)
        self.assertEqual(momentum_score(0.0)[0], 50.0)
        self.assertEqual(momentum_score(-5.0)[0], 30.0)
        self.assertEqual(momentum_score(-9.0)[0], 10.0)
        self.assertEqual(momentum_score(None)[0], 50.0)


class TestLeader(unittest.TestCase):
    def test_leader_scores(self):
        # ≥5板→90；4板→80（每降1板-10）；无连板→30；最高板炸板再-20
        z5 = pd.DataFrame({"lian_ban": [6, 2], "open_times": [0, 0]})
        self.assertEqual(leader_score(z5)[0], 90.0)
        z4 = pd.DataFrame({"lian_ban": [4, 1], "open_times": [0, 0]})
        self.assertEqual(leader_score(z4)[0], 80.0)
        zn = pd.DataFrame({"lian_ban": [1, 1], "open_times": [0, 0]})
        self.assertEqual(leader_score(zn)[0], 30.0)
        zb = pd.DataFrame({"lian_ban": [4, 1], "open_times": [2, 0]})
        self.assertEqual(leader_score(zb)[0], 60.0)      # 80-20


class TestZoneBoundaries(unittest.TestCase):
    """A8 核心：五区边界 34/47/61/74/75 分类正确。"""

    def test_boundaries(self):
        cases = [
            (0, "冰点区"), (34, "冰点区"), (35, "退潮区"), (47, "退潮区"),
            (48, "震荡区"), (61, "震荡区"), (62, "偏强区"), (74, "偏强区"),
            (75, "高热区"), (100, "高热区"),
        ]
        for temp, expect in cases:
            z = classify_zone(temp, CFG["zones"])
            self.assertEqual(z["name"], expect, msg=f"temp={temp}")

    def test_zone_position_caps(self):
        caps = {z["name"]: z["max_total"] for z in CFG["zones"]}
        self.assertEqual(caps, {"冰点区": 20, "退潮区": 40, "震荡区": 50, "偏强区": 70, "高热区": 50})


if __name__ == "__main__":
    unittest.main()
