# -*- coding: utf-8 -*-
"""A10 参数一致：config.yaml 默认值与规格书第11章逐项一致（抽查核对全部关键参数）。"""
import unittest
from pathlib import Path

from core.data_provider import load_config

BASE = Path(__file__).resolve().parent.parent
CFG = load_config(BASE / "config" / "config.yaml")


class TestLaofanParams(unittest.TestCase):
    """老樊引擎参数为精确校准值，逐字对照第11章（禁止偏差）。"""

    def test_ma_band(self):
        mb = CFG["ma_band"]
        self.assertEqual(mb["short_term"], [5, 8, 13])
        self.assertEqual(mb["mid_term"], [55, 60, 65])
        self.assertEqual(mb["bias_ma_period"], 60)
        self.assertEqual(mb["buy_bias_threshold"], -25)
        self.assertEqual(mb["sell_bias_threshold"], 25)
        self.assertEqual(mb["band_tolerance_pct"], 1.5)
        self.assertEqual(mb["vol_ma_period"], 20)

    def test_signal_cooldowns(self):
        sg = CFG["signals"]
        self.assertEqual(sg["buy_point_1_cooldown"], 20)
        self.assertEqual(sg["buy_point_2_cooldown"], 10)
        self.assertEqual(sg["buy_point_3_cooldown"], 10)
        self.assertEqual(sg["sell_point_1_cooldown"], 20)
        self.assertEqual(sg["sell_point_2_cooldown"], 5)
        self.assertEqual(sg["sell_point_3_cooldown"], 30)
        self.assertEqual(sg["exit_cooldown"], 15)

    def test_filters(self):
        fl = CFG["filters"]
        self.assertTrue(fl["breakout_volume_confirm"])
        self.assertEqual(fl["breakout_volume_ratio"], 1.5)
        self.assertEqual(fl["breakout_min_gain_pct"], 2.0)
        self.assertEqual(fl["breakout_confirm_bars"], 2)
        self.assertFalse(fl["breakdown_volume_confirm"])
        self.assertEqual(fl["breakdown_volume_ratio"], 1.3)
        self.assertEqual(fl["breakdown_min_loss_pct"], 2.0)
        self.assertEqual(fl["breakdown_confirm_bars"], 1)

    def test_positions(self):
        po = CFG["positions"]
        self.assertEqual(po["buy_point_1_position"], 30)
        self.assertEqual(po["buy_point_2_position"], 40)
        self.assertEqual(po["buy_point_3_position"], 30)
        self.assertEqual(po["sell_point_1_reduce"], 30)
        self.assertEqual(po["sell_point_2_reduce"], 50)
        self.assertEqual(po["sell_point_3_reduce"], 100)
        self.assertEqual(po["max_single_position"], 30)
        self.assertEqual(po["max_holdings"], 5)

    def test_model_thresholds(self):
        mt = CFG["model_thresholds"]
        self.assertEqual(mt["extreme_reversal"], 40)
        self.assertEqual(mt["pathfinder"], 45)
        self.assertEqual(mt["n_double_limitup"], 50)
        self.assertEqual(mt["whipsaw"], 40)
        self.assertEqual(mt["shrink_volatility"], 45)
        self.assertEqual(mt["island_reversal"], 45)

    def test_signal_priority(self):
        self.assertEqual(CFG["signal_priority"], ["SP3", "SP2", "SP1", "BP3", "BP2", "BP1"])


class TestLizhiyuanParams(unittest.TestCase):
    """李致远引擎权重与五区映射。"""

    def test_sentiment_weights(self):
        s = CFG["sentiment"]
        self.assertEqual(s["weight_rise_ratio"], 0.20)
        self.assertEqual(s["weight_limit_ratio"], 0.20)
        self.assertEqual(s["weight_index_temp"], 0.20)
        self.assertEqual(s["weight_zt_quality"], 0.15)
        self.assertEqual(s["weight_leader"], 0.15)
        self.assertEqual(s["weight_momentum"], 0.10)
        self.assertEqual(s["index_temp_range"], 5)
        self.assertAlmostEqual(sum(s[k] for k in s if k.startswith("weight_")), 1.0)

    def test_zones_table(self):
        expect = [
            {"name": "冰点区", "min": 0, "max": 34, "max_total": 20, "max_single": 5, "max_count": 2},
            {"name": "退潮区", "min": 35, "max": 47, "max_total": 40, "max_single": 10, "max_count": 3},
            {"name": "震荡区", "min": 48, "max": 61, "max_total": 50, "max_single": 12, "max_count": 4},
            {"name": "偏强区", "min": 62, "max": 74, "max_total": 70, "max_single": 15, "max_count": 5},
            {"name": "高热区", "min": 75, "max": 100, "max_total": 50, "max_single": 12, "max_count": 4},
        ]
        self.assertEqual([{k: z[k] for k in expect[0]} for z in CFG["zones"]], expect)

    def test_sector_stock_limitup_eod(self):
        self.assertEqual(CFG["sector"]["top_n_gate"], 10)
        self.assertEqual(CFG["sector"]["volume_ratio_good"], 1.5)
        self.assertEqual(CFG["sector"]["turnover_healthy"], [3, 15])
        self.assertEqual(CFG["stock_score"]["pool_threshold"], 70)
        self.assertEqual(CFG["stock_score"]["strong_threshold"], 85)
        self.assertEqual(CFG["stock_score"]["watch_threshold"], 55)
        self.assertEqual(CFG["stock_score"]["overheat_bias_ma20"], 19)
        self.assertEqual(CFG["limitup"]["score_threshold"], 60)
        self.assertEqual(CFG["limitup"]["max_open_times"], 1)
        self.assertEqual(CFG["limitup"]["high_position_pct"], 50)
        self.assertEqual(CFG["eod"]["gain_range"], [3, 9.5])
        self.assertEqual(CFG["eod"]["volume_ratio_min"], 1.2)
        self.assertEqual(CFG["eod"]["sector_top_n"], 10)

    def test_sell_rules(self):
        sr = CFG["sell_rules"]
        self.assertEqual(sr["stop_loss_short"], -5)
        self.assertEqual(sr["stop_loss_swing"], -8)
        self.assertEqual(sr["take_profit_short"], 15)
        self.assertEqual(sr["take_profit_swing"], 30)
        self.assertEqual(sr["score_drop_alert"], 15)
        self.assertEqual(sr["bias_ma20_take_profit"], 30)
        self.assertEqual(sr["sentiment_crash_drop"], 10)

    def test_run(self):
        self.assertEqual(CFG["run"]["data_dir"], "data")
        self.assertEqual(CFG["run"]["output_dir"], "output")
        self.assertIn(CFG["run"]["t0_signal"], ("none", "low_absorb", "high_throw"))


if __name__ == "__main__":
    unittest.main()
