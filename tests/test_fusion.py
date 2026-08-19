# -*- coding: utf-8 -*-
"""A9 融合规则：7.1 冲突矩阵全部15格 + 7.3 仓位取 min 逻辑 + 7.2 卖出双轨 + 7.4 建议模板（规格书 §7）。"""
import unittest

from core.data_provider import load_config
from core import fusion_engine as fe

CFG = load_config()


class TestBuyMatrix15Cells(unittest.TestCase):
    """7.1 情绪×买点冲突矩阵：15格全部有测试用例。"""

    def test_all_15_cells_defined(self):
        n = 0
        for zone in fe.ZONE_NAMES:
            for bp in fe.BP_TYPES:
                r = fe.fuse_buy(zone, bp)          # 未覆盖组合会 KeyError
                self.assertIn(r["action"], {"允许", "降级允许", "降级观察", "条件允许",
                                            "正常执行", "冻结", "禁止"})
                n += 1
        self.assertEqual(n, 15)

    def test_bingdian_freezes_bp2_bp3(self):
        self.assertEqual(fe.fuse_buy("冰点区", "BP1")["action"], "允许")
        self.assertEqual(fe.fuse_buy("冰点区", "BP2")["action"], "冻结")
        self.assertEqual(fe.fuse_buy("冰点区", "BP3")["action"], "冻结")

    def test_tuichaodowngrades(self):
        self.assertEqual(fe.fuse_buy("退潮区", "BP1")["action"], "降级允许")
        self.assertEqual(fe.fuse_buy("退潮区", "BP2")["action"], "降级观察")
        self.assertEqual(fe.fuse_buy("退潮区", "BP3")["action"], "条件允许")

    def test_zhendang_all_normal(self):
        for bp in fe.BP_TYPES:
            self.assertEqual(fe.fuse_buy("震荡区", bp)["action"], "正常执行")

    def test_pianqiang_all_normal(self):
        for bp in fe.BP_TYPES:
            self.assertEqual(fe.fuse_buy("偏强区", bp)["action"], "正常执行")

    def test_gaore_bans_all(self):
        """高热区禁新仓铁律（§4.2 + §7.1）。"""
        for bp in fe.BP_TYPES:
            r = fe.fuse_buy("高热区", bp)
            self.assertEqual(r["action"], "禁止")
            self.assertIn("禁新仓", r["note"])

    def test_unknown_cell_raises(self):
        with self.assertRaises(KeyError):
            fe.fuse_buy("未知区", "BP1")


class TestPositionFusion(unittest.TestCase):
    """7.3 仓位融合：min(信号仓位, 情绪单票上限, 30%)。"""

    def setUp(self):
        self.zones = {z["name"]: z for z in CFG["zones"]}

    def test_min_logic(self):
        # 冰点区单票上限5%：BP1信号30% → min(30,5,30)=5
        r = fe.fuse_position(30, self.zones["冰点区"])
        self.assertEqual(r["final"], 5.0)
        # 偏强区单票上限15%：BP2信号40% → min(40,15,30)=15
        r = fe.fuse_position(40, self.zones["偏强区"])
        self.assertEqual(r["final"], 15.0)
        # 震荡区单票上限12%：BP3信号30% → min(30,12,30)=12
        r = fe.fuse_position(30, self.zones["震荡区"])
        self.assertEqual(r["final"], 12.0)

    def test_hard_cap_30(self):
        # 情绪上限放大到50（模拟）也不会超硬上限30
        zone = {"name": "测试", "max_total": 100, "max_single": 50, "max_count": 5}
        self.assertEqual(fe.fuse_position(40, zone)["final"], 30.0)

    def test_signal_lower_than_caps(self):
        zone = self.zones["偏强区"]                     # 上限15
        self.assertEqual(fe.fuse_position(10, zone)["final"], 10.0)

    def test_total_position_constraint(self):
        z = self.zones["退潮区"]                        # 总上限40，持仓上限3
        r = fe.check_total_position([15, 15, 15], z)    # Σ45>40，3只≤3
        self.assertFalse(r["total_ok"])
        self.assertTrue(r["count_ok"])
        r2 = fe.check_total_position([10, 10, 10, 10], z)   # Σ40≤40，4只>3
        self.assertTrue(r2["total_ok"])
        self.assertFalse(r2["count_ok"])

    def test_holdings_capped_at_5(self):
        """持仓数量 ≤ min(情绪区间上限, 5只)。"""
        z = {"name": "高上限", "max_total": 100, "max_single": 30, "max_count": 9}
        r = fe.check_total_position([10] * 6, z)
        self.assertFalse(r["count_ok"])
        self.assertEqual(r["cap_count"], 5)


class TestSellRules(unittest.TestCase):
    """7.2 卖出双轨监控统一优先级 P0>P1>P2。"""

    def _sent(self, temp):
        return {"temperature": temp}

    def test_li_stop_loss_p0(self):
        st = {"code": "600000", "name": "测试", "cost": 10.0, "price": 9.4, "horizon": "短线"}
        alerts = fe.evaluate_sell_rules(st, self._sent(60), None, CFG)
        p0 = [a for a in alerts if a["priority"] == "P0"]
        self.assertTrue(any(a["signal"] == "硬止损" for a in p0))       # -6%≤-5%

    def test_swing_stop_loss_uses_minus8(self):
        st = {"code": "600000", "name": "测试", "cost": 10.0, "price": 9.3, "horizon": "波段"}
        alerts = fe.evaluate_sell_rules(st, self._sent(60), None, CFG)
        # -7% > -8% 不触发硬止损
        self.assertFalse(any(a["signal"] == "硬止损" for a in alerts))

    def test_break_ma20_p0(self):
        st = {"code": "600000", "name": "测试", "cost": 10.0, "price": 9.8,
              "ma20": 10.0, "horizon": "波段"}
        alerts = fe.evaluate_sell_rules(st, self._sent(60), None, CFG)
        self.assertTrue(any(a["signal"] == "跌破MA20" and a["priority"] == "P0" for a in alerts))

    def test_score_drop_p1(self):
        st = {"code": "600000", "name": "测试", "cost": 10.0, "price": 12.0,
              "score": 60, "peak_score": 80}                          # 降20分>15
        alerts = fe.evaluate_sell_rules(st, self._sent(60), None, CFG)
        self.assertTrue(any(a["signal"] == "信号卖出(评分滑坡)" and a["priority"] == "P1" for a in alerts))

    def test_board_drop_p1(self):
        st = {"code": "600000", "name": "测试", "cost": 10.0, "price": 12.0, "board_rank": 12}
        alerts = fe.evaluate_sell_rules(st, self._sent(60), None, CFG)
        self.assertTrue(any(a["signal"] == "信号卖出(板块滑坡)" for a in alerts))

    def test_sentiment_crash_p1(self):
        st = {"code": "600000", "name": "测试", "cost": 10.0, "price": 10.5}
        alerts = fe.evaluate_sell_rules(st, self._sent(62), 80, CFG)   # 前日80高热→62，回落18>10
        self.assertTrue(any(a["signal"] == "情绪卖出(情绪崩塌)" and a["priority"] == "P1" for a in alerts))

    def test_take_profit_p2(self):
        st = {"code": "600000", "name": "测试", "cost": 10.0, "price": 11.8, "horizon": "短线"}
        alerts = fe.evaluate_sell_rules(st, self._sent(60), None, CFG)
        self.assertTrue(any(a["signal"] == "止盈卖出" and a["priority"] == "P2" for a in alerts))  # +18%≥15%

    def test_priority_ordering(self):
        """多信号时 P0 排在最前。"""
        class Sig:
            def __init__(self, t, r, p):
                self.type, self.reason, self.action_pct = t, r, p
        st = {"code": "600000", "name": "测试", "cost": 10.0, "price": 8.9,
              "horizon": "短线",
              "laofan_sells": [Sig("SP3", "跌破中期带下沿", 100), Sig("SP2", "跌破短期带", 50)]}
        alerts = fe.evaluate_sell_rules(st, self._sent(60), None, CFG)
        self.assertEqual(alerts[0]["priority"], "P0")
        prios = [a["priority"] for a in alerts]
        self.assertEqual(prios, sorted(prios, key=lambda p: {"P0": 0, "P1": 1, "P2": 2}[p]))


class TestAdviceTemplate(unittest.TestCase):
    """7.4 条件化建议：正向条件 + 反向不执行条件必含。"""

    def test_advice_contains_reverse_conditions(self):
        st = {"code": "603019", "name": "中科曙光", "score": 78, "board": "算力",
              "board_rank": 3, "laofan_summary": "BUILDING · 多头排列 · BIAS60=+12.3%"}
        sent = {"temperature": 62}
        zone = {"name": "偏强区", "max_total": 70, "max_single": 15, "max_count": 5}
        buy = fe.fuse_buy("偏强区", "BP2")
        pos = fe.fuse_position(40, zone)
        txt = fe.build_advice(st, sent, zone, buy, pos)
        self.assertIn("【中科曙光 603019】", txt)
        self.assertIn("建议", txt)
        self.assertIn("反向条件", txt)
        self.assertIn("不追", txt)                      # 反向：高开>5%不追
        self.assertIn("SP2", txt)                       # 反向：破短期带减仓
        self.assertIn("SP3", txt)                       # 反向：破中期带清仓

    def test_advice_banned_signal_notes_reason(self):
        st = {"code": "600000", "name": "测试"}
        sent = {"temperature": 80}
        zone = {"name": "高热区", "max_total": 50, "max_single": 12, "max_count": 4}
        txt = fe.build_advice(st, sent, zone, fe.fuse_buy("高热区", "BP2"), None)
        self.assertIn("禁止", txt)
        self.assertIn("反向条件", txt)


class TestT0Synergy(unittest.TestCase):
    """6.8 T0协同矩阵抽查 + 乖离风险修正。"""

    def test_matrix_spot_checks(self):
        self.assertEqual(fe.t0_synergy("多头趋势", "low_absorb")["action"], "加仓低吸")
        self.assertEqual(fe.t0_synergy("空头趋势", "low_absorb")["confidence"], 35)   # 50-15
        self.assertEqual(fe.t0_synergy("空头趋势", "high_throw")["confidence"], 65)   # 50+15
        self.assertEqual(fe.t0_synergy("震荡整理", "none")["confidence"], 55)

    def test_bias_risk_correction(self):
        # BIAS60≥+50%：低吸置信度-20（60+15-20=55）；高抛+10（60-10+10=60）
        self.assertEqual(fe.t0_synergy("多头趋势", "low_absorb", bias60=55)["confidence"], 55)
        self.assertEqual(fe.t0_synergy("多头趋势", "high_throw", bias60=55)["confidence"], 60)
        # 无乖离时不修正
        self.assertEqual(fe.t0_synergy("多头趋势", "low_absorb", bias60=10)["confidence"], 75)

    def test_confidence_bounds(self):
        self.assertLessEqual(fe.t0_synergy("多头趋势", "low_absorb")["confidence"], 95)
        self.assertGreaterEqual(fe.t0_synergy("空头趋势", "low_absorb")["confidence"], 35)


if __name__ == "__main__":
    unittest.main()
