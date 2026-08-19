# -*- coding: utf-8 -*-
"""A3 买点触发 / A4 卖点触发+EXITED拦截 / A5 状态机约束 / A6 冷却期（规格书 §6.3-6.7）。"""
import unittest

from core.data_provider import load_config
from core.laofan_signals import (ALLOWED_STATES, LaofanSignalEngine, POSITION_STATES,
                                 SIGNAL_PRIORITY, StockState)
from core.ma_band_v2 import MABandV2
from tests.helpers import make_df

CFG = load_config()


def flat_then(extra, base=100.0, n_flat=70, vols=None):
    closes = [base] * n_flat + list(extra)
    return make_df(closes, vols=vols)


class TestBP1(unittest.TestCase):
    """A2/A3：BIAS60≤-25% 首次触达 → BP1（置信72，仓位30%，状态约束）。"""

    @classmethod
    def setUpClass(cls):
        cls.eng = LaofanSignalEngine(CFG)
        closes = [100.0] * 70 + [73.0]
        cls.df = make_df(closes)
        cls.b = MABandV2(CFG).compute(cls.df)

    def test_bp1_triggers(self):
        st = StockState(code="600000", status="EMPTY")
        det = self.eng.detect(self.b, st, 70)
        self.assertEqual(len(det["signals"]), 1)
        s = det["signals"][0]
        self.assertEqual(s.type, "BP1")
        self.assertEqual(s.confidence, 72)
        self.assertEqual(s.action_pct, 30)
        self.assertEqual(s.cooldown_days, 20)
        # BP1 本身不被状态拦截（同日 SP3 因 EMPTY 状态被拦截属正常，见状态机测试）
        self.assertFalse(any(bl[0] == "BP1" for bl in det["blocked"]))

    def test_bp1_not_first_touch(self):
        # 前一日已 ≤-25%（未首次触达）→ 不触发
        closes = [100.0] * 70 + [73.0, 73.0]
        b = MABandV2(CFG).compute(make_df(closes))
        det = self.eng.detect(b, StockState(code="600000", status="EMPTY"), 71)
        self.assertEqual([s for s in det["signals"] if s.type == "BP1"], [])

    def test_bp1_state_constraint(self):
        # A5：BP1 仅允许 EMPTY/EXITED（EXITED 且不在离场冷却内）
        for status in ("BUILDING", "HOLDING", "REDUCING"):
            det = self.eng.detect(self.b, StockState(code="600000", status=status), 70)
            self.assertTrue(any(bl[0] == "BP1" for bl in det["blocked"]),
                            f"BP1在{status}应被拦截")

    def test_bp1_cooldown(self):
        # A6：20天冷却期内不重复触发
        st = StockState(code="600000", status="EMPTY")
        closes = [100.0] * 70 + [73.0, 73.0, 90.0, 70.0]
        b = MABandV2(CFG).compute(make_df(closes))
        st.last_signal_dates["BP1"] = str(b["date"].iloc[70])[:10]  # 3天前触发过
        det = self.eng.detect(b, st, 73)
        self.assertEqual(det["signals"], [])
        self.assertTrue(any(bl[0] == "BP1" and "冷却" in bl[1] for bl in det["blocked"]))


class TestBP2(unittest.TestCase):
    """A3：突破中期带上沿≥2% + 量比≥1.5 + 站稳2根K线（三重过滤）。"""

    @classmethod
    def setUpClass(cls):
        cls.eng = LaofanSignalEngine(CFG)
        cls.closes = [100.0] * 75 + [103.0, 103.0]
        cls.vols = [10000.0] * 75 + [20000.0, 20000.0]
        cls.b = MABandV2(CFG).compute(make_df(cls.closes, vols=cls.vols))

    def _sig(self, b=None, i=None, st=None):
        return self.eng.detect(self.b if b is None else b,
                               StockState(code="600000", status="EMPTY") if st is None else st,
                               76 if i is None else i)

    def test_bp2_triggers_on_second_bar(self):
        det = self._sig()
        bps = [s for s in det["signals"] if s.type == "BP2"]
        self.assertEqual(len(bps), 1)
        s = bps[0]
        self.assertEqual((s.confidence, s.action_pct, s.cooldown_days), (78, 40, 10))
        # 第1根突破日（i=75）不触发：需连续2日收于带上沿上方
        det75 = self._sig(i=75)
        self.assertEqual([s for s in det75["signals"] if s.type == "BP2"], [])

    def test_bp2_filter_no_volume(self):
        vols = [10000.0] * 77
        b = MABandV2(CFG).compute(make_df(self.closes, vols=vols))
        det = self._sig(b=b)
        self.assertEqual([s for s in det["signals"] if s.type == "BP2"], [])

    def test_bp2_filter_amplitude(self):
        closes = [100.0] * 75 + [101.0, 101.5]   # 突破幅度<2%
        b = MABandV2(CFG).compute(make_df(closes, vols=self.vols[:76] + [20000.0]))
        det = self._sig(b=b)
        self.assertEqual([s for s in det["signals"] if s.type == "BP2"], [])

    def test_bp2_state_and_cooldown(self):
        # A5：HOLDING/REDUCING 禁止；A6：10天冷却
        for status in ("HOLDING", "REDUCING"):
            det = self.eng.detect(self.b, StockState(code="600000", status=status), 76)
            self.assertTrue(any(bl[0] == "BP2" for bl in det["blocked"]))
        st = StockState(code="600000", status="EMPTY")
        st.last_signal_dates["BP2"] = str(self.b["date"].iloc[70])[:10]  # 6天前
        det = self.eng.detect(self.b, st, 76)
        self.assertTrue(any(bl[0] == "BP2" and "冷却" in bl[1] for bl in det["blocked"]))


class TestBP3(unittest.TestCase):
    """A3：多头排列 + 近30日BP2型突破 + >3天 + 距带0.5-3% + 缩量 + 仅BUILDING。"""

    @classmethod
    def setUpClass(cls):
        cls.eng = LaofanSignalEngine(CFG)
        band = MABandV2(CFG)
        closes = [50 + 0.52 * i for i in range(65)]          # 50→83.28 温和上行
        closes += [83.3] * 60                                 # 平台期（中期带收敛至83.3附近）
        closes += [86.5, 86.3, 86.1, 85.9]                    # 放量突破后小幅回落
        closes += [85.0]                                      # 今日回踩（定点迭代至带上沿+2%）
        vols = [10000.0] * 125 + [20000.0] + [10000.0] * 3 + [6000.0]
        for _ in range(6):                                    # 不动点迭代 close=mid_upper×1.02
            b = band.compute(make_df(closes, vols=vols))
            closes[-1] = round(float(b["mid_upper"].iloc[-1]) * 1.02, 4)
        cls.b = band.compute(make_df(closes, vols=vols))

    def test_bp3_triggers(self):
        j = MABandV2(CFG).judge(self.b)
        self.assertTrue(j["多头排列"])
        self.assertTrue(0.5 <= j["距中期带上沿pct"] <= 3.0)
        st = StockState(code="600000", status="BUILDING")
        det = self.eng.detect(self.b, st, -1)
        bps = [s for s in det["signals"] if s.type == "BP3"]
        self.assertEqual(len(bps), 1)
        s = bps[0]
        self.assertEqual((s.confidence, s.action_pct, s.cooldown_days), (82, 30, 10))

    def test_bp3_only_building(self):
        # A5：BP3 仅 BUILDING 允许
        for status in ("EMPTY", "HOLDING", "REDUCING", "EXITED"):
            det = self.eng.detect(self.b, StockState(code="600000", status=status), -1)
            self.assertTrue(any(bl[0] == "BP3" for bl in det["blocked"]),
                            f"BP3在{status}应被拦截")

    def test_bp3_filter_volume(self):
        # 回踩未缩量（量比≥0.8）→ 不触发
        vols = [10000.0] * 125 + [20000.0] + [10000.0] * 4
        b2 = MABandV2(CFG).compute(make_df(list(self.b["close"]), vols=vols))
        det = self.eng.detect(b2, StockState(code="600000", status="BUILDING"), -1)
        self.assertEqual([s for s in det["signals"] if s.type == "BP3"], [])


class TestSellPoints(unittest.TestCase):
    """A4：SP2/SP3 触发 + SP3后EXITED与15天拦截。"""

    @classmethod
    def setUpClass(cls):
        cls.eng = LaofanSignalEngine(CFG)

    def test_sp2_triggers(self):
        closes = [200 - 2 * i for i in range(65)]            # 200→72
        closes += [72.0] * 63 + [72.5, 71.0]                 # 平台→昨微升→今破短期带
        b = MABandV2(CFG).compute(make_df(closes))
        st = StockState(code="600000", status="BUILDING", position_pct=70)
        det = self.eng.detect(b, st, -1)
        sigs = [s.type for s in det["signals"]]
        self.assertIn("SP2", sigs)
        sp2 = next(s for s in det["signals"] if s.type == "SP2")
        self.assertEqual((sp2.confidence, sp2.action_pct, sp2.cooldown_days), (75, 50, 5))
        self.assertNotIn("SP3", sigs)  # 本序列只破短期带不破中期带≥2%

    def test_sp3_triggers_and_exit_block(self):
        closes = [100.0] * 79 + [97.5]
        b = MABandV2(CFG).compute(make_df(closes))
        st = StockState(code="600000", status="BUILDING", position_pct=60)
        det = self.eng.detect(b, st, 79)
        # SP3 优先级最高
        self.assertEqual(det["signals"][0].type, "SP3")
        sp3 = det["signals"][0]
        self.assertEqual((sp3.confidence, sp3.action_pct, sp3.cooldown_days), (88, 100, 30))
        # 执行 → EXITED，清仓
        st2 = self.eng.apply_signal(st, sp3, str(b["date"].iloc[79])[:10])
        self.assertEqual(st2.status, "EXITED")
        self.assertEqual(st2.position_pct, 0.0)
        self.assertIsNotNone(st2.last_exit_date)
        # 15天内 BP 信号被拦截（A4）：次日构造 BP1 条件
        closes2 = closes + [74.0]
        b2 = MABandV2(CFG).compute(make_df(closes2))
        det2 = self.eng.detect(b2, st2, 80)
        self.assertEqual(det2["signals"], [])
        self.assertTrue(any(bl[0] == "BP1" and "离场观察" in bl[1] for bl in det2["blocked"]))

    def test_exit_cooldown_expiry(self):
        # 冷却15天满 → EMPTY
        st = StockState(code="600000", status="EXITED", last_exit_date="2026-01-01")
        st = self.eng.refresh_exit_cooldown(st, "2026-01-16")   # 15天，未满
        self.assertEqual(st.status, "EXITED")
        st = self.eng.refresh_exit_cooldown(st, "2026-01-17")   # 16天>15 → EMPTY
        self.assertEqual(st.status, "EMPTY")

    def test_sp1_first_touch(self):
        # 温和上行后末日急拉：BIAS60 由 +18.2% 首次突破 +25%（→+41.8%）
        closes = [100.0] * 70 + [110.0, 112.0, 114.0, 116.0, 118.0, 120.0, 145.0]
        b = MABandV2(CFG).compute(make_df(closes))
        bias = float(b["bias60"].iloc[-1])
        prev = float(b["bias60"].iloc[-2])
        self.assertGreaterEqual(bias, 25.0)
        self.assertLess(prev, 25.0)                     # 首次触达
        st = StockState(code="600000", status="HOLDING", position_pct=100)
        det = self.eng.detect(b, st, -1)
        self.assertIn("SP1", [s.type for s in det["signals"]])
        # A5：EMPTY 禁止 SP1
        det2 = self.eng.detect(b, StockState(code="600000", status="EMPTY"), -1)
        self.assertTrue(any(bl[0] == "SP1" for bl in det2["blocked"]))


class TestStateMachineA5(unittest.TestCase):
    """A5：信号-状态约束表全覆盖 + 状态转移 + 优先级。"""

    def test_constraint_table_covers_all(self):
        for sig, allowed in ALLOWED_STATES.items():
            self.assertTrue(allowed & set(POSITION_STATES))
            for st in POSITION_STATES:
                if st not in allowed:
                    self.assertIn(st, {"BUILDING", "HOLDING", "REDUCING", "EMPTY", "EXITED"})
        self.assertEqual(set(ALLOWED_STATES), {"BP1", "BP2", "BP3", "SP1", "SP2", "SP3"})

    def test_priority_order(self):
        self.assertEqual(SIGNAL_PRIORITY, ["SP3", "SP2", "SP1", "BP3", "BP2", "BP1"])

    def test_transitions(self):
        eng = LaofanSignalEngine(CFG)
        from core.laofan_signals import Signal
        st = StockState(code="600000", status="EMPTY")
        bp1 = Signal("BP1", 72, "建仓30%", 30, 20, "", "买入")
        st = eng.apply_signal(st, bp1, "2026-01-01")
        self.assertEqual((st.status, st.position_pct), ("BUILDING", 30))
        bp2 = Signal("BP2", 78, "加仓40%", 40, 10, "", "买入")
        st = eng.apply_signal(st, bp2, "2026-01-02")
        self.assertEqual((st.status, st.position_pct), ("BUILDING", 70))
        bp3 = Signal("BP3", 82, "加仓30%至满仓", 30, 10, "", "买入")
        st = eng.apply_signal(st, bp3, "2026-01-03")
        self.assertEqual((st.status, st.position_pct), ("HOLDING", 100))
        sp2 = Signal("SP2", 75, "减仓50%", 50, 5, "", "卖出")
        st = eng.apply_signal(st, sp2, "2026-01-04")
        self.assertEqual((st.status, st.position_pct), ("REDUCING", 50))
        sp3 = Signal("SP3", 88, "清仓100%", 100, 30, "", "卖出")
        st = eng.apply_signal(st, sp3, "2026-01-05")
        self.assertEqual((st.status, st.position_pct, st.last_exit_date),
                         ("EXITED", 0.0, "2026-01-05"))


class TestCooldownsA6(unittest.TestCase):
    """A6：各信号冷却期边界（>cd 允许，≤cd 拦截）。"""

    def test_all_cooldown_boundaries(self):
        eng = LaofanSignalEngine(CFG)
        cds = {"BP1": 20, "BP2": 10, "BP3": 10, "SP1": 20, "SP2": 5, "SP3": 30}
        for sig, cd in cds.items():
            st = StockState(code="600000", status="BUILDING")
            st.last_signal_dates[sig] = "2026-01-01"
            ok, _ = eng._cooldown_ok(st, sig, f"2026-01-{1 + cd:02d}")   # 恰好cd天 → 拦截
            self.assertFalse(ok, f"{sig} {cd}天应仍在冷却期")
            ok2, _ = eng._cooldown_ok(st, sig, f"2026-01-{2 + cd:02d}")  # cd+1天 → 允许
            self.assertTrue(ok2, f"{sig} {cd + 1}天应已过冷却期")


if __name__ == "__main__":
    unittest.main()
