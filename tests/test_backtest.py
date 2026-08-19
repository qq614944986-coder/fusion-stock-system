# -*- coding: utf-8 -*-
"""A12 历史回放：合成行情注入回放器（离线），验证信号时间轴、净值曲线与状态机无死锁。

规格书 §9.2：任选3只真实股票近250日回放为人工抽查项（需联网，命令行执行）：
    python backtest.py --code 603019 --days 250
本测试用3段合成行情（对应3只股票）离线覆盖同一代码路径。
"""
import unittest
from pathlib import Path

from backtest import run_backtest
from core.laofan_signals import POSITION_STATES
from tests.helpers import make_df

BASE = Path(__file__).resolve().parent.parent


class FakeProvider:
    """离线数据源：直接返回合成日K（结构与 akshare 封装输出一致）。"""

    def __init__(self, df):
        self.df = df
        self.missing = []

    def get_stock_daily(self, code, days=None):
        d = self.df.tail(days) if days else self.df
        return d.reset_index(drop=True)


def scenario_breakout():
    """场景1：70日平台 → 深跌10日（BIAS60=-26.7% 触发BP1）→ 反弹10日。"""
    closes = [100.0] * 70 + [95.0, 90.0, 85.0, 80.0, 75.0, 72.0, 70.0, 68.0, 66.0, 64.0]
    closes += [66.0, 68.0, 71.0, 75.0, 80.0, 86.0, 93.0, 100.0, 105.0, 110.0]
    vols = [10000.0] * 70 + [8000.0] * 10
    vols += [12000.0, 13000.0, 15000.0, 17000.0, 19000.0, 21000.0, 24000.0, 27000.0, 30000.0, 33000.0]
    return make_df(closes, vols=vols)


def scenario_crash():
    """场景2：70日平台 → 深跌（BP1建仓）→ 反弹 → 二次崩跌破中期带（SP3清仓→EXITED）。"""
    closes = [100.0] * 70 + [95.0, 90.0, 85.0, 80.0, 75.0, 72.0]
    closes += [74.0, 76.0, 79.0, 83.0, 88.0, 94.0, 100.0]
    closes += [98.0, 95.0, 92.0, 88.0, 84.0, 80.0, 76.0, 72.0]
    return make_df(closes)


def scenario_quiet():
    """场景3：长期窄幅横盘，无信号。"""
    return make_df([100.0 + (i % 5) * 0.1 for i in range(120)])


class TestBacktest(unittest.TestCase):

    def _run(self, df, code):
        return run_backtest(code, days=250, base_dir=BASE, dp=FakeProvider(df))

    def test_breakout_scenario_bp1(self):
        res = self._run(scenario_breakout(), "600001")
        tl, sg = res["timeline"], res["signals"]
        self.assertEqual(len(tl), 90)
        # 深跌触发BP1（idx75，BIAS60首次≤-25%）
        types = list(sg["signal"]) if not sg.empty else []
        self.assertIn("BP1", types)
        # 状态机无死锁：所有状态合法，仓位∈[0,100]
        self.assertTrue(set(tl["state"]).issubset(set(POSITION_STATES)))
        self.assertTrue(((tl["position_pct"] >= 0) & (tl["position_pct"] <= 100)).all())
        # 净值曲线为正
        self.assertTrue((tl["equity"] > 0).all())

    def test_crash_scenario_sp3(self):
        res = self._run(scenario_crash(), "600002")
        tl, sg = res["timeline"], res["signals"]
        self.assertFalse(sg.empty)
        self.assertIn("BP1", list(sg["signal"]))     # 先建仓
        self.assertIn("SP3", list(sg["signal"]))     # 后破带清仓
        # SP3后进入EXITED，净值曲线连续为正
        self.assertIn("EXITED", set(tl["state"]))
        self.assertTrue((tl["equity"] > 0).all())

    def test_quiet_scenario_no_signal(self):
        res = self._run(scenario_quiet(), "600003")
        self.assertTrue(res["signals"].empty)
        self.assertEqual(res["final_state"], "空仓")

    def test_state_machine_no_deadlock_all(self):
        """三场景回放后终态均合法（无死锁卡死态）。"""
        for df, code in [(scenario_breakout(), "600001"), (scenario_crash(), "600002"),
                         (scenario_quiet(), "600003")]:
            res = self._run(df, code)
            self.assertIn(res["final_state"], {"空仓", "建仓中", "满仓持有", "减仓中", "离场观察"})

    def test_outputs_written(self):
        res = self._run(scenario_breakout(), "600001")
        self.assertTrue((BASE / "output" / "backtest_600001_timeline.csv").exists())
        if not res["signals"].empty:
            self.assertTrue((BASE / "output" / "backtest_600001_signals.csv").exists())


if __name__ == "__main__":
    unittest.main()
