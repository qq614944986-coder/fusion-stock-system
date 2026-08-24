# -*- coding: utf-8 -*-
"""概率引擎单元测试（core/probability.py）。"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.probability import next_day_probability, MIN_SAMPLES


def _mkhist(n: int, step: float = 1.0):
    """构造单调略升的日K（pct_chg 恒为 1%，收盘递增 → 次日大概率收红）。"""
    close = np.linspace(10.0, 10.0 + step * n, n)
    df = pd.DataFrame({
        "close": close,
        "open": close * 1.001,
        "high": close * 1.03,
        "low": close * 0.99,
        "pct_chg": np.full(n, 1.0),
    })
    return df


class NextDayProbabilityTest(unittest.TestCase):

    def test_hist_none(self):
        r = next_day_probability(None, None, None)
        self.assertEqual(r["samples"], 0)
        self.assertIsNone(r["red_rate"])
        self.assertEqual(r["tendency"], "—")

    def test_insufficient_samples(self):
        df = _mkhist(MIN_SAMPLES - 2)          # 匹配样本 < MIN_SAMPLES
        r = next_day_probability(df, None, 1.0)
        self.assertEqual(r["samples"], MIN_SAMPLES - 3)
        self.assertIsNone(r["red_rate"])

    def test_bull_tendency_and_full_red(self):
        df = _mkhist(70)                        # 全涨 → 收红率 100%
        r = next_day_probability(df, None, 1.0)
        self.assertGreaterEqual(r["samples"], MIN_SAMPLES)
        self.assertEqual(r["red_rate"], 100.0)
        self.assertEqual(r["tendency"], "偏多")
        self.assertGreater(r["avg_ret"], 0)

    def test_pct_tolerance_filters(self):
        """涨跌幅不在容忍度内的样本被剔除，但总样本仍足够 → 有统计。"""
        df = _mkhist(80)
        r = next_day_probability(df, None, -5.0)   # 与所有样本 pct=1% 差 6%，超 ±3 容忍度
        # 历史无 -5% 样本 → 匹配不到 → 样本不足 None
        self.assertIsNone(r["red_rate"])

    def test_bias_match(self):
        df = _mkhist(60)
        r = next_day_probability(df, bias_now=0.0, pct_now=1.0)
        # bias 维度也加入过滤，命中数量可能下降但不报错
        self.assertTrue("red_rate" in r and "tendency" in r)


if __name__ == "__main__":
    unittest.main()