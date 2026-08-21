# -*- coding: utf-8 -*-
"""新增模块测试：观察池复盘引擎 / 宏观大盘视图 / S级冲高形态判定 / 复盘记录文案。"""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pool_review import PoolReview
from core.macro_view import build_macro
from main import _s_grade_check, _enrich_review_rows
from tests.helpers import make_df


# ---------------------------------------------------------------- PoolReview

class TestPoolReview(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.prv = PoolReview(self.tmp, data_dir="data")

    def _hist(self, closes, start="2026-08-10", **kw):
        return make_df(closes, start=start, **kw)

    def test_backfill_win_lose(self):
        """次日表现回填：收涨=win，收跌=lose；开盘/最高/收盘收益计算正确。"""
        # 入选日 2026-08-10 收盘10.0，次日(08-11) 开10.5 高11.0 收10.2 → win
        # 入选日 2026-08-12 收盘10.2，次日(08-13) 开10.0 高10.4 收9.8 → lose
        hist = self._hist([10.0, 10.2, 9.8, 10.0],
                          opens=[10.0, 10.5, 10.0, 9.9], highs=[10.1, 11.0, 10.4, 10.1])
        self.prv.data["短线"] = [
            {"code": "600001", "name": "甲", "added_date": "2026-08-10",
             "score": 88, "add_price": 10.0, "reason": "", "model": "", "status": "active"},
            {"code": "600002", "name": "乙", "added_date": "2026-08-12",
             "score": 86, "add_price": 10.2, "reason": "", "model": "", "status": "active"},
        ]
        n = self.prv.backfill({"600001": hist, "600002": hist}, "2026-08-14")
        self.assertEqual(n, 2)
        r1, r2 = self.prv.data["短线"]
        self.assertEqual(r1["status"], "win")
        self.assertEqual(r1["next_day"]["open_ret"], 5.0)      # (10.5-10)/10
        self.assertEqual(r1["next_day"]["high_ret"], 10.0)     # (11.0-10)/10
        self.assertEqual(r1["next_day"]["close_ret"], 2.0)     # (10.2-10)/10
        self.assertEqual(r2["status"], "lose")

    def test_backfill_skip_today_and_last_bar(self):
        """入选日=今日（未走完）与入选日不在日K中的记录不回填。"""
        hist = self._hist([10.0, 10.2], start="2026-08-13")
        self.prv.data["短线"] = [
            {"code": "600003", "name": "丙", "added_date": "2026-08-14",
             "score": 88, "add_price": 10.2, "reason": "", "model": "", "status": "active"},
            {"code": "600004", "name": "丁", "added_date": "2020-01-01",
             "score": 88, "add_price": 10.0, "reason": "", "model": "", "status": "active"},
        ]
        n = self.prv.backfill({"600003": hist, "600004": hist}, "2026-08-14")
        self.assertEqual(n, 0)

    def test_prune_short_break_ma5(self):
        """短线删除规则：收盘破MA5 → 删除；未破 → 保留。"""
        # 股A：连续上涨，收盘10 ≥ MA5 → 保留；股B：跌破 → 删除
        up = self._hist([8, 8.5, 9, 9.5, 10], start="2026-08-10")
        down = self._hist([10, 10, 10, 10, 9.0], start="2026-08-10")
        self.prv.data["短线"] = [
            {"code": "600005", "name": "A", "added_date": "2026-08-10", "score": 88,
             "add_price": 8.0, "reason": "", "model": "", "status": "win",
             "next_day": {"date": "2026-08-11", "open": 8, "high": 8.6, "close": 8.5,
                          "open_ret": 0, "high_ret": 7.5, "close_ret": 6.25}},
            {"code": "600006", "name": "B", "added_date": "2026-08-10", "score": 88,
             "add_price": 10.0, "reason": "", "model": "", "status": "lose",
             "next_day": {"date": "2026-08-11", "open": 10, "high": 10.1, "close": 10.0,
                          "open_ret": 0, "high_ret": 1.0, "close_ret": 0.0}},
        ]
        removed = self.prv.prune({"600005": up, "600006": down}, today_iso="2026-08-15")
        self.assertEqual(len(removed), 1)
        self.assertIn("600006", removed[0])
        codes = [r["code"] for r in self.prv.data["短线"]]
        self.assertEqual(codes, ["600005"])

    def test_prune_mid_score_slide_and_ma20(self):
        """中线删除规则：评分<60 或 收盘破MA20 → 删除。"""
        down = self._hist([20] * 25 + [15.0], start="2026-07-01")
        rec = {"code": "600007", "name": "C", "added_date": "2026-07-01", "score": 75,
               "add_price": 20.0, "reason": "", "model": "", "status": "win",
               "next_day": {"date": "2026-07-02", "open": 20, "high": 20.2, "close": 20.1,
                            "open_ret": 0, "high_ret": 1.0, "close_ret": 0.5}}
        # 评分滑坡（55<60）
        self.prv.data["中线"] = [dict(rec)]
        removed = self.prv.prune({"600007": down}, score_map={"600007": 55}, today_iso="2026-08-15")
        self.assertEqual(len(removed), 1)
        self.assertIn("评分", removed[0])
        # 破MA20（评分正常）
        self.prv.data["中线"] = [dict(rec)]
        removed2 = self.prv.prune({"600007": down}, score_map={"600007": 75}, today_iso="2026-08-15")
        self.assertEqual(len(removed2), 1)
        self.assertIn("MA20", removed2[0])

    def test_append_dedup_and_persist(self):
        """append 去重（同码不重复入选）+ 持久化往返。"""
        self.prv.append("中线", [{"code": "600008", "name": "D", "score": 72,
                                  "add_price": 10.0, "reason": "r", "model": "m"}], "2026-08-15")
        self.prv.append("中线", [{"code": "600008", "name": "D", "score": 73,
                                  "add_price": 10.5, "reason": "r2", "model": "m"}], "2026-08-15")
        self.assertEqual(len(self.prv.data["中线"]), 1)
        self.prv.save()
        prv2 = PoolReview(self.tmp, data_dir="data")
        self.assertEqual(len(prv2.data["中线"]), 1)
        self.assertEqual(prv2.data["中线"][0]["score"], 72)

    def test_stats_small_sample_hidden(self):
        """样本<10 不给统计（伪精确防护）；样本≥10 给胜率/平均收益。"""
        st = self.prv.stats()
        self.assertEqual(st["短线"]["samples"], 0)
        self.assertIsNone(st["短线"]["win_rate"])
        rows = []
        for i in range(10):
            win = i < 7   # 7胜3负
            add, close = (10.0, 10.5) if win else (10.0, 9.5)
            rows.append({"code": f"6000{i:02d}", "name": "x", "added_date": "2026-08-01",
                         "score": 80, "add_price": add, "reason": "", "model": "",
                         "status": "win" if win else "lose",
                         "next_day": {"date": "2026-08-02", "open": add, "high": add + 1.0,
                                      "close": close, "open_ret": 0.0,
                                      "high_ret": 10.0, "close_ret": 5.0 if win else -5.0}})
        self.prv.data["长线"] = rows
        st = self.prv.stats()
        self.assertEqual(st["长线"]["samples"], 10)
        self.assertEqual(st["长线"]["win_rate"], 70.0)
        self.assertEqual(st["长线"]["spike_rate"], 100.0)
        self.assertEqual(st["长线"]["avg_high_ret"], 10.0)

    def test_yesterday_top_and_active(self):
        self.prv.data["短线"] = [
            {"code": "600009", "name": "E", "added_date": "2026-08-14", "score": 90,
             "add_price": 10.0, "reason": "", "model": "", "status": "active"},
            {"code": "600010", "name": "F", "added_date": "2026-08-14", "score": 85,
             "add_price": 10.0, "reason": "", "model": "", "status": "active"},
            {"code": "600011", "name": "G", "added_date": "2026-08-13", "score": 99,
             "add_price": 10.0, "reason": "", "model": "", "removed_reason": "收盘破MA5"},
        ]
        yd = self.prv.yesterday_top("短线", "2026-08-14")
        self.assertEqual([r["code"] for r in yd], ["600009", "600010"])
        act = self.prv.active("短线")
        self.assertEqual([r["code"] for r in act], ["600009", "600010"])


# ---------------------------------------------------------------- macro_view

class _FakeDP:
    """离线伪 DataProvider：指数含成交额（7根，可算5日均量）；外围指数可配置。"""

    def __init__(self, global_ok=True):
        idx = pd.DataFrame({
            "date": pd.to_datetime([f"2026-08-{10 + i:02d}" for i in range(7)]),
            "close": [3000.0] * 6 + [3030.0],
            "amount": [4e11] * 6 + [5e11],
        })
        self._idx = idx
        self.global_ok = global_ok
        self.missing = []

    def get_index_daily(self, sym):
        return self._idx

    def get_global_indices(self):
        if not self.global_ok:
            return None
        return pd.DataFrame([{"name": "日经225", "price": 38000.0, "pct_chg": -1.2},
                             {"name": "韩国KOSPI", "price": 2600.0, "pct_chg": 0.8}])


class TestMacroView(unittest.TestCase):

    def test_build_macro_full(self):
        m = build_macro(_FakeDP(), {"上证指数": "sh000001", "深证成指": "sz399001"})
        self.assertEqual(m["indices"][0]["name"], "上证指数")
        self.assertEqual(m["indices"][0]["pct_chg"], 1.0)          # (3030-3000)/3000
        self.assertEqual(m["indices"][0]["amount_yi"], 5000)       # 5e11/1e8
        self.assertEqual(m["turnover_yi"], 10000)                  # 沪+深
        self.assertEqual(m["turnover_ratio_5d"], 1.25)             # 5e11/前5日均4e11
        self.assertEqual(len(m["global_indices"]), 2)
        self.assertIn("日经", json.dumps(m["global_indices"], ensure_ascii=False))
        self.assertTrue(m["summary"])

    def test_build_macro_global_missing(self):
        m = build_macro(_FakeDP(global_ok=False), {"上证指数": "sh000001"})
        self.assertEqual(m["global_indices"], [])
        self.assertTrue(any("外围" in x for x in m["missing"]))


# ---------------------------------------------------------------- S级判定

class TestSGradeCheck(unittest.TestCase):

    def _check(self, closes, res, **kw):
        hist = make_df(closes, **kw)
        return _s_grade_check(res, hist, hist["close"].astype(float))

    def test_score_below_85_rejected(self):
        ok, why = self._check([10] * 25, {"score": 80, "spot_vr": 2.0, "bias60": 0})
        self.assertFalse(ok)

    def test_double_limit_up_pattern(self):
        # 近5日3个涨幅>9.5%（11%/10.7%/9.7%）
        closes = [10.0] * 20 + [10.0, 11.1, 11.2, 12.4, 13.6]
        ok, why = self._check(closes, {"score": 88, "spot_vr": 1.0, "bias60": 0})
        self.assertTrue(ok)
        self.assertIn("N式双涨停", why)

    def test_breakout_20d_high_with_volume(self):
        closes = [10 + 0.05 * i for i in range(25)]   # 温和上行，末日创新高
        ok, why = self._check(closes, {"score": 88, "spot_vr": 2.0, "bias60": 5})
        self.assertTrue(ok)
        self.assertIn("突破20日新高", why)

    def test_extreme_reversal(self):
        # 深跌后乖离-30 且当日上涨
        closes = [20 - 0.3 * i for i in range(24)] + [14.0]
        ok, why = self._check(closes, {"score": 88, "spot_vr": 1.0, "bias60": -30.0,
                                       "pct_chg": 2.0})
        self.assertTrue(ok)
        self.assertIn("极值反转", why)

    def test_no_pattern_no_grade(self):
        closes = [10] * 25
        ok, why = self._check(closes, {"score": 88, "spot_vr": 0.5, "bias60": 0,
                                       "pct_chg": 0.0})
        self.assertFalse(ok)
        self.assertEqual(why, "")


# ---------------------------------------------------------------- 复盘文案

class TestEnrichReviewRows(unittest.TestCase):

    def test_win_row_summary(self):
        rows = _enrich_review_rows([{
            "code": "600001", "name": "甲", "added_date": "2026-08-10", "score": 88,
            "add_price": 10.0, "reason": "", "model": "", "status": "win",
            "next_day": {"open": 10.5, "high": 11.0, "close": 10.2,
                         "open_ret": 5.0, "high_ret": 10.0, "close_ret": 2.0},
        }])
        self.assertIn("收红", rows[0]["status_summary"])
        self.assertIn("高开低走", rows[0]["status_summary"])   # 开盘+5% 收盘+2% < 开盘

    def test_pending_row_summary(self):
        rows = _enrich_review_rows([{"code": "600002", "name": "乙", "added_date": "2026-08-10",
                                     "score": 88, "add_price": 10.0, "reason": "", "model": "",
                                     "status": "active"}])
        self.assertIn("待回填", rows[0]["status_summary"])


if __name__ == "__main__":
    unittest.main()
