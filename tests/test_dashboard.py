# -*- coding: utf-8 -*-
"""A11 仪表盘：本地生成 HTML、六个面板齐全、K线双带渲染、无模板错误（规格书 §8）。"""
import json
import tempfile
import unittest
from pathlib import Path

from main import render_dashboard

BASE = Path(__file__).resolve().parent.parent


def make_ctx():
    """构造与 main.run() 输出同构的最小上下文。"""
    return {
        "date": "2026-08-18",
        "sentiment": {
            "date": "2026-08-18",
            "factors": {"上涨占比": 60, "涨跌停比": 70, "指数温度": 55, "涨停活跃度": 65,
                        "情绪龙头": 60, "情绪驱动力": 50},
            "temperature": 60.0,
            "zone": {"name": "震荡区", "min": 48, "max": 61, "max_total": 50,
                     "max_single": 12, "max_count": 4, "label": "只做最强", "rule": "只加仓最强龙头"},
            "missing": [],
        },
        "sectors": {
            "boards": [{"rank": 1, "name": "算力", "total": 82.5, "attack": "进攻",
                        "dims": {"连板节奏": 70, "上攻意愿": 85, "主买占比": 60,
                                 "换手率": 90, "量能比": 95, "排名趋势": 88}}],
            "attack": ["算力"], "defend": ["银行"],
        },
        "pools": {
            "中线": [{"code": "603019", "name": "中科曙光", "score": 78, "grade": "主升进行"}],
            "短线": [{"code": "000001", "name": "打板示例", "score": 75}],
            "长线": [], "精选": [],
        },
        "eod": [],
        "positions": [{
            "code": "603019", "name": "中科曙光", "state": "BUILDING", "state_cn": "建仓中",
            "position_pct": 30, "trend": "多头排列", "bias60": 12.3, "dist_mid_upper": 4.2,
            "advice": "建议：正常执行\n反向条件：若直接高开涨超5%，不追",
            "kline": {
                "dates": ["2026-08-14", "2026-08-15", "2026-08-18"],
                "k": [[100, 102, 103, 99], [102, 101, 104, 100], [101, 105, 106, 100]],
                "short_band": {"upper": [103, 103.5, 104], "lower": [99, 99.5, 100], "middle": [101, 101.5, 102]},
                "mid_band": {"upper": [98, 98, 98], "lower": [96, 96, 96], "middle": [97, 97, 97]},
                "vol_ratio": [1.0, 1.2, 1.6],
                "signals": [{"date": "2026-08-18", "type": "BP2", "name": "突破买点"}],
            },
            "sells": [],
        }],
        "risks": {"missing": [], "cooldown": [], "discipline": ["尾盘纪律：买入时间14:50-14:55"]},
    }


class TestDashboard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.out = cls.tmp / "dashboard_20260818.html"
        render_dashboard(make_ctx(), cls.out)
        cls.html = cls.out.read_text(encoding="utf-8")

    def test_html_file_generated(self):
        self.assertTrue(self.out.exists() and self.out.stat().st_size > 1000)

    def test_six_panels_present(self):
        """面板齐全（UI重排后：总结置顶 + 宏观 + 情绪 + 板块 + 三线总揽 + 复盘 + 打板 + 持仓 + 日历 + 风险）。"""
        for panel in ["⓪ 总结决策", "① 宏观大盘", "② 情绪周期走势", "③ 板块埋伏建议",
                      "④ 短中长线评分总揽", "⑤ 短线观察池复盘", "⑥ 中长线观察池复盘",
                      "⑦ 打板筛选", "⑧ 三池Top3交叉验证精选", "⑨ 持仓监控",
                      "⑩ 信号日历", "⑪ 风险提示"]:
            self.assertIn(panel, self.html)

    def test_kline_dual_band_rendering(self):
        """K线图含双带（短期带+中期带）渲染代码。"""
        self.assertIn("short_band", self.html)
        self.assertIn("mid_band", self.html)

    def test_data_json_embedded_and_valid(self):
        start = self.html.index("const D = ") + len("const D = ")
        end = self.html.index(";\n", start)
        data = json.loads(self.html[start:end])
        self.assertEqual(data["sentiment"]["temperature"], 60.0)
        self.assertEqual(data["positions"][0]["kline"]["short_band"]["upper"], [103, 103.5, 104])

    def test_no_undefined_template_vars(self):
        self.assertNotIn("Undefined", self.html)

    def test_disclaimer_present(self):
        self.assertIn("不构成投资建议", self.html)


if __name__ == "__main__":
    unittest.main()
