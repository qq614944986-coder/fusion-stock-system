# -*- coding: utf-8 -*-
"""观察池复盘引擎：短线/中线/长线三池的「入选→次日验证→去弱留强」闭环。

设计（用户需求 + 老樊体系融合）：
- 入选记录：盘后运行时记录（入选价=当日收盘价），次日盘后运行时回填真实表现；
- 次日表现：开盘收益（高开/低开）、最高收益（冲高兑现空间）、收盘收益（收红/收绿）；
- 去弱留强删除规则（跨引擎一致）：
    短线：收盘破 MA5（老樊短期攻击带生命线）→ 删除；
    中线：主升评分 <60 或 收盘破 MA20 → 删除；
    长线：收盘破中期带下沿（MA55/60/65 下沿）→ 删除；
- 统计聚合：各池次日胜率、平均开盘/收盘/最高收益（样本<10 显示"—"，数据诚实纪律）；
- 涨跌概率（收红率/冲高率）来源：本池历史统计，随每日运行滚动自更新。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

POOLS = ("短线", "中线", "长线")


class PoolReview:
    def __init__(self, base_dir: Path, data_dir: str = "data"):
        self.file = Path(base_dir) / data_dir / "watch_pools.json"
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict = {p: [] for p in POOLS}
        self._load()

    # ---------------- 持久化

    def _load(self) -> None:
        if self.file.exists():
            try:
                raw = json.loads(self.file.read_text(encoding="utf-8"))
                for p in POOLS:
                    self.data[p] = raw.get(p, [])
            except (json.JSONDecodeError, OSError):
                self.data = {p: [] for p in POOLS}

    def save(self) -> None:
        self.file.write_text(json.dumps(self.data, ensure_ascii=False, indent=1),
                             encoding="utf-8")

    # ---------------- 次日表现回填

    def backfill(self, hist_map: dict, today_iso: str) -> int:
        """对 added_date < today 且未回填的记录，用日K回填次日表现。

        hist_map: {code: DataFrame(date/open/high/low/close)}，最后一根为今日。
        返回回填条数。
        """
        n = 0
        for pool in POOLS:
            for rec in self.data[pool]:
                if rec.get("next_day") or rec.get("added_date", "") >= today_iso:
                    continue
                hist = hist_map.get(rec["code"])
                if hist is None or hist.empty:
                    continue
                dates = [str(d)[:10] for d in hist["date"]]
                if rec["added_date"] not in dates:
                    continue
                i = dates.index(rec["added_date"])
                if i + 1 >= len(hist):   # 入选日是最新一根，次日还没走完
                    continue
                nxt = hist.iloc[i + 1]
                add_price = float(rec.get("add_price") or 0)
                if not add_price:
                    continue
                o, h, c = float(nxt["open"]), float(nxt["high"]), float(nxt["close"])
                rec["next_day"] = {
                    "date": dates[i + 1],
                    "open": round(o, 2), "high": round(h, 2), "close": round(c, 2),
                    "open_ret": round((o - add_price) / add_price * 100, 2),
                    "high_ret": round((h - add_price) / add_price * 100, 2),
                    "close_ret": round((c - add_price) / add_price * 100, 2),
                }
                rec["status"] = "win" if c > add_price else "lose"
                n += 1
        return n

    # ---------------- 去弱留强（删除规则）

    def prune(self, hist_map: dict, band_map: Optional[dict] = None,
              score_map: Optional[dict] = None, today_iso: str = "") -> list[str]:
        """按各池删除规则清理在池记录。返回删除说明列表（供复盘展示）。"""
        removed: list[str] = []
        for pool in POOLS:
            keep: list[dict] = []
            for rec in self.data[pool]:
                # 未回填次日数据的（刚入选或数据缺失）先保留
                if not rec.get("next_day"):
                    keep.append(rec)
                    continue
                reason = self._prune_reason(pool, rec, hist_map.get(rec["code"]),
                                            (score_map or {}).get(rec["code"]),
                                            (band_map or {}).get(rec["code"]))
                if reason:
                    rec["removed_reason"] = reason
                    rec.pop("active", None)
                    removed.append(f"{pool}池删除 {rec['name']}({rec['code']})：{reason}")
                else:
                    keep.append(rec)
            self.data[pool] = keep
        return removed

    @staticmethod
    def _prune_reason(pool: str, rec: dict, hist: Optional[pd.DataFrame],
                      score_now, bands: Optional[pd.DataFrame]) -> Optional[str]:
        if hist is None or len(hist) < 2:
            return None
        close = float(hist["close"].iloc[-1])
        # 短线：破5日线
        if pool == "短线":
            if len(hist) >= 5:
                ma5 = float(hist["close"].astype(float).rolling(5).mean().iloc[-1])
                if close < ma5:
                    return f"收盘{close:.2f}破MA5({ma5:.2f})"
        # 中线：评分滑坡 或 破MA20
        elif pool == "中线":
            if score_now is not None and score_now < 60:
                return f"主升评分{score_now:.0f}<60"
            if len(hist) >= 20:
                ma20 = float(hist["close"].astype(float).rolling(20).mean().iloc[-1])
                if close < ma20:
                    return f"收盘{close:.2f}破MA20({ma20:.2f})"
        # 长线：破中期带下沿
        elif pool == "长线" and bands is not None and len(bands):
            low_edge = bands["mid_lower"].iloc[-1]
            if not pd.isna(low_edge) and close < float(low_edge):
                return f"收盘{close:.2f}破中期带下沿({float(low_edge):.2f})"
        return None

    # ---------------- 今日入选

    def append(self, pool: str, entries: list[dict], today_iso: str) -> None:
        """entries: [{code,name,score,add_price,reason,model}]（add_price=今日收盘）。"""
        if pool not in POOLS:
            return
        existing = {r["code"] for r in self.data[pool] if not r.get("removed_reason")}
        for e in entries:
            if e["code"] in existing:
                continue
            self.data[pool].append({
                "code": e["code"], "name": e.get("name", ""),
                "added_date": today_iso,
                "score": e.get("score"),
                "add_price": e.get("add_price"),
                "reason": e.get("reason", ""),
                "model": e.get("model", ""),
                "status": "active",
            })
            existing.add(e["code"])

    # ---------------- 统计聚合

    def stats(self) -> dict:
        """各池：次日胜率、平均开盘/收盘/最高收益、样本数、收红率、冲高率。

        样本 <10 时数值置 None（前端显示"—"）：小样本统计是伪精确。
        """
        out: dict = {}
        for pool in POOLS:
            done = [r for r in self.data[pool] if r.get("next_day")]
            n = len(done)
            st = {"samples": n, "win_rate": None, "avg_open_ret": None,
                  "avg_close_ret": None, "avg_high_ret": None, "red_rate": None,
                  "spike_rate": None}
            if n >= 10:
                wins = sum(1 for r in done if r["status"] == "win")
                st["win_rate"] = round(wins / n * 100, 1)
                st["red_rate"] = st["win_rate"]
                st["avg_open_ret"] = round(sum(r["next_day"]["open_ret"] for r in done) / n, 2)
                st["avg_close_ret"] = round(sum(r["next_day"]["close_ret"] for r in done) / n, 2)
                st["avg_high_ret"] = round(sum(r["next_day"]["high_ret"] for r in done) / n, 2)
                st["spike_rate"] = round(sum(1 for r in done if r["next_day"]["high_ret"] > 0) / n * 100, 1)
            out[pool] = st
        return out

    def yesterday_top(self, pool: str, yesterday_iso: str, top_n: int = 5) -> list[dict]:
        """昨日入选的前N（按评分），供复盘明细表。"""
        rows = [r for r in self.data[pool] if r.get("added_date") == yesterday_iso]
        rows.sort(key=lambda r: -(r.get("score") or 0))
        return rows[:top_n]

    def active(self, pool: str) -> list[dict]:
        """当前在池（未删除）记录，按入选日倒序。"""
        rows = [r for r in self.data[pool] if not r.get("removed_reason")]
        rows.sort(key=lambda r: r.get("added_date", ""), reverse=True)
        return rows
