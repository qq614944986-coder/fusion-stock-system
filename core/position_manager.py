# -*- coding: utf-8 -*-
"""持仓与仓位记录：状态机持久化（data/positions_state.json）+ 评分峰值历史。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .laofan_signals import LaofanSignalEngine, Signal, StockState


class PositionManager:
    def __init__(self, cfg: dict, base_dir: Optional[Path] = None):
        self.cfg = cfg
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
        data_dir = self.base_dir / cfg.get("run", {}).get("data_dir", "data")
        data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = data_dir / "positions_state.json"
        self.score_hist_file = data_dir / "score_history.csv"
        self.states: dict[str, StockState] = {}
        self._load()

    # ---------------- 持仓状态

    def _load(self) -> None:
        if self.state_file.exists():
            try:
                raw = json.loads(self.state_file.read_text(encoding="utf-8"))
                for code, d in raw.items():
                    self.states[str(code).zfill(6)] = StockState.from_dict(d)
            except (json.JSONDecodeError, OSError):
                self.states = {}

    def save(self) -> None:
        data = {code: st.to_dict() for code, st in self.states.items()}
        self.state_file.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                   encoding="utf-8")

    def ensure_state(self, code: str, name: str = "", is_position: bool = True) -> StockState:
        code = str(code).zfill(6)
        if code not in self.states:
            self.states[code] = StockState(code=code, name=name, status="EMPTY", position_pct=0.0)
        st = self.states[code]
        st.code = code
        if name:
            st.name = name
        if not is_position and st.status in ("EMPTY",) and st.position_pct == 0:
            pass  # 关注股只出信号不出仓位建议
        return st

    def apply(self, engine: LaofanSignalEngine, code: str, sig: Signal, today_iso: str) -> StockState:
        st = self.ensure_state(code)
        engine.apply_signal(st, sig, today_iso)
        return st

    def refresh_all(self, engine: LaofanSignalEngine, today_iso: str) -> None:
        for st in self.states.values():
            engine.refresh_exit_cooldown(st, today_iso)

    # ---------------- 评分峰值历史（7.2 信号卖出：主升评分较峰值降>15分）

    def record_scores(self, date_iso: str, scores: dict) -> None:
        """scores: {code: 今日主升评分}"""
        rows = []
        if self.score_hist_file.exists():
            try:
                hist = pd.read_csv(self.score_hist_file, dtype={"code": str})
                hist["code"] = hist["code"].str.zfill(6)
                rows.append(hist)
            except (OSError, pd.errors.ParserError):
                pass
        today = pd.DataFrame({"date": [date_iso] * len(scores),
                              "code": [str(c).zfill(6) for c in scores.keys()],
                              "score": list(scores.values())})
        rows.append(today)
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(self.score_hist_file, index=False, encoding="utf-8-sig")

    def peak_score(self, code: str, before_date: Optional[str] = None) -> Optional[float]:
        code = str(code).zfill(6)
        if not self.score_hist_file.exists():
            return None
        try:
            hist = pd.read_csv(self.score_hist_file, dtype={"code": str})
        except (OSError, pd.errors.ParserError):
            return None
        hist["code"] = hist["code"].str.zfill(6)
        sub = hist[hist["code"] == code]
        if before_date:
            sub = sub[sub["date"] < before_date]
        if sub.empty:
            return None
        return float(sub["score"].max())
