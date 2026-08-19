# -*- coding: utf-8 -*-
"""老樊引擎 · 三大买点 × 三大卖点 + 持仓状态机 + 冷却期（规格书 §6.3/6.4/6.6/6.7）。

参数为精确校准值（config 第11章），禁止调整：
- BP1 乖离买点：BIAS60≤-25% 首次触达，置信72%，仓位30%，冷却20天，状态 EMPTY/EXITED
- BP2 突破买点：突破中期带上沿≥2% + 量比≥1.5 + 站稳2根K线，置信78%，仓位40%，冷却10天
- BP3 回踩买点：多头排列+近30日BP2型突破+回踩带沿缩量，置信82%，仓位30%至满仓，冷却10天，仅BUILDING
- SP1 乖离卖点：BIAS60≥+25% 首次触达，置信72%，减30%，冷却20天
- SP2 破短期带卖点：跌破短期带下沿≥1%，置信75%，减50%，冷却5天
- SP3 破中期带卖点：跌破中期带下沿≥2% + 1根K线确认，置信88%，清仓100%，冷却30天+15天离场观察
- 优先级：SP3 > SP2 > SP1 > BP3 > BP2 > BP1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from .ma_band_v2 import MABandV2

POSITION_STATES = ["EMPTY", "BUILDING", "HOLDING", "REDUCING", "EXITED"]

STATE_CN = {
    "EMPTY": "空仓", "BUILDING": "建仓中", "HOLDING": "满仓持有",
    "REDUCING": "减仓中", "EXITED": "离场观察",
}

# 信号-状态约束表（§6.6，严格）
ALLOWED_STATES = {
    "BP1": {"EMPTY", "EXITED"},
    "BP2": {"EMPTY", "BUILDING", "EXITED"},
    "BP3": {"BUILDING"},
    "SP1": {"HOLDING", "BUILDING"},
    "SP2": {"HOLDING", "BUILDING", "REDUCING"},
    "SP3": {"HOLDING", "BUILDING", "REDUCING"},
}

# 信号优先级（§6.7，高→低）
SIGNAL_PRIORITY = ["SP3", "SP2", "SP1", "BP3", "BP2", "BP1"]

# 基准置信度（§6.3/6.4）
BASE_CONFIDENCE = {"BP1": 72, "BP2": 78, "BP3": 82, "SP1": 72, "SP2": 75, "SP3": 88}

SIGNAL_CN = {
    "BP1": "乖离买点", "BP2": "突破买点", "BP3": "回踩买点",
    "SP1": "乖离卖点", "SP2": "破短期带卖点", "SP3": "破中期带卖点",
}


@dataclass
class StockState:
    """单只股票的状态机持久化记录（data/positions_state.json 每股一条）。"""
    code: str = ""
    name: str = ""
    status: str = "EMPTY"                    # EMPTY/BUILDING/HOLDING/REDUCING/EXITED
    position_pct: float = 0.0                # 当前仓位比例(%)
    last_signal_dates: dict = field(default_factory=dict)   # {信号: ISO日期}
    last_exit_date: Optional[str] = None     # 最近一次 SP3 清仓日期（EXITED 冷却计时）

    def to_dict(self) -> dict:
        return {
            "code": self.code, "name": self.name, "status": self.status,
            "position_pct": self.position_pct,
            "last_signal_dates": self.last_signal_dates,
            "last_exit_date": self.last_exit_date,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StockState":
        return cls(
            code=str(d.get("code", "")), name=str(d.get("name", "")),
            status=d.get("status", "EMPTY"),
            position_pct=float(d.get("position_pct", 0.0)),
            last_signal_dates=dict(d.get("last_signal_dates", {}) or {}),
            last_exit_date=d.get("last_exit_date"),
        )


@dataclass
class Signal:
    type: str            # BP1/BP2/BP3/SP1/SP2/SP3
    confidence: int
    position_action: str  # 建仓30% / 加仓40% / 加仓30%至满仓 / 减仓30% / 减仓50% / 清仓100%
    action_pct: int       # 仓位动作幅度(%)
    cooldown_days: int
    reason: str
    direction: str        # 买入/卖出

    def to_dict(self) -> dict:
        return {
            "type": self.type, "name": SIGNAL_CN.get(self.type, self.type),
            "direction": self.direction, "confidence": self.confidence,
            "position_action": self.position_action, "action_pct": self.action_pct,
            "cooldown_days": self.cooldown_days, "reason": self.reason,
        }


def _days_between(d1: str, d2: str) -> int:
    """自然日间隔（无交易日历时退化为自然日）。"""
    try:
        a = datetime.strptime(str(d1)[:10], "%Y-%m-%d")
        b = datetime.strptime(str(d2)[:10], "%Y-%m-%d")
        return abs((b - a).days)
    except (ValueError, TypeError):
        return 10 ** 9


class LaofanSignalEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.band = MABandV2(cfg)
        sg = cfg["signals"]
        self.cooldowns = {
            "BP1": int(sg["buy_point_1_cooldown"]),
            "BP2": int(sg["buy_point_2_cooldown"]),
            "BP3": int(sg["buy_point_3_cooldown"]),
            "SP1": int(sg["sell_point_1_cooldown"]),
            "SP2": int(sg["sell_point_2_cooldown"]),
            "SP3": int(sg["sell_point_3_cooldown"]),
        }
        self.exit_cooldown = int(sg["exit_cooldown"])
        self.filters = cfg["filters"]
        self.pos_cfg = cfg["positions"]

    # ------------------------------------------------------------ 工具

    @staticmethod
    def _val(b: pd.DataFrame, col: str, i: int):
        if i < 0 or i >= len(b):
            return None
        v = b[col].iloc[i]
        return None if pd.isna(v) else float(v)

    def _cooldown_ok(self, state: StockState, sig: str, today_iso: str) -> tuple[bool, str]:
        last = state.last_signal_dates.get(sig)
        if not last:
            return True, ""
        gap = _days_between(last, today_iso)
        cd = self.cooldowns[sig]
        if gap > cd:
            return True, ""
        return False, f"{sig}冷却期内(距上次{gap}天<{cd}天)"

    def _exit_block(self, state: StockState, today_iso: str) -> tuple[bool, str]:
        """SP3 后 15 天离场观察期内，BP1/BP2 一律拦截（§6.4 SP3 规则4）。"""
        if state.status == "EXITED" and state.last_exit_date:
            gap = _days_between(state.last_exit_date, today_iso)
            if gap <= self.exit_cooldown:
                return True, f"离场观察期(第{gap}天/{self.exit_cooldown}天)，不考虑重新进场"
        return False, ""

    # ------------------------------------------------------------ 买点检测

    def detect(self, bands: pd.DataFrame, state: StockState, i: int = -1) -> dict:
        """返回 {"signals": [Signal...], "blocked": [(信号,原因)...]}（同日多信号全列出，执行按优先级）。"""
        triggered: list[Signal] = []
        blocked: list[tuple[str, str]] = []
        if len(bands) == 0:
            return {"signals": [], "blocked": []}
        if i < 0:
            i += len(bands)
        today_iso = str(bands["date"].iloc[i])[:10]
        status = state.status

        # ---- BP1 乖离买点（超跌反弹）
        bias_now = self._val(bands, "bias60", i)
        bias_prev = self._val(bands, "bias60", i - 1)
        cd_ok, cd_msg = self._cooldown_ok(state, "BP1", today_iso)
        ex_block, ex_msg = self._exit_block(state, today_iso)
        if bias_now is not None and bias_prev is not None:
            cond1 = bias_now <= self.band.buy_bias            # BIAS60 ≤ -25%
            cond2 = bias_prev > self.band.buy_bias            # 前一日 > -25%（首次触达）
            if cond1 and cond2:
                if status not in ALLOWED_STATES["BP1"]:
                    blocked.append(("BP1", f"状态{status}不在允许范围{sorted(ALLOWED_STATES['BP1'])}"))
                elif not cd_ok:
                    blocked.append(("BP1", cd_msg))
                elif ex_block:
                    blocked.append(("BP1", ex_msg))
                else:
                    triggered.append(Signal(
                        "BP1", BASE_CONFIDENCE["BP1"], f"建仓{self.pos_cfg['buy_point_1_position']}%",
                        int(self.pos_cfg["buy_point_1_position"]), self.cooldowns["BP1"],
                        f"BIAS60={bias_now:.1f}%首次触达{self.band.buy_bias}%超卖区", "买入"))

        # ---- BP2 突破买点（趋势跟踪，三重过滤）
        close = self._val(bands, "close", i)
        close_prev = self._val(bands, "close", i - 1)
        mu = self._val(bands, "mid_upper", i)
        mu_prev = self._val(bands, "mid_upper", i - 1)
        vr = self._val(bands, "vol_ratio", i)
        cd_ok, cd_msg = self._cooldown_ok(state, "BP2", today_iso)
        if all(v is not None for v in (close, close_prev, mu, mu_prev, vr)):
            gain_req = float(self.filters["breakout_min_gain_pct"])       # ≥2%
            vol_req = float(self.filters["breakout_volume_ratio"])        # ≥1.5
            bars_req = int(self.filters["breakout_confirm_bars"])         # 站稳2根K线
            cond1 = close >= mu * (1 + gain_req / 100.0)                  # 突破幅度≥2%
            cond2 = vr >= vol_req if self.filters["breakout_volume_confirm"] else True
            stood = all(
                (self._val(bands, "close", i - k) is not None
                 and self._val(bands, "mid_upper", i - k) is not None
                 and self._val(bands, "close", i - k) > self._val(bands, "mid_upper", i - k))
                for k in range(bars_req)
            )
            if cond1 and cond2 and stood:
                if status not in ALLOWED_STATES["BP2"]:
                    blocked.append(("BP2", f"状态{status}不在允许范围{sorted(ALLOWED_STATES['BP2'])}"))
                elif not cd_ok:
                    blocked.append(("BP2", cd_msg))
                elif ex_block:
                    blocked.append(("BP2", ex_msg))
                else:
                    triggered.append(Signal(
                        "BP2", BASE_CONFIDENCE["BP2"], f"加仓{self.pos_cfg['buy_point_2_position']}%",
                        int(self.pos_cfg["buy_point_2_position"]), self.cooldowns["BP2"],
                        f"收盘{close:.2f}突破中期带上沿{mu:.2f}幅度≥{gain_req}%，量比{vr:.2f}，站稳{bars_req}根K线", "买入"))

        # ---- BP3 回踩买点（确认加仓，仅 BUILDING）
        judge = self.band.judge(bands, i)
        cd_ok, cd_msg = self._cooldown_ok(state, "BP3", today_iso)
        if not judge["数据不足"] and judge["多头排列"]:
            brk = self._last_breakout(bands, i, 30, vol_req=float(self.filters["breakout_volume_ratio"]))
            dist = judge["距中期带上沿pct"]
            if brk is not None:
                brk_idx, brk_gap = brk
                cond_dist = dist is not None and 0.5 <= dist <= 3.0 and close is not None and close > mu
                cond_vol = vr is not None and vr < 0.8                    # 回踩缩量
                if brk_idx >= 0 and brk_gap > 3 and cond_dist and cond_vol:
                    if status != "BUILDING":
                        blocked.append(("BP3", f"状态{status}不在允许范围['BUILDING']"))
                    elif not cd_ok:
                        blocked.append(("BP3", cd_msg))
                    else:
                        triggered.append(Signal(
                            "BP3", BASE_CONFIDENCE["BP3"],
                            f"加仓{self.pos_cfg['buy_point_3_position']}%至满仓",
                            int(self.pos_cfg["buy_point_3_position"]), self.cooldowns["BP3"],
                            f"多头排列成立，{brk_gap}天前放量突破，今缩量(量比{vr:.2f})回踩中期带上沿{dist:.1f}%",
                            "买入"))

        # ---- SP1 乖离卖点（超买止盈，今日首次）
        cd_ok, cd_msg = self._cooldown_ok(state, "SP1", today_iso)
        if bias_now is not None and bias_prev is not None:
            if bias_now >= self.band.sell_bias and bias_prev < self.band.sell_bias:
                if status not in ALLOWED_STATES["SP1"]:
                    blocked.append(("SP1", f"状态{status}不在允许范围{sorted(ALLOWED_STATES['SP1'])}"))
                elif not cd_ok:
                    blocked.append(("SP1", cd_msg))
                else:
                    triggered.append(Signal(
                        "SP1", BASE_CONFIDENCE["SP1"], f"减仓{self.pos_cfg['sell_point_1_reduce']}%",
                        int(self.pos_cfg["sell_point_1_reduce"]), self.cooldowns["SP1"],
                        f"BIAS60={bias_now:.1f}%首次触达+{self.band.sell_bias}%超买区", "卖出"))

        # ---- SP2 破短期带卖点
        sl = self._val(bands, "short_lower", i)
        sl_prev = self._val(bands, "short_lower", i - 1)
        cd_ok, cd_msg = self._cooldown_ok(state, "SP2", today_iso)
        if all(v is not None for v in (close, close_prev, sl, sl_prev)):
            cond1 = close_prev > sl_prev                                  # 前一日收于短期带下沿之上
            cond2 = close <= sl * 0.99                                    # 今日跌破≥1%
            if cond1 and cond2:
                if status not in ALLOWED_STATES["SP2"]:
                    blocked.append(("SP2", f"状态{status}不在允许范围{sorted(ALLOWED_STATES['SP2'])}"))
                elif not cd_ok:
                    blocked.append(("SP2", cd_msg))
                else:
                    triggered.append(Signal(
                        "SP2", BASE_CONFIDENCE["SP2"], f"减仓{self.pos_cfg['sell_point_2_reduce']}%",
                        int(self.pos_cfg["sell_point_2_reduce"]), self.cooldowns["SP2"],
                        f"收盘{close:.2f}跌破短期带下沿{sl:.2f}幅度≥1%", "卖出"))

        # ---- SP3 破中期带卖点（保命信号，1根K线确认，放量默认不要求）
        ml = self._val(bands, "mid_lower", i)
        cd_ok, cd_msg = self._cooldown_ok(state, "SP3", today_iso)
        if ml is not None and close is not None:
            loss_req = float(self.filters["breakdown_min_loss_pct"])      # ≥2%
            bars_req = int(self.filters["breakdown_confirm_bars"])        # 1根确认
            broke = all(
                (self._val(bands, "close", i - k) is not None
                 and self._val(bands, "mid_lower", i - k) is not None
                 and self._val(bands, "close", i - k) <= self._val(bands, "mid_lower", i - k) * (1 - loss_req / 100.0))
                for k in range(bars_req)
            )
            vol_ok = True
            if self.filters["breakdown_volume_confirm"] and vr is not None:
                vol_ok = vr >= float(self.filters["breakdown_volume_ratio"])
            if broke and vol_ok:
                if status not in ALLOWED_STATES["SP3"]:
                    blocked.append(("SP3", f"状态{status}不在允许范围{sorted(ALLOWED_STATES['SP3'])}"))
                elif not cd_ok:
                    blocked.append(("SP3", cd_msg))
                else:
                    triggered.append(Signal(
                        "SP3", BASE_CONFIDENCE["SP3"], "清仓100%",
                        100, self.cooldowns["SP3"],
                        f"收盘{close:.2f}跌破中期带下沿{ml:.2f}幅度≥{loss_req}%，{bars_req}根K线确认，进入{self.exit_cooldown}天离场观察",
                        "卖出"))

        triggered.sort(key=lambda s: SIGNAL_PRIORITY.index(s.type))
        return {"signals": triggered, "blocked": blocked}

    def _last_breakout(self, bands: pd.DataFrame, i: int, lookback: int, vol_req: float):
        """近 lookback 天内最近一次 BP2 型放量突破（收盘>中期带上沿 且 量比≥vol_req）。
        返回 (突破行号, 距今天数) 或 None。"""
        best = None
        for j in range(max(0, i - lookback), i):
            c = self._val(bands, "close", j)
            mu = self._val(bands, "mid_upper", j)
            vr = self._val(bands, "vol_ratio", j)
            if None in (c, mu, vr):
                continue
            if c > mu and vr >= vol_req:
                best = (j, i - j)
        return best

    # ------------------------------------------------------------ 状态机（§6.6）

    def apply_signal(self, state: StockState, sig: Signal, today_iso: str) -> StockState:
        """信号驱动状态转移（返回新状态对象；调用方负责持久化）。"""
        if sig.type in ("BP1", "BP2"):
            add = sig.action_pct
            state.position_pct = min(100.0, state.position_pct + add)
            state.status = "BUILDING"                       # EMPTY/EXITED→BUILDING，BUILDING维持
        elif sig.type == "BP3":
            state.position_pct = min(100.0, state.position_pct + sig.action_pct)
            state.status = "HOLDING"                        # BUILDING→HOLDING（满仓）
        elif sig.type == "SP1":
            state.position_pct = max(0.0, state.position_pct * (1 - sig.action_pct / 100.0))
            state.status = "REDUCING"
        elif sig.type == "SP2":
            state.position_pct = max(0.0, state.position_pct * (1 - sig.action_pct / 100.0))
            state.status = "REDUCING"
        elif sig.type == "SP3":
            state.position_pct = 0.0
            state.status = "EXITED"                         # 任一持仓态→EXITED
            state.last_exit_date = today_iso
        state.last_signal_dates[sig.type] = today_iso
        return state

    def refresh_exit_cooldown(self, state: StockState, today_iso: str) -> StockState:
        """EXITED 冷却 15 天满 → EMPTY。"""
        if state.status == "EXITED" and state.last_exit_date:
            if _days_between(state.last_exit_date, today_iso) > self.exit_cooldown:
                state.status = "EMPTY"
        return state
