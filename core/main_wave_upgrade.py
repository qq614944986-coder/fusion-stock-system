# -*- coding: utf-8 -*-
"""右轨 · 主升质量门槛（规格 v1.0 §四）。

把旧右轨"见强就上"的毒瘤用五道质量门槛修去；通过阈值才允许趋势跟随：
  1. 板块趋势前置 —— 板块 20 日向上 或 板块当日上涨（剔除孤立逆势）
  2. 量价健康     —— 排除放量滞涨(量比>1.8且涨<1%)、极端换手(>25%)
  3. 乖离约束     —— BIAS20 <19%（过热不追高）
  4. 板块共振     —— 同板块涨停/走强 ≥1 或 板块排名进前10
  5. 产业锚       —— 可映射到赛道产业驱动（否则降级）
右轨确认 = 核心门(1/2/3)全过 且 总过门数 ≥4。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def bias20_of(hist: Optional[pd.DataFrame]) -> Optional[float]:
    if hist is None or len(hist) < 20:
        return None
    close = hist["close"].astype(float)
    ma20 = float(close.rolling(20).mean().iloc[-1])
    if not ma20:
        return None
    return (float(close.iloc[-1]) - ma20) / ma20 * 100.0


def evaluate(hist: Optional[pd.DataFrame], rrow: Optional[dict], bctx: dict,
             track: Optional[str], cfg: dict) -> dict:
    """bctx: {board, board_pct_chg, board_rank, board_zt_count, board_alpha(板块20日涨幅,可None)}。"""
    gates: dict[str, dict] = {}
    reasons: list[str] = []

    # ---- 1 板块趋势前置
    trend_ok = False
    alpha = bctx.get("board_alpha")
    if _num(alpha) is not None:
        trend_ok = _num(alpha) > 0
        reasons.append(f"板块20日α={_num(alpha):+.1f}%" if trend_ok else f"板块20日α={_num(alpha):+.1f}%（弱）")
    else:
        bp = _num(bctx.get("board_pct_chg"))
        trend_ok = bp is not None and bp > 0
        reasons.append(f"板块当日{bp:+.1f}%" if bp is not None else "板块趋势数据缺失")
    gates["板块趋势"] = trend_ok

    # ---- 2 量价健康
    vh = True; vs = []
    close = hist["close"].astype(float) if hist is not None and len(hist) else None
    vol = hist["volume"].astype(float) if hist is not None and len(hist) else None
    vr = None
    if vol is not None and len(vol) >= 21 and vol.iloc[-21:-1].mean():
        vr = float(vol.iloc[-1] / vol.iloc[-21:-1].mean())
    pct_today = _num(rrow.get("pct_chg")) if rrow else None
    if vr is not None and pct_today is not None and vr > 1.8 and pct_today < 1.0:
        vh = False; vs.append(f"放量滞涨(量比{vr:.1f}涨{pct_today:+.1f}%)")
    tover = _num(rrow.get("turnover")) if rrow else None
    if tover is not None and tover > 25:
        vh = False; vs.append(f"极端换手{tover:.0f}%")
    reasons.extend(vs or ["量价健康"])
    gates["量价健康"] = vh

    # ---- 3 乖离约束
    b20 = bias20_of(hist)
    bias_ok = b20 is not None and b20 < 19.0
    reasons.append(f"BIAS20={b20:+.1f}%" if b20 is not None else "BIAS20不足" + ("（未过热）" if bias_ok else "（过热，降级）"))
    gates["乖离约束"] = bias_ok

    # ---- 4 板块共振
    rc = False
    zt = int(_num(bctx.get("board_zt_count")) or 0)
    rank = _num(bctx.get("board_rank"))
    if zt >= 1 or (rank is not None and rank <= 10):
        rc = True
        reasons.append(f"板块共振（涨停{zt}/排名#{rank or '-'}）")
    gates["板块共振"] = rc

    # ---- 5 产业锚
    ea = bool(track)
    reasons.append(f"产业锚：{'在' if track else '不在'}赛道" if track else "产业锚：未匹配赛道（降级）")
    gates["产业锚"] = ea

    passed = gates["板块趋势"] and gates["量价健康"] and gates["乖离约束"] and sum(gates.values()) >= 4
    score = sum(gates.values()) * 20
    return {"passes": passed, "gates": gates, "score": score, "reasons": reasons, "vr": vr}