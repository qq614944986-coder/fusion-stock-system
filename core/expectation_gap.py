# -*- coding: utf-8 -*-
"""左轨 · 预期差校验 + 温和左侧布局判据（规格 v1.0 §三）。

三锚共振才进左轨机会池：
  1. est_ok 估值锚    —— 价格处自身低位分位 且 业绩方向为正（估值的代理，不含 PE）
  2. forecast_ok 一致预期锚 —— 研报盈利预测收益为正 且 评级未卖出（研报一致预期）
  3. trend_ok 产业趋势锚 —— 标的处于赛道景气环节（由 track_signals 提供）
通过 ≥2 锚 且 无硬否决 → 左轨候选。

温和左侧布局判据（L2/L3 先手执行，早于老樊 BP1）：
  - 价格低分位(<40%) + 缩量企稳(未创新低) + 资金/基本面任一转正
  → 打"温和左侧布局"标，供 main 以轻仓执行；老樊 BP1 退为更深兜底。
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


def price_percentile(hist: Optional[pd.DataFrame]) -> Optional[float]:
    """近3年价格分位（近期收盘价所处的近750日位置，0-100 越高位）。"""
    if hist is None or len(hist) < 60:
        return None
    n = min(len(hist), 750)
    close = hist["close"].astype(float)
    win = close.iloc[-n:]
    lo, hi = float(win.min()), float(win.max())
    if hi <= lo:
        return 50.0
    c = float(close.iloc[-1])
    return (c - lo) / (hi - lo) * 100.0


def evaluate(hist: Optional[pd.DataFrame], ds: dict, track_result: Optional[dict],
             cfg: dict) -> dict:
    """返回 {anchors:{est_ok,forecast_ok,trend_ok}, mild_left, evidence, passed}。"""
    code = ds.get("code", "")
    anchors = {"est_ok": False, "forecast_ok": False, "trend_ok": bool(track_result and track_result.get("belongs"))}
    evidence: list[str] = []

    # ---- 估值锚（价格位置 + 业绩方向）
    pp = price_percentile(hist)
    est_ok = False
    ni = None
    yjbb = (ds.get("yjbb_by_code") or {}).get(code) or []
    if yjbb:
        ni = _num(yjbb[0].get("净利润-同比增长"))
    yjyg = (ds.get("yjyg_by_code") or {}).get(code) or []
    yt = str(yjyg[0].get("预告类型", "")) if yjyg else ""
    good_fund = ni is not None and ni > 0
    good_guide = any(k in yt for k in ("预增", "扭亏", "减亏"))
    if pp is not None and pp < 40:
        est_ok = True
        evidence.append(f"价格分位{pp:.0f}%（低位）")
        if good_fund or good_guide:
            evidence.append(f"业绩{('净利+'+str(round(ni,1))+'%' if ni is not None else '预告'+(yt or '好转'))}")
        else:
            evidence.append("业绩方向待确认")
    anchors["est_ok"] = est_ok

    # ---- 一致预期锚（研报盈利预测）
    forecast_ok = False
    fc = (ds.get("resfc_by_code") or {}).get(code)
    if fc is not None and not fc.empty:
        # 取最新一条盈利预测收益
        yr = "2026-盈利预测-收益"
        if yr in fc.columns:
            v = fc[yr].dropna()
            if not v.empty and _num(v.iloc[-1]) is not None and _num(v.iloc[-1]) > 0:
                rating = str(fc.get("评级") and fc["评级"].iloc[-1])
                if "卖出" not in rating:
                    forecast_ok = True
                    evidence.append(f"研报盈利预测合理（{yr}收益为正）")
    anchors["forecast_ok"] = forecast_ok
    if track_result and track_result.get("belongs"):
        evidence.append(f"赛道：{track_result['track']}·{track_result.get('stage','')}")

    # 通过 ≥2 锚 且 无硬否决
    hard_block = ds.get("hard_block")   # 调用方可注入（如近期大股东减持）
    passed = (sum(anchors.values()) >= 2) and not hard_block

    # ---- 温和左侧布局判据
    mild_left = False
    ml_note = ""
    if pp is not None and pp < 40:
        # 缩量企稳（近5日未跌破前策略性低点 + 量能不放大）
        if hist is not None and len(hist) >= 6:
            close = hist["close"].astype(float)
            low_prev5 = min(hist["low"].astype(float).iloc[-6:-1]) if "low" in hist.columns else None
            not_new_low = low_prev5 is None or float(close.iloc[-1]) >= low_prev5
            vol = hist["volume"].astype(float)
            shrink = False
            if len(vol) >= 25:
                ma5v = float(vol.iloc[-5:].mean())
                ma_prev = float(vol.iloc[-25:-5].mean())
                shrink = ma_prev > 0 and (ma5v / ma_prev) < 0.9   # 近5日阶段缩量
            # 资金/基本面任一转正
            lhb_rows = (ds.get("lhb_by_code") or {}).get(code) or []
            lhb_net = _num(lhb_rows[0].get("龙虎榜净买额")) if lhb_rows else None
            fund_in = (lhb_net is not None and lhb_net > 0) or good_fund or good_guide
            if not_new_low and shrink and fund_in:
                mild_left = True
                ml_note = f"低分位{pp:.0f}%+阶段缩量+{'资金' if lhb_net is not None and lhb_net>0 else '业绩/预告'}+企稳"

    return {
        "anchors": anchors,
        "mild_left": mild_left,
        "mild_left_note": ml_note,
        "evidence": evidence,
        "passed": passed,
        "hard_block": hard_block,
    }