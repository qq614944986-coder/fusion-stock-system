# -*- coding: utf-8 -*-
"""李致远引擎 · 五区情绪温度（规格书 §4）。

情绪温度 = 20%×上涨占比 + 20%×涨跌停比 + 20%×指数温度
         + 15%×涨停活跃度 + 15%×情绪龙头 + 10%×情绪驱动力

情绪驱动力依赖前一日温度：为避免循环依赖，先以其余五因子合成"基础温度"
（权重归一化），今日与昨日基础温度之差 Δ 映射驱动力分值，再合成最终温度。
前日温度缺失时驱动力取中性 50 并标注（数据诚实，不编造）。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

# 五区操作标签与纪律（§4.2 硬规则文字）
ZONE_LABELS = {
    "冰点区": {"label": "等待止跌", "rule": "禁止加仓，只减不加"},
    "退潮区": {"label": "控制风险", "rule": "仅盈利头寸小幅加仓"},
    "震荡区": {"label": "只做最强", "rule": "只加仓最强龙头"},
    "偏强区": {"label": "适度参与", "rule": "主升浪确认股可加仓"},
    "高热区": {"label": "防止见顶", "rule": "禁止开新仓，逐步减仓"},
}

# 情绪驱动力映射：Δ≥+5→90；(0,+5]→70；[-3,0]→50；(-8,-3)→30；Δ≤-8→10
MOMENTUM_MAP = [
    (5.0, 90, "ge"), (0.0, 70, "gt"), (-3.0, 50, "ge"), (-8.0, 30, "gt"),
]


def momentum_score(delta: Optional[float]) -> tuple[float, str]:
    if delta is None:
        return 50.0, "前日温度缺失，驱动力取中性50[数据缺失]"
    if delta >= 5:
        return 90.0, f"Δ={delta:+.1f}≥+5 → 90"
    if delta > 0:
        return 70.0, f"Δ={delta:+.1f}∈(0,+5] → 70"
    if delta >= -3:
        return 50.0, f"Δ={delta:+.1f}∈[-3,0] → 50"
    if delta > -8:
        return 30.0, f"Δ={delta:+.1f}∈(-8,-3) → 30"
    return 10.0, f"Δ={delta:+.1f}≤-8 → 10"


def leader_score(zt_pool: Optional[pd.DataFrame]) -> tuple[float, str]:
    """情绪龙头：最高连板高度≥5得90；每降1板-10；最高板炸板再-20；无连板（全断板）得30。"""
    if zt_pool is None or zt_pool.empty or "lian_ban" not in zt_pool.columns:
        return 30.0, "涨停池数据缺失，按全断板取30[数据缺失]"
    h = pd.to_numeric(zt_pool["lian_ban"], errors="coerce").fillna(0)
    max_h = int(h.max())
    if max_h < 2:
        return 30.0, "无连板（全断板）→ 30"
    eff_h = min(max_h, 5)
    score = 90 - (5 - eff_h) * 10
    note = f"最高连板{max_h}板 → {score}"
    # 最高板炸板再-20
    top = zt_pool[h == max_h]
    if "open_times" in top.columns:
        ot = pd.to_numeric(top["open_times"], errors="coerce").fillna(0)
        if float(ot.max()) > 0:
            score -= 20
            note += "；最高板当日炸板 -20"
    return float(score), note


def zt_quality_score(zt_pool: Optional[pd.DataFrame]) -> tuple[float, str]:
    """涨停活跃度：非一字板且炸板次数=0 的涨停股占比（0-100）。"""
    if zt_pool is None or zt_pool.empty:
        return 50.0, "涨停池数据缺失，取中性50[数据缺失]"
    n = len(zt_pool)
    one_word = pd.Series(False, index=zt_pool.index)
    if "first_seal_time" in zt_pool.columns:
        t = zt_pool["first_seal_time"].astype(str).str.replace(":", "")
        one_word = t.str.startswith("0925") | t.str.startswith("925")
    open0 = pd.Series(True, index=zt_pool.index)
    if "open_times" in zt_pool.columns:
        open0 = pd.to_numeric(zt_pool["open_times"], errors="coerce").fillna(0) == 0
    active = (~one_word) & open0
    ratio = float(active.sum()) / n * 100.0
    return ratio, f"活跃涨停{int(active.sum())}/{n} = {ratio:.0f}%"


def compute_sentiment(data: dict, cfg: dict, prev_base_temp: Optional[float]) -> dict:
    """data 键：rise/fall/flat/limit_up/limit_down(家数)、index_pct(三指数涨跌幅dict)、
    zt_pool(可选DataFrame)。返回六因子、温度、五区与仓位约束。"""
    s = cfg["sentiment"]
    missing: list[str] = []

    # ---- 上涨占比
    rise, fall, flat = data.get("rise"), data.get("fall"), data.get("flat")
    if None in (rise, fall, flat) or (rise + fall + flat) <= 0:
        f_rise, w_rise = 50.0, 0.0
        missing.append("上涨占比[数据缺失]")
    else:
        f_rise = rise / (rise + fall + flat) * 100.0
        w_rise = float(s["weight_rise_ratio"])

    # ---- 涨跌停比（跌停为0时取100）
    lu, ld = data.get("limit_up"), data.get("limit_down")
    if lu is None or ld is None:
        f_limit, w_limit = 50.0, 0.0
        missing.append("涨跌停比[数据缺失]")
    elif lu == 0 and ld == 0:
        f_limit, w_limit = 50.0, float(s["weight_limit_ratio"])
        missing.append("涨跌停家数均为0，取中性50[数据缺失]")
    elif ld == 0:
        f_limit, w_limit = 100.0, float(s["weight_limit_ratio"])
    else:
        f_limit, w_limit = lu / (lu + ld) * 100.0, float(s["weight_limit_ratio"])

    # ---- 指数温度：(三指数均值 + range)/ (2×range) ×100，截断[0,100]
    idx = data.get("index_pct") or {}
    vals = [v for v in idx.values() if v is not None]
    rng = float(s["index_temp_range"])
    if not vals:
        f_idx, w_idx = 50.0, 0.0
        missing.append("指数温度[数据缺失]")
    else:
        x = sum(vals) / len(vals)
        f_idx = (x + rng) / (2 * rng) * 100.0
        f_idx = max(0.0, min(100.0, f_idx))
        w_idx = float(s["weight_index_temp"])

    # ---- 涨停活跃度 / 情绪龙头
    zt = data.get("zt_pool")
    f_zt, note_zt = zt_quality_score(zt)
    if "[数据缺失]" in note_zt:
        w_zt = 0.0
    else:
        w_zt = float(s["weight_zt_quality"])
    f_ldr, note_ldr = leader_score(zt)
    if "[数据缺失]" in note_ldr and "涨停池数据缺失" in note_ldr:
        w_ldr = 0.0
    else:
        w_ldr = float(s["weight_leader"])

    # ---- 基础温度（五因子权重归一化）
    w_sum = w_rise + w_limit + w_idx + w_zt + w_ldr
    if w_sum <= 0:
        base = 50.0
        missing.append("全部基础因子缺失[数据缺失]")
    else:
        base = (f_rise * w_rise + f_limit * w_limit + f_idx * w_idx
                + f_zt * w_zt + f_ldr * w_ldr) / w_sum

    # ---- 情绪驱动力（基于基础温度变化，避免循环）
    delta = None if prev_base_temp is None else base - prev_base_temp
    f_mom, note_mom = momentum_score(delta)
    w_mom = float(s["weight_momentum"])

    # ---- 最终温度（有缺失因子时权重归一化，保证0-100）
    total_w = w_sum + w_mom
    temp = base * (w_sum / total_w) + f_mom * (w_mom / total_w) if total_w > 0 else 50.0

    # ---- 五区分类
    zone = classify_zone(temp, cfg["zones"])
    zinfo = dict(zone)
    zinfo.update(ZONE_LABELS.get(zinfo["name"], {}))

    return {
        "date": data.get("date"),
        "factors": {
            "上涨占比": round(f_rise, 1), "涨跌停比": round(f_limit, 1),
            "指数温度": round(f_idx, 1), "涨停活跃度": round(f_zt, 1),
            "情绪龙头": round(f_ldr, 1), "情绪驱动力": round(f_mom, 1),
        },
        "factor_notes": {"涨停活跃度": note_zt, "情绪龙头": note_ldr, "情绪驱动力": note_mom},
        "base_temp": round(base, 2),
        "delta": None if delta is None else round(delta, 2),
        "temperature": round(temp, 1),
        "zone": zinfo,
        "missing": missing,
    }


def classify_zone(temp: float, zones: list) -> dict:
    """五区分类：按配置区间 min≤分数≤max。"""
    for z in zones:
        if float(z["min"]) <= temp <= float(z["max"]):
            return dict(z)
    return dict(zones[-1])
