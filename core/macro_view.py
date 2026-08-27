# -*- coding: utf-8 -*-
"""宏观大盘模块：指数组 + 两市量能 + 外围日韩（用户需求：决策前先看大盘）。

盘后运行，所有数据为收盘终值（非盘中实时）——次日执行时的参照基准。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def build_macro(dp, index_map: dict) -> dict:
    """组装宏观大盘视图。

    dp: DataProvider；index_map: index_symbol_map()。
    返回：{indices: [{name, close, pct_chg, amount_yi}], turnover_yi, turnover_ratio_5d,
           turnover_delta_prev（较前日变化%）, turnover_series（近10日成交额序列，图表用）,
           turnover_trend（趋势描述文案）, global_indices: [{name, pct_chg}], summary: str, missing: []}
    """
    out: dict = {"indices": [], "turnover_yi": None, "turnover_ratio_5d": None,
                 "turnover_delta_prev": None, "turnover_series": [],
                 "turnover_trend": "", "global_indices": [], "summary": "", "missing": []}

    # ---------- 指数组（收盘涨跌 + 成交额）
    idx_amounts: dict[str, float] = {}
    for cname, sym in index_map.items():
        idx = dp.get_index_daily(sym)
        if idx is None or len(idx) < 2:
            out["missing"].append(f"[数据缺失] 指数 {cname}")
            continue
        c0, c1 = float(idx["close"].iloc[-2]), float(idx["close"].iloc[-1])
        pct = (c1 - c0) / c0 * 100.0 if c0 else None
        amt = None
        if "amount" in idx.columns:
            try:
                amt = float(idx["amount"].iloc[-1])
            except (TypeError, ValueError):
                amt = None
        out["indices"].append({"name": cname, "close": round(c1, 2),
                               "pct_chg": None if pct is None else round(pct, 2),
                               "amount_yi": None if amt is None else round(amt / 1e8, 0)})
        if amt is not None:
            idx_amounts[cname] = amt

    # ---------- 两市量能序列（近10日，沪+深指数成交额；缺失日剔除）与衍生指标
    sh_idx = dp.get_index_daily(index_map["上证指数"]) if "上证指数" in index_map else None
    sz_idx = dp.get_index_daily(index_map["深证成指"]) if "深证成指" in index_map else None
    series: list[dict] = []
    if sh_idx is not None and sz_idx is not None and len(sh_idx) and len(sz_idx):
        n = min(len(sh_idx), len(sz_idx), 10)
        for i in range(n, 0, -1):
            try:
                d = str(sh_idx["date"].iloc[-i])[:10]
                a = pd.to_numeric(sh_idx["amount"], errors="coerce").iloc[-i]
                b = pd.to_numeric(sz_idx["amount"], errors="coerce").iloc[-i]
                if pd.notna(a) and pd.notna(b) and (a + b) > 0:
                    series.append({"date": d, "yi": round(float(a + b) / 1e8, 0)})
            except (KeyError, TypeError, ValueError):
                continue
    if series:
        out["turnover_series"] = series
        out["turnover_yi"] = series[-1]["yi"]
        # 较前日
        if len(series) >= 2:
            prev, today = series[-2]["yi"], series[-1]["yi"]
            if prev > 0:
                out["turnover_delta_prev"] = round((today - prev) / prev * 100, 1)
        # 量能比（vs 5日均，沿用序列口径）
        if len(series) >= 6:
            ma5 = sum(s["yi"] for s in series[-6:-1]) / 5
            if ma5 > 0:
                out["turnover_ratio_5d"] = round(series[-1]["yi"] / ma5, 2)
        out["turnover_trend"] = _trend_desc(series)
    else:
        # 先做当日成交额快照兜底（校准锚点）
        if out["turnover_yi"] is None:
            get_spot = getattr(dp, "get_spot", None)
            spot = get_spot() if callable(get_spot) else None
            if spot is not None and not spot.empty and "amount" in spot.columns:
                codes = spot["code"].astype(str).str.zfill(6)
                hs = spot[codes.str[:2].isin(["60", "68", "00", "30"])]
                total = pd.to_numeric(hs["amount"], errors="coerce").sum()
                if total > 0:
                    out["turnover_yi"] = round(float(total) / 1e8, 0)
        # 降级：指数无成交额列（东财封禁走新浪，新浪指数仅 volume）→ 用两市成交量序列，
        # 以"当日总成交额（含快照兜底）"校准成估算额（方向与比例等价，标注估算口径）
        vol_series: list[dict] = []
        if sh_idx is not None and sz_idx is not None and len(sh_idx) and len(sz_idx):
            n = min(len(sh_idx), len(sz_idx), 10)
            for i in range(n, 0, -1):
                try:
                    d = str(sh_idx["date"].iloc[-i])[:10]
                    va = pd.to_numeric(sh_idx["volume"], errors="coerce").iloc[-i]
                    vb = pd.to_numeric(sz_idx["volume"], errors="coerce").iloc[-i]
                    if pd.notna(va) and pd.notna(vb) and (va + vb) > 0:
                        vol_series.append({"date": d, "v": float(va + vb)})
                except (KeyError, TypeError, ValueError):
                    continue
        if vol_series:
            # 当日总成交额（快照兜底已算）作为校准锚点
            today_amt = out["turnover_yi"]
            if today_amt:
                scale = float(today_amt) / vol_series[-1]["v"]
                series = [{"date": s["date"], "yi": round(s["v"] * scale, 0)} for s in vol_series]
                out["turnover_series"] = series
                if len(series) >= 2:
                    prev, today = series[-2]["yi"], series[-1]["yi"]
                    if prev > 0:
                        out["turnover_delta_prev"] = round((today - prev) / prev * 100, 1)
                if len(series) >= 6:
                    ma5 = sum(s["yi"] for s in series[-6:-1]) / 5
                    if ma5 > 0:
                        out["turnover_ratio_5d"] = round(series[-1]["yi"] / ma5, 2)
                out["turnover_trend"] = _trend_desc(series) + "〔量估算：按当日额校准的量能序列〕"

    if out["turnover_yi"] is None:
        # 指数成交额缺失（东财封禁、新浪指数无额）→ 全市场快照聚合兜底（仅当日，无序列）
        get_spot = getattr(dp, "get_spot", None)
        spot = get_spot() if callable(get_spot) else None
        if spot is not None and not spot.empty and "amount" in spot.columns:
            codes = spot["code"].astype(str).str.zfill(6)
            hs = spot[codes.str[:2].isin(["60", "68", "00", "30"])]
            total = pd.to_numeric(hs["amount"], errors="coerce").sum()
            if total > 0:
                out["turnover_yi"] = round(float(total) / 1e8, 0)

    # ---------- 外围：日经225 / 韩国KOSPI
    gi = dp.get_global_indices()
    if gi is not None and not gi.empty:
        for _, r in gi.iterrows():
            out["global_indices"].append({"name": str(r["name"]),
                                          "pct_chg": None if r.get("pct_chg") is None else round(float(r["pct_chg"]), 2)})
    else:
        out["missing"].append("[数据缺失] 外围指数（日经/KOSPI）")

    # ---------- 一句话总结
    out["summary"] = _summarize(out)
    return out


def _trend_desc(series: list[dict]) -> str:
    """量能趋势描述：较前日方向 + 近5日连续性（连续放量/连续缩量/起伏）。"""
    if len(series) < 3:
        return ""
    diffs = [series[i]["yi"] - series[i - 1]["yi"] for i in range(1, len(series))]
    ups = sum(1 for d in diffs if d > 0)
    downs = sum(1 for d in diffs if d < 0)
    tail = diffs[-5:] if len(diffs) >= 5 else diffs
    t_up = sum(1 for d in tail if d > 0)
    t_dn = sum(1 for d in tail if d < 0)
    if t_up >= 4:
        pat = "近5日持续放量"
    elif t_dn >= 4:
        pat = "近5日持续缩量"
    elif t_up >= 3:
        pat = "近5日量能震荡走高"
    elif t_dn >= 3:
        pat = "近5日量能震荡走低"
    else:
        pat = "量能无方向"
    first, last = series[0]["yi"], series[-1]["yi"]
    overall = (last - first) / first * 100 if first > 0 else 0
    return f"较10日前{overall:+.0f}%，{pat}（{downs}降/{ups}升）"


def _summarize(m: dict) -> str:
    parts: list[str] = []
    # 指数基调
    if m["indices"]:
        up = [i for i in m["indices"] if (i["pct_chg"] or 0) > 0]
        if len(up) == len(m["indices"]):
            parts.append("A股全线收涨")
        elif not up:
            parts.append("A股全线收跌")
        else:
            parts.append(f"指数{len(up)}/{len(m['indices'])}收涨")
    # 量能
    if m["turnover_yi"] is not None:
        t = m["turnover_yi"]
        lvl = "万亿级" if t >= 10000 else ("高位" if t >= 8000 else ("偏暖" if t >= 6000 else "缩量"))
        parts.append(f"两市成交{t:.0f}亿（{lvl}）")
    if m.get("turnover_delta_prev") is not None:
        dlt = m["turnover_delta_prev"]
        parts.append(f"量能较前日{'放量' if dlt > 0 else ('缩量' if dlt < 0 else '持平')}{abs(dlt):.0f}%")
    if m["turnover_ratio_5d"] is not None:
        r = m["turnover_ratio_5d"]
        parts.append(f"较5日均{'放量' if r > 1.1 else ('缩量' if r < 0.9 else '持平')}{abs(r - 1) * 100:.0f}%")
    if m.get("turnover_trend"):
        parts.append(m["turnover_trend"])
    # 外围
    if m["global_indices"]:
        gs = "、".join(f"{g['name']}{g['pct_chg']:+.1f}%" for g in m["global_indices"] if g["pct_chg"] is not None)
        if gs:
            parts.append(f"外围 {gs}")
    return "；".join(parts) if parts else "宏观数据缺失"
