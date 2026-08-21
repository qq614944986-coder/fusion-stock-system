# -*- coding: utf-8 -*-
"""宏观大盘模块：指数组 + 两市量能 + 外围日韩（用户需求：决策前先看大盘）。

盘后运行，所有数据为收盘终值（非盘中实时）——次日执行时的参照基准。
"""
from __future__ import annotations

from typing import Optional


def build_macro(dp, index_map: dict) -> dict:
    """组装宏观大盘视图。

    dp: DataProvider；index_map: index_symbol_map()。
    返回：{indices: [{name, close, pct_chg, amount_yi}], turnover_yi, turnover_ratio_5d,
           global_indices: [{name, pct_chg}], summary: str, missing: []}
    """
    out: dict = {"indices": [], "turnover_yi": None, "turnover_ratio_5d": None,
                 "global_indices": [], "summary": "", "missing": []}

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

    # ---------- 两市成交额（沪+深）与量能比（vs 5日均）
    sh, sz = idx_amounts.get("上证指数"), idx_amounts.get("深证成指")
    if sh is not None and sz is not None:
        out["turnover_yi"] = round((sh + sz) / 1e8, 0)
        # 量能比：用上证指数近5日成交额均值近似（两市结构稳定，单指数代理可行）
        sh_idx = dp.get_index_daily(index_map["上证指数"])
        if sh_idx is not None and "amount" in sh_idx.columns and len(sh_idx) >= 6:
            try:
                ma5 = float(sh_idx["amount"].iloc[-6:-1].mean())
                today = float(sh_idx["amount"].iloc[-1])
                if ma5 > 0:
                    out["turnover_ratio_5d"] = round(today / ma5, 2)
            except (TypeError, ValueError):
                pass

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
    if m["turnover_ratio_5d"] is not None:
        r = m["turnover_ratio_5d"]
        parts.append(f"量能较5日均{'放量' if r > 1.1 else ('缩量' if r < 0.9 else '持平')}{abs(r - 1) * 100:.0f}%")
    # 外围
    if m["global_indices"]:
        gs = "、".join(f"{g['name']}{g['pct_chg']:+.1f}%" for g in m["global_indices"] if g["pct_chg"] is not None)
        if gs:
            parts.append(f"外围 {gs}")
    return "；".join(parts) if parts else "宏观数据缺失"
