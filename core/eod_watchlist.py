# -*- coding: utf-8 -*-
"""李致远引擎 · 尾盘观察池（次日关注，盘后回算，规格书 §5.4）。

入选条件（全部满足）：
1. 当日涨幅 3%-9.5%（未涨停）或涨停稳封（炸板次数=0）；
2. 成交量 > 5日均量×1.2；
3. 所属板块当日排名前 10；
4. 14:30 后不回落（日频版简化：收盘价 > 当日均价，均价=成交额/成交量）；
5. 中大阳线实体饱满（实体/振幅 > 60%）。

尾盘纪律提示（输出到建议）：买入时间 14:50-14:55；单票 ≤15% 仓位；
次日高开>3% 可止盈；涨超8% 不追。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

DISCIPLINE_NOTES = [
    "买入时间 14:50-14:55",
    "单票 ≤15% 仓位",
    "次日高开>3% 可止盈",
    "涨超8% 不追",
]


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def check_eod(row: dict, hist: Optional[pd.DataFrame], board_rank: Optional[int],
              cfg: dict) -> dict:
    """row：spot 单行（code/name/price/pct_chg/open/high/low/volume/amount/vr/pre_close）。
    hist：日K（用于5日均量，优先精确计算；缺失时降级用spot量比近似并标注）。"""
    ecfg = cfg["eod"]
    code = str(row.get("code", "")).zfill(6)
    name = str(row.get("name", ""))
    pct = _num(row.get("pct_chg"))
    price = _num(row.get("price"))
    o, h, l = _num(row.get("open")), _num(row.get("high")), _num(row.get("low"))
    vol = _num(row.get("volume"))
    amt = _num(row.get("amount"))
    notes: list[str] = []

    ok = True
    reason = []

    # 1 涨幅区间（未涨停）或涨停稳封
    lo, hi = float(ecfg["gain_range"][0]), float(ecfg["gain_range"][1])
    if pct is None:
        ok, _ = False, reason.append("涨幅数据缺失")
    elif lo <= pct <= hi:
        pass
    elif pct > hi:
        # 涨停：需要稳封（spot 无法直接看炸板 → 用收盘=最高近似：收盘仍封死）
        if price is not None and h is not None and abs(price - h) < 1e-6:
            notes.append("涨停稳封（收盘=最高）")
        else:
            ok = False
            reason.append(f"涨停但尾盘未稳封(收{price}/高{h})")
    else:
        ok = False
        reason.append(f"涨幅{pct}%不在{lo}%-{hi}%")

    # 2 成交量 > 5日均量×1.2（hist 优先；缺失降级 spot 量比）
    vr_min = float(ecfg["volume_ratio_min"])
    if hist is not None and len(hist) >= 6:
        v5 = float(hist["volume"].astype(float).iloc[-6:-1].mean())
        vol_ratio = vol / v5 if (v5 > 0 and vol is not None) else None
    else:
        vol_ratio = _num(row.get("vr"))
        if vol_ratio is not None:
            notes.append("5日均量以spot量比近似[数据降级]")
    if vol_ratio is None:
        ok = False
        reason.append("量能数据缺失")
    elif not vol_ratio > vr_min:
        ok = False
        reason.append(f"量比{vol_ratio:.2f}≤{vr_min}")

    # 3 板块排名前10
    if board_rank is None:
        ok = False
        reason.append("板块排名数据缺失")
    elif board_rank > int(ecfg["sector_top_n"]):
        ok = False
        reason.append(f"板块排名#{board_rank}未进前{int(ecfg['sector_top_n'])}")

    # 4 收盘价 > 当日均价（spot: 成交额元 / (成交量手×100)）
    if None in (amt, vol) or vol <= 0:
        ok = False
        reason.append("均价数据缺失")
    else:
        avg_price = amt / (vol * 100)
        if price is None or price <= avg_price:
            ok = False
            reason.append(f"收盘{price}≤当日均价{avg_price:.2f}（14:30后回落）")

    # 5 中大阳线实体饱满：实体/振幅>60%
    if None in (o, h, l, price) or (h - l) <= 0:
        ok = False
        reason.append("K线数据缺失")
    else:
        body_ratio = abs(price - o) / (h - l)
        if body_ratio <= 0.6:
            ok = False
            reason.append(f"实体/振幅{body_ratio:.0%}≤60%")

    return {
        "code": code, "name": name, "pct_chg": pct, "price": price,
        "vol_ratio": None if vol_ratio is None else round(vol_ratio, 2),
        "board_rank": board_rank, "selected": ok,
        "reason": "；".join(reason) if not ok else "全部条件满足",
        "notes": notes, "discipline": DISCIPLINE_NOTES,
    }
