# -*- coding: utf-8 -*-
"""李致远引擎 · 打板质量评分（八条硬排除 + 六维评分，规格书 §5.3）。

硬排除 R1-R8 任一命中直接排除，不进评分：
R1 ST/*ST；R2 上市未满60交易日；R3 一字涨停板；R4 当日炸板≥2次；
R5 股价距20日低点涨幅>50%；R6 无量涨停；R7 纯公告利好无板块协同；R8 独苗涨停。

六维评分（满分100，≥60进打板观察池）：封板时间20/封板强度20/量能结构15/
板块协同20/连板高度15/市场情绪10。

量化说明（规格未细化处的实现约定，已在 docstring 标注）：
- 封板时间：10:00前20；10:00-11:00 15；11:00-14:00 10（规格未列11:00-13:00档，
  并入10分档）；14:00后5。
- R6 无量涨停：量比<0.5。
- R7 板块协同近似判定（无公告语义数据）：板块涨幅≥2% 或 板块主力净流入>0
  视为有协同，否则排除并标注"近似判定"（同板块涨停数由 R8 单独把关）。
- 板块协同维度：涨停梯队模式20分；板块共振模式18分（源文档基准57/61分，
  保序映射到本维度20分制）。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def score_limitup(row: dict, hist: Optional[pd.DataFrame], board_ctx: dict,
                  sentiment_temp: Optional[float], cfg: dict) -> dict:
    """row：涨停池单行（code/name/lian_ban/open_times/first_seal_time/price/amount...）。
    board_ctx：board/board_zt_count/board_pct_chg/board_main_net_pct/board_ladder(板块内连板档数)。"""
    lcfg = cfg["limitup"]
    code = str(row.get("code", "")).zfill(6)
    name = str(row.get("name", ""))
    notes: list[str] = []

    # ---------------- 硬排除
    def _exclude(reason: str) -> dict:
        return {"code": code, "name": name, "excluded": True, "exclude_reason": reason,
                "score": 0, "dims": {}, "in_pool": False}

    if "ST" in name.upper():                                    # R1
        return _exclude("R1: ST/*ST")
    if hist is not None and len(hist) < 60:                     # R2
        return _exclude(f"R2: 次新股（日K仅{len(hist)}条<60交易日）")
    fst = str(row.get("first_seal_time", "") or "")
    ot = int(_num(row.get("open_times")) or 0)
    if fst.replace(":", "").startswith(("0925", "925")) and ot == 0:   # R3 一字板
        return _exclude("R3: 一字涨停板（开盘封板未打开）")
    if ot > int(lcfg["max_open_times"]):                        # R4 炸板≥2次
        return _exclude(f"R4: 当日炸板{ot}次≥2次")
    price = _num(row.get("price"))
    if hist is not None and price is not None and len(hist) >= 20:   # R5 高位接力
        low20 = float(hist["low"].astype(float).iloc[-20:].min())
        if low20 > 0 and (price - low20) / low20 * 100 > float(lcfg["high_position_pct"]):
            return _exclude(f"R5: 距20日低点涨幅{(price - low20) / low20 * 100:.0f}%>50%")
    vr = None
    if hist is not None and len(hist) >= 21:
        vma20 = float(hist["volume"].astype(float).rolling(20).mean().iloc[-1])
        if vma20 > 0:
            vr = float(hist["volume"].astype(float).iloc[-1]) / vma20
    if vr is not None and vr < 0.5:                             # R6 无量涨停
        return _exclude(f"R6: 无量涨停（量比{vr:.2f}<0.5）")
    # R7 / R8 板块协同
    zt_cnt = int(board_ctx.get("board_zt_count") or 0)
    board_pct = _num(board_ctx.get("board_pct_chg"))
    board_net = _num(board_ctx.get("board_main_net_pct"))
    if zt_cnt <= 1:                                             # R8 独苗涨停
        return _exclude("R8: 板块内无其他个股联动（独苗涨停）")
    synergy = (board_pct is not None and board_pct >= 2) or (board_net is not None and board_net > 0)
    if not synergy:                                             # R7 近似判定
        return _exclude("R7: 疑似纯公告利好、无板块协同（近似判定）")

    # ---------------- 六维评分
    dims: dict = {}

    # 封板时间（20）
    t = fst.replace(":", "")
    if t < "1000":
        dims["封板时间"] = 20.0
    elif t < "1100":
        dims["封板时间"] = 15.0
    elif t < "1400":
        dims["封板时间"] = 10.0
    else:
        dims["封板时间"] = 5.0
    dims["封板时间_说明"] = f"首封{fst}"

    # 封板强度（20）：开板0次20；1次10
    dims["封板强度"] = 20.0 if ot == 0 else 10.0

    # 量能结构（15）：涨停前放量、封板后缩量为佳 → 按量比分档
    if vr is None:
        dims["量能结构"] = 7.5
        dims["量能结构_说明"] = "量能数据缺失取中值"
    elif vr >= 1.5:
        dims["量能结构"] = 15.0
    elif vr >= 1.2:
        dims["量能结构"] = 10.0
    else:
        dims["量能结构"] = 5.0

    # 板块协同（20）：梯队20；共振18
    if board_ctx.get("board_ladder", 0) >= 3:
        dims["板块协同"] = 20.0
        dims["板块协同_说明"] = "涨停梯队模式（板块内1/2/3板完整梯队）"
    else:
        dims["板块协同"] = 18.0
        dims["板块协同_说明"] = f"板块共振模式（同板块{zt_cnt}只涨停）"

    # 连板高度（15）：首板5；2连板10；≥3连板15
    lb = int(_num(row.get("lian_ban")) or 1)
    dims["连板高度"] = 5.0 if lb == 1 else (10.0 if lb == 2 else 15.0)

    # 市场情绪（10）：≥62满分；48-61得6；<48得0
    if sentiment_temp is None:
        dims["市场情绪"] = 0.0
        dims["市场情绪_说明"] = "情绪数据缺失"
    elif sentiment_temp >= 62:
        dims["市场情绪"] = 10.0
    elif sentiment_temp >= 48:
        dims["市场情绪"] = 6.0
    else:
        dims["市场情绪"] = 0.0

    score = round(sum(v for k, v in dims.items()
                      if not k.endswith("_说明") and isinstance(v, float)), 1)
    in_pool = score >= float(lcfg["score_threshold"])
    if notes:
        dims["备注"] = "；".join(notes)

    return {
        "code": code, "name": name, "excluded": False, "exclude_reason": "",
        "score": score, "dims": dims, "in_pool": in_pool,
        "lian_ban": lb, "open_times": ot, "first_seal_time": fst,
        "vr": None if vr is None else round(vr, 2), "board": board_ctx.get("board"),
    }


def board_ladder_count(zt_pool: Optional[pd.DataFrame], board: Optional[str],
                       stock_board_map: dict) -> int:
    """板块内涨停连板档数（1板/2板/3板及以上 各算一档，≥3档为完整梯队）。"""
    if zt_pool is None or zt_pool.empty or "code" not in zt_pool.columns:
        return 0
    lbs = set()
    for _, r in zt_pool.iterrows():
        if stock_board_map.get(str(r["code"]).zfill(6)) != board:
            continue
        lb = int(_num(r.get("lian_ban")) or 1)
        lbs.add(min(lb, 3))
    return len(lbs)
