# -*- coding: utf-8 -*-
"""李致远引擎 · 打板质量评分（用户口径重构版，原规格书 §5.3 基础上调整）。

用户口径（与原规格差异处已标注）：
- 只看主板（60/00）——用户合规约束 [新增]
- 首半优先、最多三板——原"连板越高分越高"改为板位递减 [口径变更]
- 必须收盘仍封住涨停（收盘涨幅≥9.8%）[新增]
- 换手率 2%-25%：下限防无量板，上限防极端分歧 [新增]
- 成交额≥2亿（原主升池5000万门槛，打板更严格）[新增]
- 量能不过度失控（量比≤5）[新增]
- 主力净流入>0 [新增]
- 孤立涨停：降级扣15分（原R8直接排除，改为宽容处理保留独立走强通道）[口径变更]

硬排除清单（任一命中出局）：
R1 ST/*ST；R2 上市未满60交易日；R3 一字涨停板；R4 炸板≥2次或收盘未封住；
R5 距20日低点涨幅>50%；R6 非主板；R7 连板>3板；R8 换手率越界；
R9 成交额<2亿；R10 量比>5（失控）；R11 主力净流出。

六维评分（满分100，≥60进打板观察池）：
封板时间20（首封越早越好，"首半优先"的时间维度）/ 板位高度15（首板15>二板10>三板5）/
量能结构15 / 换手结构15 / 主力净额15 / 板块共振20（梯队>共振>孤立5分并总分-15）。

执行与放弃条件（池纪律，输出到仪表盘）：
仅在涨停价仍封住且能正常成交时观察；炸板不回封 / 封单快速衰减 / 板块前排转弱 → 放弃。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .data_provider import is_main_board


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def precheck_exclude(row: dict, cfg: dict) -> Optional[str]:
    """涨停池预筛（纯行字段硬排除，无需日K）——供 main.py 在拉取日K前剪枝，
    避免对注定出局的股票（非主板/连板>3/换手越界/成交额不足等）浪费请求。

    仅覆盖 R1/R3/R4/R6/R7/R8/R9（不依赖 hist 的规则）；
    R2/R5/R10/R11 等仍由 score_limitup 完整判定（预筛只是提前剪枝，非最终守门）。
    """
    lcfg = cfg["limitup"]
    name = str(row.get("name", ""))
    code = str(row.get("code", "")).zfill(6)
    if "ST" in name.upper():                                    # R1
        return "R1: ST"
    fst = str(row.get("first_seal_time", "") or "")
    ot = int(_num(row.get("open_times")) or 0)
    if fst.replace(":", "").startswith(("0925", "925")) and ot == 0:   # R3
        return "R3: 一字板"
    if ot > int(lcfg["max_open_times"]):                        # R4a
        return f"R4: 炸板{ot}次"
    pct = _num(row.get("pct_chg"))
    if pct is not None and pct < 9.8:                           # R4b
        return "R4: 收盘未封住"
    if lcfg.get("main_board_only") and not is_main_board(code):        # R6
        return "R6: 非主板"
    lb = int(_num(row.get("lian_ban")) or 1)
    if lb > int(lcfg["max_lianban"]):                           # R7
        return f"R7: {lb}连板"
    t_over = _num(row.get("turnover"))                          # R8
    t_lo, t_hi = float(lcfg["turnover_range"][0]), float(lcfg["turnover_range"][1])
    if t_over is not None and not (t_lo <= t_over <= t_hi):
        return "R8: 换手越界"
    amt = _num(row.get("amount"))                               # R9
    if amt is not None and amt < float(lcfg["min_amount"]):
        return "R9: 成交额不足"
    return None


def score_limitup(row: dict, hist: Optional[pd.DataFrame], board_ctx: dict,
                  sentiment_temp: Optional[float], cfg: dict) -> dict:
    """row：涨停池单行（code/name/lian_ban/open_times/first_seal_time/price/amount/
    turnover/pct_chg/main_net_pct...）。board_ctx：board/board_zt_count/board_pct_chg/
    board_main_net_pct/board_ladder。"""
    lcfg = cfg["limitup"]
    code = str(row.get("code", "")).zfill(6)
    name = str(row.get("name", ""))

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
    if ot > int(lcfg["max_open_times"]):                        # R4a 炸板≥2次
        return _exclude(f"R4: 当日炸板{ot}次≥2次")
    pct = _num(row.get("pct_chg"))
    if pct is not None and pct < 9.8:                           # R4b 收盘未封住
        return _exclude(f"R4: 收盘未封住涨停（{pct:.1f}%）")
    price = _num(row.get("price"))
    if hist is not None and price is not None and len(hist) >= 20:   # R5 高位
        low20 = float(hist["low"].astype(float).iloc[-20:].min())
        if low20 > 0 and (price - low20) / low20 * 100 > float(lcfg["high_position_pct"]):
            return _exclude(f"R5: 距20日低点涨幅{(price - low20) / low20 * 100:.0f}%>50%")
    if lcfg.get("main_board_only") and not is_main_board(code):        # R6 非主板
        return _exclude("R6: 非主板（用户约束只做主板）")
    lb = int(_num(row.get("lian_ban")) or 1)
    if lb > int(lcfg["max_lianban"]):                           # R7 连板>3
        return _exclude(f"R7: {lb}连板>{lcfg['max_lianban']}板（首半优先口径）")
    t_over = _num(row.get("turnover"))                          # R8 换手越界
    t_lo, t_hi = float(lcfg["turnover_range"][0]), float(lcfg["turnover_range"][1])
    if t_over is not None and not (t_lo <= t_over <= t_hi):
        return _exclude(f"R8: 换手率{t_over:.1f}%越界[{t_lo:.0f}%,{t_hi:.0f}%]")
    amt = _num(row.get("amount"))                               # R9 成交额
    if amt is not None and amt < float(lcfg["min_amount"]):
        return _exclude(f"R9: 成交额{amt / 1e8:.2f}亿<2亿")
    # 量比（近20日均量基准）
    vr = None
    if hist is not None and len(hist) >= 21:
        vma20 = float(hist["volume"].astype(float).rolling(20).mean().iloc[-1])
        if vma20 > 0:
            vr = float(hist["volume"].astype(float).iloc[-1]) / vma20
    if vr is not None and vr > float(lcfg["max_volume_ratio"]):        # R10 失控
        return _exclude(f"R10: 量比{vr:.1f}>5（量能过度失控）")
    main_net_pct = _num(row.get("main_net_pct"))                # R11 主力净流出
    if lcfg.get("require_main_inflow"):
        if main_net_pct is None:
            return _exclude("R11: 主力资金数据缺失（要求净流入验证）")
        if main_net_pct <= 0:
            return _exclude(f"R11: 主力净流出{main_net_pct:.1f}%")

    # ---------------- 六维评分
    dims: dict = {}

    # 封板时间（20）：首半优先的时间维度
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

    # 板位高度（15）：首半优先——首板15 / 二板10 / 三板5
    dims["板位高度"] = {1: 15.0, 2: 10.0, 3: 5.0}.get(lb, 5.0)
    dims["板位高度_说明"] = f"{lb}板（首半优先）"

    # 量能结构（15）：涨停放量适度为佳
    if vr is None:
        dims["量能结构"] = 7.5
        dims["量能结构_说明"] = "量能数据缺失取中值"
    elif 1.5 <= vr <= 3:
        dims["量能结构"] = 15.0
        dims["量能结构_说明"] = f"量比{vr:.2f}（健康放量）"
    elif 1.2 <= vr < 1.5 or 3 < vr <= 5:
        dims["量能结构"] = 10.0
        dims["量能结构_说明"] = f"量比{vr:.2f}"
    else:
        dims["量能结构"] = 5.0
        dims["量能结构_说明"] = f"量比{vr:.2f}（偏弱）"

    # 换手结构（15）：5-15%活跃适中
    if t_over is None:
        dims["换手结构"] = 7.5
    elif 5 <= t_over <= 15:
        dims["换手结构"] = 15.0
    elif 2 <= t_over < 5 or 15 < t_over <= 20:
        dims["换手结构"] = 10.0
    else:
        dims["换手结构"] = 5.0
    if t_over is not None:
        dims["换手结构_说明"] = f"换手{t_over:.1f}%"

    # 主力净额（15）：净流入占比越高越好
    if main_net_pct is None:
        dims["主力净额"] = 7.5
    elif main_net_pct >= 3:
        dims["主力净额"] = 15.0
    elif main_net_pct > 0:
        dims["主力净额"] = 10.0
    if main_net_pct is not None:
        dims["主力净额_说明"] = f"主力净占比{main_net_pct:+.1f}%"

    # 板块共振（20）：梯队 > 共振 > 孤立（孤立不排除，降级）
    zt_cnt = int(board_ctx.get("board_zt_count") or 0)
    isolated = zt_cnt <= 1
    if board_ctx.get("board_ladder", 0) >= 3 and not isolated:
        dims["板块共振"] = 20.0
        dims["板块共振_说明"] = "涨停梯队模式（板块内1/2/3板完整梯队）"
    elif not isolated:
        dims["板块共振"] = 18.0
        dims["板块共振_说明"] = f"板块共振（同板块{zt_cnt}只涨停）"
    else:
        dims["板块共振"] = 5.0
        dims["板块共振_说明"] = "孤立涨停（独立走强，降级处理）"

    score = round(sum(v for k, v in dims.items()
                      if not k.endswith("_说明") and isinstance(v, float)), 1)
    # 孤立涨停降级扣分
    if isolated:
        score = round(max(0.0, score - float(lcfg["isolated_penalty"])), 1)

    in_pool = score >= float(lcfg["score_threshold"])
    return {
        "code": code, "name": name, "excluded": False, "exclude_reason": "",
        "score": score, "dims": dims, "in_pool": in_pool,
        "lian_ban": lb, "open_times": ot, "first_seal_time": fst,
        "vr": None if vr is None else round(vr, 2), "board": board_ctx.get("board"),
        "main_net_pct": main_net_pct, "turnover": t_over,
    }


# 打板池执行/放弃纪律（用户口径，输出到仪表盘）
LIMITUP_DISCIPLINE = [
    "执行：仅在涨停价仍封住且能正常成交时观察（排板，不追高开）",
    "放弃：炸板不回封 / 封单快速衰减 / 板块前排转弱 → 任一出现即放弃",
    "首半优先：首板>二板>三板，三板以上不碰（低吸埋伏定位）",
]


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
