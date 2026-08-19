# -*- coding: utf-8 -*-
"""李致远引擎 · 主升浪个股评分（八因子，满分100，规格书 §5.2）。

入池标准：评分≥70 且通过板块门槛 → 主升候选池（中线维度）。
等级：85-100 极强主升（可重仓）；70-84 主升进行（标准仓）；
     55-69 观察关注；<55 不推荐。
特殊标记：短线过热(MA20乖离>19%) / 突破确认 / 回踩买点 / 板块门槛未过 / 形态扣分。

实现说明：扣分规则的分档边界规格未逐一给出，按"扣分区间中值"落地并注释标注；
本模块使用常规 MA5/10/20（李致远体系），与老樊均线带（MA5/8/13、MA55/60/65）互不影响。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def _v(df: pd.DataFrame, col: str, i: int = -1) -> Optional[float]:
    if df is None or len(df) == 0 or col not in df.columns:
        return None
    v = df[col].iloc[i]
    return None if pd.isna(v) else float(v)


def score_stock(hist: Optional[pd.DataFrame], ctx: dict, cfg: dict) -> dict:
    """hist：日K（含pct_chg/volume）；ctx：code/name/board/board_rank/board_zt_count/
    board_rank_in_stock(板块内涨幅名次)/main_net_pct/sentiment_temp/spot_vr/turnover。"""
    scfg = cfg["stock_score"]
    code = ctx.get("code", "")
    name = ctx.get("name", "")
    markers: list[str] = []
    detail: dict = {}

    # 本模块均线（MA5/10/20）
    close_s = hist["close"].astype(float) if hist is not None and len(hist) else pd.Series(dtype=float)
    ma5 = close_s.rolling(5).mean().iloc[-1] if len(close_s) >= 5 else None
    ma10 = close_s.rolling(10).mean().iloc[-1] if len(close_s) >= 10 else None
    ma20 = close_s.rolling(20).mean().iloc[-1] if len(close_s) >= 20 else None
    close = _v(hist, "close") if hist is not None else None
    if close is None:
        close = ctx.get("price")

    # ---- 1 趋势强度（20分）
    s1 = 20.0
    if None in (ma5, ma10, ma20, close):
        s1, note1 = 10.0, "均线数据不足，取中值"
    else:
        bull = ma5 > ma10 > ma20
        above20 = close >= ma20
        if bull and above20:
            note1 = "MA5>MA10>MA20多头排列且价在MA20上方"
        elif not above20:
            s1 = max(0.0, 20 - 15)
            note1 = "跌破MA20扣15"
        else:
            inv = int(ma5 <= ma10) + int(ma10 <= ma20)
            s1 = max(0.0, 20 - (10 if inv >= 2 else 5))
            note1 = f"均线缠绕扣{10 if inv >= 2 else 5}"
    detail["趋势强度"] = (s1, note1)

    # ---- 2 量价配合（18分）
    s2 = 18.0
    turnover = ctx.get("turnover")
    vr = ctx.get("spot_vr")
    if vr is None and hist is not None and len(hist) >= 21:
        vma20 = hist["volume"].astype(float).rolling(20).mean().iloc[-1]
        vol_today = _v(hist, "volume")
        vr = vol_today / vma20 if vma20 and vma20 > 0 else None
    pct = ctx.get("pct_chg", 0.0) or 0.0
    note2 = []
    if vr is not None and vr > 1.2 and pct < 1.0:
        s2 -= 8; note2.append("放量滞涨扣8")
    if vr is not None and vr < 0.8 and pct > 2.0:
        s2 -= 5; note2.append("缩量上涨扣5")
    if turnover is not None and turnover > 20:
        s2 -= 10; note2.append("换手>20%扣10")
    if not note2:
        note2 = ["量价正常"]
    s2 = max(0.0, s2)
    detail["量价配合"] = (s2, "；".join(note2))

    # ---- 3 形态识别（15分）
    s3 = 15.0
    note3 = "形态识别正常"
    if hist is not None and len(hist) >= 21 and ma20 is not None and close is not None:
        hi20 = hist["high"].astype(float).iloc[-21:-1].max()
        low5 = hist["low"].astype(float).iloc[-5:]
        touch_ma20 = bool((low5 <= ma20 * 1.02).any()) and close > ma20
        amp10 = (hist["high"].astype(float).iloc[-10:].max() - hist["low"].astype(float).iloc[-10:].min()) / close * 100
        if close > hi20:
            note3 = "突破形态（创20日新高）"
            markers.append("突破确认")
        elif touch_ma20:
            note3 = "回踩MA20确认"
            markers.append("回踩买点")
        elif amp10 < 8 and pct > 2:
            note3 = "平台整理后放量启动"
        elif vr is not None and vr > 1.2 and pct < 1.0:
            s3 = max(0.0, 15 - 17)
            note3 = "假突破（冲高回落），扣17"
            markers.append("形态扣分")
        else:
            s3 = max(0.0, 15 - 17)
            note3 = "形态未完成，扣17"
            markers.append("形态扣分")
    else:
        s3 = 7.5
        note3 = "历史数据不足，取中值"
    detail["形态识别"] = (s3, note3)

    # ---- 4 板块共振（15分）
    board_rank = ctx.get("board_rank")          # 板块当日排名（1起）
    s4 = 15.0
    if board_rank is None:
        s4, note4 = 7.5, "板块数据缺失，取中值"
    elif board_rank <= 5:
        note4 = f"板块排名前5(#{board_rank})"
        if (ctx.get("board_zt_count") or 0) >= 3:
            note4 += "，板块内多股涨停"
    elif board_rank <= 10:
        s4 = 10.0
        note4 = f"板块排名6-10(#{board_rank})扣5"
    else:
        s4 = 2.0
        note4 = f"板块排名靠后(#{board_rank})扣13"
        markers.append("板块门槛未过")
    detail["板块共振"] = (s4, note4)

    # ---- 5 资金流向（12分）
    mnp = ctx.get("main_net_pct")
    if mnp is None:
        s5, note5 = 6.0, "资金流数据缺失，取中值"
    elif mnp > 0:
        s5, note5 = 12.0, f"主力净流入{mnp:.2f}%"
    elif mnp > -3:
        s5, note5 = 6.0, f"主力小幅净流出{mnp:.2f}%"
    else:
        s5, note5 = 2.0, f"主力净流出{mnp:.2f}%>3%，扣10"
    detail["资金流向"] = (s5, note5)

    # ---- 6 位置评估（8分）
    if ma20 is not None and close is not None:
        bias20 = (close - ma20) / ma20 * 100
        if bias20 > float(scfg["overheat_bias_ma20"]):
            s6, note6 = 2.0, f"MA20乖离{bias20:.1f}%>{scfg['overheat_bias_ma20']}%"
            markers.append("短线过热")
        elif bias20 > 15:
            s6, note6 = 5.0, f"MA20乖离{bias20:.1f}%偏高"
        else:
            s6, note6 = 8.0, f"MA20乖离{bias20:.1f}%健康"
    else:
        s6, note6 = 4.0, "位置数据不足"
    detail["位置评估"] = (s6, note6)

    # ---- 7 龙头地位（7分）
    bris = ctx.get("board_rank_in_stock")       # 板块内涨幅名次（1起）
    lianban = ctx.get("lianban") or 1
    if (bris is not None and bris <= 3) or (lianban or 1) >= 2:
        s7, note7 = 7.0, "板块内涨幅前3/连板龙头"
    elif bris is not None and bris <= 10:
        s7, note7 = 2.0, "跟风股扣5"
    else:
        s7, note7 = 0.0, "跟风股扣7"
    detail["龙头地位"] = (s7, note7)

    # ---- 8 情绪加成（5分）
    st = ctx.get("sentiment_temp")
    if st is None:
        s8, note8 = 0.0, "情绪数据缺失"
    elif st >= 62:
        s8, note8 = 5.0, f"情绪温度{st}≥62加分"
    elif st >= 48:
        s8, note8 = 3.0, f"情绪温度{st}∈[48,62)"
    else:
        s8, note8 = 0.0, f"情绪温度{st}<48为0"
    detail["情绪加成"] = (s8, note8)

    total = round(sum(s for s, _ in detail.values()), 1)

    if total >= float(scfg["strong_threshold"]):
        grade = "极强主升"
    elif total >= float(scfg["pool_threshold"]):
        grade = "主升进行"
    elif total >= float(scfg["watch_threshold"]):
        grade = "观察关注"
    else:
        grade = "不推荐"

    passed_gate = board_rank is not None and board_rank <= int(cfg["sector"]["top_n_gate"])
    in_pool = passed_gate and total >= float(scfg["pool_threshold"])

    return {
        "code": code, "name": name, "score": total, "grade": grade,
        "detail": {k: {"score": v[0], "note": v[1]} for k, v in detail.items()},
        "markers": markers, "board": ctx.get("board"),
        "board_rank": board_rank, "passed_gate": passed_gate, "in_pool": in_pool,
        "horizon": "中线" if total >= 70 else ("长线参考" if total >= 60 else ""),
    }
