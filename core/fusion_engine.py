# -*- coding: utf-8 -*-
"""融合决策层（规格书 §7 + §6.8 T0协同）。

- 7.1 买入融合：情绪×老樊买点冲突矩阵（5区×3买点，15格）
- 7.2 卖出融合：双轨监控统一优先级 P0 > P1 > P2
- 7.3 仓位融合：个股 = min(老樊信号仓位, 情绪单票上限, 30%)；总仓位与持仓数约束
- 7.4 条件化建议：正向条件 + 反向不执行条件（用户硬性要求）
- 6.8 T0 协同（MVP：人工输入 T0 信号，按日线方向输出协同结论）
"""
from __future__ import annotations

from typing import Optional

# ---------------- 7.1 买入冲突矩阵（15格全部枚举，禁止隐式默认）

BUY_MATRIX = {
    ("冰点区", "BP1"): {"action": "允许", "note": "允许，仓位≤2成试错（熊市只做买点1，与老樊共振）"},
    ("冰点区", "BP2"): {"action": "冻结", "note": "冻结，等待情绪修复"},
    ("冰点区", "BP3"): {"action": "冻结", "note": "冻结，等待情绪修复"},
    ("退潮区", "BP1"): {"action": "降级允许", "note": "降级允许，仓位≤3成"},
    ("退潮区", "BP2"): {"action": "降级观察", "note": "降级为观察：逆情绪建仓，需极强理由"},
    ("退潮区", "BP3"): {"action": "条件允许", "note": "仅 BUILDING 状态持仓股可加（老樊原规则）"},
    ("震荡区", "BP1"): {"action": "正常执行", "note": "正常执行"},
    ("震荡区", "BP2"): {"action": "正常执行", "note": "正常执行"},
    ("震荡区", "BP3"): {"action": "正常执行", "note": "正常执行"},
    ("偏强区", "BP1"): {"action": "正常执行", "note": "正常执行"},
    ("偏强区", "BP2"): {"action": "正常执行", "note": "正常执行（积极）"},
    ("偏强区", "BP3"): {"action": "正常执行", "note": "正常执行（积极）"},
    ("高热区", "BP1"): {"action": "禁止", "note": "禁新仓（高热>75 铁律）"},
    ("高热区", "BP2"): {"action": "禁止", "note": "禁新仓（铁律：情绪>75禁开新仓）"},
    ("高热区", "BP3"): {"action": "禁止", "note": "禁新仓（高热>75 铁律）"},
}

ZONE_NAMES = ["冰点区", "退潮区", "震荡区", "偏强区", "高热区"]
BP_TYPES = ["BP1", "BP2", "BP3"]


def fuse_buy(zone_name: str, bp_type: str) -> dict:
    """查询 7.1 冲突矩阵（未知组合显式报错，不静默默认）。"""
    key = (zone_name, bp_type)
    if key not in BUY_MATRIX:
        raise KeyError(f"冲突矩阵未覆盖：{key}")
    return {"zone": zone_name, "bp": bp_type, **BUY_MATRIX[key]}


def fuse_position(signal_pct: float, zone: dict, max_single: float = 30.0) -> dict:
    """7.3 仓位融合：个股建议仓位 = min(老樊信号仓位, 情绪单票上限, 单票≤30%)。"""
    cap_zone = float(zone["max_single"])
    final = min(float(signal_pct), cap_zone, float(max_single))
    return {
        "signal_pct": float(signal_pct), "zone_cap": cap_zone,
        "hard_cap": float(max_single), "final": round(final, 1),
        "formula": f"min(信号{signal_pct}%, 情绪上限{cap_zone}%, 硬上限{max_single}%) = {final:.1f}%",
    }


def check_total_position(positions_pct: list, zone: dict, max_holdings: int = 5) -> dict:
    """总仓位约束：Σ个股仓位 ≤ 情绪区间总仓位上限；持仓数 ≤ min(区间上限, 5)。"""
    total = sum(positions_pct)
    cap_total = float(zone["max_total"])
    cap_count = min(int(zone["max_count"]), int(max_holdings))
    return {
        "total": round(total, 1), "cap_total": cap_total,
        "count": len(positions_pct), "cap_count": cap_count,
        "total_ok": total <= cap_total, "count_ok": len(positions_pct) <= cap_count,
    }


# ---------------- 7.2 卖出双轨监控

def evaluate_sell_rules(stock: dict, sentiment: dict, prev_sentiment_temp: Optional[float],
                        cfg: dict) -> list[dict]:
    """stock：{code,name,cost,price,horizon(短线/波段),laofan_sells:[Signal...],
    ma20,score,peak_score,board_rank,bias60}。返回按 P0>P1>P2 排序的卖出提示。"""
    sr = cfg["sell_rules"]
    alerts: list[dict] = []

    # ---- 老樊轨
    for sig in stock.get("laofan_sells", []):
        p = {"SP3": "P0", "SP2": "P1", "SP1": "P2"}[sig.type]
        alerts.append({"priority": p, "source": "老樊", "signal": sig.type,
                       "name": sig.reason,
                       "action": "无条件清仓，进入15天离场观察" if sig.type == "SP3"
                       else (f"减仓{sig.action_pct}%" if sig.type != "SP1" else f"减仓30%")})

    # ---- 李致远轨
    cost, price = stock.get("cost"), stock.get("price")
    pnl = None
    if cost and price and cost > 0:
        pnl = (price - cost) / cost * 100.0
    horizon = stock.get("horizon") or "波段"
    stop_line = float(sr["stop_loss_short"] if horizon == "短线" else sr["stop_loss_swing"])
    tp_line = float(sr["take_profit_short"] if horizon == "短线" else sr["take_profit_swing"])
    ma20 = stock.get("ma20")

    if pnl is not None and pnl <= stop_line:
        alerts.append({"priority": "P0", "source": "李致远", "signal": "硬止损",
                       "name": f"亏损{pnl:.1f}%≤{stop_line}%（{horizon}仓）",
                       "action": "无条件立即执行"})
    if ma20 and price and price < ma20 and (pnl is None or pnl < 0):
        alerts.append({"priority": "P0", "source": "李致远", "signal": "跌破MA20",
                       "name": f"现价{price:.2f}跌破MA20({ma20:.2f})且无盈利保护",
                       "action": "无条件立即执行"})

    score, peak = stock.get("score"), stock.get("peak_score")
    if score is not None and peak is not None and peak - score > float(sr["score_drop_alert"]):
        alerts.append({"priority": "P1", "source": "李致远", "signal": "信号卖出(评分滑坡)",
                       "name": f"主升评分{score}较峰值{peak}降{peak - score:.0f}分>15分",
                       "action": "减仓1/2观察"})
    if stock.get("board_rank") is not None and stock["board_rank"] > 10:
        alerts.append({"priority": "P1", "source": "李致远", "signal": "信号卖出(板块滑坡)",
                       "name": f"板块排名跌出前10(#{stock['board_rank']})",
                       "action": "减仓1/2观察"})

    # 情绪崩塌：情绪温度自高热区单日回落>10分 → 清仓
    if prev_sentiment_temp is not None:
        drop = prev_sentiment_temp - float(sentiment["temperature"])
        if prev_sentiment_temp >= 75 and drop > float(sr["sentiment_crash_drop"]):
            alerts.append({"priority": "P1", "source": "李致远", "signal": "情绪卖出(情绪崩塌)",
                           "name": f"前日高热{prev_sentiment_temp}→今日{sentiment['temperature']}，回落{drop:.0f}分>10分",
                           "action": "清仓"})

    if pnl is not None and pnl >= tp_line:
        alerts.append({"priority": "P2", "source": "李致远", "signal": "止盈卖出",
                       "name": f"盈利{pnl:.1f}%≥{tp_line}%（{horizon}仓）",
                       "action": "分批止盈（先1/2，剩余移动止盈）"})
    bias20 = stock.get("bias_ma20")
    if bias20 is not None and bias20 > float(sr["bias_ma20_take_profit"]):
        alerts.append({"priority": "P2", "source": "李致远", "signal": "止盈卖出(MA20乖离)",
                       "name": f"MA20乖离{bias20:.1f}%>30%",
                       "action": "分批止盈（先1/2，剩余移动止盈）"})

    order = {"P0": 0, "P1": 1, "P2": 2}
    alerts.sort(key=lambda a: order[a["priority"]])
    return alerts


# ---------------- 7.4 条件化建议（必须：正向条件 + 反向不执行条件）

def build_advice(stock: dict, sentiment: dict, zone: dict, fusion_buy: Optional[dict],
                 pos_info: Optional[dict]) -> str:
    """按 7.4 模板生成条件化建议文本。反向条件必含（用户硬性要求）。"""
    code = stock.get("code", "")
    name = stock.get("name", "")
    score = stock.get("score")
    board, board_rank = stock.get("board"), stock.get("board_rank")
    temp = sentiment["temperature"]
    lines = []

    head = f"【{name} {code}】"
    parts = []
    if score is not None:
        parts.append(f"主升评分{score}")
    if board and board_rank:
        parts.append(f"板块#{board_rank}({board})")
    parts.append(f"情绪{temp}({zone['name']})")
    lines.append(head + " · ".join(parts))

    lf = stock.get("laofan_summary")
    if lf:
        lines.append(f"老樊状态：{lf}")

    # 正向条件
    if fusion_buy and fusion_buy["action"] not in ("禁止", "冻结"):
        pos_txt = f"（仓位上限{pos_info['final']:.0f}%）" if pos_info else ""
        lines.append(f"建议：{fusion_buy['note']} → 触发{fusion_buy['bp']}，按信号执行{pos_txt}")
    elif fusion_buy:
        lines.append(f"建议：{fusion_buy['bp']}信号被融合规则{fusion_buy['action']}——{fusion_buy['note']}")
    else:
        lines.append("建议：今日无老樊买卖点信号，维持现状观察")

    # 反向不执行条件（必含）
    reverse = []
    dist = stock.get("dist_to_mid_upper")
    if dist is not None and 0 <= dist <= 6:
        reverse.append(f"若放量跌破中期带上沿且幅度>2% → 观望，不追")
    reverse.append("若直接高开涨超5%，不追，等待回踩")
    reverse.append("若跌破短期带下沿≥1% → SP2，减仓50%")
    reverse.append("若跌破中期带下沿≥2% → SP3，无条件清仓")
    lines.append("反向条件：" + "；".join(reverse))
    return "\n".join(lines)


# ---------------- 6.8 T0 协同（MVP 降级版）

T0_MATRIX = {
    ("多头趋势", "low_absorb"): ("加仓低吸", 15, 95, "日线多头+T0低吸=加仓良机，止损设中期带下方"),
    ("多头趋势", "high_throw"): ("谨慎高抛", -10, 40, "仅部分仓位高抛，避免卖飞"),
    ("多头趋势", "none"): ("持股观望", 0, 60, "持股为主，回调是低吸机会"),
    ("空头趋势", "high_throw"): ("减仓高抛", 15, 95, "减仓良机，控制风险"),
    ("空头趋势", "low_absorb"): ("谨慎低吸", -15, 35, "小仓位快进快出，避免接飞刀"),
    ("空头趋势", "none"): ("观望为主", 0, 50, "控制仓位，不急于抄底"),
    ("震荡整理", "low_absorb"): ("正常做T/区间操作", 0, 55, "震荡市最适合做T"),
    ("震荡整理", "high_throw"): ("正常做T/区间操作", 0, 55, "震荡市最适合做T"),
    ("震荡整理", "none"): ("正常做T/区间操作", 0, 55, "震荡市最适合做T"),
}


def t0_synergy(trend: str, t0_signal: str, bias60: Optional[float] = None) -> dict:
    """trend：多头趋势/空头趋势/震荡整理；t0_signal：low_absorb/high_throw/none。
    乖离风险修正：BIAS60≥+50% → 低吸-20%（提示追高风险）；高抛+10%。"""
    if (trend, t0_signal) not in T0_MATRIX:
        raise KeyError(f"T0矩阵未覆盖：{(trend, t0_signal)}")
    action, adj, bound, note = T0_MATRIX[(trend, t0_signal)]
    base = 60 if trend == "多头趋势" else (50 if trend == "空头趋势" else 55)
    conf = base + adj
    extra = ""
    if bias60 is not None and bias60 >= 50:
        if t0_signal == "low_absorb":
            conf -= 20
            extra = "；乖离过高（BIAS60≥+50%），追高风险大，置信度-20%"
        elif t0_signal == "high_throw":
            conf += 10
            extra = "；高抛止盈是正确选择，置信度+10%"
    conf = max(35, min(95, conf)) if t0_signal != "none" else min(conf, bound)
    return {"trend": trend, "t0_signal": t0_signal, "action": action,
            "confidence": conf, "note": note + extra}
