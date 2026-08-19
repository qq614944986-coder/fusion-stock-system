# -*- coding: utf-8 -*-
"""李致远引擎 · 板块筛选（六维度排名，规格书 §5.1）。

六维度：连板节奏 / 上攻意愿 / 主买占比 / 换手率 / 量能比 / 排名趋势，
各自标准化到 0-100 后等权合成板块总分。
板块门槛：个股所属板块当日排名未进前 10 → 标记"板块门槛未过"。
进攻/防御：偏强/高热期提示进攻板块（量能比>1.5 且板块连板高度≥3）；
          冰点/退潮期提示防御板块（当日涨幅为正且换手率<3%）。

量化说明（规格未细化处的实现约定）：
- 换手率维度：板块聚合换手率 = Σ成分股成交额 / Σ流通市值 ×100（成交额加权），
  区间 [3,15] 得满分，区间外线性扣分至 0。
- 创20日新高个股数：仅对本地已有日K缓存的个股精确计算（候选池/自选/涨停池），
  覆盖范围外不计数并在 missing 中标注 —— 不编造数据。
- 排名趋势：按前一日板块涨幅重排名次，上升→100 / 持平→50 / 下降→0。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def _minmax(values: dict) -> dict:
    """min-max 标准化到 0-100（全相等时取 50）。"""
    if not values:
        return {}
    vs = list(values.values())
    lo, hi = min(vs), max(vs)
    if hi - lo < 1e-12:
        return {k: 50.0 for k in values}
    return {k: (v - lo) / (hi - lo) * 100.0 for k, v in values.items()}


def _turnover_score(t: Optional[float], healthy: tuple) -> float:
    """换手率维度得分：[3,15] 满分，区间外按偏离程度扣分。"""
    lo, hi = healthy
    if t is None:
        return 50.0
    if lo <= t <= hi:
        return 100.0
    if t < lo:
        return max(0.0, t / lo * 100.0)
    return max(0.0, 100.0 - (t - hi) / hi * 100.0)


def _volume_ratio_score(ratio: Optional[float], good: float) -> float:
    """量能比维度：>1.5 满分，否则线性。"""
    if ratio is None:
        return 50.0
    return min(100.0, ratio / good * 100.0)


def screen_sectors(boards: Optional[pd.DataFrame],
                   fund_flow: Optional[pd.DataFrame],
                   cons_map: dict,
                   board_hist_map: dict,
                   zt_pool: Optional[pd.DataFrame],
                   hist_cache: dict,
                   cfg: dict,
                   zone_name: str = "") -> dict:
    """纯函数：输入由 main.py 组装。cons_map/board_hist_map 键为板块名。"""
    scfg = cfg["sector"]
    missing: list[str] = []

    if boards is None or boards.empty:
        return {"boards": [], "stock_board_map": {}, "gate_top_n": [],
                "attack_boards": [], "defend_boards": [], "missing": ["行业板块列表[数据缺失]"]}

    ff_map: dict = {}
    if fund_flow is not None and not fund_flow.empty:
        for _, r in fund_flow.iterrows():
            ff_map[str(r["board"])] = {
                "main_net_pct": pd.to_numeric(r.get("main_net_pct"), errors="coerce"),
                "main_net_inflow": pd.to_numeric(r.get("main_net_inflow"), errors="coerce"),
            }
    else:
        missing.append("板块资金流[数据缺失]")

    # 股票→板块 映射（来自已拉取的成分股）
    stock_board_map: dict = {}
    for bd, cons in cons_map.items():
        if cons is None or cons.empty or "code" not in cons.columns:
            continue
        for c in cons["code"].astype(str).str.zfill(6):
            stock_board_map[c] = bd

    # 板块内涨停股（按股票→板块映射）
    zt_by_board: dict = {}
    if zt_pool is not None and not zt_pool.empty and "code" in zt_pool.columns:
        for _, r in zt_pool.iterrows():
            code = str(r["code"]).zfill(6)
            bd = stock_board_map.get(code)
            if bd:
                zt_by_board.setdefault(bd, []).append(r)

    # 创20日新高（仅本地缓存覆盖的个股，诚实降级）
    new_high_by_board: dict = {}
    covered = 0
    for code, hist in hist_cache.items():
        if hist is None or len(hist) < 21:
            continue
        bd = stock_board_map.get(str(code).zfill(6))
        if not bd:
            continue
        covered += 1
        h = hist.tail(21).reset_index(drop=True)
        if float(h["close"].iloc[-1]) >= float(h["high"].max()):
            new_high_by_board[bd] = new_high_by_board.get(bd, 0) + 1
    if covered < len(stock_board_map):
        missing.append(f"创20日新高仅覆盖本地缓存{covered}只个股（未覆盖不计入）")

    raw = {}
    for _, r in boards.iterrows():
        bd = str(r["board"])
        cons = cons_map.get(bd)
        # ---- 维度1 连板节奏：2板×1 + 3板×2 + 4板×3 + ≥5板×5
        lianban_score = 0.0
        max_lianban = 0
        for zr in zt_by_board.get(bd, []):
            lb = int(pd.to_numeric(zr.get("lian_ban"), errors="coerce") or 1)
            max_lianban = max(max_lianban, lb)
            if lb >= 5:
                lianban_score += 5
            elif lb == 4:
                lianban_score += 3
            elif lb == 3:
                lianban_score += 2
            elif lb == 2:
                lianban_score += 1
        # ---- 维度2 上攻意愿：涨幅>3%占比×100 + 创20日新高数
        if cons is not None and not cons.empty and "pct_chg" in cons.columns:
            pc = pd.to_numeric(cons["pct_chg"], errors="coerce").dropna()
            attack = (pc > 3).mean() * 100.0 + new_high_by_board.get(bd, 0) if len(pc) else 0.0
        else:
            attack = 0.0
            missing.append(f"板块[{bd}]成分股[数据缺失]")
        # ---- 维度3 主买占比
        main_pct = ff_map.get(bd, {}).get("main_net_pct")
        if main_pct is None or pd.isna(main_pct):
            main_pct = None
        # ---- 维度4 换手率（聚合）
        if cons is not None and not cons.empty and {"amount", "float_cap"} <= set(cons.columns):
            amt = pd.to_numeric(cons["amount"], errors="coerce").sum()
            fc = pd.to_numeric(cons["float_cap"], errors="coerce").sum()
            turnover = float(amt / fc * 100) if fc and fc > 0 else None
        else:
            t = pd.to_numeric(r.get("turnover"), errors="coerce")
            turnover = None if pd.isna(t) else float(t)
        # ---- 维度5 量能比（板块指数当日成交额 / 5日均额）
        bh = board_hist_map.get(bd)
        vr = None
        if bh is not None and len(bh) >= 6 and "amount" in bh.columns:
            amt_s = pd.to_numeric(bh["amount"], errors="coerce").dropna()
            if len(amt_s) >= 6 and float(amt_s.iloc[-2]) > 0:
                vr = float(amt_s.iloc[-1]) / float(pd.Series(amt_s.iloc[-6:-1]).mean())
        # ---- 维度6 排名趋势
        raw[bd] = {
            "连板节奏": lianban_score, "上攻意愿": attack,
            "主买占比": None if main_pct is None else float(main_pct),
            "换手率": turnover, "量能比": vr,
            "当日涨幅": float(pd.to_numeric(r.get("pct_chg"), errors="coerce") or 0.0),
            "max_lianban": max_lianban, "stock_count": 0 if cons is None else len(cons),
        }

    # 前一日板块涨幅排名（来自板块指数历史倒数第2根）
    prev_pct: dict = {}
    for bd, bh in board_hist_map.items():
        if bh is not None and len(bh) >= 2 and "pct_chg" in bh.columns:
            v = pd.to_numeric(bh["pct_chg"], errors="coerce").iloc[-2]
            if not pd.isna(v):
                prev_pct[bd] = float(v)
    prev_rank = {bd: i + 1 for i, (bd, _) in enumerate(sorted(prev_pct.items(), key=lambda kv: -kv[1]))}
    today_rank = {bd: i + 1 for i, (bd, _) in enumerate(sorted(raw.items(), key=lambda kv: -kv[1]["当日涨幅"]))}

    dim_scores: dict = {bd: {} for bd in raw}
    for dim in ("连板节奏", "上攻意愿", "主买占比"):
        vals = {bd: raw[bd][dim] for bd in raw if raw[bd][dim] is not None}
        norm = _minmax(vals)
        for bd in raw:
            dim_scores[bd][dim] = round(norm.get(bd, 50.0), 1)
    for bd in raw:
        dim_scores[bd]["换手率"] = round(_turnover_score(raw[bd]["换手率"], tuple(scfg["turnover_healthy"])), 1)
        dim_scores[bd]["量能比"] = round(_volume_ratio_score(raw[bd]["量能比"], float(scfg["volume_ratio_good"])), 1)
        pr, tr = prev_rank.get(bd), today_rank.get(bd)
        if pr is None:
            dim_scores[bd]["排名趋势"] = 50.0
        elif tr < pr:
            dim_scores[bd]["排名趋势"] = 100.0
        elif tr == pr:
            dim_scores[bd]["排名趋势"] = 50.0
        else:
            dim_scores[bd]["排名趋势"] = 0.0

    rows = []
    for bd, ds in dim_scores.items():
        total = sum(ds.values()) / 6.0
        rows.append({
            "board": bd, "total": round(total, 1), "dims": ds,
            "量能比raw": None if raw[bd]["量能比"] is None else round(raw[bd]["量能比"], 2),
            "排名趋势dir": {100.0: "↑", 50.0: "→", 0.0: "↓"}[ds["排名趋势"]],
            "当日涨幅": round(raw[bd]["当日涨幅"], 2),
            "换手率raw": None if raw[bd]["换手率"] is None else round(raw[bd]["换手率"], 2),
            "max_lianban": raw[bd]["max_lianban"], "stock_count": raw[bd]["stock_count"],
        })
    rows.sort(key=lambda x: -x["total"])
    for i, row in enumerate(rows):
        row["rank"] = i + 1

    gate_n = int(scfg["top_n_gate"])
    gate_top_n = [r["board"] for r in rows[:gate_n]]

    # 进攻/防御标签（按情绪区间）
    attack_boards, defend_boards = [], []
    for r in rows:
        if zone_name in ("偏强区", "高热区"):
            if (r["量能比raw"] or 0) > float(scfg["volume_ratio_good"]) and r["max_lianban"] >= 3:
                r["tag"] = "进攻"
                attack_boards.append(r["board"])
            else:
                r["tag"] = ""
        elif zone_name in ("冰点区", "退潮区"):
            if r["当日涨幅"] > 0 and (r["换手率raw"] is not None and r["换手率raw"] < 3):
                r["tag"] = "防御"
                defend_boards.append(r["board"])
            else:
                r["tag"] = ""
        else:
            r["tag"] = ""

    return {
        "boards": rows, "stock_board_map": stock_board_map,
        "gate_top_n": gate_top_n, "attack_boards": attack_boards,
        "defend_boards": defend_boards, "missing": missing,
    }
