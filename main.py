# -*- coding: utf-8 -*-
"""双引擎选股择时系统 · 每日主流程（规格书 §2.2）。

用法：
    python main.py                     # 每日收盘后运行
    python main.py --t0-signal low_absorb|high_throw|none   # 人工输入做T信号（默认 none）
输出：output/dashboard_YYYYMMDD.html（单文件 HTML 仪表盘）
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from core.data_provider import DataProvider, load_config, load_watchlist, index_symbol_map, is_main_board
from core.sentiment_engine import compute_sentiment
from core.sector_screener import screen_sectors
from core.stock_scorer import score_stock
from core.limitup_scorer import score_limitup, board_ladder_count, LIMITUP_DISCIPLINE
from core.eod_watchlist import check_eod
from core.ma_band_v2 import MABandV2
from core.laofan_signals import LaofanSignalEngine, Signal, StockState, STATE_CN
from core.laofan_models import LaofanModels
from core.position_manager import PositionManager
from core.macro_view import build_macro
from core.pool_review import PoolReview, POOLS
from core import fusion_engine as fe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("main")

BASE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- 渲染

def render_dashboard(ctx: dict, out_path: Path) -> None:
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(str(BASE / "dashboard")))
    tpl = env.get_template("template.html.j2")
    html = tpl.render(date=ctx["date"], generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      data_json=json.dumps(ctx, ensure_ascii=False, default=str))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info("仪表盘已生成: %s", out_path)


# ---------------------------------------------------------------- S级冲高形态判定

def _s_grade_check(res: dict, hist: Optional[pd.DataFrame], close_s) -> tuple[bool, str]:
    """S级冲高优先形态（用户需求 + 老樊强势模型近似）。

    判定：评分≥85 且 命中以下强势形态之一（主板已在上游过滤）：
    - N式双涨停近似：近5日≥2个涨幅>9.5%的交易日；
    - 突破形态近似：收盘创20日新高 且 量比≥1.5；
    - 极值反转近似：BIAS60≤-25% 且回升（当日上涨）。
    """
    score = res.get("score") or 0
    if score < 85:
        return False, ""
    if hist is None or len(hist) < 21:
        return False, ""
    try:
        pct5 = pd.to_numeric(hist["pct_chg"], errors="coerce").tail(5)
        zt_days = int((pct5 > 9.5).sum())
        if zt_days >= 2:
            return True, f"N式双涨停（近5日{zt_days}个涨停级涨幅）"
        if float(close_s.iloc[-1]) >= float(close_s.tail(20).max()):
            vr = res.get("spot_vr")
            if vr is not None and vr >= 1.5:
                return True, "突破20日新高且放量"
        b60 = res.get("bias60")
        if b60 is not None and b60 <= -25 and (res.get("pct_chg") or 0) > 0:
            return True, "极值反转（深乖离回升）"
    except (KeyError, TypeError, ValueError):
        return False, ""
    return False, ""


def _enrich_review_rows(rows: list[dict]) -> list[dict]:
    """复盘记录附加状态总结文案（用户需求：股票状态总结/是否高开低走）。"""
    out: list[dict] = []
    for r in rows:
        r = dict(r)
        nd = r.get("next_day")
        if nd:
            verdict = "收红" if nd["close_ret"] > 0 else "收绿"
            parts = [f"次日{verdict}", f"开盘{nd['open_ret']:+.1f}%",
                     f"最高{nd['high_ret']:+.1f}%", f"收盘{nd['close_ret']:+.1f}%"]
            if nd["open_ret"] > 0 and nd["close_ret"] < nd["open_ret"]:
                parts.append("高开低走")
            r["status_summary"] = "，".join(parts)
        else:
            r["status_summary"] = "在池观察中（次日数据待回填）"
        out.append(r)
    return out


# ---------------------------------------------------------------- 主流程

def run(t0_signal: str = "none") -> Path:
    cfg = load_config()
    cfg["run"]["t0_signal"] = t0_signal
    wl = load_watchlist()
    today = datetime.now()
    today_iso = today.strftime("%Y-%m-%d")
    date_compact = today.strftime("%Y%m%d")

    dp = DataProvider(cfg, base_dir=BASE, trade_date=date_compact)
    pm = PositionManager(cfg, base_dir=BASE)
    band_eng = MABandV2(cfg)
    sig_eng = LaofanSignalEngine(cfg)
    models = LaofanModels(cfg)
    prv = PoolReview(BASE, cfg["run"]["data_dir"])

    # ========== 1. 宏观大盘（指数组 + 两市量能 + 外围日韩） ==========
    macro = build_macro(dp, index_symbol_map())
    log.info("宏观：%s", macro["summary"])

    # ========== 2. 情绪温度 ==========
    act = dp.get_market_activity() or {}
    idx_pct = {}
    for cname, sym in index_symbol_map().items():
        idx = dp.get_index_daily(sym)
        if idx is not None and len(idx) >= 2:
            c0, c1 = float(idx["close"].iloc[-2]), float(idx["close"].iloc[-1])
            idx_pct[cname] = (c1 - c0) / c0 * 100.0 if c0 else None
    zt_pool = dp.get_zt_pool()

    def _act_int(key: str):
        for k, v in (act or {}).items():
            if key in str(k):
                try:
                    return int(float(v))
                except (TypeError, ValueError):
                    return None
        return None

    sent_hist_file = BASE / cfg["run"]["data_dir"] / "sentiment_history.csv"
    prev_base = prev_temp = None
    if sent_hist_file.exists():
        sh = pd.read_csv(sent_hist_file)
        sh = sh[sh["date"] < today_iso]
        if not sh.empty:
            prev_base = float(sh["base_temp"].iloc[-1])
            prev_temp = float(sh["temperature"].iloc[-1])

    sentiment = compute_sentiment({
        "date": today_iso,
        "rise": _act_int("上涨"), "fall": _act_int("下跌"), "flat": _act_int("平盘"),
        "limit_up": _act_int("涨停"), "limit_down": _act_int("跌停"),
        "index_pct": idx_pct, "zt_pool": zt_pool,
    }, cfg, prev_base)
    zone = sentiment["zone"]
    log.info("情绪温度 %.1f → %s（%s）", sentiment["temperature"], zone["name"], zone["label"])

    # 情绪龙头具体化（用户需求：日级情绪龙头=名称+连板数）
    sentiment["leader_stock"] = None
    if zt_pool is not None and not zt_pool.empty and "lian_ban" in zt_pool.columns:
        try:
            zp = zt_pool.copy()
            zp["lian_ban"] = pd.to_numeric(zp["lian_ban"], errors="coerce").fillna(1).astype(int)
            top = zp.sort_values("lian_ban", ascending=False).iloc[0]
            sentiment["leader_stock"] = {
                "name": str(top.get("name", "")), "code": str(top.get("code", "")).zfill(6),
                "lian_ban": int(top["lian_ban"]),
            }
        except (KeyError, IndexError, TypeError):
            pass

    # 涨跌家数 / 涨停跌停（用户需求：实时涨跌家数比、涨停跌停数——盘后为收盘终值）
    sentiment["breadth_rt"] = {
        "rise": _act_int("上涨"), "fall": _act_int("下跌"),
        "limit_up": _act_int("涨停"), "limit_down": _act_int("跌停"),
    }

    # 追加情绪历史
    if sent_hist_file.exists():
        hist_df = pd.read_csv(sent_hist_file)
    else:
        hist_df = pd.DataFrame(columns=["date", "base_temp", "temperature"])
    hist_df = pd.concat([hist_df, pd.DataFrame([{
        "date": today_iso, "base_temp": sentiment["base_temp"],
        "temperature": sentiment["temperature"]}])], ignore_index=True)
    hist_df.to_csv(sent_hist_file, index=False, encoding="utf-8-sig")

    # ========== 3. 板块筛选 ==========
    boards = dp.get_industry_boards()
    fund_flow = dp.get_sector_fund_flow()
    scan_limit = int(cfg["run"].get("sector_scan_limit", 30) or 0)
    cons_map: dict = {}
    board_hist_map: dict = {}
    if boards is not None and not boards.empty:
        boards_sorted = boards.sort_values("pct_chg", ascending=False)
        scan_list = boards_sorted["board"].tolist()
        if scan_limit > 0:
            scan_list = scan_list[:scan_limit]
        for bd in scan_list:
            cons_map[bd] = dp.get_board_cons(bd)
            board_hist_map[bd] = dp.get_board_hist(bd)

    hist_cache: dict = {}   # code → 日K DataFrame（本地缓存共享）

    def get_hist(code: str):
        code = str(code).zfill(6)
        if code not in hist_cache:
            hist_cache[code] = dp.get_stock_daily(code)
        return hist_cache[code]

    sector = screen_sectors(boards, fund_flow, cons_map, board_hist_map, zt_pool,
                            hist_cache, cfg, zone["name"])
    board_rank_map = {r["board"]: r["rank"] for r in sector["boards"]}
    sbm = sector["stock_board_map"]

    # 板块内涨幅名次（龙头地位因子）
    rank_in_board: dict = {}
    for bd, cons in cons_map.items():
        if cons is None or cons.empty or "pct_chg" not in cons.columns:
            continue
        cc = cons.sort_values("pct_chg", ascending=False).reset_index(drop=True)
        for i, r in cc.iterrows():
            rank_in_board[str(r["code"]).zfill(6)] = i + 1

    # 板块涨停统计（打板协同上下文）
    zt_by_board: dict = {}
    if zt_pool is not None and not zt_pool.empty and "code" in zt_pool.columns:
        for _, r in zt_pool.iterrows():
            bd = sbm.get(str(r["code"]).zfill(6)) or (r.get("industry") if isinstance(r.get("industry"), str) else None)
            if bd:
                zt_by_board.setdefault(bd, []).append(r)
    boards_pct_map = {}
    if boards is not None and not boards.empty:
        for _, r in boards.iterrows():
            boards_pct_map[str(r["board"])] = pd.to_numeric(r.get("pct_chg"), errors="coerce")
    ff_map = {}
    if fund_flow is not None and not fund_flow.empty:
        for _, r in fund_flow.iterrows():
            ff_map[str(r["board"])] = pd.to_numeric(r.get("main_net_pct"), errors="coerce")

    # ========== 4. 三线候选池（板块优先 + 主板过滤 + S级冲高形态判定） ==========
    cons_row_map: dict = {}   # code → 成分股行（替代全市场快照：价格/涨幅/换手板块接口自带）
    for bd, cons in cons_map.items():
        if cons is None or cons.empty:
            continue
        for _, r in cons.iterrows():
            cons_row_map[str(r["code"]).zfill(6)] = r

    mid_rows: list[dict] = []     # 中线：主升确认（≥70 且过板块门槛）
    long_rows: list[dict] = []    # 长线：观察区（60-69）
    short_rows: list[dict] = []   # 短线：S级冲高优先形态（≥85 + 强势形态，进短线观察池）
    per_board = int(cfg["run"].get("per_board_stock_limit", 4) or 4)
    total_limit = int(cfg["run"].get("stock_score_universe_limit", 24) or 24)
    universe: list = []
    if sector["gate_top_n"]:
        for bd in sector["gate_top_n"]:
            cons = cons_map.get(bd)
            if cons is None or cons.empty:
                continue
            rows = []
            for _, r in cons.iterrows():
                code = str(r["code"]).zfill(6)
                name = str(r.get("name", ""))
                amt = pd.to_numeric(r.get("amount"), errors="coerce")
                pct = pd.to_numeric(r.get("pct_chg"), errors="coerce")
                # 预筛：非ST、成交额≥5000万、当日上涨、主板（用户合规约束）
                if "ST" in name.upper() or pd.isna(amt) or amt < 5e7 or pd.isna(pct) or pct <= 0:
                    continue
                if not is_main_board(code):
                    continue
                rows.append((code, name, bd, float(amt)))
            rows.sort(key=lambda u: -u[3])
            universe.extend(rows[:per_board])
        universe.sort(key=lambda u: -u[3])
        universe = universe[:total_limit]

    # 兜底宇宙：成分股接口全挂（东财push2被封等）时用全市场快照构建
    # （主板+非ST+当日上涨+成交额≥5000万，按成交额排序取前N；板块字段缺失→中线门槛降级，短线/长线照常）
    if not universe:
        spot = dp.get_spot()
        if spot is not None and not spot.empty:
            for _, r in spot.iterrows():
                code = str(r["code"]).zfill(6)
                name = str(r.get("name", ""))
                amt = pd.to_numeric(r.get("amount"), errors="coerce")
                pct = pd.to_numeric(r.get("pct_chg"), errors="coerce")
                if "ST" in name.upper() or pd.isna(amt) or amt < 5e7 or pd.isna(pct) or pct <= 0:
                    continue
                if not is_main_board(code):
                    continue
                universe.append((code, name, None, float(amt)))
            universe.sort(key=lambda u: -u[3])
            universe = universe[:total_limit]
            log.info("候选宇宙走快照兜底：%d 只（成分股接口不可用）", len(universe))

    # 主力净流入：全市场批量一次拉取（腾讯快照/同花顺资金流，替代逐股东财请求）；
    # 批量源全挂时才逐股东财兜底
    main_net_map: dict = dp.get_main_net_map()
    if main_net_map:
        log.info("主力净流入走批量源：覆盖 %d 只", len(main_net_map))
    else:
        for code, _, _, _ in universe:
            ff = dp.get_stock_fund_flow(code)
            if ff is not None and not ff.empty and "main_net_pct" in ff.columns:
                v = pd.to_numeric(ff["main_net_pct"], errors="coerce").iloc[-1]
                main_net_map[code] = None if pd.isna(v) else float(v)
    for code, name, bd, _ in universe:
        hist = get_hist(code)
        crow = cons_row_map.get(code)
        if hist is None or hist.empty:
            continue
        close_s = hist["close"].astype(float)
        vol_s = hist["volume"].astype(float)
        vr = None
        if len(vol_s) >= 21:
            ma20v = vol_s.rolling(20).mean().iloc[-2]
            if ma20v and not pd.isna(ma20v) and ma20v > 0:
                vr = round(float(vol_s.iloc[-1] / ma20v), 2)
        res = score_stock(hist, {
            "code": code, "name": name, "board": bd,
            "board_rank": board_rank_map.get(bd),
            "board_zt_count": len(zt_by_board.get(bd, [])),
            "board_rank_in_stock": rank_in_board.get(code),
            "main_net_pct": main_net_map.get(code),
            "sentiment_temp": sentiment["temperature"],
            "spot_vr": vr,
            "turnover": None if crow is None else (None if pd.isna(crow.get("turnover")) else float(crow.get("turnover"))),
            "pct_chg": None if crow is None else (None if pd.isna(crow.get("pct_chg")) else float(crow.get("pct_chg"))),
            "price": None if crow is None else (None if pd.isna(crow.get("price")) else float(crow.get("price"))),
        }, cfg)
        # 附加展示字段（用户需求：主力流入强度/换手/乖离/涨跌概率倾向）
        res["main_net_pct"] = main_net_map.get(code)
        res["price"] = float(close_s.iloc[-1])
        res["pct_chg"] = None if crow is None or pd.isna(crow.get("pct_chg")) else float(crow.get("pct_chg"))
        res["turnover"] = None if crow is None or pd.isna(crow.get("turnover")) else float(crow.get("turnover"))
        res["bias20"] = None
        res["bias60"] = None
        if len(close_s) >= 60:
            ma60 = close_s.rolling(60).mean().iloc[-1]
            res["bias60"] = round(float((close_s.iloc[-1] - ma60) / ma60 * 100), 1)
        if len(close_s) >= 20:
            ma20 = close_s.rolling(20).mean().iloc[-1]
            res["bias20"] = round(float((close_s.iloc[-1] - ma20) / ma20 * 100), 1)
        # S级冲高形态判定（用户需求：S级进短线候选池）
        s_grade, s_reason = _s_grade_check(res, hist, close_s)
        res["s_grade"] = s_grade
        res["s_reason"] = s_reason
        if s_grade:
            short_rows.append(res)
        elif res["score"] >= float(cfg["stock_score"]["pool_threshold"]) and res["passed_gate"]:
            mid_rows.append(res)
        elif res["score"] >= 60:
            long_rows.append(res)
    mid_rows.sort(key=lambda x: -x["score"])
    long_rows.sort(key=lambda x: -x["score"])
    short_rows.sort(key=lambda x: -x["score"])
    # 三线各取前5（用户需求：宁缺毋滥，不足5只如实显示）
    mid_rows = mid_rows[:5]
    long_rows = long_rows[:5]
    short_rows = short_rows[:5]
    log.info("三线池：短线S级 %d / 中线 %d / 长线 %d", len(short_rows), len(mid_rows), len(long_rows))

    # ========== 4.5 观察池复盘闭环（入选→次日验证→去弱留强→滚动统计） ==========
    # 池内仍在观察的代码补拉日K（历史入选可能不在今日候选宇宙内）
    pool_active_codes = {rec["code"] for p in POOLS for rec in prv.data[p]
                         if not rec.get("removed_reason")}
    for code in pool_active_codes:
        get_hist(code)
    hist_map = {c: h for c, h in hist_cache.items() if h is not None and not h.empty}

    # 上一交易日（从上证指数日K推导，盘后运行时最后一根=今日）
    prev_trade_day = ""
    sh_idx = dp.get_index_daily(index_symbol_map()["上证指数"])
    if sh_idx is not None and len(sh_idx):
        past = [str(d)[:10] for d in sh_idx["date"] if str(d)[:10] < today_iso]
        if past:
            prev_trade_day = past[-1]

    # 1) 次日表现回填（对 added_date < today 且未回填的记录）
    backfilled = prv.backfill(hist_map, today_iso)
    # 2) 去弱留强（短线破MA5 / 中线评分滑坡或破MA20 / 长线破中期带下沿）
    band_map: dict = {}
    for rec in prv.data["长线"]:
        h = hist_map.get(rec["code"])
        if h is not None and len(h) >= 66:
            try:
                band_map[rec["code"]] = band_eng.compute(h)
            except (KeyError, ValueError, TypeError):
                pass
    score_map = {r["code"]: r["score"] for r in mid_rows + long_rows}
    removed_log = prv.prune(hist_map, band_map=band_map, score_map=score_map, today_iso=today_iso)
    # 3) 今日新入选（入选价=当日收盘）
    prv.append("短线", [{
        "code": r["code"], "name": r["name"], "score": r["score"],
        "add_price": float(hist_map[r["code"]]["close"].iloc[-1]) if r["code"] in hist_map else None,
        "reason": f"S级冲高形态：{r.get('s_reason', '')}",
        "model": r.get("grade", ""),
    } for r in short_rows], today_iso)
    prv.append("中线", [{
        "code": r["code"], "name": r["name"], "score": r["score"],
        "add_price": float(hist_map[r["code"]]["close"].iloc[-1]) if r["code"] in hist_map else None,
        "reason": f"主升确认（评分{r['score']:.0f}·板块#{r.get('board_rank') or '—'}）",
        "model": r.get("grade", ""),
    } for r in mid_rows], today_iso)
    prv.append("长线", [{
        "code": r["code"], "name": r["name"], "score": r["score"],
        "add_price": float(hist_map[r["code"]]["close"].iloc[-1]) if r["code"] in hist_map else None,
        "reason": f"观察区（评分{r['score']:.0f}，等主升确认升级）",
        "model": r.get("grade", ""),
    } for r in long_rows], today_iso)
    prv.save()

    # 4) 滚动统计（次日胜率/平均开盘收盘/平均最高收益；样本<10 显示"—"）
    prv_stats = prv.stats()
    review_payload = {
        "stats": prv_stats,
        "yesterday_date": prev_trade_day,
        "yesterday": {p: _enrich_review_rows(prv.yesterday_top(p, prev_trade_day)) for p in POOLS},
        "active": {p: _enrich_review_rows(prv.active(p)) for p in POOLS},
        "removed": removed_log,
        "backfilled": backfilled,
    }
    log.info("观察池复盘：回填 %d 条 · 删除 %d 条", backfilled, len(removed_log))

    # ========== 5. 打板观察池 ==========
    zb_pool = dp.get_zb_pool()
    zt_rows: list[dict] = []
    if zt_pool is not None and not zt_pool.empty:
        for _, r in zt_pool.iterrows():
            code = str(r["code"]).zfill(6)
            bd = sbm.get(code) or (r.get("industry") if isinstance(r.get("industry"), str) else None)
            bctx = {
                "board": bd, "board_zt_count": len(zt_by_board.get(bd, [])),
                "board_pct_chg": boards_pct_map.get(bd),
                "board_main_net_pct": ff_map.get(bd),
                "board_ladder": board_ladder_count(zt_pool, bd, sbm),
            }
            # R11 主力净流入验证：涨停池自带字段缺失，从批量主力净占比映射注入
            ztres = score_limitup({**r.to_dict(), "main_net_pct": main_net_map.get(code)},
                                  get_hist(code), bctx, sentiment["temperature"], cfg)
            if ztres["in_pool"]:
                # 近5日涨幅（打板评分展示用）
                pct5 = None
                _h = get_hist(code)
                if _h is not None and len(_h):
                    pct5 = round(float(pd.to_numeric(_h["pct_chg"], errors="coerce").tail(5).sum()), 1)
                zt_rows.append({
                    "code": code, "name": ztres["name"], "price": r.get("price"),
                    "pct_chg": r.get("pct_chg"),
                    "score": ztres["score"], "zt_score": ztres["score"],
                    "board_rank": board_rank_map.get(bd), "markers": [f"{ztres['lian_ban']}连板"],
                    "laofan_sig": "", "action": f"打板观察（{ztres['dims'].get('板块协同_说明', '')}）",
                    "grade": f"打板{ztres['score']}分",
                    # 打板明细（用户需求：量能/换手/主力净额/近5日涨幅/板块共振）
                    "lian_ban": ztres["lian_ban"], "first_seal_time": ztres["first_seal_time"],
                    "open_times": ztres["open_times"], "vr": ztres["vr"],
                    "turnover": ztres["turnover"], "main_net_pct": ztres["main_net_pct"],
                    "pct5": pct5, "board": bd,
                    "board_zt_count": bctx["board_zt_count"], "board_ladder": bctx["board_ladder"],
                })
    zt_rows.sort(key=lambda x: -x["score"])

    # ========== 6. 尾盘观察池（成分股涨幅粗筛 → 日K构造完整行情行） ==========
    eod_rows: list[dict] = []
    codes_checked = set()
    for bd in sector["gate_top_n"]:
        cons = cons_map.get(bd)
        if cons is None or cons.empty:
            continue
        for _, r in cons.iterrows():
            code = str(r["code"]).zfill(6)
            if code in codes_checked:
                continue
            pct = pd.to_numeric(r.get("pct_chg"), errors="coerce")
            if pd.isna(pct) or not (3 <= pct):
                continue
            codes_checked.add(code)
            hist = get_hist(code)
            if hist is None or len(hist) < 2:
                continue
            # 盘后运行：日K最后一根即今日行情，从中构造 check_eod 所需字段
            last = hist.iloc[-1]
            prev = hist.iloc[-2]
            vol_ma5 = hist["volume"].astype(float).tail(6).head(5).mean()
            vr = round(float(last["volume"] / vol_ma5), 2) if vol_ma5 and vol_ma5 > 0 else None
            res = check_eod({
                "code": code, "name": str(r.get("name", "")),
                "price": float(last["close"]), "pct_chg": float(pct),
                "open": float(last["open"]), "high": float(last["high"]), "low": float(last["low"]),
                "volume": float(last["volume"]),
                "amount": float(last["amount"]) if not pd.isna(last.get("amount")) else None,
                "vr": vr, "pre_close": float(prev["close"]),
            }, hist, board_rank_map.get(bd), cfg)
            if res["selected"]:
                eod_rows.append({
                    "code": code, "name": res["name"], "price": res["price"],
                    "pct_chg": res["pct_chg"], "score": None, "zt_score": None,
                    "board_rank": res["board_rank"], "markers": [f"量比{res['vol_ratio']}"],
                    "laofan_sig": "", "action": "尾盘关注（次日14:50-14:55执行）",
                    "grade": "尾盘池", "discipline": res["discipline"],
                })
    log.info("打板池 %d 只 · 尾盘池 %d 只", len(zt_rows), len(eod_rows))

    # ========== 7. 老樊引擎：自选持仓 + 候选池 ==========
    pm.refresh_all(sig_eng, today_iso)
    position_codes = [str(p["code"]).zfill(6) for p in wl["positions"]]
    watching_codes = [str(w["code"]).zfill(6) for w in wl["watching"]]
    candidate_codes = [r["code"] for r in (short_rows + mid_rows + long_rows)[:10]]
    all_codes = list(dict.fromkeys(position_codes + watching_codes + candidate_codes))

    pos_payload: list[dict] = []
    risks_cooldown: list[str] = []
    review_rows: list[dict] = []
    advice_map: dict = {}

    for code in all_codes:
        name = next((p.get("name", "") for p in wl["positions"] if str(p["code"]).zfill(6) == code), "") or \
               next((w.get("name", "") for w in wl["watching"] if str(w["code"]).zfill(6) == code), "")
        is_position = code in position_codes
        hist = get_hist(code)
        st = pm.ensure_state(code, name, is_position)

        entry: dict = {"code": code, "name": name or st.name, "state": st.status,
                       "state_cn": STATE_CN[st.status], "position_pct": st.position_pct}
        if hist is None or len(hist) < 66:
            entry.update({"advice": "[数据缺失] 日K数据不足（<66条），老樊引擎无法计算均线带",
                          "signals": [], "signal_marks": []})
            pos_payload.append(entry)
            continue

        bands = band_eng.compute(hist)
        judge = band_eng.judge(bands)
        det = sig_eng.detect(bands, st)
        today_sigs = det["signals"]
        for b in det["blocked"]:
            risks_cooldown.append(f"{entry['name']}({code}) {b[0]}：{b[1]}")

        # 模型评分（联动信号）
        model_res = models.evaluate_all(bands, st, today_sigs, code=code)
        entry["models"] = [m for m in model_res if m["triggered"]]

        # 融合：买入
        fusion_buy = None
        pos_info = None
        if today_sigs and today_sigs[0].direction == "买入":
            bp = today_sigs[0].type
            fusion_buy = fe.fuse_buy(zone["name"], bp)
            pos_info = fe.fuse_position(today_sigs[0].action_pct, zone,
                                        float(cfg["positions"]["max_single_position"]))
        # 融合：卖出双轨
        mid_res = next((r for r in short_rows + mid_rows + long_rows if r["code"] == code), None)
        crow = cons_row_map.get(code)
        price = None
        if hist is not None and len(hist):
            price = float(hist["close"].iloc[-1])   # 盘后运行：日K收盘即最新价
        elif crow is not None and not pd.isna(crow.get("price")):
            price = float(crow.get("price"))
        cost = next((p.get("cost") for p in wl["positions"] if str(p["code"]).zfill(6) == code and p.get("cost")), None)
        ma20 = float(hist["close"].astype(float).rolling(20).mean().iloc[-1]) if len(hist) >= 20 else None
        bias20 = None
        if ma20 and price:
            bias20 = (price - ma20) / ma20 * 100
        sell_alerts = []
        if is_position:
            sell_alerts = fe.evaluate_sell_rules({
                "code": code, "name": entry["name"], "cost": cost, "price": price,
                "horizon": "短线" if any(z["code"] == code for z in zt_rows) else "波段",
                "laofan_sells": [s for s in today_sigs if s.direction == "卖出"],
                "ma20": ma20, "score": mid_res["score"] if mid_res else None,
                "peak_score": pm.peak_score(code, today_iso),
                "board_rank": mid_res["board_rank"] if mid_res else None,
                "bias_ma20": bias20,
            }, sentiment, prev_temp, cfg)

        # T0 协同
        t0 = None
        if t0_signal != "none":
            trend = "多头趋势" if judge["多头排列"] else ("空头趋势" if judge["空头排列"] else "震荡整理")
            t0 = fe.t0_synergy(trend, t0_signal, judge["bias60"])

        # 条件化建议（7.4）
        laofan_summary = (f"{STATE_CN[st.status]} · "
                          f"{'多头排列' if judge['多头排列'] else ('空头排列' if judge['空头排列'] else ('均线粘合(变盘信号)' if judge['粘合信号'] else '震荡整理'))}"
                          f" · BIAS60={judge['bias60']:+.1f}%（{judge['bias60_zone']}）"
                          if judge["bias60"] is not None else f"{STATE_CN[st.status]} · 数据不足")
        if judge["距中期带上沿pct"] is not None:
            laofan_summary += f" · 距中期带上沿{judge['距中期带上沿pct']:+.1f}%"
        advice = fe.build_advice({
            "code": code, "name": entry["name"],
            "score": mid_res["score"] if mid_res else None,
            "board": mid_res["board"] if mid_res else sbm.get(code),
            "board_rank": mid_res["board_rank"] if mid_res else None,
            "laofan_summary": laofan_summary,
            "dist_to_mid_upper": judge["距中期带上沿pct"],
        }, sentiment, zone, fusion_buy, pos_info)
        if t0:
            advice += f"\nT0协同（{t0_signal}）：{t0['action']}（置信{t0['confidence']}%）— {t0['note']}"
        if not is_position:
            advice += "\n（关注股：只出信号，不出仓位建议）"

        # K线 payload（近130根）
        kl = bands.tail(130).reset_index(drop=True)
        dates = [str(d)[:10] for d in kl["date"]]
        ma_payload = {f"MA{p}": [None if pd.isna(v) else round(float(v), 3) for v in kl[f"MA{p}"]]
                      for p in band_eng.short_periods + band_eng.mid_periods}
        signal_marks = []
        for sig_type, d_iso in (st.last_signal_dates or {}).items():
            if d_iso and d_iso[:10] in dates:
                bi = dates.index(d_iso[:10])
                signal_marks.append({"date": d_iso[:10], "type": sig_type, "name": sig_type,
                                     "price": round(float(kl["close"].iloc[bi]), 2)})
        for s in today_sigs:
            signal_marks.append({"date": today_iso, "type": s.type, "name": s.type,
                                 "price": round(float(kl["close"].iloc[-1]), 2)})
            review_rows.append({"date": today_iso, "code": code, "name": entry["name"],
                                "signal": s.type, "confidence": s.confidence,
                                "action": s.position_action})

        entry.update({
            "bias60": None if judge["bias60"] is None else round(judge["bias60"], 1),
            "bias_zone": judge["bias60_zone"],
            "trend_cn": ("多头排列" if judge["多头排列"] else "空头排列" if judge["空头排列"]
                         else "均线粘合(变盘)" if judge["粘合信号"] else "震荡整理"),
            "dist_mid_upper": None if judge["距中期带上沿pct"] is None else round(judge["距中期带上沿pct"], 1),
            "signals": [dict(s.to_dict(), is_today=True) for s in today_sigs],
            "sell_alerts": sell_alerts, "t0": t0, "advice": advice,
            "cost": cost,
            "pnl_pct": None if not (cost and price) else round((price - cost) / cost * 100, 1),
            "price": price,
            "kline": {
                "dates": dates,
                "k": [[round(float(a), 2) for a in (kl["open"].iloc[i], kl["close"].iloc[i],
                                                    kl["low"].iloc[i], kl["high"].iloc[i])]
                      for i in range(len(kl))],
                "ma": ma_payload,
                "short_band": {"upper": [None if pd.isna(v) else round(float(v), 3) for v in kl["short_upper"]],
                               "lower": [None if pd.isna(v) else round(float(v), 3) for v in kl["short_lower"]]},
                "mid_band": {"upper": [None if pd.isna(v) else round(float(v), 3) for v in kl["mid_upper"]],
                             "lower": [None if pd.isna(v) else round(float(v), 3) for v in kl["mid_lower"]]},
                "bias60": [None if pd.isna(v) else round(float(v), 2) for v in kl["bias60"]],
            },
            "signal_marks": signal_marks,
        })
        advice_map[code] = advice
        pos_payload.append(entry)

    # ========== 8. 融合输出：三线候选池建议表 ==========
    default_actions = {
        "短线": "S级冲高形态：次日冲高兑现为主（收盘破MA5离场）",
        "中线": "入池观察（等老樊买点）",
        "长线": "观察区（等主升确认后升级中线）",
    }
    for pool_name, pool_rows in (("短线", short_rows), ("中线", mid_rows), ("长线", long_rows)):
        for r in pool_rows:
            r["laofan_sig"] = ""
            r["action"] = default_actions[pool_name]
            if pool_name == "短线" and r.get("s_reason"):
                r["action"] += f"〔{r['s_reason']}〕"
            lf_entry = next((e for e in pos_payload if e["code"] == r["code"]), None)
            if lf_entry:
                sigs = lf_entry.get("signals") or []
                if sigs:
                    fb = fe.fuse_buy(zone["name"], sigs[0]["type"])
                    r["laofan_sig"] = f"{sigs[0]['type']}(置信{sigs[0]['confidence']}%)"
                    r["action"] = f"{fb['action']}：{fb['note']}"
                else:
                    r["laofan_sig"] = "无信号"

    # 三池Top3交叉验证精选：短线S级/中线主升/长线观察 各取Top3，
    # 叠加打板/尾盘两个协同源 —— 同一股票被≥2个源同时选中才算"交叉验证精选"
    #（三线池按评分分层互斥，但S级形态股常同时命中打板池、主升股常同时命中尾盘池，交叉由此产生）
    src_lists = (("短线S级", short_rows), ("中线主升", mid_rows), ("长线观察", long_rows),
                 ("打板", zt_rows), ("尾盘", eod_rows))
    src_map: dict = {}
    for lbl, rows in src_lists:
        for r in rows:
            src_map.setdefault(r["code"], []).append(lbl)
    top3_codes = [r["code"] for _, rows in src_lists for r in rows[:3]]
    cross = Counter(top3_codes)
    picked = [c for c, n in cross.items() if n >= 2]
    if not picked:
        # 无交叉命中 → 按情绪区间回退单池Top3（冰点/退潮埋伏长线，偏强/高热做短线，中间态做中线）；
        # 首选池为空（如高热区但今日无S级短线）时顺位补位，避免"精选0只"
        order = ([short_rows] if zone["name"] in ("偏强区", "高热区") else
                 [long_rows] if zone["name"] in ("冰点区", "退潮区") else [mid_rows])
        order += [mid_rows, long_rows, short_rows]
        fb = next((rows for rows in order if rows), [])
        picked = [r["code"] for r in fb[:3]]
    picked_set = set(picked)
    jx_rows: list[dict] = []
    seen = set()
    for _, rows in src_lists:
        for r in rows:
            c = r["code"]
            if c in picked_set and c not in seen:
                seen.add(c)
                row = dict(r)
                row["cross_src"] = "+".join(src_map.get(c, []))
                row["cross_n"] = len(src_map.get(c, []))
                jx_rows.append(row)
    jx_rows.sort(key=lambda x: (-x.get("cross_n", 0), -(x.get("score") or 0)))
    log.info("三池交叉验证精选 %d 只：%s", len(jx_rows),
             "、".join(f"{r['name']}×{r['cross_n']}" for r in jx_rows) or "无")

    # ========== 复盘日志（9.1 信号触发记录） ==========
    review_file = BASE / cfg["run"]["data_dir"] / "review_log.csv"
    if review_rows:
        rdf = pd.DataFrame(review_rows)
        if review_file.exists():
            rdf = pd.concat([pd.read_csv(review_file, dtype={"code": str}), rdf], ignore_index=True)
        rdf.to_csv(review_file, index=False, encoding="utf-8-sig")
    # 评分峰值记录（供 7.2 信号卖出用）
    pm.record_scores(today_iso, {**{r["code"]: r["score"] for r in short_rows + mid_rows + long_rows}})
    pm.save()

    # ========== 9. 渲染 ==========
    discipline = [
        f"情绪区间：{zone['name']}（{zone['label']}）—— {zone['rule']}",
        f"总仓位上限 ≤{zone['max_total']}%，单票 ≤{zone['max_single']}%，持仓数 ≤{zone['max_count']}只",
    ]
    if zone["name"] == "高热区":
        discipline.append("高热区铁律：情绪>75 禁开新仓，逐步减仓")
    if eod_rows:
        discipline.append("尾盘纪律：" + "；".join(eod_rows[0].get("discipline", [])))

    # ========== 8.5 总结决策与操作建议（用户需求：使用者第一眼先看这段） ==========
    delta_txt = "—" if sentiment.get("delta") is None else f"{sentiment['delta']:+.1f}分"
    leader = sentiment.get("leader_stock") or {}
    summary_lines = [
        f"① 市场环境：{macro['summary']}",
        f"② 情绪周期：{sentiment['temperature']:.0f}分（{zone['name']}·{zone['label']}），较前日{delta_txt}"
        + (f"；情绪龙头 {leader.get('name','')}({leader.get('lian_ban','')}连板)" if leader else ""),
        f"③ 仓位纪律：总仓位≤{zone['max_total']}%，单票≤{zone['max_single']}%，持仓≤{zone['max_count']}只"
        + ("；高热区禁开新仓" if zone["name"] == "高热区" else ""),
    ]
    if sector["ambush"]["attack"]:
        a = sector["ambush"]["attack"][0]
        summary_lines.append(f"④ 板块埋伏：进攻 {a['board']}（{a['basis']}）→ {a['action']}")
    elif sector["ambush"]["defend"]:
        d = sector["ambush"]["defend"][0]
        summary_lines.append(f"④ 板块防守：{d['board']}（{d['basis']}）→ {d['action']}")
    else:
        summary_lines.append(f"④ 板块埋伏：{sector['ambush']['watch']}")
    jx_names = "、".join(f"{r['name']}({r['code']})×{r['cross_n']}源"
                         for r in jx_rows[:3]) or "今日无交叉验证精选"
    summary_lines.append(f"⑤ 三池精选：{jx_names}")
    summary_lines.append(f"⑥ 打板纪律：{LIMITUP_DISCIPLINE[1]}")
    # 数据缺失汇总：成分股类缺失同根因（东财封禁、无替代源）会有数十条重复记录，压缩为一条
    _all_missing = dp.missing + sentiment["missing"] + sector["missing"]
    _cons_miss = [m for m in _all_missing if "成分股" in m]
    missing_list = [m for m in _all_missing if "成分股" not in m]
    if _cons_miss:
        missing_list.append(
            f"[数据缺失] 板块成分股接口不可用（涉及{_cons_miss.__len__()}个板块，同根因：东财封禁、"
            f"无替代源；候选宇宙已走快照兜底，板块宽度/换手维度降级）")
    n_miss = len(missing_list)
    if n_miss:
        summary_lines.append(f"⑦ 数据缺失 {n_miss} 项（详见风险区标注，不影响其他模块）")

    ctx = {
        "date": today_iso,
        "summary": summary_lines,
        "macro": macro,
        "sentiment": sentiment,
        "sentiment_history": _sentiment_history(cfg, today_iso, sentiment["temperature"]),
        "sectors": {"boards": sector["boards"], "attack": sector["attack_boards"],
                    "defend": sector["defend_boards"], "ambush": sector["ambush"]},
        "pools": {"短线": short_rows, "中线": mid_rows, "长线": long_rows,
                  "打板": zt_rows, "尾盘": eod_rows, "精选": jx_rows},
        "pool_review": review_payload,
        "limitup_discipline": LIMITUP_DISCIPLINE,
        "eod": eod_rows,
        "positions": pos_payload,
        "risks": {"missing": missing_list,
                  "cooldown": risks_cooldown, "discipline": discipline},
    }
    out = BASE / cfg["run"]["output_dir"] / f"dashboard_{date_compact}.html"
    render_dashboard(ctx, out)
    # 供本地 Web 应用读取（server.py /api/data）
    result_json = BASE / cfg["run"]["data_dir"] / "last_result.json"
    try:
        result_json.write_text(json.dumps(ctx, ensure_ascii=False, default=str), encoding="utf-8")
    except OSError as e:
        log.warning("last_result.json 写入失败: %s", e)
    return out


def _sentiment_history(cfg: dict, today_iso: str, today_temp: float) -> list[dict]:
    """近60日情绪温度序列（情绪周期图表数据；今日已写入 CSV 后读回，含今日）。"""
    f = BASE / cfg["run"]["data_dir"] / "sentiment_history.csv"
    rows: list[dict] = []
    try:
        if f.exists():
            h = pd.read_csv(f)
            h = h.dropna(subset=["temperature"]).tail(60)
            rows = [{"date": str(d)[:10], "temp": round(float(t), 1)}
                    for d, t in zip(h["date"], h["temperature"])]
    except (OSError, ValueError, KeyError) as e:
        log.warning("情绪历史读取失败: %s", e)
    if not rows or rows[-1]["date"] != today_iso:
        rows.append({"date": today_iso, "temp": round(float(today_temp), 1)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="双引擎选股择时系统 · 每日主流程")
    ap.add_argument("--t0-signal", choices=["low_absorb", "high_throw", "none"], default="none",
                    help="做T人工信号（默认 none）")
    args = ap.parse_args()
    run(t0_signal=args.t0_signal)


if __name__ == "__main__":
    main()
