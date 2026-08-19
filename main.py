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
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.data_provider import DataProvider, load_config, load_watchlist, index_symbol_map
from core.sentiment_engine import compute_sentiment
from core.sector_screener import screen_sectors
from core.stock_scorer import score_stock
from core.limitup_scorer import score_limitup, board_ladder_count
from core.eod_watchlist import check_eod
from core.ma_band_v2 import MABandV2
from core.laofan_signals import LaofanSignalEngine, SIGNAL_CN, STATE_CN
from core.laofan_models import LaofanModels
from core.position_manager import PositionManager
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

    # ========== 4. 主升候选池 ==========
    spot = dp.get_spot()
    spot_map = {}
    if spot is not None and not spot.empty:
        for _, r in spot.iterrows():
            spot_map[str(r["code"]).zfill(6)] = r

    mid_rows: list[dict] = []
    long_rows: list[dict] = []
    if sector["gate_top_n"]:
        universe = []
        for bd in sector["gate_top_n"]:
            cons = cons_map.get(bd)
            if cons is None or cons.empty:
                continue
            for _, r in cons.iterrows():
                code = str(r["code"]).zfill(6)
                amt = pd.to_numeric(r.get("amount"), errors="coerce")
                universe.append((code, str(r.get("name", "")), bd, 0.0 if pd.isna(amt) else float(amt)))
        # 预筛：非ST、成交额≥5000万、当日上涨
        def _up(c):
            sp = spot_map.get(c)
            if sp is None:
                return False
            p = pd.to_numeric(sp.get("pct_chg"), errors="coerce")
            return (not pd.isna(p)) and p > 0
        universe = [u for u in universe
                    if "ST" not in u[1].upper() and u[3] >= 5e7 and _up(u[0])]
        universe.sort(key=lambda u: -u[3])
        universe = universe[:int(cfg["run"].get("stock_score_universe_limit", 50) or 50)]
        for code, name, bd, _ in universe:
            hist = get_hist(code)
            sp = spot_map.get(code)
            ff = dp.get_stock_fund_flow(code)
            main_net_pct = None
            if ff is not None and not ff.empty and "main_net_pct" in ff.columns:
                v = pd.to_numeric(ff["main_net_pct"], errors="coerce").iloc[-1]
                main_net_pct = None if pd.isna(v) else float(v)
            res = score_stock(hist, {
                "code": code, "name": name, "board": bd,
                "board_rank": board_rank_map.get(bd),
                "board_zt_count": len(zt_by_board.get(bd, [])),
                "board_rank_in_stock": rank_in_board.get(code),
                "main_net_pct": main_net_pct,
                "sentiment_temp": sentiment["temperature"],
                "spot_vr": None if sp is None else (None if pd.isna(sp.get("vr")) else float(sp.get("vr"))),
                "turnover": None if sp is None else (None if pd.isna(sp.get("turnover")) else float(sp.get("turnover"))),
                "pct_chg": None if sp is None else (None if pd.isna(sp.get("pct_chg")) else float(sp.get("pct_chg"))),
                "price": None if sp is None else (None if pd.isna(sp.get("price")) else float(sp.get("price"))),
            }, cfg)
            if res["score"] >= float(cfg["stock_score"]["pool_threshold"]) and res["passed_gate"]:
                mid_rows.append(res)
            elif res["score"] >= 60:
                long_rows.append(res)
    mid_rows.sort(key=lambda x: -x["score"])
    long_rows.sort(key=lambda x: -x["score"])
    log.info("主升候选池 %d 只（≥70 且板块门槛过）", len(mid_rows))

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
            ztres = score_limitup({**r.to_dict()}, get_hist(code), bctx, sentiment["temperature"], cfg)
            if ztres["in_pool"]:
                sp = spot_map.get(code)
                zt_rows.append({
                    "code": code, "name": ztres["name"], "price": ztres.get("price") or (sp and sp.get("price")),
                    "pct_chg": None if sp is None else sp.get("pct_chg"),
                    "score": ztres["score"], "zt_score": ztres["score"],
                    "board_rank": board_rank_map.get(bd), "markers": [f"{ztres['lian_ban']}连板"],
                    "laofan_sig": "", "action": f"打板观察（{ztres['dims'].get('板块协同_说明','')}）",
                    "grade": f"打板{ztres['score']}分",
                })
    zt_rows.sort(key=lambda x: -x["score"])

    # ========== 6. 尾盘观察池 ==========
    eod_rows: list[dict] = []
    if spot is not None and not spot.empty:
        codes_checked = set()
        for bd in sector["gate_top_n"]:
            cons = cons_map.get(bd)
            if cons is None or cons.empty:
                continue
            for _, r in cons.iterrows():
                code = str(r["code"]).zfill(6)
                if code in codes_checked:
                    continue
                sp = spot_map.get(code)
                if sp is None:
                    continue
                p = pd.to_numeric(sp.get("pct_chg"), errors="coerce")
                if pd.isna(p) or not (3 <= p):
                    continue
                codes_checked.add(code)
                res = check_eod({**sp.to_dict()}, get_hist(code), board_rank_map.get(bd), cfg)
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
    candidate_codes = [r["code"] for r in mid_rows[:10]]
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
        mid_res = next((r for r in mid_rows + long_rows if r["code"] == code), None)
        sp = spot_map.get(code)
        price = None if sp is None else (None if pd.isna(sp.get("price")) else float(sp.get("price")))
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

    # ========== 8. 融合输出：候选池建议表 ==========
    for r in mid_rows:
        r["laofan_sig"] = ""
        r["action"] = "入池观察（等老樊买点）"
        lf_entry = next((e for e in pos_payload if e["code"] == r["code"]), None)
        if lf_entry:
            sigs = lf_entry.get("signals") or []
            if sigs:
                fb = fe.fuse_buy(zone["name"], sigs[0]["type"])
                r["laofan_sig"] = f"{sigs[0]['type']}(置信{sigs[0]['confidence']}%)"
                r["action"] = f"{fb['action']}：{fb['note']}"
            else:
                r["laofan_sig"] = "无信号"

    # 精选：各维度 Top3 交叉验证
    top3_mid = [r["code"] for r in mid_rows[:3]]
    top3_zt = [r["code"] for r in zt_rows[:3]]
    top3_eod = [r["code"] for r in eod_rows[:3]]
    from collections import Counter
    cross = Counter(top3_mid + top3_zt + top3_eod)
    picked = [c for c, n in cross.items() if n >= 2] or top3_mid[:3]
    picked_set = set(picked)
    jx_rows = [r for r in (mid_rows + zt_rows + eod_rows) if r["code"] in picked_set]
    seen = set()
    jx_rows = [r for r in jx_rows if not (r["code"] in seen or seen.add(r["code"]))]

    # ========== 复盘日志（9.1 信号触发记录） ==========
    review_file = BASE / cfg["run"]["data_dir"] / "review_log.csv"
    if review_rows:
        rdf = pd.DataFrame(review_rows)
        if review_file.exists():
            rdf = pd.concat([pd.read_csv(review_file, dtype={"code": str}), rdf], ignore_index=True)
        rdf.to_csv(review_file, index=False, encoding="utf-8-sig")
    # 评分峰值记录（供 7.2 信号卖出用）
    pm.record_scores(today_iso, {**{r["code"]: r["score"] for r in mid_rows + long_rows}})
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

    ctx = {
        "date": today_iso,
        "sentiment": sentiment,
        "sectors": {"boards": sector["boards"], "attack": sector["attack_boards"],
                    "defend": sector["defend_boards"]},
        "pools": {"中线": mid_rows, "短线": zt_rows + eod_rows, "长线": long_rows, "精选": jx_rows},
        "eod": eod_rows,
        "positions": pos_payload,
        "risks": {"missing": dp.missing + sentiment["missing"] + sector["missing"],
                  "cooldown": risks_cooldown, "discipline": discipline},
    }
    out = BASE / cfg["run"]["output_dir"] / f"dashboard_{date_compact}.html"
    render_dashboard(ctx, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="双引擎选股择时系统 · 每日主流程")
    ap.add_argument("--t0-signal", choices=["low_absorb", "high_throw", "none"], default="none",
                    help="做T人工信号（默认 none）")
    args = ap.parse_args()
    run(t0_signal=args.t0_signal)


if __name__ == "__main__":
    main()
