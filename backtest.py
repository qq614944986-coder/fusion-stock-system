# -*- coding: utf-8 -*-
"""历史回放（规格书 §9.2）：给定股票代码+日期区间，逐日重放老樊引擎。

用法：
    python backtest.py --code 603019 --days 250
    python backtest.py --code 603019 --start 2025-06-01 --end 2026-08-01

逐日重放：均线带 / BIAS60 / 三买三卖 / 状态机 / 冷却期；
输出：信号时间轴（控制台+CSV）与净值曲线（CSV）。
净值口径（MVP）：信号按当日收盘价成交；权益 = 现金 + 仓位% × 当日涨跌幅贡献。
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.data_provider import DataProvider, load_config
from core.ma_band_v2 import MABandV2
from core.laofan_signals import (LaofanSignalEngine, StockState, SIGNAL_CN, STATE_CN,
                                 SIGNAL_PRIORITY)

BASE = Path(__file__).resolve().parent


def run_backtest(code: str, start: str = "", end: str = "", days: int = 250,
                 base_dir: Path = BASE, dp=None) -> dict:
    cfg = load_config(base_dir / "config" / "config.yaml")
    if dp is None:
        dp = DataProvider(cfg, base_dir=base_dir)
    band_eng = MABandV2(cfg)
    eng = LaofanSignalEngine(cfg)

    hist = dp.get_stock_daily(code, days=max(days + 120, 400))
    if hist is None or len(hist) < 66:
        raise RuntimeError(f"[数据缺失] {code} 日K不足（{0 if hist is None else len(hist)}条 < 66）")

    if start:
        hist = hist[hist["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    if end:
        hist = hist[hist["date"] <= pd.Timestamp(end)].reset_index(drop=True)
    # 回放窗口前的历史用于预热均线
    warmup = dp.get_stock_daily(code, days=max(days + 400, 700))
    if warmup is not None and len(warmup) > len(hist):
        pre = warmup[warmup["date"] < (hist["date"].iloc[0] if len(hist) else pd.Timestamp(start))]
        hist_full = pd.concat([pre.tail(70), hist], ignore_index=True) if len(hist) else hist
    else:
        hist_full = hist
    bands = band_eng.compute(hist_full).reset_index(drop=True)

    offset = len(hist_full) - len(hist) if len(hist_full) > len(hist) else 0
    state = StockState(code=str(code).zfill(6), status="EMPTY", position_pct=0.0)

    rows: list[dict] = []
    signals_log: list[dict] = []
    equity = 1.0
    prev_close = None

    for i in range(offset, len(bands)):
        date_iso = str(bands["date"].iloc[i])[:10]
        close = float(bands["close"].iloc[i])
        state = eng.refresh_exit_cooldown(state, date_iso)
        det = eng.detect(bands, state, i)
        executed = None
        if det["signals"]:
            executed = det["signals"][0]                 # 按优先级 SP3>SP2>SP1>BP3>BP2>BP1
            state = eng.apply_signal(state, executed, date_iso)
            signals_log.append({
                "date": date_iso, "signal": executed.type, "name": SIGNAL_CN[executed.type],
                "direction": executed.direction, "confidence": executed.confidence,
                "action": executed.position_action, "price": round(close, 2),
                "state_after": STATE_CN[state.status], "position_after": state.position_pct,
                "reason": executed.reason,
            })
        # 净值（按收盘成交，仓位比例近似）
        if prev_close and close:
            day_ret = close / prev_close - 1.0
            equity *= (1 - state.position_pct / 100.0) + (state.position_pct / 100.0) * (1 + day_ret)
        prev_close = close
        bias = bands["bias60"].iloc[i]
        rows.append({
            "date": date_iso, "close": round(close, 2),
            "bias60": None if pd.isna(bias) else round(float(bias), 2),
            "state": state.status, "position_pct": state.position_pct,
            "signal": executed.type if executed else "",
            "equity": round(equity, 4),
        })

    out_dir = base_dir / cfg["run"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    code6 = str(code).zfill(6)
    tl = pd.DataFrame(rows)
    sg = pd.DataFrame(signals_log)
    tl.to_csv(out_dir / f"backtest_{code6}_timeline.csv", index=False, encoding="utf-8-sig")
    if not sg.empty:
        sg.to_csv(out_dir / f"backtest_{code6}_signals.csv", index=False, encoding="utf-8-sig")

    return {"timeline": tl, "signals": sg, "final_state": STATE_CN[state.status],
            "final_equity": round(equity, 4), "missing": dp.missing}


def main() -> None:
    ap = argparse.ArgumentParser(description="老樊引擎历史回放")
    ap.add_argument("--code", required=True, help="股票代码，如 603019")
    ap.add_argument("--start", default="", help="开始日期 YYYY-MM-DD")
    ap.add_argument("--end", default="", help="结束日期 YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=250, help="回放交易日天数（默认250）")
    args = ap.parse_args()
    res = run_backtest(args.code, args.start, args.end, args.days)
    tl, sg = res["timeline"], res["signals"]
    print(f"\n===== 回放结果：{args.code} =====")
    print(f"区间：{tl['date'].iloc[0]} ~ {tl['date'].iloc[-1]}（{len(tl)}个交易日）")
    if not sg.empty:
        print("\n--- 信号时间轴 ---")
        for _, r in sg.iterrows():
            print(f"{r['date']}  {r['signal']:<4}{r['name']:<6} {r['direction']} "
                  f"@{r['price']:>8.2f}  {r['action']:<10} → {r['state_after']}({r['position_after']:.0f}%)")
    else:
        print("区间内无信号触发")
    print(f"\n期末状态：{res['final_state']} · 净值：{res['final_equity']:.4f}")
    if res["missing"]:
        print("数据缺失：", "; ".join(res["missing"]))
    print("明细已输出到 output/backtest_*_timeline.csv / _signals.csv")


if __name__ == "__main__":
    main()
