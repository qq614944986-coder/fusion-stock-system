# -*- coding: utf-8 -*-
"""机会卡片组装（规格 v1.0 §七）：把左轨/右轨结果打包成带产业逻辑的卡片。

exec（老樊介入执行）映射：
  - 双轨      ：温和左侧布局先行 → 右轨 BP2/BP3 升级
  - 左轨 温和 ：温和左侧布局（轻仓）→ 老樊 BP1 兜底
  - 左轨 预期差：左轨预期差（BP1 兜底）
  - 右轨 主升 ：右轨主升跟随（BP2/BP3）
thesis（产业逻辑模板）：按赛道给出 景气逻辑 + 需人工验证项 + 典型风险，
是"导航"，不代替使用者终判。可细化到产品环节（stage）。
"""
from __future__ import annotations

from typing import Optional

_THESIS: dict[str, dict] = {
    "AI算力": {
        "logic": ("算力需求沿 芯片→光模块→PCB/CCL→服务器→IDC→电力/温控 逐环节传导，存在1-2季度滞后；"
                  "环节景气与股价位置背离处即机会（景气已起而未定价）。"),
        "verify": ["云厂商(微软/谷歌/Meta/亚马逊)最新资本开支指引及同比",
                   "该环节是否已现涨价、订单能见度或产能利用率提升",
                   "国产算力(昇腾/海光生态)集采/招标落地进度",
                   "环节估值分位与涨幅是否已充分反映需求",
                   "公司订单/产能/毛利率是否与股价同步走强"],
        "risk": ["算力资本开支不及预期或下修", "高景气环节易情绪透支、快涨后负乖离修正", "国产替代兑现度低于预期"],
    },
    "创新药": {
        "logic": ("创新药价值来自管线与BD(license-out/数据读出)，估值弹性在事件而非常态；"
                  "在关键临床数据读出/FDA审评/出海落地前的分歧期，是非共识埋伏窗口。"),
        "verify": ["近期有无III期/关键II期数据读出、NDA/BLA审评节点(公告/招股书/行业日历)",
                   "有无BD/license-out合作及其首付款/里程碑条款",
                   "医保谈判/集采对该品种是正面还是负面",
                   "在手资金能否支撑管线(烧钱风险)",
                   "数据风险：读数失败则股价可能大幅下挫"],
        "risk": ["临床数据读出失败/不及预期", "BD落地后利好兑现回落", "研发投入大、盈利兑现周期长"],
    },
    "互联网": {
        "logic": ("互联网中期看 利润拐点(降本增效/回购) + 政策周期(平台监管/游戏版号) + AI应用重估；"
                  "利润率修复往往先于收入，是先行的机会信号。"),
        "verify": ["近季毛利率/营业利润率是否拐点向上(收入未增但利润先修复)",
                   "政策边际(版号发放节奏/平台监管)是否缓和",
                   "回购/分红/注销力度是否加码(公司对低估的确认)",
                   "AI应用是否带来真实变现(广告/云/AI营收占比)而非纯主题",
                   "大厂资本开支/回购作为行业景气先行指标"],
        "risk": ["政策再度收紧", "AI重估仅主题炒作、无落地收入", "流量红利见顶、竞争加剧致利润率再承压"],
    },
}


def _fmt_thesis(track: str, stage: Optional[str]) -> str:
    t = _THESIS.get(track)
    if not t:
        return "未匹配赛道，需人工结合基本面定位产业逻辑。"
    head = f"【{track}" + (f"·{stage}" if stage else "") + "】"
    logic = t["logic"]
    verify = "；".join(f"{i+1}.{v}" for i, v in enumerate(t["verify"]))
    risk = "；".join(t["risk"])
    return f"{head} 景气逻辑：{logic} 需人工验证：{verify} 典型风险：{risk}"


def build_card(code: str, name: str, track: Optional[str], anomalies: list,
               eg: dict, wave: dict, cfg: dict, stage: Optional[str] = None) -> dict:
    """组装机会卡片。eg/wave 为 expectation_gap / main_wave_upgrade 的输出 dict。"""
    mild = bool(eg.get("mild_left") and eg.get("passed"))
    wave_ok = bool(wave.get("passes"))
    exec_kind, exec_sig, risk_off = "仅异动观察", "—", "不满足介入位，仅跟踪"
    if mild and wave_ok:
        exec_kind, exec_sig = "双轨", "温和布局→右轨BP2/BP3升级"
    elif wave_ok:
        exec_kind, exec_sig = "右轨主升跟随", "BP2/BP3"
    elif eg.get("passed"):
        exec_kind, exec_sig = "左轨预期差低吸", "温和布局+BP1兜底"
    elif eg.get("mild_left"):
        exec_kind, exec_sig = "温和左侧布局", "轻仓"
    risk_off = "破MA5或板块龙头转弱则放弃；情绪高热不追" if wave_ok else \
        ("若跌破前期低点或业绩下修则放弃" if mild else "暂不介入，等缩量企稳+资金回补")

    good_signals = [s for s in anomalies if s.get("name") != "放量滞涨"]
    return {
        "code": code, "name": name,
        "track": track or "未匹配赛道",
        "stage": stage or "",
        "signals": good_signals,
        "anchors": eg.get("anchors", {}),
        "mild_left": mild,
        "mild_left_note": eg.get("mild_left_note", ""),
        "wave": {"passes": wave_ok, "gates": wave.get("gates", {}), "score": wave.get("score", 0)},
        "thesis": _fmt_thesis(track, stage) if track else "未匹配赛道，需人工定位产业逻辑。",
        "exec_kind": exec_kind, "exec_sig": exec_sig,
        "risk_off": risk_off,
        "evidence": eg.get("evidence", []),
        "hard_block": eg.get("hard_block"),
        "position_note": _position_note(eg, wave),
    }


def _position_note(eg: dict, wave: dict) -> str:
    parts = []
    if wave.get("passes"):
        parts.append("右轨：板块+量价+乖离门槛过，可跟随")
    if eg.get("mild_left"):
        parts.append(f"左轨温和：{eg.get('mild_left_note','')}")
    if eg.get("passed") and not eg.get("mild_left"):
        parts.append("左轨预期差：三锚≥2过，等 BP1 深位兜底")
    return "；".join(parts) or "位置待定（未达介入门槛）"