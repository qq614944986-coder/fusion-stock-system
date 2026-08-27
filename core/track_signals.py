# -*- coding: utf-8 -*-
"""赛道信号库（规格 v1.0 §五 / §六）：三大赛道主板标的名录 + 环节归属 + 赛道判定。

名录聚焦 AI 算力产业链 / 创新药 / 互联网 的主板（60/00，含原中小板002/001）可交易标的，
按产业链环节（stage）细分，供机会卡片产业定位。classify 返回赛道，classify_detail 返回(赛道, 环节)。
仅收录主板标的；创业板300/301、科创板688、北交所均被主流程过滤，此处不列。
名录为人工维护的产业标签，须数据诚实：不把非该环节的股票误标（避免误导 classify）。
"""
from __future__ import annotations

from typing import Optional

# 赛道 → 环节 → 主板标的代码（合集，已核查板块归属）
TRACK_STOCKS: dict[str, dict[str, dict[str, set[str]]]] = {
    "AI算力": {
        "光模块/光通信": {"000988", "002281", "000063"},          # 华工科技 光迅科技 中兴通讯
        "CCL/PCB": {"002463", "002916", "600183", "603228", "001389", "002815"},  # 沪电 深南 生益 景旺 广合 崇达
        "服务器/整机": {"601138", "000977", "603019", "000938", "000034", "000066"},  # 工业富联 浪潮 曙光 紫光 神州 长城
        "铜连接/连接器": {"002130", "002897", "600577"},          # 沃尔核材 意华股份 精达股份(铜缆)
        "液冷/温控": {"002837", "002272"},                        # 英维克 川润股份
        "电源/磁性元件": {"002851", "002885", "002335"},          # 麦格米特 京泉华 科华数据
        "IDC/算力运营": {"600845", "603881", "002335"},           # 宝信软件 数据港 科华数据
        "网络设备/交换机": {"000063", "600498"},                   # 中兴通讯 烽火通信
    },
    "创新药": {
        "创新药企": {"600276", "002262", "600079", "002653", "600380", "000963", "600196", "601607"},
        "生长激素/生物制品": {"000661", "603392", "002007"},
        "CXO/外包服务": {"603259", "002821", "603127"},
        "原料药/中间体": {"600521", "000739", "605116", "603520", "002332", "002020"},
    },
    "互联网": {
        "游戏/内容": {"002555", "603444", "002624", "002517", "002602", "002558", "002174", "603258", "002605"},
        "广告/营销": {"002027", "002131"},
        "平台/应用": {"601360", "002195"},
        "电商/跨境": {"600415", "002315", "002095"},
    },
}

# code → 环节（供 classify_detail 返回）
STAGE_BY_CODE: dict[str, str] = {
    c: stage
    for track, stages in TRACK_STOCKS.items()
    for stage, codes in stages.items()
    for c in codes
}


def classify(code: str, name: str = "") -> Optional[str]:
    """返回赛道（AI算力/创新药/互联网），无匹配返回 None。仅依赖主代码。"""
    code = str(code).zfill(6)
    stage = STAGE_BY_CODE.get(code)
    if not stage:
        return None
    for track, stages in TRACK_STOCKS.items():
        if any(code in codes for codes in stages.values()):
            return track
    return None


def classify_detail(code: str, name: str = "") -> Optional[tuple]:
    """返回 (赛道, 环节) 或 None。"""
    code = str(code).zfill(6)
    stage = STAGE_BY_CODE.get(code)
    if not stage:
        return None
    for track, stages in TRACK_STOCKS.items():
        if code in stages.get(stage, set()):
            return track, stage
    return None


def track_meta() -> dict:
    """口径展示：赛道 → 主板标的数 / 环节数。"""
    return {t: {"stocks": sum(len(c) for c in s.values()), "stages": len(s)}
            for t, s in TRACK_STOCKS.items()}


# 东财行业板块名 → 使用者熟悉的概念名（界面显示为"行业名(概念)"）。
# 根因：用户按概念找板块（PCB/光纤），系统按东财行业名（元件/通信设备）展示，认不出。
BOARD_ALIAS: dict[str, str] = {
    "元件": "PCB",
    "光学光电子": "光模块/面板",
    "通信设备": "光通信/算力网络",
    "半导体": "算力芯片",
    "电子化学品": "半导体材料",
    "计算机设备": "算力硬件",
    "软件开发": "软件/AI应用",
    "互联网服务": "互联网",
    "游戏": "互联网·游戏",
    "文化传媒": "传媒/内容",
    "通信服务": "运营商",
    "消费电子": "消费电子",
    "汽车零部件": "汽车零部件",
    "电池": "锂电池",
    "光伏设备": "光伏",
    "能源金属": "锂/钴资源",
    "小金属": "稀土/小金属",
    "贵金属": "黄金",
    "工业金属": "铜铝",
    "航天航空": "军工",
    "船舶制造": "军工·船舶",
    "生物制品": "疫苗/血制品",
    "化学制药": "创新药/仿制药",
    "医疗器械": "医疗器械",
    "医疗服务": "CXO/医疗服务",
    "中药": "中药",
    "医药商业": "医药流通",
    "电源设备": "电力设备",
    "电网设备": "电网/特高压",
    "证券": "券商",
    "保险": "保险",
    "银行": "银行",
    "房地产开发": "地产",
    "酿酒行业": "白酒",
    "食品饮料": "食品饮料",
}


def board_display_name(board: str) -> str:
    """板块显示名：有映射时'行业(概念)'，否则原名。"""
    alias = BOARD_ALIAS.get(board)
    return f"{board}({alias})" if alias else board