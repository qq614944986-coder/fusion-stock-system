# -*- coding: utf-8 -*-
"""本地 Web 应用服务（零第三方依赖，Python 标准库实现）。

用法：
    python server.py            # 启动后自动打开浏览器 http://127.0.0.1:8199

设计：数据获取完全手动——界面点「行情刷新」按钮才触发分析（POST /api/refresh），
平时打开页面只读上一次结果（GET /api/data），不产生任何网络请求。

路由：
    GET  /                    → webapp/index.html
    GET  /api/data            → data/last_result.json（上次分析结果）
    POST /api/refresh         → 重新拉取数据并分析（后台线程执行 main.run）
    GET  /api/watchlist       → 自选股列表
    POST /api/watchlist       → 添加自选 {code, name, mode: watching|positions, cost?, shares?}
    POST /api/watchlist/remove→ 移除自选 {code}
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import yaml

BASE = Path(__file__).resolve().parent
PORT = 8199

_analyze_lock = threading.Lock()
_analyze_status = {"running": False, "error": None}


# ---------------------------------------------------------------- 自选股读写

def _watchlist_path() -> Path:
    return BASE / "config" / "watchlist.yaml"


def read_watchlist() -> dict:
    p = _watchlist_path()
    if not p.exists():
        return {"positions": [], "watching": []}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data.setdefault("positions", [])
    data.setdefault("watching", [])
    return data


def _write_watchlist(data: dict) -> None:
    _watchlist_path().write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def add_watchlist(code: str, name: str, mode: str = "watching",
                  cost: float | None = None, shares: int | None = None) -> dict:
    code = str(code).strip().zfill(6)
    if not (code.isdigit() and len(code) == 6):
        return {"ok": False, "msg": "股票代码应为6位数字"}
    wl = read_watchlist()
    for sec in ("positions", "watching"):
        wl[sec] = [r for r in wl[sec] if str(r.get("code", "")).zfill(6) != code]
    entry = {"code": code, "name": name or code}
    if mode == "positions":
        if cost is not None:
            entry["cost"] = float(cost)
        if shares is not None:
            entry["shares"] = int(shares)
        wl["positions"].append(entry)
    else:
        wl["watching"].append(entry)
    _write_watchlist(wl)
    return {"ok": True, "msg": f"{name or code} 已加入{'持仓' if mode == 'positions' else '关注'}"}


def remove_watchlist(code: str) -> dict:
    code = str(code).strip().zfill(6)
    wl = read_watchlist()
    before = sum(len(wl[s]) for s in ("positions", "watching"))
    for sec in ("positions", "watching"):
        wl[sec] = [r for r in wl[sec] if str(r.get("code", "")).zfill(6) != code]
    after = sum(len(wl[s]) for s in ("positions", "watching"))
    if after == before:
        return {"ok": False, "msg": f"{code} 不在自选列表中"}
    _write_watchlist(wl)
    return {"ok": True, "msg": f"{code} 已移除"}


# ---------------------------------------------------------------- 分析触发

def run_analysis(t0_signal: str = "none") -> dict:
    """在锁保护下执行 main.run()，返回状态。"""
    if not _analyze_lock.acquire(blocking=False):
        return {"ok": False, "msg": "已有分析在进行中，请稍候"}
    _analyze_status.update(running=True, error=None)
    try:
        import main as main_mod
        main_mod.run(t0_signal=t0_signal)
        return {"ok": True, "msg": "分析完成"}
    except Exception as e:  # noqa: BLE001 — 任何异常都要反馈给前端
        _analyze_status["error"] = f"{type(e).__name__}: {e}"
        return {"ok": False, "msg": _analyze_status["error"]}
    finally:
        _analyze_status["running"] = False
        _analyze_lock.release()


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def log_message(self, fmt, *args):  # 精简控制台输出
        print(f"[{self.command}] {urlparse(self.path).path}")

    # ---------------- GET

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            fp = BASE / "webapp" / "index.html"
            if fp.exists():
                self._send(200, fp.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, "webapp/index.html 缺失".encode(), "text/plain; charset=utf-8")
        elif path == "/api/data":
            fp = BASE / "data" / "last_result.json"
            if fp.exists():
                self._send(200, fp.read_bytes(), "application/json; charset=utf-8")
            else:
                self._json({"empty": True, "msg": "尚无分析结果，请点击「行情刷新」"})
        elif path == "/api/status":
            self._json(_analyze_status)
        elif path == "/api/watchlist":
            self._json(read_watchlist())
        else:
            self._json({"error": "not found"}, 404)

    # ---------------- POST

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._body()
        if path == "/api/refresh":
            t0 = body.get("t0_signal", "none")
            threading.Thread(target=run_analysis, kwargs={"t0_signal": t0}, daemon=True).start()
            self._json({"ok": True, "msg": "已开始拉取数据并分析（约1-2分钟），完成后自动刷新",
                        "t0_signal": t0})
        elif path == "/api/watchlist":
            self._json(add_watchlist(body.get("code", ""), body.get("name", ""),
                                     body.get("mode", "watching"),
                                     body.get("cost"), body.get("shares")))
        elif path == "/api/watchlist/remove":
            self._json(remove_watchlist(body.get("code", "")))
        else:
            self._json({"error": "not found"}, 404)


def main() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"双引擎选股择时系统 · 本地服务已启动: {url}  (Ctrl+C 退出)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")


if __name__ == "__main__":
    main()
