"""网页演示控制台：把状态机、识别画面、平衡曲线、行人安全域放到一个页面上。

启动（在 vision_arm 目录下）：
    python run.py demo
然后浏览器打开 http://127.0.0.1:8765

为什么用浏览器而不是 OpenCV 窗口（答辩要讲的）：
    1. OpenCV 窗口不能排版，也没法同时显示曲线和日志；
    2. 网页可以直接给评委看，甚至能远程投屏；
    3. 曲线是前端 Canvas 画的，不依赖 matplotlib，少一个装不上的依赖。
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .brain import DemoEngine

WEB_DIR = Path(__file__).resolve().parent / "web"


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


class Handler(BaseHTTPRequestHandler):
    engine: DemoEngine = None          # type: ignore[assignment]
    root: Path = None                  # type: ignore[assignment]
    server_version = "DormArmDemo/1.0"

    # ---- 基础工具 ----
    def log_message(self, fmt, *args):  # 关掉逐条请求日志，演示时刷屏很难看
        pass

    def _send(self, code: int, body: bytes, ctype: str = "text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # ---- 路由 ----
    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            page = WEB_DIR / "index.html"
            if not page.exists():
                self._send(500, b"web/index.html not found", "text/plain; charset=utf-8")
                return
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            return
        if u.path == "/api/state":
            self._json(self.engine.snapshot())
            return
        if u.path == "/api/frame.jpg":
            data = self.engine.frame_jpeg()
            if not data:
                self._send(503, b"", "image/jpeg")
                return
            self._send(200, data, "image/jpeg")
            return
        if u.path == "/api/balance":
            self._json(self._balance_payload())
            return
        self._send(404, b"not found")

    def do_POST(self):  # noqa: N802
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
        q = parse_qs(raw)
        if u.path == "/api/command":
            text = (q.get("text") or [""])[0]
            self._json(self.engine.command(text))
            return
        if u.path == "/api/thresholds":
            try:
                normal = float((q.get("normal") or ["0.8"])[0])
                slow = float((q.get("slow") or ["0.5"])[0])
                stop = float((q.get("stop") or ["0.3"])[0])
            except ValueError:
                self._json({"ok": False, "message": "参数不是数字"}, 400)
                return
            stop = max(0.05, min(stop, slow - 0.01))
            slow = max(stop + 0.01, min(slow, normal - 0.01))
            self.engine.set_safety(normal, slow, stop)
            self._json({"ok": True, "message": f"安全域已更新：{normal}/{slow}/{stop} m"})
            return
        if u.path == "/api/reset":
            self.engine.reset()
            self._json({"ok": True, "message": "已复位"})
            return
        self._send(404, b"not found")

    # ---- 平衡对照数据 ----
    def _balance_payload(self) -> dict:
        p = self.root / "out" / "balance_compare.json"
        plan = self.root / "out" / "balance_plan.json"
        out: dict = {"compare": None, "plan": None}
        if p.exists():
            try:
                out["compare"] = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                out["compare"] = None
        if plan.exists():
            try:
                out["plan"] = json.loads(plan.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                out["plan"] = None
        return out


def serve(root: Path, data_dir: Path, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True) -> None:
    Handler.root = root
    Handler.engine = DemoEngine(data_dir=data_dir, base_dir=root)
    Handler.engine.start()

    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    print("=" * 66)
    print("  宿舍取件机械臂 —— 可视化演示控制台")
    print("=" * 66)
    print(f"  本机访问 : {url}")
    if host in ("0.0.0.0", ""):
        print(f"  局域网   : http://{_lan_ip()}:{port}   （手机/其他电脑可看）")
    print("  按 Ctrl+C 停止")
    print("=" * 66)

    if open_browser:
        threading.Timer(0.8, lambda: __import__("webbrowser").open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        httpd.shutdown()
        Handler.engine.stop()
        print("已停止")
