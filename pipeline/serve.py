"""Serve the reports locally so any browser can play the synced video.

Opening report.html as a file works in Chrome, but Safari blocks a local page
from loading the recording in data/raw/ and needs video sent in byte ranges.
This serves the project folder over http://localhost with Range support.

    python -m pipeline.serve                     # list of reports
    python -m pipeline.serve runs/match/<video>  # open one report
"""
from __future__ import annotations

import argparse
import html
import os
import re
import threading
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Handler(SimpleHTTPRequestHandler):
    """Static files with HTTP Range support (needed for video seeking in Safari)."""

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._index()
        rng = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not rng or not path.is_file():
            return super().do_GET()
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
        size = path.stat().st_size
        if not m or (m.group(1) == "" and m.group(2) == ""):
            return super().do_GET()
        if m.group(1) == "":                       # suffix range: last N bytes
            start, end = max(0, size - int(m.group(2))), size - 1
        else:
            start = int(m.group(1))
            end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
        if start >= size or start > end:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass                                # the browser cancelled while seeking

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def _index(self):
        rows = []
        for rep in sorted((ROOT / "runs" / "match").glob("*/report.html")):
            rel = rep.relative_to(ROOT).as_posix()
            rows.append(f'<li><a href="/{html.escape(rel)}">{html.escape(rep.parent.name)}</a></li>')
        body = ("<!doctype html><meta charset=utf-8><title>Reports</title>"
                "<body style='font:16px system-ui;padding:16px'><h1>Match reports</h1><ul>"
                + "".join(rows) + "</ul>").encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Serve match reports on localhost.")
    ap.add_argument("report", nargs="?", type=Path, help="a match folder (runs/match/<video>) to open")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT)))
    url = f"http://localhost:{args.port}/"
    if args.report:
        url += (args.report.resolve() / "report.html").relative_to(ROOT).as_posix()
    print(f"Serving reports at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
