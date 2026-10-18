"""Start the web app and the background worker together.

    python -m server            # this computer only: http://localhost:8000
    python -m server --lan      # also reachable from your phone on the same Wi-Fi
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys

import uvicorn


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the analyzer web app + worker.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--lan", action="store_true",
                    help="listen on your local network so your phone can open it (no password: home Wi-Fi only)")
    args = ap.parse_args()
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    worker = subprocess.Popen([sys.executable, "-m", "server.worker"])
    print(f"Open http://localhost:{args.port}" + (f"  (phone: http://{lan_ip()}:{args.port})" if args.lan else ""))
    try:
        uvicorn.run("server.app:app", host=host, port=args.port, log_level="warning")
    finally:
        worker.terminate()
        worker.wait(timeout=10)


if __name__ == "__main__":
    main()
