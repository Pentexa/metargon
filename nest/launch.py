#!/usr/bin/env python3
"""METARGON NEST — launcher.

Usage:
    python nest/launch.py            # default: http://127.0.0.1:8765
    python nest/launch.py --port 9000 --no-browser

Windows: double-click or run from the repo root. Ctrl+C stops.
"""
from __future__ import annotations

import argparse
import threading
import webbrowser

from nest.server import make_server
from nest.command_interpreter import OpenAICompatibleAdapter


def main(argv=None):
    p = argparse.ArgumentParser(description="METARGON NEST")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    adapter = OpenAICompatibleAdapter()   # enabled only if METARGON_NEST_API_KEY set
    server, app = make_server(host=args.host, port=args.port, api_adapter=adapter)
    url = f"http://{args.host}:{args.port}"
    print(f"METARGON NEST listening on {url}")
    print(f"  storage : {app.storage.root}")
    print(f"  api mode: {app.config()['api_mode']} "
          f"(set METARGON_NEST_API_KEY to enable provider-assisted language understanding)")
    print("  Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
