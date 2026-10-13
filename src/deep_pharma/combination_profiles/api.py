"""Loopback-only read API. No model fit, uploads, arbitrary paths or external calls."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import yaml

from .datastore import ProfileStore

ASSETS = Path(__file__).parent / "assets"


def make_server(store: ProfileStore, port=8766) -> ThreadingHTTPServer:
    @lru_cache(maxsize=128)
    def comparison(payload: str):
        return store.compare(json.loads(payload))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Selection bodies and molecule IDs are not written to access logs.

        def respond(self, status: int, body: bytes, mime="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob:; "
                "style-src 'self'; script-src 'self'; connect-src 'self'; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def local_request(self):
            host = self.headers.get("Host", "")
            valid = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if host not in valid:
                self.respond(403, b'{"error":"Invalid host"}')
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + h for h in valid}:
                self.respond(403, b'{"error":"Invalid origin"}')
                return False
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                self.respond(403, b'{"error":"Cross-site request refused"}')
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            path = urlparse(self.path).path
            if path == "/api/catalog":
                self.respond(
                    200, json.dumps(store.catalog(), ensure_ascii=False, allow_nan=False).encode()
                )
                return
            files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
            }
            if path not in files:
                self.respond(404, b'{"error":"Not found"}')
                return
            filename, mime = files[path]
            self.respond(200, (ASSETS / filename).read_bytes(), mime)

        def do_POST(self):
            if not self.local_request():
                return
            if self.path != "/api/compare":
                self.respond(404, b'{"error":"Not found"}')
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 32768:
                    raise ValueError("Requête trop volumineuse/vide")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("Objet JSON attendu")
                payload = json.dumps(request, sort_keys=True, allow_nan=False)
                result = comparison(payload)
                self.respond(200, json.dumps(result, ensure_ascii=False, allow_nan=False).encode())
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, json.dumps({"error": str(exc)}, ensure_ascii=False).encode())

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description="DeepPharma descriptive combination explorer")
    parser.add_argument("--config", type=Path, default=Path("configs/combination_profiles.yaml"))
    parser.add_argument("--port", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parent.parent

    def path(key):
        value = config.get(key)
        return root / value if value else None

    store = ProfileStore(
        path("train"), oof_dir=path("oof_dir"), manifest=path("manifest"), profiles=path("profiles")
    )
    output = path("audit")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(store.audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.audit_only:
        print(json.dumps(store.audit, ensure_ascii=True, indent=2))
        return
    server = make_server(store, args.port if args.port is not None else config.get("port", 8766))
    print(f"DeepPharma : http://127.0.0.1:{server.server_port} — Ctrl+C pour arrêter", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
