#!/usr/bin/env python
"""Browser GUI for testing the final AST submission models."""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.submission_serving import DEFAULT_OUTPUT_DIR, SubmissionModelService


STATIC_ROOT = PROJECT_ROOT / "web_ui"


class SubmissionUIHandler(BaseHTTPRequestHandler):
    service: SubmissionModelService

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(STATIC_ROOT / "index.html")
            return
        if parsed.path == "/api/models":
            self._send_json(
                {
                    "models": self.service.available_models(),
                    "metrics": self.service.metrics_summary(),
                    "dynamic_vocab": self.service.dynamic_vocab_summary(),
                    "output_dir": str(self.service.output_dir),
                }
            )
            return
        if parsed.path == "/api/metrics":
            self._send_json({"metrics": self.service.metrics_summary()})
            return
        if parsed.path.startswith("/static/"):
            safe_name = parsed.path.removeprefix("/static/")
            self._send_file(STATIC_ROOT / "static" / safe_name)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(STATIC_ROOT / "index.html", include_body=False)
            return
        if parsed.path.startswith("/static/"):
            safe_name = parsed.path.removeprefix("/static/")
            self._send_file(STATIC_ROOT / "static" / safe_name, include_body=False)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/predict":
                self._send_json(
                    self.service.predict(
                        str(payload.get("text") or ""),
                        mode=str(payload.get("mode") or "text_ast_fgm"),
                        model_name=str(payload.get("model") or "cnn"),
                    )
                )
                return
            if parsed.path == "/api/compare":
                self._send_json({"rows": self.service.compare_models(str(payload.get("text") or ""))})
                return
            if parsed.path == "/api/attack":
                self._send_json(
                    self.service.attack_search(
                        str(payload.get("text") or ""),
                        mode=str(payload.get("mode") or "text_ast_fgm"),
                        model_name=str(payload.get("model") or "cnn"),
                        label=str(payload.get("label") or "spam"),
                        max_variants=int(payload["max_variants"]) if payload.get("max_variants") else None,
                        strength=str(payload.get("strength") or "mild"),
                    )
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # pragma: no cover - surfaced to browser
            self._send_json({"error": type(exc).__name__, "message": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, include_body: bool = True) -> None:
        path = path.resolve()
        try:
            path.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".js":
            content_type = "application/javascript"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the AST submission testing GUI.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Final submission artifact directory.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = SubmissionModelService(Path(args.output_dir))
    SubmissionUIHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), SubmissionUIHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Submission testing UI: {url}")
    print(f"Artifacts: {Path(args.output_dir).resolve()}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
