"""bookkit.server — a zero-dependency review UI for a scanned book.

The loop this server exists for::

    agent finds damaged OCR  ->  browser shows the pixels  ->  human (or
    agent) types what the ink really says  ->  /api/repair  ->  exports change

Routes
------
``/``                      the review UI
``/api/book``              info, outline, sections, items, OCR summary
``/api/page/<n>``          paragraphs, lines, flags, ink boxes for one page
``/api/page/<n>.png``      page raster (``?dpi=150``)
``/api/crop/<n>.png``      region raster (``?line=3`` or ``?bbox=x0,y0,x1,y1``)
``/api/repairs``           GET the applied list, POST one ``{page,line,before,after}``
``/api/repair``            POST the same (alias kept for readability)
``/api/rescan``            POST: rebuild the model from the file and re-run QA
``/export/<kind>``         ``book.md`` | ``book.json`` | ``chunks.jsonl`` | ``ocr.json``

Only ``http.server`` from the standard library is used, so the UI runs anywhere
Python does.  Everything is bound to ``0.0.0.0`` so a sandboxed preview proxy
can reach it.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from . import __version__, ink, quality, render, structure
from .core import Book

WEB_DIR = Path(__file__).parent / "web"
JSON_HEADERS = {"Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store"}


class State:
    """Shared, lock-protected server state."""

    def __init__(self, book: Book, out_dir: Optional[Path] = None):
        self.book = book
        self.out_dir = out_dir or book.path.parent / "bookkit-out"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.repairs: list[quality.Repair] = []
        self.report = quality.scan(book)
        self.repairs_file = self.out_dir / "repairs.jsonl"
        if self.repairs_file.exists():
            try:
                self.repairs = quality.load_repairs(str(self.repairs_file))
                quality.apply_repairs(book, self.repairs)
                self.report = quality.scan(book)
            except Exception as exc:  # pragma: no cover
                print(f"! could not load {self.repairs_file}: {exc}")

    def save_repair(self, page: int, line: Optional[int], before: str, after: str) -> int:
        with self.lock:
            rep = quality.Repair(page=page, line=line, before=before, after=after,
                                 reason="web review", verified=True)
            quality.apply_repairs(self.book, [rep])
            self.repairs.append(rep)
            quality.save_repairs(self.repairs, str(self.repairs_file))
            self.report = quality.scan(self.book)
            return len(self.repairs)

    def rescan(self) -> None:
        with self.lock:
            from .core import load as _load

            self.book.close()
            self.book = _load(self.book.path)
            quality.apply_repairs(self.book, self.repairs)
            self.report = quality.scan(self.book)


class Handler(BaseHTTPRequestHandler):
    state: State

    server_version = f"bookkit/{__version__}"
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
        pass

    def _send(self, status: int, body: bytes, headers: dict[str, str]) -> None:
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False,
                                      default=str).encode("utf-8"), JSON_HEADERS)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message, "status": status}, status)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    # -- routes -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path.split("/", 2)[2])
            if path == "/api/book":
                return self._api_book()
            if path.startswith("/api/page/"):
                rest = path[len("/api/page/"):]
                if rest.endswith(".png"):
                    return self._page_png(rest[:-4], query)
                return self._api_page(rest, query)
            if path.startswith("/api/crop/"):
                rest = path[len("/api/crop/"):]
                return self._crop_png(rest.removesuffix(".png"), query)
            if path == "/api/repairs":
                return self._json([r.to_dict() for r in self.state.repairs])
            if path.startswith("/export/"):
                return self._export(path.split("/")[-1], query)
            return self._error(404, "not found")
        except Exception as exc:  # pragma: no cover - defensive
            return self._error(500, f"{type(exc).__name__}: {exc}")

    def do_HEAD(self) -> None:  # noqa: N802
        return self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        data = self._body()
        try:
            if path in ("/api/repair", "/api/repairs"):
                page = int(data.get("page") or 0)
                if not page:
                    return self._error(400, "page is required")
                total = self.state.save_repair(
                    page=page,
                    line=(int(data["line"]) if data.get("line") is not None else None),
                    before=str(data.get("before", "")),
                    after=str(data.get("after", "")),
                )
                return self._json({"ok": True, "repairs": total,
                                   "ocr": self.state.report.summary()})
            if path == "/api/rescan":
                self.state.rescan()
                return self._json({"ok": True, "ocr": self.state.report.summary()})
            return self._error(404, "not found")
        except Exception as exc:  # pragma: no cover - defensive
            return self._error(500, f"{type(exc).__name__}: {exc}")

    # -- handlers ---------------------------------------------------------
    def _static(self, name: str) -> None:
        candidate = (WEB_DIR / name).resolve()
        if not str(candidate).startswith(str(WEB_DIR.resolve())) or not candidate.exists():
            return self._error(404, "no such asset")
        ctype = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if candidate.suffix in (".html", ".css", ".js", ".svg"):
            ctype += "; charset=utf-8"
        self._send(200, candidate.read_bytes(),
                   {"Content-Type": ctype, "Cache-Control": "no-store"})

    def _api_book(self) -> None:
        book = self.state.book
        info = book.info()
        self._json({
            "bookkit": __version__,
            "info": info,
            "outline": [e.to_dict() for e in structure.outline(book)],
            "sections": structure.sections(book),
            "items": [i.to_dict(text=False) for i in structure.items(book)],
            "ocr": self.state.report.summary(),
            "page_verdicts": {p.number: {"verdict": p.verdict, "score": p.score,
                                         "flags": len(p.flags)}
                              for p in self.state.report.pages},
            "repairs": len(self.state.repairs),
        })

    def _api_page(self, raw: str, query: dict[str, list[str]]) -> None:
        try:
            n = int(raw)
        except ValueError:
            return self._error(400, "bad page number")
        book = self.state.book
        page = book.page(n)
        rep = next((p for p in self.state.report.pages if p.number == n), None)
        item_map: dict[int, list[dict[str, Any]]] = {}
        for it in structure.items(book):
            if it.page == n:
                item_map.setdefault(it.page, []).append(it.to_dict(text=False))
        self._json({
            "page": {
                "number": page.number,
                "width": page.width,
                "height": page.height,
                "chars": page.chars,
                "raster": page.raster,
                "raster_px": list(page.raster_px),
                "header": page.header,
                "folio": page.folio,
            },
            "verdict": rep.verdict if rep else "unknown",
            "score": rep.score if rep else 0.0,
            "paragraphs": [p.to_dict() for p in page.paragraphs],
            "lines": [{"i": l.index, "text": l.text,
                       "bbox": [round(v, 1) for v in l.bbox],
                       "junk": l.junk} for l in page.lines],
            "flags": [f.to_dict() for f in (rep.flags if rep else [])],
            "boxes": [b.to_dict() for b in ink.boxes(book, n)],
            "bars": [b.to_dict() for b in ink.bars(book, n)],
            "items": item_map.get(n, []),
        })

    def _page_png(self, raw: str, query: dict[str, list[str]]) -> None:
        dpi = int((query.get("dpi") or ["150"])[0])
        png = self.state.book.render(int(raw), dpi=max(40, min(dpi, 400)))
        self._send(200, png, {"Content-Type": "image/png",
                              "Cache-Control": "no-store"})

    def _crop_png(self, raw: str, query: dict[str, list[str]]) -> None:
        n = int(raw)
        dpi = max(60, min(int((query.get("dpi") or ["300"])[0]), 600))
        pad = float((query.get("pad") or ["6"])[0])
        book = self.state.book
        if query.get("line") is not None:
            png = book.line_crop(n, int(query["line"][0]), dpi=dpi, pad=pad)
        elif query.get("bbox"):
            x0, y0, x1, y1 = (float(v) for v in query["bbox"][0].split(","))
            png = book.crop(n, (x0, y0, x1, y1), dpi=dpi, pad=pad)
        else:
            png = book.render(n, dpi=dpi)
        self._send(200, png, {"Content-Type": "image/png",
                              "Cache-Control": "no-store"})

    def _export(self, kind: str, query: dict[str, list[str]]) -> None:
        book = self.state.book
        report = self.state.report
        if kind in ("book.md", "chapter.md", "md"):
            data = render.to_markdown(book, report).encode("utf-8")
            name, ctype = "book.md", "text/markdown; charset=utf-8"
        elif kind in ("book.json", "json"):
            data = json.dumps(render.payload(book, report), ensure_ascii=False,
                              indent=2, default=str).encode("utf-8")
            name, ctype = "book.json", "application/json; charset=utf-8"
        elif kind in ("chunks.jsonl", "jsonl"):
            data = render.chunks_jsonl(book, report).encode("utf-8")
            name, ctype = "chunks.jsonl", "application/x-ndjson; charset=utf-8"
        elif kind in ("ocr.json", "ocr"):
            data = json.dumps(report.to_dict(), ensure_ascii=False,
                              indent=2).encode("utf-8")
            name, ctype = "ocr.json", "application/json; charset=utf-8"
        elif kind in ("digest.json", "digest"):
            data = json.dumps(render.digest(book, report), ensure_ascii=False,
                              indent=2).encode("utf-8")
            name, ctype = "digest.json", "application/json; charset=utf-8"
        else:
            return self._error(404, "unknown export")
        sticker = query.get("save", ["0"])[0] in ("1", "true", "yes")
        if sticker:
            (self.state.out_dir / name).write_bytes(data)
        self._send(200, data, {"Content-Type": ctype,
                               "Content-Disposition": f'inline; filename="{name}"',
                               "Cache-Control": "no-store"})


def serve(book: Book, host: str = "0.0.0.0", port: int = 8765,
          out_dir: Optional[Path] = None) -> None:
    """Run the review UI until interrupted."""
    state = State(book, out_dir=out_dir)
    handler = type("BoundHandler", (Handler,), {"state": state})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    summary = state.report.summary()
    print(f"bookkit {__version__} — {book.path.name}")
    print(f"  pages {summary['pages']}  ocr score {summary['score']}  "
          f"flags {summary['flag_counts']}")
    print(f"  listening on http://{host}:{port}   (exports -> {state.out_dir})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()
