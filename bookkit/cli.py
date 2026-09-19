"""bookkit.cli — command line interface.

    bookkit info     book.pdf                 # metadata + page census
    bookkit outline  book.pdf                 # headings with pages
    bookkit text     book.pdf -p 6            # one page, layout-aware
    bookkit items    book.pdf --kind problem  # problems/exercises/solutions
    bookkit md       book.pdf -o chapter.md   # full Markdown export
    bookkit json     book.pdf -o chapter.json # structured export
    bookkit jsonl    book.pdf -o chunks.jsonl # retrieval chunks
    bookkit digest   book.pdf                 # compact agent-sized map
    bookkit ocr      book.pdf                 # OCR quality report + queue
    bookkit crop     book.pdf -p 6 --line 3 -o line.png     # pixels on demand
    bookkit find     book.pdf "Problem 1.7" -o crops/       # crop every hit
    bookkit fix      book.pdf --repairs fixes.json -o fixed.md
    bookkit serve    book.pdf -o out/         # live review UI

Every command supports ``--json`` for machine consumption.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from . import __version__, ink, quality, render, structure
from .core import Book, load


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _open(args: argparse.Namespace) -> Book:
    return load(args.pdf, password=getattr(args, "password", None))


def _emit(payload: Any, as_json: bool, text_fn=None) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    elif text_fn is not None:
        print(text_fn())
    else:
        print(payload)


def _write_or_print(data: str, out: Optional[str]) -> None:
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data, encoding="utf-8")
        print(f"wrote {p} ({len(data):,} bytes)")
    else:
        sys.stdout.write(data)


def _write_bytes(data: bytes, out: Optional[str]) -> None:
    if not out:
        sys.stdout.buffer.write(data)
        return
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    print(f"wrote {p} ({len(data):,} bytes)")


def _parse_range(spec: Optional[str], total: int) -> list[int]:
    """``3`` / ``2-5`` / ``1,4,9`` -> list of 1-based page numbers."""
    if not spec:
        return list(range(1, total + 1))
    pages: list[int] = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, _, b = chunk.partition("-")
            pages.extend(range(int(a), int(b) + 1))
        elif chunk:
            pages.append(int(chunk))
    return [p for p in dict.fromkeys(pages) if 1 <= p <= total]


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    with _open(args) as book:
        info = book.info()
        info["path"] = str(book.path)
        if args.json:
            print(json.dumps(info, ensure_ascii=False, indent=2, default=str))
            return 0
        print(f"{book.path.name}")
        print(f"  pages        : {info['pages']}  (scanned: {info['scanned_pages']})")
        print(f"  raster       : {info['raster_px'][0]}x{info['raster_px'][1]} px/page")
        print(f"  chars        : {info['total_chars']:,}  "
              f"(median {info['median_chars_per_page']:.0f}/page)")
        for key in ("title", "author", "producer"):
            if info.get(key):
                print(f"  {key:<13}: {info[key]}")
        print(f"  encrypted    : {info['encrypted']}")
    return 0


def cmd_outline(args: argparse.Namespace) -> int:
    with _open(args) as book:
        entries = [e.to_dict() for e in structure.outline(book)]
        if args.json:
            print(json.dumps({"outline": entries, "sections": structure.sections(book)},
                             ensure_ascii=False, indent=2))
            return 0
        for e in structure.outline(book):
            pad = "  " * max(0, e.level - 1)
            print(f"{pad}[L{e.level}] {e.title}   (p{e.page}, y={e.y:.0f})")
    return 0


def cmd_text(args: argparse.Namespace) -> int:
    with _open(args) as book:
        pages = _parse_range(args.pages, len(book))
        out: list[str] = []
        for n in pages:
            page = book.page(n)
            out.append(f"===== page {n}  ({page.width:.0f}x{page.height:.0f}pt, "
                       f"{page.chars} chars, raster={page.raster}) =====")
            for par in page.paragraphs:
                if args.body_only and par.role not in ("body",):
                    continue
                mark = {
                    "heading": "#", "problem": "P", "exercise": "E", "solution": "S",
                    "example": "X", "label": "L", "theorem": "T", "noise": "-",
                }.get(par.role, " ")
                out.append(f"[{mark}] {par.text}")
        if args.json:
            print(json.dumps(render.payload(book, page=pages[0]), ensure_ascii=False, indent=2))
        else:
            print("\n".join(out))
    return 0


def cmd_items(args: argparse.Namespace) -> int:
    with _open(args) as book:
        items = structure.items(book)
        if args.kind:
            kinds = set(args.kind)
            items = [i for i in items if i.kind in kinds]
        if args.section:
            items = [i for i in items if (i.section_number or "") == args.section]
        if args.page:
            lo, _, hi = args.page.partition("-")
            lo_i, hi_i = int(lo), int(hi or lo)
            items = [i for i in items if lo_i <= i.page <= hi_i]
        if args.json:
            print(json.dumps([i.to_dict() for i in items], ensure_ascii=False, indent=2))
            return 0
        for it in items:
            head = f"{it.tag:<16} p{it.page}" + (f"-{it.end_page}" if it.end_page != it.page else "")
            bits = [head, f"box={'Y' if it.in_box else 'n'}"]
            if it.hints:
                bits.append("hints=" + ",".join(it.hints))
            if it.duplicate_of is not None:
                bits.append(f"dup#{it.duplicate_of}")
            print(f"[{it.kind:<8}] {'  '.join(bits)}")
            print(f"    {it.text[:230]}")
    return 0


def cmd_md(args: argparse.Namespace) -> int:
    with _open(args) as book:
        report = quality.scan(book)
        opts = render.MarkdownOptions(
            include_flags=not args.no_flags,
            include_outline=not args.no_outline,
            page_breaks=not args.no_page_breaks,
        )
        md = render.to_markdown(book, None if args.no_flags else report, opts)
        _write_or_print(md, args.out)
    return 0


def cmd_json(args: argparse.Namespace) -> int:
    with _open(args) as book:
        report = quality.scan(book)
        payload = render.payload(book, report)
        data = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        _write_or_print(data, args.out)
    return 0


def cmd_jsonl(args: argparse.Namespace) -> int:
    with _open(args) as book:
        report = quality.scan(book)
        data = render.chunks_jsonl(book, report)
        _write_or_print(data, args.out)
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    with _open(args) as book:
        report = quality.scan(book)
        if args.json:
            print(json.dumps(render.digest(book, report), ensure_ascii=False, indent=2))
            return 0
        print(render.digest_text(book, report))
    return 0


def cmd_ocr(args: argparse.Namespace) -> int:
    with _open(args) as book:
        report = quality.scan(book)
        if args.json:
            print(json.dumps(report.to_dict(with_flags=args.flags),
                             ensure_ascii=False, indent=2))
            return 0
        print(report.text(top=args.top))
        if args.queue:
            print()
            print("queue (JSON, feed to `bookkit crop` or a vision model):")
            print(json.dumps([f.to_dict() for f in report.queue(limit=args.top)],
                             ensure_ascii=False, indent=2))
        if args.out:
            _write_or_print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                            args.out)
    return 0


def cmd_crop(args: argparse.Namespace) -> int:
    with _open(args) as book:
        page = book.page(args.page)
        if args.line is not None:
            png = book.line_crop(args.page, args.line, dpi=args.dpi, pad=args.pad)
            where = f"page {args.page} line {args.line}: {page.lines[args.line].text[:60]}"
        elif args.bbox:
            x0, y0, x1, y1 = (float(v) for v in args.bbox.split(","))
            png = book.crop(args.page, (x0, y0, x1, y1), dpi=args.dpi, pad=args.pad)
            where = f"page {args.page} bbox {args.bbox}"
        else:
            png = book.render(args.page, dpi=args.dpi)
            where = f"page {args.page} (full)"
        print(f"# crop: {where} @ {args.dpi} dpi", file=sys.stderr)
        _write_bytes(png, args.out)
    return 0


def cmd_find(args: argparse.Namespace) -> int:
    with _open(args) as book:
        hits = book.find_bbox(args.needle, pad_lines=args.pad_lines)
        if not hits:
            print(f"no match for {args.needle!r}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(hits, ensure_ascii=False, indent=2))
            return 0
        out_dir = Path(args.out or "crops")
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, hit in enumerate(hits):
            png = book.crop(hit["page"], tuple(hit["bbox"]), dpi=args.dpi, pad=4.0)
            name = out_dir / f"p{hit['page']:03d}-l{hit['line']:03d}-{idx}.png"
            name.write_bytes(png)
            print(f"{name}  <- {hit['text'][:70]}")
        print(f"\n{len(hits)} crop(s) in {out_dir}")
    return 0


def cmd_fix(args: argparse.Namespace) -> int:
    """Apply agent-authored repairs, or auto-repair safe patterns only."""
    with _open(args) as book:
        repairs = quality.load_repairs(args.repairs) if args.repairs else []
        applied = quality.apply_repairs(book, repairs) if repairs else 0
        before = quality.scan(book)
        if args.out:
            md = render.to_markdown(book, before)
            _write_or_print(md, args.out)
        else:
            print(f"applied {applied} repair(s)")
            print(before.text(top=args.top))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    with _open(args) as book:
        serve(book, host=args.host, port=args.port,
              out_dir=Path(args.out) if args.out else None)
    return 0


def cmd_pages(args: argparse.Namespace) -> int:
    with _open(args) as book:
        rows = book.stats_rows()
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
            return 0
        print(f"{'page':>4} {'chars':>6} {'lines':>6} {'paras':>6} {'imgs':>5} "
              f"{'raster':>6} {'midsz':>6}  header")
        for r in rows:
            print(f"{r['page']:>4} {r['chars']:>6} {r['lines']:>6} {r['paragraphs']:>6} "
                  f"{r['images']:>5} {str(r['raster']):>6} {r['median_size']:>6}  "
                  f"{r['header'][:40]}")
    return 0


def cmd_ink(args: argparse.Namespace) -> int:
    """Debug view: rules/bars/boxes plus an optional overlay PNG."""
    with _open(args) as book:
        pages = _parse_range(args.pages, len(book))
        if args.json:
            data = {
                str(n): {
                    "rules": [r.to_dict() for r in ink.rules(book, n)],
                    "boxes": [b.to_dict() for b in ink.boxes(book, n)],
                    "bars": [b.to_dict() for b in ink.bars(book, n)],
                }
                for n in pages
            }
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            for n in pages:
                print(f"page {n}: {len(ink.rules(book, n))} rules, "
                      f"{len(ink.boxes(book, n))} boxes, {len(ink.bars(book, n))} bars")
                for r in ink.rules(book, n):
                    print(f"   {r.orient} pos={r.pos:7.1f} {r.start:6.1f}-{r.end:6.1f} "
                          f"th={r.thickness:4.1f} ink={r.ink_ratio:.2f} "
                          f"{'dashed' if r.dashed else 'solid'}")
        if args.overlay:
            import pymupdf

            for n in pages:
                page = book.doc[n - 1]
                for r in ink.rules(book, n):
                    colour = (1, 0, 0) if r.orient == "h" else (0, 0, 1)
                    if r.orient == "h":
                        page.draw_line(pymupdf.Point(r.start, r.pos),
                                       pymupdf.Point(r.end, r.pos), color=colour, width=0.7)
                    else:
                        page.draw_line(pymupdf.Point(r.pos, r.start),
                                       pymupdf.Point(r.pos, r.end), color=colour, width=0.7)
                for b in ink.boxes(book, n):
                    page.draw_rect(pymupdf.Rect(b.x0, b.y0, b.x1, b.y1),
                                   color=(0, 0.7, 0), width=1.0)
                for b in ink.bars(book, n):
                    page.draw_rect(pymupdf.Rect(b.x0, b.y0, b.x1, b.y1),
                                   color=(0.8, 0, 0.8), width=1.0)
                out = Path(args.overlay)
                if len(pages) > 1:
                    out = out.with_name(f"{out.stem}-p{n}{out.suffix}")
                out.parent.mkdir(parents=True, exist_ok=True)
                page.get_pixmap(dpi=args.dpi).save(out)
                print(f"overlay -> {out}")
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bookkit",
        description="Layout-aware toolkit for scanned book PDFs (Markdown, items, "
                    "OCR QA, crops).",
    )
    p.add_argument("--version", action="version", version=f"bookkit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def book_arg(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("pdf", help="path to the PDF")
        sp.add_argument("--password", help="PDF password, if any")

    sp = sub.add_parser("info", help="metadata and page census")
    book_arg(sp); sp.add_argument("--json", action="store_true"); sp.set_defaults(fn=cmd_info)

    sp = sub.add_parser("pages", help="per-page statistics")
    book_arg(sp); sp.add_argument("--json", action="store_true"); sp.set_defaults(fn=cmd_pages)

    sp = sub.add_parser("outline", help="headings with page positions")
    book_arg(sp); sp.add_argument("--json", action="store_true"); sp.set_defaults(fn=cmd_outline)

    sp = sub.add_parser("text", help="reconstructed text of selected pages")
    book_arg(sp)
    sp.add_argument("-p", "--pages", help="1 | 2-5 | 1,4,9 (default: all)")
    sp.add_argument("--body-only", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_text)

    sp = sub.add_parser("items", help="problems / exercises / solutions")
    book_arg(sp)
    sp.add_argument("--kind", nargs="*", default=None, help="problem exercise solution …")
    sp.add_argument("--section", help="filter by section number, e.g. 1.3")
    sp.add_argument("--page", help="filter by page or page range")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_items)

    sp = sub.add_parser("md", help="export Markdown")
    book_arg(sp)
    sp.add_argument("-o", "--out")
    sp.add_argument("--no-flags", action="store_true", help="omit OCR warnings")
    sp.add_argument("--no-outline", action="store_true")
    sp.add_argument("--no-page-breaks", action="store_true")
    sp.set_defaults(fn=cmd_md)

    sp = sub.add_parser("json", help="export structured JSON")
    book_arg(sp); sp.add_argument("-o", "--out"); sp.set_defaults(fn=cmd_json)

    sp = sub.add_parser("jsonl", help="export retrieval chunks (JSON Lines)")
    book_arg(sp); sp.add_argument("-o", "--out"); sp.set_defaults(fn=cmd_jsonl)

    sp = sub.add_parser("digest", help="compact map for limited context windows")
    book_arg(sp); sp.add_argument("--json", action="store_true"); sp.set_defaults(fn=cmd_digest)

    sp = sub.add_parser("ocr", help="OCR quality report and repair queue")
    book_arg(sp)
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--flags", action="store_true", help="include every flag in JSON")
    sp.add_argument("--queue", action="store_true", help="print the review queue as JSON")
    sp.add_argument("--top", type=int, default=25)
    sp.add_argument("-o", "--out")
    sp.set_defaults(fn=cmd_ocr)

    sp = sub.add_parser("crop", help="render a page, a line or a bbox to PNG")
    book_arg(sp)
    sp.add_argument("-p", "--page", type=int, required=True)
    sp.add_argument("--line", type=int, help="line index on that page")
    sp.add_argument("--bbox", help="x0,y0,x1,y1 in PDF points")
    sp.add_argument("--dpi", type=int, default=300)
    sp.add_argument("--pad", type=float, default=6.0)
    sp.add_argument("-o", "--out", required=True)
    sp.set_defaults(fn=cmd_crop)

    sp = sub.add_parser("find", help="crop every occurrence of a string")
    book_arg(sp)
    sp.add_argument("needle")
    sp.add_argument("-o", "--out", help="output directory (default: crops/)")
    sp.add_argument("--dpi", type=int, default=300)
    sp.add_argument("--pad-lines", type=int, default=1)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_find)

    sp = sub.add_parser("fix", help="apply repairs and re-export")
    book_arg(sp)
    sp.add_argument("--repairs", help="JSON/JSONL of {page,line,before,after}")
    sp.add_argument("-o", "--out")
    sp.add_argument("--top", type=int, default=10)
    sp.set_defaults(fn=cmd_fix)

    sp = sub.add_parser("ink", help="debug: rules, boxes, bars (+ overlay PNG)")
    book_arg(sp)
    sp.add_argument("-p", "--pages", help="page or range (default: all)")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--overlay", help="write an overlay PNG here")
    sp.add_argument("--dpi", type=int, default=90)
    sp.set_defaults(fn=cmd_ink)

    sp = sub.add_parser("serve", help="live review UI")
    book_arg(sp)
    sp.add_argument("-o", "--out", help="output directory")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8765)
    sp.set_defaults(fn=cmd_serve)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except FileNotFoundError as exc:
        print(f"bookkit: no such file: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
