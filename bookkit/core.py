"""bookkit.core — PDF loading, page model and layout-aware text reconstruction.

Why not just ``page.get_text()``?
--------------------------------
For a scanned book the embedded text layer is OCR output: line order can be
scrambled, font sizes are jittery per glyph, and headings look exactly like
body text to a naive extractor.  We therefore rebuild the page from geometry:

* glyph spans -> lines (with union bboxes and a *median* line size),
* lines -> paragraphs using vertical gaps, indentation and sentence endings,
* repeated top/bottom lines -> running headers / folio numbers (stripped),
* blocks of raster covering the page -> "scanned page" detection.

Nothing is thrown away: every reconstructed line keeps its bbox so a later
stage can render that exact region as an image and re-read it with vision.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

try:  # pymupdf >= 1.24 exposes the modern name
    import pymupdf
except ImportError:  # pragma: no cover - legacy fallback
    import fitz as pymupdf  # type: ignore

BBox = tuple[float, float, float, float]

# --------------------------------------------------------------------------
# text hygiene
# --------------------------------------------------------------------------

_WS = re.compile(r"[\s\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_SOFT_HYPHEN = "\u00ad"

#: Patterns that *hint* a line starts a new block of content.  Used only to
#: avoid gluing a heading onto the paragraph below it; real classification
#: happens in :mod:`bookkit.structure`.
HEADING_HINTS = (
    re.compile(r"^CHAPTER\s+[0-9IVXLC]+\b", re.I),
    re.compile(r"^\d+(?:\.\d+)*\s+\S"),
    re.compile(r"^[A-Z][A-Z0-9 ,'\-&]{3,60}$"),  # ALL CAPS line
    re.compile(r"^(Problems?|Solutions?|Hints?|Important|Definition|Theorem|"
               r"Lemma|Corollary|Example|Exercise|Note|Remark|Proof)\b(?=$|[:\s])"),
)


def clean_text(s: str) -> str:
    """Collapse all whitespace runs to one space and drop invisible junk."""
    s = s.replace(_SOFT_HYPHEN, "")
    s = _WS.sub(" ", s)
    return s.strip()


def looks_like_heading(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 90:
        return False
    return any(p.match(t) for p in HEADING_HINTS)


def dehyphenate(prev: str, nxt: str) -> bool:
    """True when ``nxt`` should be joined to ``prev`` without a hyphen/space."""
    if not prev.endswith("-"):
        return False
    # keep real compounds such as "one- to" or em-dash usage
    if prev.endswith("--") or prev.endswith(" -"):
        return False
    return bool(nxt) and nxt[0].islower()


def join_lines(parts: Iterable[str]) -> str:
    """Join consecutive line strings of one paragraph into flowing text."""
    out = ""
    for raw in parts:
        piece = clean_text(raw)
        if not piece:
            continue
        if not out:
            out = piece
            continue
        if dehyphenate(out, piece) and not out.endswith("--"):
            out = out[:-1] + piece
        else:
            out += " " + piece
    return clean_text(_fix_inline_spacing(out))


_INLINE = [
    (re.compile(r"\s+([,.;:!?%)\]}])"), r"\1"),
    (re.compile(r"([(\[{])\s+"), r"\1"),
    (re.compile(r"\s{2,}"), " "),
]


def _fix_inline_spacing(s: str) -> str:
    for pat, rep in _INLINE:
        s = pat.sub(rep, s)
    return s


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


@dataclass
class Line:
    """One visual line of text with its source geometry."""

    text: str
    bbox: BBox
    size: float
    page_no: int
    index: int = 0
    block: int = 0
    indent: float = 0.0
    n_chars: int = 0
    span_count: int = 1
    size_min: float = 0.0
    size_max: float = 0.0
    junk: list[str] = field(default_factory=list)

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "i": self.index,
            "page": self.page_no,
            "text": self.text,
            "bbox": [round(v, 2) for v in self.bbox],
            "size": round(self.size, 2),
            "indent": round(self.indent, 2),
            "junk": self.junk,
        }


@dataclass
class Block:
    """A pymupdf text block (a rough visual region), kept for fidelity."""

    index: int
    lines: list[Line]
    bbox: BBox

    @property
    def text(self) -> str:
        return join_lines(l.text for l in self.lines)


@dataclass
class Paragraph:
    """Reflowed text: one logical paragraph / heading / labelled box."""

    text: str
    lines: list[Line]
    page_no: int
    bbox: BBox
    role: str = "body"          # body|heading|problem|exercise|label|box|header|folio|noise
    level: Optional[int] = None
    number: Optional[str] = None
    tag: Optional[str] = None   # "Problem 1.7" / "1.2.5"
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def line_numbers(self) -> list[int]:
        return [l.index for l in self.lines]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "page": self.page_no,
            "role": self.role,
            "text": self.text,
            "bbox": [round(v, 2) for v in self.bbox],
        }
        if self.level is not None:
            d["level"] = self.level
        if self.number:
            d["number"] = self.number
        if self.tag:
            d["tag"] = self.tag
        if self.attrs:
            d["attrs"] = self.attrs
        return d


@dataclass
class Page:
    number: int                 # 1-based
    width: float
    height: float
    lines: list[Line]
    blocks: list[Block]
    paragraphs: list[Paragraph]
    image_count: int = 0
    raster: bool = False        # a single raster image covers the page -> scan
    raster_px: tuple[int, int] = (0, 0)
    header: str = ""
    folio: str = ""
    median_size: float = 0.0
    left_margin: float = 0.0
    junk: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(l.n_chars for l in self.lines)

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)

    def lines_to_dict(self) -> list[dict[str, Any]]:
        return [l.to_dict() for l in self.lines]


# --------------------------------------------------------------------------
# reconstruction
# --------------------------------------------------------------------------


def _join_spans(spans: Sequence[dict[str, Any]]) -> tuple[str, float, float, float]:
    """Concatenate glyph spans, inserting a space only where geometry needs it."""
    parts: list[str] = []
    sizes: list[float] = []
    prev: Optional[dict[str, Any]] = None
    for sp in spans:
        txt = sp.get("text", "")
        if not txt:
            continue
        sizes.append(float(sp.get("size", 0.0)))
        if prev is not None:
            need_space = (
                not parts[-1].endswith((" ", "-", "/", "(", "["))
                and not txt.startswith((" ", "-", "/", ")", "]", ",", ".", ":", ";"))
                and txt[:1].isalnum()
                and parts[-1][-1:].isalnum()
            )
            if need_space:
                gap = float(sp["bbox"][0]) - float(prev["bbox"][2])
                em = max(float(sp.get("size", 10.0)), 1.0) * 0.34
                if gap > em:
                    parts.append(" ")
        parts.append(txt)
        prev = sp
    text = clean_text("".join(parts))
    size = statistics.median(sizes) if sizes else 0.0
    return text, round(size, 2), (round(min(sizes), 2) if sizes else 0.0), \
        (round(max(sizes), 2) if sizes else 0.0)


def _extract_lines(page: Any, page_no: int) -> tuple[list[Line], list[Block]]:
    raw = page.get_text("dict")
    lines: list[Line] = []
    blocks: list[Block] = []
    for bi, blk in enumerate(raw.get("blocks", [])):
        if blk.get("type") != 0:
            continue
        blk_lines: list[Line] = []
        for ln in blk.get("lines", []):
            text, size, smin, smax = _join_spans(ln.get("spans", []))
            if not text:
                continue
            x0, y0, x1, y1 = (float(v) for v in ln["bbox"])
            line = Line(
                text=text,
                bbox=(x0, y0, x1, y1),
                size=size,
                page_no=page_no,
                index=len(lines),
                block=bi,
                n_chars=len(text),
                span_count=len(ln.get("spans", [])),
                size_min=smin,
                size_max=smax,
            )
            lines.append(line)
            blk_lines.append(line)
        if blk_lines:
            xs0 = min(l.bbox[0] for l in blk_lines)
            ys0 = min(l.bbox[1] for l in blk_lines)
            xs1 = max(l.bbox[2] for l in blk_lines)
            ys1 = max(l.bbox[3] for l in blk_lines)
            blocks.append(Block(bi, blk_lines, (xs0, ys0, xs1, ys1)))
    lines.sort(key=lambda l: (round(l.y0, 1), l.x0))
    for i, ln in enumerate(lines):
        ln.index = i
    return lines, blocks


def _is_raster_page(page: Any) -> tuple[bool, int, tuple[int, int]]:
    """A page whose ink is one big image is a scan."""
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        infos = []
    page_area = max(page.rect.width * page.rect.height, 1.0)
    best = 0.0
    px = (0, 0)
    for info in infos:
        x0, y0, x1, y1 = info.get("bbox", (0, 0, 0, 0))
        area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
        cover = area / page_area
        if cover > best:
            best = cover
            w = info.get("width") or 0
            h = info.get("height") or 0
            px = (int(w), int(h))
    count = len(page.get_images(full=True))
    return (best >= 0.9 and px[0] >= 800, count, px)


def _median_gap(lines: Sequence[Line]) -> float:
    gaps = [b.y0 - a.y1 for a, b in zip(lines, lines[1:]) if b.y0 > a.y1]
    gaps = [g for g in gaps if g >= -1.0]
    if not gaps:
        return 6.0
    med = statistics.median(gaps)
    return max(med, 1.0)


_INDENT_MIN = 8.0
_SENTENCE_END = (".", "!", "?", ":", "”", '"')


def _build_paragraphs(lines: Sequence[Line], left_margin: float) -> list[Paragraph]:
    """Group lines into paragraphs using geometry (gaps, indent, endings)."""
    if not lines:
        return []
    med_gap = _median_gap(lines)
    groups: list[list[Line]] = []
    cur: list[Line] = []

    def flush() -> None:
        nonlocal cur
        if cur:
            groups.append(cur)
            cur = []

    for ln in lines:
        if not cur:
            cur.append(ln)
            continue
        prev = cur[-1]
        gap = ln.y0 - prev.y1
        indented = ln.x0 - prev.x0 > _INDENT_MIN and ln.x0 - left_margin > _INDENT_MIN
        new_block = ln.block != prev.block
        prev_heading = looks_like_heading(prev.text)
        cur_heading = looks_like_heading(ln.text)

        brk = False
        if prev_heading or cur_heading:
            brk = True
        elif gap > max(med_gap * 1.45, 3.0):
            brk = True
        elif indented and prev.text.endswith(_SENTENCE_END):
            brk = True
        elif new_block and gap > med_gap * 0.75:
            brk = True

        if brk:
            flush()
        cur.append(ln)
    flush()

    paras: list[Paragraph] = []
    for grp in groups:
        x0 = min(l.bbox[0] for l in grp)
        y0 = min(l.bbox[1] for l in grp)
        x1 = max(l.bbox[2] for l in grp)
        y1 = max(l.bbox[3] for l in grp)
        paras.append(
            Paragraph(
                text=join_lines(l.text for l in grp),
                lines=list(grp),
                page_no=grp[0].page_no,
                bbox=(x0, y0, x1, y1),
            )
        )
    return paras


def _guess_margins(lines: Sequence[Line]) -> tuple[float, float]:
    if not lines:
        return 0.0, 0.0
    xs = sorted(round(l.x0, 1) for l in lines)
    # 10th percentile is a decent left text margin for book pages
    left = xs[max(0, int(len(xs) * 0.10))]
    sizes = [l.size for l in lines if l.size > 0]
    return left, (statistics.median(sizes) if sizes else 0.0)


# --------------------------------------------------------------------------
# running headers / folios
# --------------------------------------------------------------------------

_FOLIO = re.compile(r"^(?:page\s+)?[ivxlcdm]*\d{1,4}$", re.I)


_CHAPTER_HEAD = re.compile(r"^CHAPTER\b", re.I)


def furniture_key(text: str) -> str:
    """Normalise a running head so OCR jitter still matches.

    The same head comes back as ``1.2. COUNTING LISTS OF NUMBERS`` on one page
    and ``1.2 COUNTING LISTS OF NUMBERS`` on the next, so punctuation is
    dropped — but case and digits are kept, because *case* is what separates a
    running head from the real heading in the body text below it.
    """
    lowered = text.strip().lower()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", lowered).split())


def is_caps_line(text: str) -> bool:
    """True for ALL-CAPS lines — the typographic mark of a running head."""
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(1 for c in letters if c.isupper()) / len(letters) > 0.8


def _detect_furniture(pages: Sequence[Page]) -> None:
    """Strip repeated top lines (running heads) and bottom folio numbers."""
    if not pages:
        return
    n = len(pages)
    top_seen: dict[str, int] = {}
    for p in pages:
        if p.lines:
            for ln in p.lines[:2]:
                if ln.y0 < p.height * 0.12 and (
                        is_caps_line(ln.text) or _CHAPTER_HEAD.match(ln.text)):
                    key = furniture_key(ln.text)
                    if key:
                        top_seen[key] = top_seen.get(key, 0) + 1
    repeated = {k for k, c in top_seen.items() if c >= max(3, n * 0.12)}

    for p in pages:
        drop: set[int] = set()  # ids of lines to remove (identity, not index)
        for ln in p.lines[:2]:
            if ln.y0 < p.height * 0.12 and (
                furniture_key(ln.text) in repeated
                or (_CHAPTER_HEAD.match(ln.text) and is_caps_line(ln.text))
            ):
                p.header = ln.text
                drop.add(id(ln))
        for ln in reversed(p.lines[-2:]):
            if ln.y1 > p.height * 0.92 and _FOLIO.match(ln.text):
                p.folio = ln.text
                drop.add(id(ln))
            elif ln.y1 > p.height * 0.94 and furniture_key(ln.text) in repeated:
                p.header = ln.text
                drop.add(id(ln))
        if not drop:
            continue
        p.lines = [l for l in p.lines if id(l) not in drop]
        for i, ln in enumerate(p.lines):
            ln.index = i
        kept_blocks: list[Block] = []
        for b in p.blocks:
            kept = [l for l in b.lines if id(l) not in drop]
            if kept:
                kept_blocks.append(Block(b.index, kept, b.bbox))
        p.blocks = kept_blocks


# --------------------------------------------------------------------------
# book
# --------------------------------------------------------------------------


class Book:
    """A loaded PDF plus the reconstructed, per-page text model."""

    def __init__(self, path: str | Path, password: Optional[str] = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.doc = pymupdf.open(self.path)
        if self.doc.needs_pass and password:
            if not self.doc.authenticate(password):
                raise ValueError("wrong password")
        self._pages: Optional[list[Page]] = None

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        try:
            self.doc.close()
        except Exception:
            pass

    def __enter__(self) -> "Book":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __len__(self) -> int:
        return self.doc.page_count

    def __iter__(self) -> Iterator[Page]:
        return iter(self.pages)

    # -- model -------------------------------------------------------------
    @property
    def pages(self) -> list[Page]:
        if self._pages is None:
            self._pages = self._build_pages()
        return self._pages

    def _build_pages(self) -> list[Page]:
        out: list[Page] = []
        for i in range(self.doc.page_count):
            raw = self.doc[i]
            lines, blocks = _extract_lines(raw, i + 1)
            raster, nimg, px = _is_raster_page(raw)
            left, med_size = _guess_margins(lines)
            for ln in lines:
                ln.indent = round(ln.x0 - left, 2)
            out.append(
                Page(
                    number=i + 1,
                    width=round(raw.rect.width, 2),
                    height=round(raw.rect.height, 2),
                    lines=lines,
                    blocks=blocks,
                    paragraphs=_build_paragraphs(lines, left),
                    image_count=nimg,
                    raster=raster,
                    raster_px=px,
                    median_size=med_size,
                    left_margin=left,
                )
            )
        # publish the model *before* the structure pass: annotate() reads
        # book.pages and would otherwise recurse into _build_pages forever.
        self._pages = out
        _detect_furniture(out)
        # paragraphs must be rebuilt once furniture is gone
        for p in out:
            p.paragraphs = _build_paragraphs(p.lines, p.left_margin)
        from . import structure  # local import avoids a cycle at import time

        structure.annotate(self)
        return out

    # -- convenience -------------------------------------------------------
    def page(self, n: int) -> Page:
        if not 1 <= n <= len(self):
            raise IndexError(f"page {n} out of range 1..{len(self)}")
        return self.pages[n - 1]

    @property
    def median_chars(self) -> float:
        vals = [p.chars for p in self.pages if p.lines]
        return statistics.median(vals) if vals else 0.0

    def info(self) -> dict[str, Any]:
        meta = {k: v for k, v in (self.doc.metadata or {}).items() if v}
        return {
            "path": str(self.path),
            "pages": len(self),
            "title": meta.get("title") or "",
            "author": meta.get("author") or "",
            "producer": meta.get("producer") or "",
            "encrypted": bool(self.doc.needs_pass),
            "metadata": meta,
            "median_chars_per_page": round(self.median_chars, 1),
            "scanned_pages": sum(1 for p in self.pages if p.raster),
            "total_chars": sum(p.chars for p in self.pages),
            "raster_px": next((p.raster_px for p in self.pages if p.raster_px[0]), (0, 0)),
        }

    # -- pixels ------------------------------------------------------------
    def render(self, n: int, dpi: int = 150, clip: Optional[BBox] = None) -> bytes:
        """PNG bytes for a whole page or a region of it."""
        page = self.doc[n - 1]
        rect = pymupdf.Rect(*clip) if clip else None
        pix = page.get_pixmap(dpi=dpi, clip=rect, alpha=False)
        return pix.tobytes("png")

    def crop(self, n: int, bbox: BBox, dpi: int = 300, pad: float = 6.0) -> bytes:
        """High-DPI PNG of ``bbox`` grown by ``pad`` points (clamped to page)."""
        page = self.doc[n - 1]
        clip = pymupdf.Rect(
            max(0.0, bbox[0] - pad),
            max(0.0, bbox[1] - pad),
            min(page.rect.width, bbox[2] + pad),
            min(page.rect.height, bbox[3] + pad),
        )
        return page.get_pixmap(dpi=dpi, clip=clip, alpha=False).tobytes("png")

    def line_crop(self, n: int, line_index: int, dpi: int = 300, pad: float = 8.0) -> bytes:
        line = self.page(n).lines[line_index]
        return self.crop(n, line.bbox, dpi=dpi, pad=pad)

    def find(self, n: int, needle: str, limit: int = 50) -> list[BBox]:
        """Locate ``needle`` on page ``n`` and return its rectangles."""
        try:
            rects = self.doc[n - 1].search_for(needle, quads=False)
        except Exception:
            return []
        return [tuple(float(v) for v in r) for r in rects[:limit]]  # type: ignore[return-value]

    def find_bbox(self, needle: str, pages: Optional[Sequence[int]] = None,
                  pad_lines: int = 1) -> list[dict[str, Any]]:
        """Find every occurrence of ``needle`` and return page + tight bbox."""
        hits: list[dict[str, Any]] = []
        for p in self.pages:
            if pages and p.number not in pages:
                continue
            for ln in p.lines:
                if needle in ln.text:
                    idx = ln.index
                    lo = max(0, idx - pad_lines)
                    hi = min(len(p.lines) - 1, idx)
                    y0 = min(p.lines[i].y0 for i in range(lo, hi + 1))
                    y1 = max(p.lines[i].y1 for i in range(lo, hi + 1))
                    hits.append({
                        "page": p.number,
                        "line": idx,
                        "text": ln.text,
                        "bbox": [p.left_margin - 2, y0, max(l.bbox[2] for l in p.lines[lo:hi + 1]), y1],
                    })
        return hits

    def crop_text(self, needle: str, dpi: int = 300, pad_lines: int = 1
                  ) -> list[tuple[dict[str, Any], bytes]]:
        """Crops of every place ``needle`` appears — the self-healing loop."""
        out: list[tuple[dict[str, Any], bytes]] = []
        for hit in self.find_bbox(needle, pad_lines=pad_lines):
            out.append((hit, self.crop(hit["page"], tuple(hit["bbox"]), dpi=dpi, pad=4.0)))
        return out

    # -- reports -----------------------------------------------------------
    def stats_rows(self) -> list[dict[str, Any]]:
        rows = []
        for p in self.pages:
            rows.append({
                "page": p.number,
                "chars": p.chars,
                "lines": len(p.lines),
                "paragraphs": len(p.paragraphs),
                "images": p.image_count,
                "raster": p.raster,
                "median_size": round(p.median_size, 2),
                "header": p.header,
                "folio": p.folio,
                "junk": len(p.junk),
            })
        return rows


def load(path: str | Path, password: Optional[str] = None) -> Book:
    """Open a PDF and build the layout-aware model (lazy: pages on access)."""
    return Book(path, password=password)
