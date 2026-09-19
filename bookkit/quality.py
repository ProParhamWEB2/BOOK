"""bookkit.quality — find the places where OCR lies.

A scanned book's text layer is *optimistic*: it always produces something,
even where it read ``3 2/3`` as ``3층`` or merged a fraction into ``21 -2``.
This module scans the reconstructed text, rates every suspicious line, and
returns a work queue of regions that only a *look at the pixels* can settle.

Detection strategies
--------------------
* ``script_mix``   — CJK/Hangul/Cyrillic/Arabic glyphs inside an otherwise
  Latin book: the classic signature of a mis-read maths region.
* ``junk_glyph``   — control chars, replacement chars, private-use area.
*``span_outlier`` — one glyph ~2x the page's reading size: OCR glue.
* ``glued_words``  — ``wordword`` runs with a capital in the middle.
* ``dense_symbols``— lines that are mostly punctuation/symbols.
* ``isolated_marks`` / ``repeat_char`` — stray ``~``, ``|``, ``\\\\``, ``???``.
* ``empty_region`` — a large blank band with no text (a figure or a table).
* ``low_text_page``— page text far below the book median (layout not read).
* ``unreliable_metrics`` — the page's glyph sizes jitter wildly, so *no*
  size-based inference (headings vs body) can be trusted there.

Every flag carries a bbox and a ``crop`` route, so the agent can render the
original ink and repair the text itself — the loop this toolkit exists for.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

from . import ink
from .render import is_mathish

if TYPE_CHECKING:  # pragma: no cover
    from .core import Book, Line, Page, Paragraph

# --------------------------------------------------------------------------
# character classes
# --------------------------------------------------------------------------

SCRIPT_RANGES = {
    "cjk": [(0x3000, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    "hangul": [(0x1100, 0x11FF), (0xAC00, 0xD7AF)],
    "cyrillic": [(0x0400, 0x04FF)],
    "arabic": [(0x0600, 0x06FF), (0xFB50, 0xFDFF)],
    "hebrew": [(0x0590, 0x05FF)],
    "greek": [(0x0370, 0x03FF)],
    "thai": [(0x0E00, 0x0E7F)],
    "devanagari": [(0x0900, 0x097F)],
}

#: scripts that are plausible inside a maths text (variables, π, θ …)
MATH_FRIENDLY = {"greek"}

BAD_CHARS = re.compile(
    "[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\ufffd\ue000-\uf8ff\ufff0-\uffff]"
)
GLUED_WORD = re.compile(r"\b[a-z]{3,}[A-Z][a-z]{2,}\b")
#: a line that is nothing but symbol noise — but real typography such as
#: "…", "……" or a long dash rule must not be swept up in it
DENSE_SYMBOLS = re.compile(r"^(?=[^\w\s]{3,}$)(?![\.…]+$)(?![-–—―]+$)[^\w\s]+$")
REPEAT_CHAR = re.compile(r"([^\w\s])\1{2,}")
STRAY_MARKS = re.compile(r"(?<!\d)[~|\\^`](?!\d)")
ODD_PUNCT = re.compile(r"[˙¸˝˛ˇ˘°±×÷]{1,}")

ALNUM = re.compile(r"[A-Za-z0-9]")
LATIN = re.compile(r"[A-Za-z]")

FLAG_WEIGHTS = {
    "script_mix": 6.0,
    "junk_glyph": 6.0,
    "span_outlier": 3.0,
    "glued_words": 1.5,
    "dense_symbols": 2.0,
    "isolated_marks": 1.0,
    "repeat_char": 1.0,
    "odd_punct": 1.5,
    "empty_region": 1.5,
    "math_line": 0.8,
    "low_text_page": 4.0,
    "unreliable_metrics": 0.5,
    "missing_text_layer": 12.0,
    "text_without_raster": 2.0,
}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


@dataclass
class Flag:
    """One suspicious place in the book."""

    code: str
    page: int
    severity: float
    text: str = ""
    line: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    detail: str = ""
    fix_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "code": self.code,
            "page": self.page,
            "severity": round(self.severity, 2),
            "text": self.text,
        }
        if self.line is not None:
            d["line"] = self.line
        if self.bbox is not None:
            d["bbox"] = [round(v, 1) for v in self.bbox]
        if self.detail:
            d["detail"] = self.detail
        if self.fix_hint:
            d["fix_hint"] = self.fix_hint
        return d


@dataclass
class PageReport:
    number: int
    flags: list[Flag] = field(default_factory=list)
    chars: int = 0
    lines: int = 0
    ink_ratio: float = 0.0

    @property
    def score(self) -> float:
        return round(sum(f.severity for f in self.flags), 2)

    @property
    def verdict(self) -> str:
        s = self.score
        if s == 0:
            return "clean"
        if s < 3:
            return "review"
        if s < 10:
            return "suspect"
        return "broken"

    def to_dict(self, with_flags: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "page": self.number,
            "verdict": self.verdict,
            "score": self.score,
            "chars": self.chars,
            "lines": self.lines,
            "flags": len(self.flags),
        }
        if with_flags:
            d["details"] = [f.to_dict() for f in self.flags]
        return d


@dataclass
class Report:
    """The whole-book verdict plus the per-page detail."""

    path: str
    pages: list[PageReport] = field(default_factory=list)
    language: str = "unknown"
    reading_size: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def flags(self) -> list[Flag]:
        return [f for p in self.pages for f in p.flags]

    @property
    def score(self) -> float:
        return round(sum(p.score for p in self.pages), 2)

    @property
    def worst_pages(self) -> list[PageReport]:
        return sorted(self.pages, key=lambda p: -p.score)

    def summary(self) -> dict[str, Any]:
        codes = Counter(f.code for f in self.flags)
        verdicts = Counter(p.verdict for p in self.pages)
        return {
            "path": self.path,
            "pages": len(self.pages),
            "language": self.language,
            "reading_size": self.reading_size,
            "score": self.score,
            "verdicts": dict(verdicts),
            "flag_counts": dict(codes.most_common()),
            "clean_pages": sum(1 for p in self.pages if not p.flags),
        }

    def queue(self, limit: int = 40, min_severity: float = 1.0,
              codes: Optional[Sequence[str]] = None) -> list[Flag]:
        """The review work queue: worst-first, croppable regions."""
        keep = [f for f in self.flags if f.severity >= min_severity]
        if codes:
            keep = [f for f in keep if f.code in set(codes)]
        keep.sort(key=lambda f: (-f.severity, f.page, f.line or 0))
        return keep[:limit]

    def to_dict(self, with_flags: bool = True) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "notes": self.notes,
            "pages": [p.to_dict(with_flags=with_flags) for p in self.pages],
        }

    # -- rendering --------------------------------------------------------
    def text(self, top: int = 25) -> str:
        s = self.summary()
        lines = [
            f"OCR report — {s['path']}",
            f"pages={s['pages']}  language={s['language']}  "
            f"reading_size={s['reading_size']}pt  score={s['score']}",
            f"verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(s['verdicts'].items())),
            f"flags: " + ", ".join(f"{k}={v}" for k, v in s['flag_counts'].items()),
        ]
        for note in self.notes:
            lines.append(f"note: {note}")
        worst = [p for p in self.worst_pages if p.score > 0][:top]
        if worst:
            lines.append("")
            lines.append(f"worst pages (top {len(worst)}):")
            for p in worst:
                codes = Counter(f.code for f in p.flags)
                detail = ", ".join(f"{c}×{n}" for c, n in codes.most_common(4))
                lines.append(f"  p{p.number:>4}  {p.verdict:<8} score={p.score:<6}| {detail}")
        reparable = self.queue(limit=top)
        if reparable:
            lines.append("")
            lines.append("review queue (repair by cropping + vision):")
            for f in reparable:
                snippet = (f.text[:58] + "…") if len(f.text) > 58 else f.text
                lines.append(f"  p{f.page}:{f.line if f.line is not None else '-'} "
                             f"[{f.code}] {snippet}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------


def _script_of(ch: str) -> Optional[str]:
    cp = ord(ch)
    for name, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    return None


def _line_issues(line: "Line", reading_size: float) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    text = line.text
    scripts = Counter()
    junk: list[str] = []
    for ch in text:
        if BAD_CHARS.match(ch):
            junk.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L") and not LATIN.match(ch):
            sc = _script_of(ch)
            if sc and sc not in MATH_FRIENDLY:
                scripts[sc] += 1
    if junk:
        out.append(("junk_glyph", "".join(sorted(set(junk))),
                    f"{len(junk)} control/replacement glyph(s)"))
    if scripts:
        scripts.pop("greek", None)
        if scripts:
            name, n = scripts.most_common(1)[0]
            out.append(("script_mix", name, f"{n} {name} glyph(s) in a Latin line"))

    if reading_size and line.size_max >= reading_size * 2.2 and line.span_count <= 3:
        out.append(("span_outlier", f"{line.size_max:.1f}pt",
                    f"glyph ~{line.size_max / reading_size:.1f}x the reading size"))

    if GLUED_WORD.search(text):
        out.append(("glued_words", GLUED_WORD.search(text).group(0),
                    "possible missing space / merged word"))

    stripped = text.strip()
    if DENSE_SYMBOLS.match(stripped):
        out.append(("dense_symbols", stripped, "line is punctuation only"))
    if STRAY_MARKS.search(text):
        out.append(("isolated_marks", STRAY_MARKS.search(text).group(0),
                    "stray mark from OCR noise"))
    if REPEAT_CHAR.search(text):
        out.append(("repeat_char", REPEAT_CHAR.search(text).group(0),
                    "repeated punctuation run"))
    if ODD_PUNCT.search(text):
        out.append(("odd_punct", ODD_PUNCT.search(text).group(0),
                    "diacritic-like noise from a mis-read glyph"))
    return out


def _empty_regions(book: "Book", page: "Page") -> list[tuple[float, float]]:
    """Vertical bands inside the text area with no reconstructed line."""
    if not page.lines:
        return []
    top = page.lines[0].y0
    bottom = page.lines[-1].y1
    if bottom - top < 60:
        return []
    spans = sorted((l.y0, l.y1) for l in page.lines)
    merged: list[list[float]] = []
    for y0, y1 in spans:
        if merged and y0 <= merged[-1][1] + 2:
            merged[-1][1] = max(merged[-1][1], y1)
        else:
            merged.append([y0, y1])
    gaps: list[tuple[float, float]] = []
    for a, b in zip(merged, merged[1:]):
        gap = b[0] - a[1]
        if gap > 34:  # a figure/table/blank band, not just a paragraph break
            gaps.append((a[1], b[0]))
    return gaps


def scan(book: "Book", min_flag_severity: float = 0.0,
         empty_region_min: float = 40.0) -> Report:
    """Run every detector over the loaded book."""
    rep = Report(path=str(book.path))
    sizes = [l.size for p in book.pages for l in p.lines if l.size > 0]
    reading_size = statistics.median(sizes) if sizes else 0.0
    rep.reading_size = round(reading_size, 2)

    latin = sum(len(LATIN.findall(p.text)) for p in book.pages)
    total_alpha = sum(sum(1 for ch in p.text if ch.isalpha()) for p in book.pages)
    non_latin = total_alpha - latin
    if total_alpha and non_latin / max(total_alpha, 1) > 0.35:
        rep.language = "non-latin"
        rep.notes.append("most text is not Latin script — check language separately")
    else:
        rep.language = "latin"

    metrics_cv = _metrics_jitter(book)
    if metrics_cv > 0.35:
        rep.notes.append(
            f"glyph sizes jitter by {metrics_cv:.0%} (scanner metrics unreliable) — "
            "headings must be detected by pattern/layout, not font size"
        )

    median_chars = book.median_chars
    for page in book.pages:
        pr = PageReport(number=page.number, chars=page.chars, lines=len(page.lines))
        try:
            pr.ink_ratio = ink.page_profile(book, page.number)["ink_ratio"]
        except Exception:
            pass

        if metrics_cv > 0.35:
            pr.flags.append(Flag(
                code="unreliable_metrics", page=page.number,
                severity=FLAG_WEIGHTS["unreliable_metrics"],
                detail=f"span size CV {metrics_cv:.2f}",
            ))

        if median_chars and page.chars < median_chars * 0.25:
            pr.flags.append(Flag(
                code="low_text_page", page=page.number,
                severity=FLAG_WEIGHTS["low_text_page"],
                detail=f"{page.chars} chars vs {median_chars:.0f} median",
                fix_hint="render the page and read it with vision",
            ))
        if page.raster and not page.lines:
            pr.flags.append(Flag(
                code="missing_text_layer", page=page.number,
                severity=FLAG_WEIGHTS["missing_text_layer"],
                detail="scanned page with no text layer at all",
                fix_hint="OCR or read with vision",
            ))
        if page.lines and not page.raster and page.image_count == 0:
            pr.flags.append(Flag(
                code="text_without_raster", page=page.number,
                severity=FLAG_WEIGHTS["text_without_raster"],
                detail="no page image: cropped verification impossible",
            ))

        for line in page.lines:
            issues = _line_issues(line, reading_size)
            if is_mathish(line.text):
                # flattened fractions and mangled expressions are invisible to
                # pattern checks — the pixels are the only source of truth
                issues.append(("math_line", line.text,
                               "maths region: verify against the crop"))
            for code, snippet, detail in issues:
                sev = FLAG_WEIGHTS.get(code, 1.0)
                if code == "script_mix":
                    sev = FLAG_WEIGHTS["script_mix"]
                if code == "junk_glyph":
                    sev = FLAG_WEIGHTS["junk_glyph"]
                pr.flags.append(Flag(
                    code=code, page=page.number, severity=sev,
                    text=line.text, line=line.index, bbox=line.bbox,
                    detail=detail, fix_hint=_fix_hint(code),
                ))
            line.junk = [c for c, _, _ in issues]

        for y0, y1 in _empty_regions(book, page):
            if y1 - y0 < empty_region_min:
                continue
            pr.flags.append(Flag(
                code="empty_region", page=page.number,
                severity=FLAG_WEIGHTS["empty_region"],
                bbox=(page.left_margin, y0, page.width - page.left_margin, y1),
                detail=f"{y1 - y0:.0f}pt band with no text (figure or table?)",
                fix_hint="crop and look at it: figures/tables are invisible in text",
            ))

        pr.flags = [f for f in pr.flags if f.severity >= min_flag_severity]
        rep.pages.append(pr)
    return rep


def _metrics_jitter(book: "Book") -> float:
    """How much glyph sizes vary *inside* a line — the scanner artefact.

    A PDF typeset normally reports one size per line (spread ~0).  This book's
    OCR layer gives every glyph its own size, so a line that only contains one
    size is the exception, not the rule.  The median relative spread is
    therefore a direct measure of "can font size be trusted for structure?".
    """
    spreads: list[float] = []
    for page in book.pages:
        for line in page.lines:
            if line.size > 0 and line.span_count >= 2:
                spreads.append((line.size_max - line.size_min) / line.size)
    if len(spreads) < 20:
        return 0.0
    return statistics.median(spreads)


def _fix_hint(code: str) -> str:
    return {
        "script_mix": "render this line (crop dpi=400) and read the ink with vision",
        "junk_glyph": "render this line and re-read it",
        "span_outlier": "a glyph glued on: verify the numbers from the crop",
        "glued_words": "check for a missing space in the original",
        "dense_symbols": "likely a mis-read maths expression — read the crop",
        "isolated_marks": "OCR speck: verify against the crop before deleting",
        "repeat_char": "verify against the crop",
        "odd_punct": "verify against the crop",
        "low_text_page": "render the whole page and read it with vision",
        "missing_text_layer": "render the whole page and read it with vision",
        "empty_region": "inspect as a figure/table region",
        "math_line": "crop this region and re-read the maths from the ink",
    }.get(code, "verify against the crop")


# --------------------------------------------------------------------------
# repair loop helpers
# --------------------------------------------------------------------------


@dataclass
class Repair:
    """A proposed replacement of a damaged span (agent or human authored)."""

    page: int
    line: Optional[int]
    before: str
    after: str
    reason: str = ""
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page, "line": self.line, "before": self.before,
            "after": self.after, "reason": self.reason, "verified": self.verified,
        }


def apply_repairs(book: "Book", repairs: Iterable[Repair]) -> int:
    """Apply verifications to the in-memory model; returns the number applied."""
    n = 0
    by_page: dict[int, list[Repair]] = {}
    for r in repairs:
        by_page.setdefault(r.page, []).append(r)
    for page_no, items_ in by_page.items():
        page = book.page(page_no)
        for r in items_:
            targets = [l for l in page.lines if (r.line is not None and l.index == r.line)] \
                if r.line is not None else page.lines
            for line in targets:
                if r.before and r.before not in line.text:
                    continue
                # an empty "before" means: replace the whole line
                line.text = line.text.replace(r.before, r.after) if r.before else r.after
                line.junk = []
                n += 1
    if n:
        for page in book.pages:
            from .core import _build_paragraphs  # local: avoid a public cycle

            page.paragraphs = _build_paragraphs(page.lines, page.left_margin)
        from . import structure

        structure.annotate(book)
    return n


def load_repairs(path: str) -> list[Repair]:
    """Read a JSON/JSONL repair file (what the web UI writes)."""
    import json
    from pathlib import Path

    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    data: list[dict[str, Any]] = []
    if p.suffix in {".jsonl", ".ndjson"}:
        data = [json.loads(l) for l in raw.splitlines() if l.strip()]
    else:
        parsed = json.loads(raw)
        data = parsed if isinstance(parsed, list) else parsed.get("repairs", [])
    return [
        Repair(
            page=int(d.get("page", 0)),
            line=d.get("line"),
            before=d.get("before", ""),
            after=d.get("after", ""),
            reason=d.get("reason", ""),
            verified=bool(d.get("verified", True)),
        )
        for d in data
    ]


def save_repairs(repairs: Iterable[Repair], path: str) -> None:
    import json
    from pathlib import Path

    payload = [r.to_dict() for r in repairs]
    p = Path(path)
    if p.suffix in {".jsonl", ".ndjson"}:
        p.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in payload) + "\n",
                     encoding="utf-8")
    else:
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
