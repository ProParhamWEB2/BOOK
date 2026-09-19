"""bookkit.ink — geometric structure recovered from pixels.

Scanned books draw meaning that the OCR text layer throws away: the boxes
around *Problem* statements, the rules separating sections, the black bar
announcing *Problems*, the dashed boundary of an *Important* note.
We recover those from the raster with a cheap grayscale pass at 72 dpi.

How rules and boxes are told apart from text
--------------------------------------------
1. **Horizontal candidates** — rows with a lot of ink spanning a wide part
   of the page.  A printed rule is then confirmed by *thinness*: sampling
   along the line, the ink is at most ~4 px tall, while a text row is 5-9 px
   (the x-height of 10 pt type at 72 dpi).  Both neighbours 2 px above/below
   must also be quiet, so letters that merely look heavy are rejected.
2. **Boxes** — a pair of horizontal rules with matching x-extents *and* a
   verified side edge: the pixels just inside each end must run from the top
   rule to the bottom rule (duty >= 0.4).  Text columns never line up with
   both rules, so this needs no per-book tuning.
3. **Vertical rules** — reported for completeness, with the same duty/max-gap
   reasoning (a border is solid or finely dashed; text gaps are line
   spacings).
4. **Filled bars** — thick bands of near-solid ink (the *Problems* header).

Everything is cached per page under ``book._ink_cache``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

try:
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from .core import Book

#: gray -> ink map.  Book scans separate cleanly: paper sits at ~240-255,
#: text at 0-100, but *thin* borders antialias to ~150-200, so the threshold
#: has to sit high or box sides disappear.
_BIN = bytes(1 if i < 200 else 0 for i in range(256))
INK_THRESHOLD = 200

MIN_DARK_RATIO = 0.40      # row/column duty cycle to be a candidate
MIN_SPAN_RATIO = 0.22      # horizontal candidate must be this wide
MAX_THICKNESS_PX = 4       # a rule is this thin at 72 dpi
NEIGHBOUR_GAP_PX = 2       # neighbours this far away must be quiet
NEIGHBOUR_MAX_RATIO = 0.15
MIN_VERT_DUTY = 0.50       # vertical rule duty cycle
MIN_VERT_SPAN_PX = 25      # ... and minimum height
MAX_VERT_GAP_PX = 4        # gaps inside a vertical rule (dashed borders)
DASH_MAX_THICKNESS_PX = 7  # a dashed horizontal border scans thicker than a rule
EDGE_DUTY = 0.40           # a box side must be this solid
EDGE_TOL_PX = 5            # side edge may sit this far inside the rule ends
BAR_MIN_FILL = 0.85        # inside a filled bar the ink is this dense
BAR_MIN_WIDTH = 0.15       # ... over at least this much of the page width
BAR_MAX_THICKNESS = 30
SAMPLE_POINTS = 5          # thinness samples along a horizontal candidate


@dataclass(frozen=True)
class Rule:
    """A long, thin run of ink: a horizontal or vertical line on the page."""

    orient: str          # "h" | "v"
    pos: float           # y for horizontal, x for vertical (points)
    start: float         # x0 (h) or y0 (v)
    end: float           # x1 (h) or y1 (v)
    thickness: float = 1.0
    ink_ratio: float = 0.0
    dashed: bool = False

    @property
    def length(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "orient": self.orient, "pos": round(self.pos, 1),
            "start": round(self.start, 1), "end": round(self.end, 1),
            "thickness": round(self.thickness, 2), "ink_ratio": self.ink_ratio,
            "dashed": self.dashed,
        }


@dataclass(frozen=True)
class Box:
    """A bordered region of the page."""

    x0: float
    y0: float
    x1: float
    y1: float
    dashed: bool = False
    filled: bool = False
    closed: bool = True

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains_rect(self, bbox: Sequence[float], tol: float = 2.0) -> bool:
        return (bbox[0] >= self.x0 - tol and bbox[1] >= self.y0 - tol
                and bbox[2] <= self.x1 + tol and bbox[3] <= self.y1 + tol)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(v, 1) for v in (self.x0, self.y0, self.x1, self.y1)],
            "dashed": self.dashed, "filled": self.filled, "closed": self.closed,
        }


# --------------------------------------------------------------------------
# pixels
# --------------------------------------------------------------------------


class _Page:
    """Binary row/column buffers for one page (1 = ink)."""

    __slots__ = ("w", "h", "rows", "cols", "scale")

    def __init__(self, book: "Book", page_no: int, dpi: int):
        pix = book.doc[page_no - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
        self.w, self.h = pix.width, pix.height
        self.scale = dpi / 72.0
        self.rows = [pix.samples[y * self.w:(y + 1) * self.w].translate(_BIN)
                     for y in range(self.h)]
        self.cols = [bytes(c) for c in zip(*self.rows)]

    def ink(self, orient: str, i: int, j: int) -> int:
        buf = self.rows[i] if orient == "h" else self.cols[i]
        return buf[j]


def _page(book: "Book", page_no: int, dpi: int) -> _Page:
    cache = getattr(book, "_ink_cache", None)
    if cache is None:
        cache = {}
        book._ink_cache = cache  # type: ignore[attr-defined]
    key = (page_no, dpi, "page")
    if key not in cache:
        cache[key] = _Page(book, page_no, dpi)
    return cache[key]


def _row_stats(buf: bytes) -> tuple[int, int, int, int, int]:
    """(dark, first, last, longest_run, n_runs) for one row/column buffer."""
    dark = buf.count(1)
    if not dark:
        return 0, -1, -1, 0, 0
    first = buf.find(b"\x01")
    last = buf.rfind(b"\x01")
    runs = [len(r) for r in buf.split(b"\x00") if r]
    return dark, first, last, max(runs), len(runs)


def _run_positions(buf: bytes) -> list[tuple[int, int]]:
    """(start, length) of every contiguous dark run in a buffer."""
    out: list[tuple[int, int]] = []
    pos = 0
    for part in buf.split(b"\x00"):
        if part:
            out.append((pos, len(part)))
        pos += len(part) + 1
    return out


def _perpendicular_thickness(page: _Page, y: int, xs: Sequence[int],
                             limit: int = 8) -> float:
    """Median ink height of a horizontal candidate at sampled x positions."""
    heights: list[int] = []
    for x in xs:
        if not page.rows[y][x]:
            continue
        up = 0
        while up < limit and y - up - 1 >= 0 and page.rows[y - up - 1][x]:
            up += 1
        down = 0
        while down < limit and y + down + 1 < page.h and page.rows[y + down + 1][x]:
            down += 1
        heights.append(up + down + 1)
    if not heights:
        return 0.0
    heights.sort()
    return float(heights[len(heights) // 2])


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------


def rules(book: "Book", page_no: int, dpi: int = 72,
          verticals: str = "boxlike") -> list[Rule]:
    """All long horizontal/vertical rules on a page, in points.

    ``verticals="boxlike"`` (default) keeps only vertical rules that start and
    end *on* a horizontal rule — i.e. box sides, table grids and column rules.
    The strokes inside large display type also pass the raw pixel test, and
    filtering them out here is what makes :func:`boxes` trustworthy.
    """
    cache = getattr(book, "_ink_cache", None) or {}
    key = (page_no, dpi, "rules", verticals)
    if key in cache:
        return cache[key]

    pg = _page(book, page_no, dpi)
    w, h, scale = pg.w, pg.h, pg.scale
    row_stats = [_row_stats(r) for r in pg.rows]
    col_stats = [_row_stats(c) for c in pg.cols]

    found: list[Rule] = []

    # ---- horizontal: wide, thin, isolated -------------------------------
    i = 0
    while i < h:
        if row_stats[i][0] < MIN_DARK_RATIO * w:
            i += 1
            continue
        j = i
        good: list[tuple[int, int, int, int, int]] = []
        while j < h and row_stats[j][0] >= MIN_DARK_RATIO * w:
            if row_stats[j][1] >= 0 and (row_stats[j][2] - row_stats[j][1]) >= MIN_SPAN_RATIO * w:
                good.append(row_stats[j])
            j += 1
        if good:
            thickness = j - i
            above = row_stats[i - NEIGHBOUR_GAP_PX][0] if i - NEIGHBOUR_GAP_PX >= 0 else 0
            below = row_stats[j - 1 + NEIGHBOUR_GAP_PX][0] if j - 1 + NEIGHBOUR_GAP_PX < h else 0
            quiet = (above <= NEIGHBOUR_MAX_RATIO * w) and (below <= NEIGHBOUR_MAX_RATIO * w)
            ink_est = sum(g[0] for g in good) / max(
                (max(g[2] for g in good) - min(g[1] for g in good) + 1) * thickness, 1)
            max_thick = MAX_THICKNESS_PX if ink_est >= 0.8 else DASH_MAX_THICKNESS_PX
            if thickness <= max_thick and quiet:
                first = min(g[1] for g in good)
                last = max(g[2] for g in good)
                span = max(last - first + 1, 1)
                x_samples = [first + int(span * t) for t in (0.15, 0.3, 0.5, 0.7, 0.85)]
                thin = _perpendicular_thickness(pg, (i + j - 1) // 2, x_samples)
                ink = sum(g[0] for g in good) / (span * thickness)
                if thin <= MAX_THICKNESS_PX + 1 and ink >= 0.30:
                    found.append(Rule(
                        orient="h",
                        pos=((i + j - 1) / 2) / scale,
                        start=first / scale,
                        end=last / scale,
                        thickness=thickness / scale,
                        ink_ratio=round(ink, 3),
                        dashed=ink < 0.8,
                    ))
        i = j + 1

    # ---- vertical: a duty-cycle window with small gaps -------------------
    for x, buf in enumerate(pg.cols):
        span = _best_vertical_span(buf)
        if span is None:
            continue
        y0, y1, duty = span
        # the columns on both sides must not carry the same border
        left = _best_vertical_span(pg.cols[x - 1]) if x > 0 else None
        right = _best_vertical_span(pg.cols[x + 1]) if x + 1 < w else None
        neighbours_similar = any(
            n is not None and abs(n[0] - y0) <= 3 and abs(n[1] - y1) <= 3
            for n in (left, right)
        )
        if not neighbours_similar and (col_stats[x][3] < MIN_VERT_SPAN_PX):
            # a lone 1 px column is only trusted when it is really long
            continue
        found.append(Rule(
            orient="v", pos=x / scale, start=y0 / scale, end=y1 / scale,
            thickness=1 / scale, ink_ratio=round(duty, 3), dashed=duty < 0.8,
        ))

    found = _merge_touching(found)
    if verticals == "boxlike":
        found = [r for r in found if r.orient == "h"] + [
            v for v in found if v.orient == "v" and _anchored(v, found)
        ]
    found.sort(key=lambda r: (r.orient, r.pos, r.start))
    cache[key] = found
    book._ink_cache = cache  # type: ignore[attr-defined]
    return found


def _anchored(vertical: Rule, all_rules: Sequence[Rule], tol: float = 6.0) -> bool:
    """True when a vertical rule begins and ends on a horizontal rule."""
    tops = [h for h in all_rules if h.orient == "h"
            and abs(h.pos - vertical.start) <= tol
            and h.start - tol <= vertical.pos <= h.end + tol]
    bottoms = [h for h in all_rules if h.orient == "h"
               and abs(h.pos - vertical.end) <= tol
               and h.start - tol <= vertical.pos <= h.end + tol]
    return bool(tops and bottoms)


def _best_vertical_span(buf: bytes) -> Optional[tuple[int, int, float]]:
    """Longest window with rule-like duty and small gaps (border or dash)."""
    runs = _run_positions(buf)
    if not runs:
        return None
    best: Optional[tuple[int, int, float]] = None
    n = len(runs)
    for i in range(n):
        start = runs[i][0]
        end = start + runs[i][1]
        ink = runs[i][1]
        prev_end = end
        # a solid border is a single long run: evaluate the window on its own
        if end - start >= MIN_VERT_SPAN_PX and (
                best is None or (end - start) > (best[1] - best[0])):
            best = (start, end, 1.0)
        for j in range(i + 1, n):
            if runs[j][0] - prev_end > MAX_VERT_GAP_PX:
                break
            end = runs[j][0] + runs[j][1]
            prev_end = end
            ink += runs[j][1]
            span = end - start
            duty = ink / max(span, 1)
            if span >= MIN_VERT_SPAN_PX and duty >= MIN_VERT_DUTY:
                if best is None or span > (best[1] - best[0]):
                    best = (start, end, duty)
    return best


def _merge_touching(rules_: Sequence[Rule]) -> list[Rule]:
    """Fuse neighbouring columns/rows that belong to one thick rule."""
    out: list[Rule] = []
    for r in sorted(rules_, key=lambda r: (r.orient, r.pos)):
        if out:
            prev = out[-1]
            if (prev.orient == r.orient and (r.pos - prev.pos) <= MAX_THICKNESS_PX
                    and abs(prev.start - r.start) <= 4 and abs(prev.end - r.end) <= 4):
                out[-1] = Rule(
                    orient=prev.orient, pos=(prev.pos + r.pos) / 2,
                    start=min(prev.start, r.start), end=max(prev.end, r.end),
                    thickness=prev.thickness + r.thickness,
                    ink_ratio=round(min(prev.ink_ratio, r.ink_ratio), 3),
                    dashed=prev.dashed or r.dashed,
                )
                continue
        out.append(r)
    return out


# --------------------------------------------------------------------------
# boxes & bars
# --------------------------------------------------------------------------


def _edge_solid(pg: _Page, x_px: int, y0_px: int, y1_px: int,
                tol: int = EDGE_TOL_PX) -> bool:
    """Is there a drawn side edge at ``x_px`` spanning exactly this y-range?

    Only the ink *inside* the range counts — the surrounding paragraph text
    must not be able to fake a border.
    """
    if not (0 <= x_px < pg.w) or y1_px <= y0_px:
        return False
    span = y1_px - y0_px
    for cand in range(max(0, x_px - 2), min(pg.w, x_px + 3)):
        ink = 0
        first: Optional[int] = None
        last: Optional[int] = None
        for start, length in _run_positions(pg.cols[cand]):
            end = start + length
            if end <= y0_px or start >= y1_px:
                continue
            lo, hi = max(start, y0_px), min(end, y1_px)
            ink += hi - lo
            first = lo if first is None else min(first, lo)
            last = hi if last is None else max(last, hi)
        if first is None or last is None:
            continue
        if ink / span >= EDGE_DUTY and (first - y0_px) <= tol and (y1_px - last) <= tol:
            return True
    return False


def boxes(book: "Book", page_no: int, dpi: int = 72, min_height: float = 12.0,
          min_width: float = 40.0, tol: float = 6.0,
          require_closed: bool = True) -> list[Box]:
    """Bordered regions, from rule pairs verified by their side edges."""
    cache = getattr(book, "_ink_cache", None) or {}
    key = (page_no, dpi, "boxes", require_closed)
    if key in cache:
        return cache[key]

    pg = _page(book, page_no, dpi)
    scale = pg.scale
    horiz = [r for r in rules(book, page_no, dpi)
             if r.orient == "h" and r.length >= min_width]
    # a heavy/double border is detected as a *filled bar* — it still closes a
    # box, so promote bars to horizontal candidates too
    for bar in bars(book, page_no, dpi):
        if bar.width >= min_width:
            horiz.append(Rule(orient="h", pos=bar.y0, start=bar.x0, end=bar.x1,
                              thickness=bar.y1 - bar.y0, ink_ratio=1.0, dashed=False))
    horiz.sort(key=lambda r: r.pos)
    page_h = book.page(page_no).height

    out: list[Box] = []
    for i, top in enumerate(horiz):
        for bottom in horiz[i + 1:]:
            dy = bottom.pos - top.pos
            if dy < min_height or dy > page_h * 0.6:
                continue
            if abs(top.start - bottom.start) > tol or abs(top.end - bottom.end) > tol:
                continue
            x0 = max(top.start, bottom.start)
            x1 = min(top.end, bottom.end)
            if x1 - x0 < min_width:
                continue
            y0, y1 = top.pos, bottom.pos
            y0_px, y1_px = int(round(y0 * scale)), int(round(y1 * scale))
            left = _edge_solid(pg, int(round(x0 * scale)), y0_px, y1_px)
            right = _edge_solid(pg, int(round(x1 * scale)), y0_px, y1_px)
            closed = left and right
            if require_closed and not closed:
                continue
            out.append(Box(x0, y0, x1, y1,
                           dashed=top.dashed or bottom.dashed, closed=closed))

    out = _drop_nested(out)
    out.sort(key=lambda b: (b.y0, b.x0))
    cache[key] = out
    book._ink_cache = cache  # type: ignore[attr-defined]
    return out


def _drop_nested(boxes_: Sequence[Box], tol: float = 3.0) -> list[Box]:
    """Keep the innermost rectangle per region; drop duplicates and wrappers."""
    kept: list[Box] = []
    for cand in sorted(boxes_, key=lambda b: (b.area, not b.closed)):
        duplicate = any(
            abs(cand.x0 - k.x0) <= tol and abs(cand.y0 - k.y0) <= tol
            and abs(cand.x1 - k.x1) <= tol and abs(cand.y1 - k.y1) <= tol
            for k in kept
        )
        wrapper = any(
            cand.x0 <= k.x0 + tol and cand.y0 <= k.y0 + tol
            and cand.x1 >= k.x1 - tol and cand.y1 >= k.y1 - tol
            for k in kept
        )
        if duplicate or wrapper:
            continue
        kept.append(cand)
    return kept


def has_geometry(book: "Book", page_no: int, dpi: int = 72) -> bool:
    """Cheap probe: does this page carry rules/boxes at all?

    Two column profiles and one row profile cost milliseconds even on a big
    page, so plain-text PDFs (hundreds of pages) never pay for a full ink
    analysis — but a *drawn* page is still found.
    """
    cache = getattr(book, "_ink_cache", None) or {}
    key = (page_no, dpi, "geometry")
    if key in cache:
        return cache[key]
    pg = _page(book, page_no, dpi)
    result = False
    for row in pg.rows:
        dark, first, last, run, nruns = _row_stats(row)
        if first >= 0 and run >= 0.25 * pg.w and dark >= 0.4 * pg.w:
            result = True
            break
    if not result:
        for col in pg.cols:
            if _row_stats(col)[3] >= MIN_VERT_SPAN_PX:
                result = True
                break
    cache[key] = result
    book._ink_cache = cache  # type: ignore[attr-defined]
    return result


def bars(book: "Book", page_no: int, dpi: int = 72) -> list[Box]:
    """Filled rectangles, e.g. the black *Problems* header (white text on ink).

    The test is *fill inside its own extent*: within ``[first, last]`` the row
    must be at least :data:`BAR_MIN_FILL` dark.  Ordinary text rows reach ~0.55
    because letters are thin; a filled bar with knocked-out white text reaches
    ~0.95.
    """
    cache = getattr(book, "_ink_cache", None) or {}
    key = (page_no, dpi, "bars")
    if key in cache:
        return cache[key]
    pg = _page(book, page_no, dpi)
    w, h, scale = pg.w, pg.h, pg.scale
    stats = [_row_stats(r) for r in pg.rows]
    out: list[Box] = []

    def filled(y: int) -> bool:
        dark, first, last, run, nruns = stats[y]
        if first < 0:
            return False
        span = last - first + 1
        return span >= BAR_MIN_WIDTH * w and dark / span >= BAR_MIN_FILL

    i = 0
    while i < h:
        if not filled(i):
            i += 1
            continue
        j = i
        while j < h and filled(j):
            j += 1
        if 3 <= j - i <= BAR_MAX_THICKNESS:
            x0 = min(stats[k][1] for k in range(i, j) if stats[k][1] >= 0)
            x1 = max(stats[k][2] for k in range(i, j) if stats[k][2] >= 0)
            out.append(Box(x0 / scale, i / scale, x1 / scale, (j - 1) / scale, filled=True))
        i = j + 1
    cache[key] = out
    book._ink_cache = cache  # type: ignore[attr-defined]
    return out


def box_for(book: "Book", page_no: int, bbox: Sequence[float]) -> Optional[Box]:
    """The innermost box (bordered or filled) containing ``bbox``."""
    best: Optional[Box] = None
    for bx in list(boxes(book, page_no, require_closed=True)) + list(bars(book, page_no)):
        if not bx.contains_rect(bbox, tol=6.0):
            continue
        if best is None or bx.area < best.area:
            best = bx
    return best


def page_profile(book: "Book", page_no: int) -> dict[str, Any]:
    """Ink statistics used by the quality checker."""
    pg = _page(book, page_no, 72)
    ink = sum(r.count(1) for r in pg.rows)
    return {
        "page": page_no,
        "px": [pg.w, pg.h],
        "ink_ratio": round(ink / max(pg.w * pg.h, 1), 4),
        "rules": len(rules(book, page_no)),
        "boxes": len(boxes(book, page_no)),
        "bars": len(bars(book, page_no)),
    }
