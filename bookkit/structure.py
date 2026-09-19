"""bookkit.structure — roles, outline and item extraction.

The scanner OCR'd the book, so the text layer has *no* usable font metrics
(every glyph got a random size).  Structure therefore comes from three
things, in order of trust:

1. **Patterns** — ``CHAPTER n``, ``1.2.5``, ``Problem 1.7:``, ``Hints:`` …
2. **Geometry** — indent, vertical gaps, and the boxes recovered in
   :mod:`bookkit.ink` (a problem statement lives *inside* a bordered box).
3. **Repetition** — running heads and folio numbers, removed in core.

The result is a per-paragraph ``role`` plus a flat list of
:class:`Item` objects (problems, exercises, solutions, boxes) that carry
their section context, hints and page ranges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

from . import ink

if TYPE_CHECKING:  # pragma: no cover
    from .core import Book, Page, Paragraph

# --------------------------------------------------------------------------
# patterns
# --------------------------------------------------------------------------

RE_CHAPTER = re.compile(r"^CHAPTER\s+([0-9]+|[IVXLC]+)\b[.:]?\s*(.*)$", re.I)
RE_SECTION = re.compile(r"^(\d+\.\d+)\s+(.{2,90})$")
#: running-head style section line: "1.2. COUNTING LISTS OF NUMBERS"
RE_CAPS_SECTION = re.compile(r"^(\d+\.\d+)\.?\s+([A-Z][A-Z0-9 ,'\-&.:]{2,70})$")
RE_SUB = re.compile(r"^(\d+\.\d+\.\d+)\s*(?P<star>\*?)\s+(?P<body>.{2,300})$")
RE_PROBLEM = re.compile(r"^Problem\s+(\d+\.\d+)\s*[::]\s*(?P<body>.*)$", re.S)
RE_SOLUTION = re.compile(r"^Solution\s+(?:for\s+)?(?:Problem\s+)?(\d+\.\d+)\s*[::]\s*(?P<body>.*)$",
                         re.S | re.I)
RE_LABEL = re.compile(
    r"^(?P<label>Problems?|Solutions?|Hints?|Important|Definition|Def\.|Theorem|Lemma|"
    r"Corollary|Example|Examples|Note|Remark|Proof|Proofs|Exercise|Exercises|"
    r"Short\s+Answer|Summary)\b\s*[::]?",
    re.I,
)
RE_HINTS = re.compile(r"^Hints?\s*[::]?\s*(?P<rest>.*)$", re.I)
RE_INLINE_HINTS = re.compile(r"\bHints?\s*[::]\s*(?P<rest>[\d,\s]+)\s*$", re.I)
RE_NUMBERED_TAIL = re.compile(r"^(?P<num>\d+\.\d+(?:\.\d+)?)\s*$")
RE_QUESTIONISH = re.compile(
    r"^(how|what|why|when|where|which|who|is|are|does|do|can|find|show|prove|"
    r"compute|calculate|determine|given|suppose|let)\b",
    re.I,
)
QUESTION_WORDS = RE_QUESTIONISH

ROLE_BODY = "body"
ROLE_HEADING = "heading"
ROLE_LABEL = "label"
ROLE_PROBLEM = "problem"
ROLE_EXERCISE = "exercise"
ROLE_SOLUTION = "solution"
ROLE_EXAMPLE = "example"
ROLE_THEOREM = "theorem"
ROLE_BOX = "box"
ROLE_NOISE = "noise"

ITEM_ROLES = (ROLE_PROBLEM, ROLE_EXERCISE, ROLE_SOLUTION, ROLE_EXAMPLE, ROLE_THEOREM)


# --------------------------------------------------------------------------
# items
# --------------------------------------------------------------------------


@dataclass
class Item:
    """One logical unit of the book: a problem, an exercise, a solution…"""

    kind: str                       # problem|exercise|solution|example|theorem|box
    tag: str                        # "Problem 1.7" / "1.2.5"
    number: Optional[str] = None
    star: bool = False
    page: int = 0
    end_page: int = 0
    section: Optional[str] = None
    section_number: Optional[str] = None
    text: str = ""
    raw_lines: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    in_box: bool = False
    duplicate_of: Optional[int] = None
    paragraphs: list["Paragraph"] = field(default_factory=list)

    @property
    def id(self) -> str:
        num = self.number or "?"
        return f"{self.kind}:{num}"

    def line_numbers(self) -> list[int]:
        return [l.index for p in self.paragraphs for l in p.lines]

    def pages(self) -> list[int]:
        return list(range(self.page, self.end_page + 1))

    def to_dict(self, text: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "tag": self.tag,
            "number": self.number,
            "page": self.page,
            "end_page": self.end_page,
            "section": self.section,
            "section_number": self.section_number,
            "hints": self.hints,
            "bbox": [round(v, 2) for v in self.bbox],
            "lines": self.line_numbers(),
            "in_box": self.in_box,
        }
        if self.star:
            d["star"] = True
        if self.duplicate_of is not None:
            d["duplicate_of"] = self.duplicate_of
        if text:
            d["text"] = self.text
        return d


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


def _title_like(text: str) -> bool:
    """Book-title shape: 2-6 capitalised words *with* lowercase letters.

    This is what tells ``Counting Is Arithmetic`` (the title) from
    ``WWNON`` (OCR junk) and ``In this book, we'll learn…`` (body text).
    """
    words = text.strip().split()
    if not 2 <= len(words) <= 6:
        return False
    if text.strip().endswith((".", ",", ":", ";", "!", "?", "-")):
        return False
    caps = 0
    for w in words:
        core = w.strip("'’")
        if core[:1].isupper() and any(ch.islower() for ch in core):
            caps += 1
    return caps / len(words) >= 0.75


def _section_like(body: str) -> bool:
    """Does this look like a section *title* rather than a numbered exercise?

    Titles are short, unpunctuated and title-cased ("1.2 Counting Lists of
    Numbers"); exercises are sentences or questions ("1.36 Which integers n
    satisfy…?").  This is what keeps review-problem numbering (1.17, 1.28 …)
    out of the outline.
    """
    b = body.strip()
    if not b or len(b) > 70 or b.endswith((".", ":", ";", ",", "?")):
        return False
    if _is_question(b):
        return False
    words = [w for w in b.split() if w[:1].isalpha()]
    if not words:
        return False
    caps = sum(1 for w in words if w[:1].isupper())
    return caps / len(words) >= 0.6


def _is_question(text: str) -> bool:
    t = text.strip()
    if t.endswith("?"):
        return True
    return bool(RE_QUESTIONISH.match(t))


def classify(par: "Paragraph", boxes: Sequence[ink.Box] = (),
             known_sections: Optional[set[str]] = None) -> "Paragraph":
    """Assign ``role``/``level``/``number``/``tag`` to a paragraph, in place.

    ``known_sections`` is the set of real section numbers (``1.2``, ``1.5``).
    A numbered item whose parent is a known section is an *exercise*
    (``1.3.4 There are 20 cars…``); one whose parent is unknown is a
    *sub-heading* (``1.6 Summary``).
    """
    text = par.text.strip()
    par.attrs.setdefault("in_box", False)
    for bx in boxes:
        if bx.contains_rect(par.bbox, tol=6.0):
            par.attrs["in_box"] = True
            par.attrs.setdefault("box", [round(v, 1) for v in (bx.x0, bx.y0, bx.x1, bx.y1)])
            break

    m = RE_CHAPTER.match(text)
    if m:
        par.role = ROLE_HEADING
        par.level = 1
        par.number = m.group(1)
        par.tag = f"CHAPTER {m.group(1)}"
        tail = m.group(2).strip()
        if tail and len(tail) > 2:
            par.attrs["subtitle"] = tail
        return par

    m = RE_PROBLEM.match(text)
    if m:
        par.role = ROLE_PROBLEM
        par.number = m.group(1)
        par.tag = f"Problem {m.group(1)}"
        return par

    m = RE_SOLUTION.match(text)
    if m:
        par.role = ROLE_SOLUTION
        par.number = m.group(1)
        par.tag = f"Solution {m.group(1)}"
        return par

    m = RE_SUB.match(text)
    if m:
        body = m.group("body").strip()
        star = bool(m.group("star")) or body.startswith("*")
        parent = ".".join(m.group(1).split(".")[:2])
        # a numbered item inside a known section is an exercise, not a heading
        if (known_sections and parent in known_sections) or _is_question(body) or len(body) > 120:
            par.role = ROLE_EXERCISE
            par.number = m.group(1)
            par.tag = m.group(1) + ("*" if star else "")
            par.attrs["star"] = star
        else:
            par.role = ROLE_HEADING
            par.level = 3
            par.number = m.group(1)
            par.tag = m.group(1)
        return par

    m = RE_CAPS_SECTION.match(text)
    if m and len(text) < 75:
        par.role = ROLE_HEADING
        par.level = 2
        par.number = m.group(1)
        par.tag = f"{m.group(1)} {m.group(2).strip().title()}"
        return par

    m = RE_SECTION.match(text)
    if m:
        if _section_like(m.group(2)):
            par.role = ROLE_HEADING
            par.level = 2
            par.number = m.group(1)
            par.tag = f"{m.group(1)} {m.group(2).strip()}"
            return par
        if len(text) > 12:
            # "1.17 How many numbers…?" — a numbered exercise, not a section
            par.role = ROLE_EXERCISE
            par.number = m.group(1)
            par.tag = m.group(1)
            return par

    m = RE_LABEL.match(text)
    if m and _looks_like_label(text, m.group("label")):
        par.role = ROLE_LABEL
        label = m.group("label")
        par.tag = label
        lw = label.lower()
        if lw.startswith("example"):
            par.role = ROLE_EXAMPLE
        elif lw.startswith(("theorem", "lemma", "corollary", "definition", "proof")):
            par.role = ROLE_THEOREM
        return par

    if not text:
        par.role = ROLE_NOISE
        return par

    if (par.page_no == 1 and len(text) < 70 and par.y0 < 400
            and _title_like(text)):
        par.role = ROLE_HEADING
        par.level = 0
        par.tag = text
        par.attrs.setdefault("title", True)
        return par

    par.role = ROLE_BODY
    return par


def _looks_like_label(text: str, label: str) -> bool:
    """`Problems` on its own, or an inline label such as `Important: don't…`."""
    rest = text[len(label):].lstrip(": ").strip()
    if not rest:
        return True
    if label.lower() in {"important", "definition", "note", "remark", "theorem",
                         "example", "summary"}:
        return True
    return len(rest) > 0 and (text[len(label):len(label) + 1] in ":  ")


# --------------------------------------------------------------------------
# book-level passes
# --------------------------------------------------------------------------


def know_sections(book: "Book") -> set[str]:
    """Real section numbers (``1.2``, ``1.5``) discovered from headings."""
    known: set[str] = set()
    for page in book.pages:
        for par in page.paragraphs:
            text = par.text.strip()
            if RE_SUB.match(text):
                continue
            m = RE_SECTION.match(text)
            if m and _section_like(m.group(2)):
                known.add(m.group(1))
    return known


def annotate(book: "Book") -> "Book":
    """Classify every paragraph of every page (idempotent, two passes)."""
    known = know_sections(book)
    seen: dict[str, int] = {}
    for page in book.pages:
        page_boxes = ink.boxes(book, page.number) \
            if (page.raster or ink.has_geometry(book, page.number)) else []
        for par in page.paragraphs:
            classify(par, page_boxes, known)
            if par.role == ROLE_HEADING and par.number and par.level in (1, 2):
                key = f"{par.level}:{par.number}"
                prev = seen.get(key)
                if prev is None:
                    seen[key] = par
                elif _is_caps(prev.text) and not _is_caps(par.text):
                    # the running head came first: keep the real section line
                    prev.role = ROLE_NOISE
                    prev.attrs["duplicate_heading"] = True
                    seen[key] = par
                else:
                    par.role = ROLE_NOISE
                    par.attrs["duplicate_heading"] = True
    return book


def _is_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(1 for c in letters if c.isupper()) / len(letters) > 0.8


@dataclass
class OutlineEntry:
    level: int
    title: str
    number: Optional[str]
    page: int
    y: float
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "title": self.title,
            "number": self.number,
            "page": self.page,
            "y": round(self.y, 1),
            "bbox": [round(v, 1) for v in self.bbox],
        }


def outline(book: "Book") -> list[OutlineEntry]:
    """Every heading in reading order, with page and vertical position."""
    out: list[OutlineEntry] = []
    for page in book.pages:
        for par in page.paragraphs:
            if par.role == ROLE_HEADING:
                out.append(
                    OutlineEntry(
                        level=par.level if par.level is not None else 9,
                        title=par.text,
                        number=par.number,
                        page=par.page_no,
                        y=par.y0,
                        bbox=par.bbox,
                    )
                )
    return out


def _flatten(book: "Book") -> list["Paragraph"]:
    return [p for page in book.pages for p in page.paragraphs]


def _bbox_union(paras: Iterable["Paragraph"]) -> tuple[float, float, float, float]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for p in paras:
        xs0.append(p.bbox[0]); ys0.append(p.bbox[1])
        xs1.append(p.bbox[2]); ys1.append(p.bbox[3])
    if not xs0:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _parse_hints(text: str) -> list[str]:
    m = RE_HINTS.match(text.strip())
    if not m:
        return []
    rest = m.group("rest").strip()
    if not rest:
        return []
    return [h.strip() for h in re.split(r"[,;]", rest) if h.strip()]


def merge_paragraphs(paras: Sequence["Paragraph"]) -> "Paragraph":
    """Join several paragraphs into one (text, bbox and lines merged)."""
    base = paras[0]
    merged = type(base)(
        text=" ".join(p.text.strip() for p in paras if p.text.strip()),
        lines=[l for p in paras for l in p.lines],
        page_no=base.page_no,
        bbox=_bbox_union(paras),
        role=base.role,
        level=base.level,
        number=base.number,
        tag=base.tag,
        attrs=dict(base.attrs),
    )
    return merged


def items(book: "Book") -> list[Item]:
    """Extract problems/exercises/solutions with continuation + context.

    Continuation rules
    ------------------
    * inside a detected box: absorb following body paragraphs while they stay
      in the *same* box (this keeps "Problem 1.7" away from the prose below),
    * outside a box: absorb while the next paragraph does not start a new item,
      heading or label and the current text doesn't already look complete.
    """
    flat = _flatten(book)
    found: list[Item] = []
    section: Optional[str] = None
    section_no: Optional[str] = None
    section_page: Optional[int] = None

    i = 0
    while i < len(flat):
        par = flat[i]

        if par.role == ROLE_HEADING and par.number and par.level in (1, 2, 3):
            section = par.text.strip()
            section_no = par.number
            section_page = par.page_no
            i += 1
            continue

        if par.role == ROLE_LABEL and RE_HINTS.match(par.text.strip()):
            # standalone "Hints: 162" belongs to the item above it
            if found:
                found[-1].hints.extend(_parse_hints(par.text))
            i += 1
            continue

        if par.role in ITEM_ROLES:
            kind = {"problem": "problem", "exercise": "exercise",
                    "solution": "solution", "example": "example",
                    "theorem": "theorem"}[par.role]
            item = Item(
                kind=kind,
                tag=par.tag or par.text[:24],
                number=par.number,
                star=bool(par.attrs.get("star")),
                page=par.page_no,
                end_page=par.page_no,
                section=section,
                section_number=section_no,
                bbox=par.bbox,
                in_box=bool(par.attrs.get("in_box")),
                paragraphs=[par],
            )
            box_id = par.attrs.get("box")
            j = i + 1
            while j < len(flat):
                nxt = flat[j]
                if nxt.role in ITEM_ROLES or nxt.role == ROLE_HEADING:
                    break
                if nxt.role == ROLE_LABEL:
                    if RE_HINTS.match(nxt.text.strip()):
                        item.hints.extend(_parse_hints(nxt.text))
                        j += 1
                        continue
                    if nxt.attrs.get("in_box") and nxt.attrs.get("box") == box_id:
                        item.paragraphs.append(nxt)
                        j += 1
                        continue
                    break
                if box_id is not None:
                    if nxt.attrs.get("box") != box_id:
                        break
                else:
                    if nxt.attrs.get("in_box"):
                        break
                    if len(item.paragraphs) >= 6:
                        break
                    current = merge_paragraphs(item.paragraphs).text
                    indented = nxt.x0 - item.bbox[0] > 4.0
                    continues = (
                        nxt.text[:1].islower()
                        or indented
                        or current.rstrip().endswith((":", ",", ";"))
                        or len(current) < 70
                    )
                    if not continues and current.rstrip().endswith((".", "!", "?")):
                        break
                item.paragraphs.append(nxt)
                j += 1

            merged = merge_paragraphs(item.paragraphs)
            item.text = merged.text
            item.bbox = merged.bbox
            item.end_page = max(p.page_no for p in item.paragraphs)
            # inline hints at the very end of the text ("…? Hints: 123, 104")
            m = RE_INLINE_HINTS.search(item.text)
            if m:
                item.hints.extend([h.strip() for h in m.group("rest").split(",") if h.strip()])
            item.hints = [h for h in dict.fromkeys(item.hints) if h]
            found.append(item)
            i = j
            continue

        i += 1

    _mark_duplicates(found)
    return found


def _mark_duplicates(found: list[Item]) -> None:
    """Flag repeated statements (worked examples often restate a problem)."""
    seen: dict[tuple[str, str], int] = {}
    for idx, it in enumerate(found):
        key = (it.kind, (it.number or it.text[:40]))
        if key in seen:
            it.duplicate_of = seen[key]
        else:
            seen[key] = idx


def sections(book: "Book") -> list[dict[str, Any]]:
    """Section map: number/title/page-range, useful for chunk metadata."""
    entries = outline(book)
    secs: list[dict[str, Any]] = []
    for e in entries:
        if e.level in (1, 2, 3) and e.number:
            secs.append({
                "level": e.level,
                "number": e.number,
                "title": e.title,
                "page": e.page,
            })
    for idx, sec in enumerate(secs):
        nxt = next((x for x in secs[idx + 1:] if x["level"] <= sec["level"]), None)
        sec["end_page"] = (nxt["page"] - 1) if nxt and nxt["page"] > sec["page"] else \
            (nxt["page"] if nxt else len(book))
    return secs


def counts(book: "Book") -> dict[str, int]:
    """Role histogram for a whole book."""
    hist: dict[str, int] = {}
    for page in book.pages:
        for par in page.paragraphs:
            hist[par.role] = hist.get(par.role, 0) + 1
    return hist
