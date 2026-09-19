"""Tests for bookkit.

Run with pytest::

    python -m pytest tests -q

The suite has two halves:

* **synthetic** tests build small PDFs on the fly (with pymupdf) so the layout,
  ink and structure logic is exercised without shipping fixtures;
* **real book** tests run against ``Chapter1.pdf`` when it is present in the
  repository root, and are skipped otherwise.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterator

import pymupdf
import pytest

from bookkit import core, ink, quality, render, structure
from bookkit.core import Book, load

REPO = Path(__file__).resolve().parents[1]
CHAPTER = REPO / "Chapter1.pdf"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def build_pdf(tmp_path: Path, pages: list[str], font: str = "helv") -> Path:
    """A text-layer-only PDF: one paragraph box per string.

    ``font="korea"`` embeds a CJK-capable face, which is how the tests put a
    mis-recognised glyph into a real text layer.
    """
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(pymupdf.Rect(50, 60, 545, 780), text,
                            fontsize=11, fontname=font, lineheight=1.35)
    out = tmp_path / f"synthetic-{font}.pdf"
    doc.save(out)
    doc.close()
    return out


def build_book_pdf(tmp_path: Path) -> Path:
    """A page with running heads, two sections, a boxed problem and an exercise."""
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page(width=595, height=842)
        page.insert_text((60, 40), "CHAPTER 1. TESTING LAYOUT", fontsize=9)
        page.insert_text((60, 70), "1.1 Introduction", fontsize=14)
        page.insert_textbox(pymupdf.Rect(60, 90, 540, 200),
                            "This is the first paragraph of the section. " * 4,
                            fontsize=11)
        page.draw_rect(pymupdf.Rect(55, 220, 540, 280), width=1.0)
        page.insert_textbox(pymupdf.Rect(60, 228, 535, 275),
                            "Problem 1.7: How many players are taking French?", fontsize=11)
        page.insert_text((60, 320), "Some prose between blocks, " * 6, fontsize=11)
        page.insert_text((60, 390), "1.1.1 How many numbers are in the list 1, 2, 3?", fontsize=11)
        page.insert_text((60, 820), str(i + 1), fontsize=9)
    out = tmp_path / "book.pdf"
    doc.save(out)
    doc.close()
    return out


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_book_pdf(tmp_path_factory.mktemp("synth"))


@pytest.fixture(scope="module")
def chapter() -> Iterator[Book]:
    if not CHAPTER.exists():  # pragma: no cover
        pytest.skip("Chapter1.pdf not present")
    book = load(CHAPTER)
    yield book
    book.close()


# --------------------------------------------------------------------------
# text hygiene
# --------------------------------------------------------------------------


def test_clean_text_collapses_whitespace():
    assert core.clean_text("  a \t b\n c ") == "a b c"
    assert core.clean_text("soft\u00adhyphen") == "softhyphen"


def test_dehyphenate_and_join_lines():
    assert core.dehyphenate("determin-", "ing") is True
    assert core.dehyphenate("one-", "Third") is False
    assert core.dehyphenate("sub-", "problem") is True
    assert core.join_lines(["determin-", "ing the count"]) == "determining the count"
    assert core.join_lines(["the count", "is 42"]) == "the count is 42"


def test_looks_like_heading():
    assert core.looks_like_heading("1.2 Counting Lists of Numbers")
    assert core.looks_like_heading("CHAPTER 1")
    assert core.looks_like_heading("Problems")
    assert not core.looks_like_heading("This is a plain sentence about counting.")


# --------------------------------------------------------------------------
# layout model on a synthetic book
# --------------------------------------------------------------------------


def test_model_basics(synthetic: Path):
    book = load(synthetic)
    assert len(book) == 3
    page = book.page(1)
    assert page.chars > 100
    assert page.lines, "lines must be extracted"
    # lines are ordered top-to-bottom
    ys = [l.y0 for l in page.lines]
    assert ys == sorted(ys)
    # every line keeps a usable bbox
    for line in page.lines[:5]:
        assert line.bbox[2] > line.bbox[0]
        assert line.bbox[3] > line.bbox[1]


def test_running_heads_and_folios_are_stripped(synthetic: Path):
    book = load(synthetic)
    assert book.page(1).header.startswith("CHAPTER 1")
    assert book.page(2).folio == "2"
    assert not any(l.text == "2" for l in book.page(2).lines)
    assert not any(l.text.startswith("CHAPTER") for l in book.page(2).lines)


def test_paragraphs_reflow_into_sentences(synthetic: Path):
    book = load(synthetic)
    paras = [p for p in book.page(1).paragraphs if p.role == "body"]
    assert paras, "prose should be classified as body"
    longest = max(paras, key=lambda p: len(p.text))
    assert longest.text.startswith("This is the first paragraph")
    # a reflowed paragraph is longer than a single line chunk
    assert len(longest.text) > 80


def test_know_sections_and_outline(synthetic: Path):
    book = load(synthetic)
    outline = structure.outline(book)
    titles = [e.title for e in outline]
    assert "1.1 Introduction" in titles
    sections = structure.sections(book)
    assert sections[0]["number"] == "1.1"
    assert sections[0]["page"] == 1


def test_items_include_boxed_problem_and_exercise(synthetic: Path):
    book = load(synthetic)
    items = structure.items(book)
    kinds = {i.kind: i for i in items}
    assert "problem" in kinds, "the boxed Problem 1.7 must be found"
    problem = kinds["problem"]
    assert problem.number == "1.7"
    assert problem.in_box is True, "the detected border should mark it as boxed"
    assert "fresh" in problem.text or "French" in problem.text
    assert any(i.kind == "exercise" for i in items), "1.1.1 is an exercise"


# --------------------------------------------------------------------------
# ink geometry
# --------------------------------------------------------------------------


def test_ink_finds_box_and_bars(synthetic: Path):
    book = load(synthetic)
    page_no = 1
    boxes = ink.boxes(book, page_no)
    assert boxes, "the drawn rectangle must be detected as a box"
    box = max(boxes, key=lambda b: b.area)
    assert box.width == pytest.approx(485, abs=8)
    assert box.height == pytest.approx(60, abs=8)
    assert box.closed is True
    # no filled bars on this page
    assert ink.bars(book, page_no) == []


def test_box_for_lookup(synthetic: Path):
    book = load(synthetic)
    line = next(l for l in book.page(1).lines if l.text.startswith("Problem 1.7"))
    found = ink.box_for(book, 1, line.bbox)
    assert found is not None and found.width > 400


def test_ink_rejects_plain_text_pages(tmp_path: Path):
    pdf = build_pdf(tmp_path, ["Just one paragraph of ordinary text, nothing drawn here."])
    book = load(pdf)
    assert ink.boxes(book, 1) == []
    assert [r for r in ink.rules(book, 1) if r.orient == "v"] == []


# --------------------------------------------------------------------------
# quality
# --------------------------------------------------------------------------


def test_line_issues_detects_each_failure_mode():
    """Unit-level: every detector fires on its own signature."""
    line = core.Line(text="The count is 41층", bbox=(0, 0, 100, 10), size=8.0,
                     page_no=1, size_max=18.0, span_count=1)
    codes = {code for code, _, _ in quality._line_issues(line, 8.0)}
    assert {"script_mix", "span_outlier"} <= codes

    glued = core.Line(text="thewordAnotherword here", bbox=(0, 0, 100, 10),
                      size=8.0, page_no=1)
    assert "glued_words" in {c for c, _, _ in quality._line_issues(glued, 8.0)}

    noisy = core.Line(text="???", bbox=(0, 0, 100, 10), size=8.0, page_no=1)
    assert "dense_symbols" in {c for c, _, _ in quality._line_issues(noisy, 8.0)}

    ellipsis = core.Line(text="……", bbox=(0, 0, 100, 10), size=8.0, page_no=1)
    assert "dense_symbols" not in {c for c, _, _ in quality._line_issues(ellipsis, 8.0)}

    clean = core.Line(text="A perfectly ordinary sentence about counting.",
                      bbox=(0, 0, 100, 10), size=8.0, page_no=1)
    assert quality._line_issues(clean, 8.0) == []


def test_scan_flags_script_mixes(tmp_path: Path):
    pdf = build_pdf(tmp_path, ["The count is 41층 and the list continues normally."],
                    font="korea")
    book = load(pdf)
    report = quality.scan(book)
    codes = {f.code for f in report.flags}
    assert "script_mix" in codes
    flag = next(f for f in report.flags if f.code == "script_mix")
    assert flag.bbox is not None, "flags must be croppable"
    assert flag.severity >= 5


def test_repairs_change_the_model_and_the_score(tmp_path: Path):
    pdf = build_pdf(tmp_path, ["The value is 41층 exactly."], font="korea")
    book = load(pdf)
    before = quality.scan(book)
    line = next(l for l in book.page(1).lines if "층" in l.text)
    applied = quality.apply_repairs(book, [
        quality.Repair(page=1, line=line.index, before="41층", after="41"),
    ])
    after = quality.scan(book)
    assert applied == 1
    assert "층" not in book.page(1).text
    assert after.score < before.score


def test_repair_with_unknown_before_is_skipped(tmp_path: Path):
    pdf = build_pdf(tmp_path, ["The value is 41 exactly."])
    book = load(pdf)
    applied = quality.apply_repairs(book, [
        quality.Repair(page=1, line=0, before="not in this line", after="x"),
    ])
    assert applied == 0
    assert "41" in book.page(1).text


def test_repairs_roundtrip_through_disk(tmp_path: Path):
    path = tmp_path / "fixes.jsonl"
    repairs = [quality.Repair(page=2, line=3, before="a", after="b", reason="test")]
    quality.save_repairs(repairs, str(path))
    loaded = quality.load_repairs(str(path))
    assert len(loaded) == 1 and loaded[0].page == 2 and loaded[0].after == "b"


def test_mathish_detection():
    assert render.is_mathish("n x (n-1) x (n-2)")
    assert render.is_mathish("215-62 = 153")
    assert not render.is_mathish("How many numbers are in the list")


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_markdown_has_structure(synthetic: Path):
    book = load(synthetic)
    report = quality.scan(book)
    md = render.to_markdown(book, report)
    assert md.startswith("# ")
    assert "## Outline" in md
    assert "### 1.1 Introduction" in md
    assert "> **Problem 1.7:" in md
    assert "<!-- bookkit:page=2 -->" in md
    assert "bookkit" in md.splitlines()[-1]


def test_chunks_are_serialisable(synthetic: Path):
    import json

    book = load(synthetic)
    report = quality.scan(book)
    chunks = render.chunks(book, report)
    assert chunks
    for chunk in chunks:
        payload = json.dumps(chunk.to_dict(), ensure_ascii=False)
        assert json.loads(payload)["id"] == chunk.id
    problems = [c for c in chunks if c.role == "problem"]
    assert problems and problems[0].page >= 1


def test_digest_shape(synthetic: Path):
    book = load(synthetic)
    digest = render.digest(book, quality.scan(book))
    assert digest["pages"] == 3
    assert digest["outline"] and digest["sections"]
    assert "ocr" in digest


def test_payload_is_json_safe(synthetic: Path):
    import json

    book = load(synthetic)
    payload = render.payload(book, quality.scan(book), page=1)
    text = json.dumps(payload, ensure_ascii=False)
    assert json.loads(text)["pages"][0]["number"] == 1


# --------------------------------------------------------------------------
# the real scanned book
# --------------------------------------------------------------------------


def test_chapter_is_detected_as_scan(chapter: Book):
    info = chapter.info()
    assert info["pages"] == 26
    assert info["scanned_pages"] == 26
    assert info["raster_px"][0] >= 4000
    assert info["total_chars"] > 40_000


def test_chapter_outline(chapter: Book):
    titles = [e.title for e in structure.outline(chapter)]
    assert "Counting Is Arithmetic" in titles
    for expected in ("1.2 Counting Lists of Numbers", "1.5 Permutations", "1.6 Summary"):
        assert expected in titles, titles
    # review-problem numbering must not leak into the outline
    assert not any(t.startswith(("1.17", "1.28", "1.33")) for t in titles)


def test_chapter_sections_have_sane_ranges(chapter: Book):
    sections = {s["number"]: s for s in structure.sections(chapter)}
    assert sections["1.2"]["page"] == 2
    assert sections["1.2"]["end_page"] == 5
    assert sections["1.3"]["page"] == 6
    assert sections["1.3"]["end_page"] == 12
    assert sections["1.6"]["page"] == 21


def test_chapter_items(chapter: Book):
    items = structure.items(chapter)
    kinds = {}
    for it in items:
        kinds[it.kind] = kinds.get(it.kind, 0) + 1
    assert kinds.get("problem", 0) >= 30
    assert kinds.get("exercise", 0) >= 30
    assert kinds.get("solution", 0) >= 10
    by_number = {i.number: i for i in items if i.kind == "problem"}
    assert "1.7" in by_number
    problem_17 = by_number["1.7"]
    assert problem_17.section_number == "1.3"
    assert "Brown High School" in problem_17.text
    # the worked example restates the same problem on its own page
    assert any(i.duplicate_of is not None for i in items)


def test_chapter_ocr_report_is_actionable(chapter: Book):
    report = quality.scan(chapter)
    summary = report.summary()
    assert summary["pages"] == 26
    assert summary["flag_counts"].get("script_mix", 0) >= 5
    assert summary["reading_size"] == pytest.approx(8.4, abs=0.6)
    queue = report.queue(limit=10)
    assert queue and queue[0].severity >= queue[-1].severity
    assert any(f.bbox for f in queue), "the queue must be croppable"
    assert "ocr report" in report.text().lower()


def test_chapter_boxes_and_bars(chapter: Book):
    # page 6 has two bordered problem statements and a black "Problems" bar
    boxes = ink.boxes(chapter, 6)
    assert len(boxes) >= 2
    assert any(b.height > 50 and b.width > 400 for b in boxes)
    bars = ink.bars(chapter, 6)
    assert bars, "the solid Problems header should be detected"


def test_chapter_render_and_crop(chapter: Book):
    png = chapter.render(1, dpi=60)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    line = chapter.page(2).lines[0]
    crop = chapter.crop(2, line.bbox, dpi=120)
    assert crop[:8] == b"\x89PNG\r\n\x1a\n" and len(crop) > 500
    assert chapter.line_crop(2, 0, dpi=120)[:4] == b"\x89PNG"


def test_chapter_find(chapter: Book):
    hits = chapter.find_bbox("Problem 1.7")
    assert len(hits) >= 2, "Problem 1.7 appears twice (statement + worked example)"
    assert all(h["page"] >= 6 for h in hits)
    assert all(len(h["bbox"]) == 4 for h in hits)


def test_chapter_markdown_flags_bad_ocr(chapter: Book):
    report = quality.scan(chapter)
    md = render.to_markdown(chapter, report)
    assert "<!-- ⚠ OCR script_mix" in md
    assert "3층" in md  # the flow marks it, it never silently rewrites it


def test_chapter_knows_its_language_and_metrics(chapter: Book):
    report = quality.scan(chapter)
    assert report.language == "latin"
    # the scanner's per-glyph sizes are noise: the report must say so
    assert any("jitter" in n for n in report.notes)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_info_and_digest(synthetic: Path, capsys: pytest.CaptureFixture[str]):
    from bookkit.cli import main

    assert main(["info", str(synthetic)]) == 0
    out = capsys.readouterr().out
    assert "pages" in out and "3" in out

    assert main(["digest", str(synthetic), "--json"]) == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["pages"] == 3


def test_cli_items_json(synthetic: Path, capsys: pytest.CaptureFixture[str]):
    import json

    from bookkit.cli import main

    assert main(["items", str(synthetic), "--json"]) == 0
    items = json.loads(capsys.readouterr().out)
    assert any(i["kind"] == "problem" for i in items)


def test_cli_crop_writes_a_png(synthetic: Path, tmp_path: Path):
    from bookkit.cli import main

    out = tmp_path / "crop.png"
    assert main(["crop", str(synthetic), "-p", "1", "--dpi", "90", "-o", str(out)]) == 0
    assert out.exists() and out.read_bytes()[:4] == b"\x89PNG"


def test_cli_md_export(synthetic: Path, tmp_path: Path):
    from bookkit.cli import main

    out = tmp_path / "book.md"
    assert main(["md", str(synthetic), "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## Outline" in text and "Problem 1.7" in text


def test_cli_ink_overlay(synthetic: Path, tmp_path: Path):
    from bookkit.cli import main

    out = tmp_path / "overlay.png"
    assert main(["ink", str(synthetic), "-p", "1", "--overlay", str(out)]) == 0
    assert out.exists()
