"""bookkit — a layout-aware toolkit for turning scanned book PDFs into
agent-friendly Markdown, JSON and cropped images.

Built by an AI agent, for an AI agent (and its human).

Core ideas
----------
1. Scanned pages keep their raster + a (possibly broken) OCR text layer.
2. Text is reconstructed with layout awareness (blocks, gaps, indent,
   running headers/footers) instead of naive ``get_text()``.
3. Suspect OCR regions are *flagged* (never silently "fixed"), and can be
   rendered as high-DPI crops so a vision-capable agent can read the
   original pixels and repair the text itself.
4. Everything is emitted as stable, diffable artifacts: Markdown, JSON,
   JSONL chunks and a compact "digest" for limited context windows.
"""

__version__ = "0.1.0"
__all__ = ["__version__", "load"]

from .core import Book, load  # noqa: E402
