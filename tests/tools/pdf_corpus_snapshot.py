"""Snapshot how the PDF backend treats a folder of real PDFs, as numbers only.

Usage:  python tests/tools/pdf_corpus_snapshot.py <pdf-dir> <out.json> [--max-pages N]

Run it once per backend or per code version and diff the two JSON files with
tests/tools/pdf_corpus_diff.py. Output holds counts, boxes and sizes only, never page content, so snapshots are safe
to share. Only the public agent API and pypdf are used, so the script works on any backend.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import sys
import time
from pathlib import Path

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omni_ingest.agent.document import ImageExtractionAgent, RasterizeAgent  # noqa: E402
from omni_ingest.core.model import ByteContent, KnowledgeItem  # noqa: E402

DPI = 72


def text_chars(data: bytes, max_pages: int | None) -> tuple[int, int, list[int]]:
    reader = PdfReader(io.BytesIO(data))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    per_page = [len(re.sub(r"\s+", "", page.extract_text() or "")) for page in pages]
    return len(reader.pages), sum(per_page), per_page


def read_back(item: KnowledgeItem) -> bytes:
    return asyncio.run(item.content.read(None))  # type: ignore[arg-type]


def extraction(data: bytes, **config) -> tuple[list[dict], float]:
    parent = KnowledgeItem(content=ByteContent.from_bytes(data))
    start = time.perf_counter()
    items = ImageExtractionAgent(padding=0, dpi=DPI, **config)._extract_pdf(parent, data)
    elapsed = time.perf_counter() - start
    rows = [{"page": i.metadata["page"], "bbox": [round(v, 1) for v in i.metadata["bbox"]]} for i in items]
    return rows, elapsed


def mode_run(data: bytes, mode: str, max_pages: int | None) -> dict:
    parent = KnowledgeItem(content=ByteContent.from_bytes(data))
    items = ImageExtractionAgent(padding=0, dpi=DPI, mode=mode, include_vector=True)._extract_pdf(parent, data)
    out = read_back(parent)
    pages, chars, _ = text_chars(out, max_pages)
    squashed = "".join(page.extract_text() or "" for page in (PdfReader(io.BytesIO(out)).pages[:max_pages] if max_pages else PdfReader(io.BytesIO(out)).pages))
    squashed = re.sub(r"\s+", "", squashed)
    markers = sum(1 for item in items if f'<imageid="{item.id}"/>' in squashed)
    return {"pages": pages, "text_chars": chars, "items": len(items), "markers_found": markers, "bytes_ratio": round(len(out) / len(data), 3)}


def rasterize(data: bytes) -> dict:
    item = KnowledgeItem(content=ByteContent.from_bytes(data))
    start = time.perf_counter()
    RasterizeAgent(dpi=DPI)._rasterize_pdf(item, data)
    elapsed = time.perf_counter() - start
    out = read_back(item)
    reader = PdfReader(io.BytesIO(out))
    sizes = [(round(float(p.mediabox.width)), round(float(p.mediabox.height))) for p in reader.pages]
    return {"pages": len(sizes), "sizes_first_5": sizes[:5], "text_chars": sum(len(re.sub(r"\s+", "", p.extract_text() or "")) for p in reader.pages[:20]), "seconds": round(elapsed, 2)}


def stage(fn, *args, **kwargs):
    """Run one stage; a failure is recorded so the rest of the file still produces data."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}


def snapshot(path: Path, max_pages: int | None, with_modes: bool) -> dict:
    data = path.read_bytes()
    reader = PdfReader(io.BytesIO(data))
    pages, chars, per_page = text_chars(data, max_pages)
    first = reader.pages[0]
    entry: dict = {
        "bytes": len(data),
        "pages": pages,
        "page_size_first": [round(float(first.mediabox.width)), round(float(first.mediabox.height))],
        "text_chars": chars,
        "pages_without_text": sum(1 for c in per_page if c == 0),
    }
    entry["seconds"] = {}
    for key, config in (("embedded_images", {"include_vector": False}), ("vector_default", {"include_vector": True, "include_tables": False}), ("vector_with_tables", {"include_vector": True, "include_tables": True})):
        result = stage(extraction, data, **config)
        if isinstance(result, dict):
            entry[key] = result
        else:
            entry[key], seconds = result
            entry["seconds"][key] = round(seconds, 2)
    if with_modes:
        entry["redact"] = stage(mode_run, data, "redact", max_pages)
        entry["caption"] = stage(mode_run, data, "caption", max_pages)
        entry["rasterize"] = stage(rasterize, data)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-pages", type=int, default=None, help="limit text statistics to the first N pages")
    parser.add_argument("--only", default="*.pdf", help="glob of files inside pdf_dir, e.g. 'WB-*.pdf'")
    parser.add_argument("--no-modes", action="store_true", help="skip redact/caption/rasterize runs")
    args = parser.parse_args()

    result: dict = json.loads(args.out.read_text()) if args.out.exists() else {}  # merge into an existing snapshot
    for path in sorted(args.pdf_dir.glob(args.only)):
        print(f"{path.name} ...", flush=True)
        try:
            result[path.name] = snapshot(path, args.max_pages, not args.no_modes)
        except Exception as e:  # recorded, not fatal: a replacement may fail where the old backend did not
            result[path.name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        print("  done", {k: (len(v) if isinstance(v, list) else (v["error"] if isinstance(v, dict) and "error" in v else v)) for k, v in result[path.name].items() if k in ("pages", "embedded_images", "vector_default", "vector_with_tables", "redact", "caption", "rasterize", "error")}, flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
