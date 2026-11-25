"""Shared helpers for the PDF characterization tests. Backend agnostic: no PyMuPDF imports."""

from __future__ import annotations

import io
import struct
from pathlib import Path

from pypdf import PdfReader

from omni_ingest.agent.document import ImageExtractionAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource
from omni_ingest.core.pipeline import IngestionContext

PDFS = Path(__file__).parent.parent / "resources" / "pdf"
RESOURCES = Path(__file__).parent.parent / "resources"
TOL = 1.0
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def load(path: Path) -> bytes:
    return path.read_bytes()


def make_parent(data: bytes) -> KnowledgeItem:
    return KnowledgeItem(content=ByteContent.from_bytes(data))


def extract(data: bytes, **config) -> tuple[KnowledgeItem, list[KnowledgeItem]]:
    parent = make_parent(data)
    return parent, ImageExtractionAgent(**config)._extract_pdf(parent, data)


def png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == PNG_SIGNATURE
    return struct.unpack(">II", data[16:24])


async def image_bytes(item: KnowledgeItem) -> bytes:
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), items=[item])
    return await item.content.read(ctx)


def page_texts(data: bytes) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages]


def page_image_sizes(data: bytes) -> list[list[tuple[int, int]]]:
    """(width, height) in pixels of every image XObject on each page."""
    sizes = []
    for page in PdfReader(io.BytesIO(data)).pages:
        xobjects = (page.get("/Resources") or {}).get("/XObject") or {}
        found = []
        for ref in xobjects.values():
            xobject = ref.get_object()
            if xobject.get("/Subtype") == "/Image":
                found.append((int(xobject["/Width"]), int(xobject["/Height"])))
        sizes.append(found)
    return sizes


def page_sizes(data: bytes) -> list[tuple[float, float]]:
    return [(float(page.mediabox.width), float(page.mediabox.height)) for page in PdfReader(io.BytesIO(data)).pages]


def assert_bbox(actual: list[float], expected: tuple[float, float, float, float]) -> None:
    assert len(actual) == 4
    for a, e in zip(actual, expected, strict=True):
        assert abs(a - e) <= TOL, f"bbox {actual} != {expected}"
