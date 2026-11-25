"""Edge-case characterization for the PDF backend (currently PyMuPDF).

Companion to test_pdf_characterization.py. These pin what the backend does implicitly: page geometry, hidden
layers, damaged and encrypted files, what redaction destroys, what survives each mode, maths and chemistry
content, and image decoding for the markdown output.

Tests named `*_current` record CURRENT behaviour that is not necessarily desirable (data loss, odd coordinates).
A replacement backend that differs there is not automatically wrong: review the difference, then update the test
and note it in the migration PR.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest
from pypdf import PdfReader

from omni_ingest.agent.document import RasterizeAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.output import MarkdownDirOutput
from omni_ingest.core.pipeline import IngestionContext

from ._pdf_helpers import PDFS, RESOURCES, assert_bbox, extract, image_bytes, load, make_parent, page_sizes, page_texts, png_size

IMAGES = RESOURCES / "images"
DEFAULTS = {"include_vector": False, "padding": 0, "dpi": 72}


def squash(text: str) -> str:
    """Remove all whitespace so markers wrapped across lines can still be matched."""
    return re.sub(r"\s+", "", text)


def survivors(text: str, items: list[KnowledgeItem]) -> int:
    squashed = squash(text)
    return sum(1 for item in items if f'<imageid="{item.id}"/>' in squashed)


# --------------------------------------------------------------------------------------
# Page geometry
# --------------------------------------------------------------------------------------


def test_rotated_page_bbox_is_in_unrotated_space_current():
    """Page stored 400x200 and rotated 90 (displays 200x400). The box is reported in unrotated
    coordinates and then clipped by the rotated page rectangle, so it is neither the true
    box (40,60,360,140) nor its rotated equivalent."""
    _, [item] = extract(load(PDFS / "edge_rotated.pdf"), **DEFAULTS)
    assert_bbox(item.metadata["bbox"], (40, 60, 200, 140))


def test_rasterize_bakes_page_rotation_into_page_size():
    data = load(PDFS / "edge_rotated.pdf")
    item = make_parent(data)
    RasterizeAgent(dpi=72)._rasterize_pdf(item, data)
    [(w, h)] = page_sizes(item_bytes_sync(item))
    assert (round(w), round(h)) == (200, 400)


def test_cropbox_clips_boxes_and_is_origin_relative_current():
    """CropBox starts at (50,50). The 360x360 image is clipped to 100% of the crop and dropped (>80%).
    The inner image (100,100,200,200) is reported relative to the crop origin."""
    _, items = extract(load(PDFS / "edge_cropbox.pdf"), **DEFAULTS)
    [item] = items
    assert_bbox(item.metadata["bbox"], (50, 50, 150, 150))


def test_rasterize_uses_cropbox_as_page_size():
    data = load(PDFS / "edge_cropbox.pdf")
    item = make_parent(data)
    RasterizeAgent(dpi=72)._rasterize_pdf(item, data)
    [(w, h)] = page_sizes(item_bytes_sync(item))
    assert (round(w), round(h)) == (250, 250)


def test_image_on_hidden_layer_is_not_extracted():
    _, items = extract(load(PDFS / "edge_hidden_layer.pdf"), **DEFAULTS)
    assert items == []


def test_image_with_alpha_channel_is_extracted():
    _, [item] = extract(load(PDFS / "edge_alpha_image.pdf"), **DEFAULTS)
    assert_bbox(item.metadata["bbox"], (40, 60, 360, 140))


def test_image_placed_with_rotation_reports_axis_aligned_bbox():
    _, [item] = extract(load(PDFS / "edge_rotated_image.pdf"), **DEFAULTS)
    assert_bbox(item.metadata["bbox"], (100, 100, 300, 200))


# --------------------------------------------------------------------------------------
# Damaged, empty and encrypted input
# --------------------------------------------------------------------------------------


def test_truncated_pdf_is_repaired_and_extracted():
    data = load(PDFS / "one_image.pdf")
    _, items = extract(data[:-200], **DEFAULTS)
    assert len(items) == 1


def test_broken_startxref_is_repaired_and_extracted():
    data = load(PDFS / "one_image.pdf").replace(b"startxref", b"startxxxx")
    _, items = extract(data, **DEFAULTS)
    assert len(items) == 1


def test_empty_bytes_raise():
    with pytest.raises(Exception):  # noqa: B017 - exception type is backend specific
        extract(b"", **DEFAULTS)


def test_pdf_with_zero_pages_yields_nothing():
    data = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF"
    _, items = extract(data, **DEFAULTS)
    assert items == []


def test_user_password_pdf_raises():
    with pytest.raises(ValueError):
        extract(load(PDFS / "edge_encrypted_user_pw.pdf"), **DEFAULTS)


@pytest.mark.asyncio
async def test_owner_password_only_pdf_is_processed_and_output_is_decrypted_current():
    data = load(PDFS / "edge_encrypted_owner_only.pdf")
    assert PdfReader(io.BytesIO(data)).is_encrypted
    parent, [item] = extract(data, mode="redact", **DEFAULTS)
    assert item.metadata["page"] == 1
    assert not PdfReader(io.BytesIO(await image_bytes(parent))).is_encrypted


# --------------------------------------------------------------------------------------
# What redact and caption destroy or keep
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redact_deletes_text_under_the_region_but_caption_keeps_it_current():
    data = load(PDFS / "edge_overlap_text.pdf")
    source_text = page_texts(data)[0]
    assert "OVERLAPTEXT" in source_text and "OUTSIDE" in source_text

    redacted, _ = extract(data, mode="redact", **DEFAULTS)
    redacted_text = page_texts(await image_bytes(redacted))[0]
    assert "OVERLAPTEXT" not in redacted_text
    assert "OUTSIDE" in redacted_text

    captioned, _ = extract(data, mode="caption", **DEFAULTS)
    captioned_text = page_texts(await image_bytes(captioned))[0]
    assert "OVERLAPTEXT" in captioned_text and "OUTSIDE" in captioned_text


@pytest.mark.asyncio
async def test_math_content_under_a_vector_region_is_deleted_by_redact_current():
    """The text 'x+1' sits under a radical sign, which is extracted as a figure. Redact removes the text."""
    data = load(PDFS / "math_chem.pdf")
    assert "x+1" in page_texts(data)[0]

    redacted, _ = extract(data, mode="redact", padding=0, dpi=72, include_vector=True)
    text = page_texts(await image_bytes(redacted))[0]
    assert "x+1" not in text
    assert "2 H2 + O2" in text and "CH3-CH2-OH" in text and "E = mc" in text

    captioned, _ = extract(data, mode="caption", padding=0, dpi=72, include_vector=True)
    assert "x+1" in page_texts(await image_bytes(captioned))[0]


@pytest.mark.asyncio
async def test_redact_markers_for_small_regions_do_not_all_survive_current():
    """Markers are written into the region itself. Narrow regions (5pt wide brackets) cannot hold the
    marker, so some `<image id=.../>` markers are missing from the redacted PDF. Caption keeps all."""
    data = load(PDFS / "math_chem.pdf")
    redacted, items = extract(data, mode="redact", padding=0, dpi=72, include_vector=True)
    assert len(items) == 5
    assert survivors(page_texts(await image_bytes(redacted))[0], items) < len(items)

    captioned, items = extract(data, mode="caption", padding=0, dpi=72, include_vector=True)
    assert survivors(page_texts(await image_bytes(captioned))[0], items) == len(items)


def item_bytes_sync(item: KnowledgeItem) -> bytes:
    import asyncio

    return asyncio.run(image_bytes(item))


def structure(data: bytes) -> dict:
    reader = PdfReader(io.BytesIO(data))
    annots = [[str(a.get_object().get("/Subtype")) for a in (page.get("/Annots") or [])] for page in reader.pages]
    return {
        "pages": len(reader.pages),
        "outline": len(reader.outline),
        "title": (reader.metadata or {}).get("/Title"),
        "attachments": sorted(reader.attachments),
        "annots": annots,
        "fields": sorted(reader.get_fields() or {}),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["preserve", "redact", "caption"])
async def test_document_structure_survives_all_parent_modes(mode):
    """Outline, metadata, MathML attachment, links, annotations and form fields are untouched."""
    data = load(PDFS / "metadata_rich.pdf")
    before = structure(data)
    assert before == {
        "pages": 3, "outline": 2, "title": "Title", "attachments": ["equation.mathml"],
        "annots": [["/Link"], ["/Text", "/Popup", "/Widget"], []], "fields": ["f"],
    }
    parent, _ = extract(data, mode=mode, **DEFAULTS)
    assert structure(await image_bytes(parent)) == before


def test_rasterize_drops_outline_metadata_attachments_annotations_and_fields_current():
    data = load(PDFS / "metadata_rich.pdf")
    item = make_parent(data)
    RasterizeAgent(dpi=72)._rasterize_pdf(item, data)
    after = structure(item_bytes_sync(item))
    assert after == {"pages": 3, "outline": 0, "title": None, "attachments": [], "annots": [[], [], []], "fields": []}


# --------------------------------------------------------------------------------------
# Maths and chemistry
# --------------------------------------------------------------------------------------


def sorted_boxes(items: list[KnowledgeItem]) -> list[list[float]]:
    return sorted((item.metadata["bbox"] for item in items), key=lambda b: (b[1], b[0]))


def test_math_and_chemistry_vector_strokes_become_figures_current():
    """PDFs carry no MathML in page content: maths is glyph text plus thin strokes, chemistry likewise.
    Expected figures: radical sign, two matrix brackets, benzene ring (with its OH bond) and reaction arrow.
    Not extracted: fraction bar (below the 0.1% area floor), sum and integral glyphs, formulas typed as text."""
    _, items = extract(load(PDFS / "math_chem.pdf"), padding=0, dpi=72, include_vector=True)
    expected = [(150, 112, 190, 138), (355, 100, 360, 160), (420, 100, 425, 160), (250, 314, 320, 326), (85.4, 280, 179.6, 360)]
    boxes = sorted_boxes(items)
    assert len(boxes) == len(expected)
    for box, want in zip(boxes, sorted(expected, key=lambda b: (b[1], b[0])), strict=True):
        assert_bbox(box, want)
    assert all(item.metadata["page"] == 1 for item in items)


def test_math_and_chemistry_ignored_without_vector_extraction():
    _, items = extract(load(PDFS / "math_chem.pdf"), padding=0, dpi=72, include_vector=False)
    assert items == []


def test_math_and_chemistry_text_layer_is_readable_in_source():
    text = squash(page_texts(load(PDFS / "math_chem.pdf"))[0])
    for formula in ("E=mc", "a+b", "2H2+O2->2H2O", "CH3-CH2-OH+O2"):
        assert formula in text


# --------------------------------------------------------------------------------------
# Markdown output: which image bytes become PNG and which pass through
# --------------------------------------------------------------------------------------

GIF_1X1 = bytes.fromhex("47494638396101000100800000ffffff00000021f90401000000002c00000000010001000002024401003b")
SVG = b"<svg xmlns='http://www.w3.org/2000/svg' width='4' height='4'/>"


async def written(tmp_path: Path, raw: bytes, content_type: str) -> tuple[str, bytes]:
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")))
    ctx.items = [KnowledgeItem(content=ByteContent.from_bytes(raw, content_type=content_type, content_encoding="binary"), metadata={"page": 1})]
    assert (await MarkdownDirOutput(str(tmp_path)).write(ctx)).status == StepStatus.SUCCESS
    [path] = list((tmp_path / "out").iterdir())
    assert json.loads((tmp_path / "doc_metadata.json").read_text())["images"][0]["file"].endswith(path.suffix)
    return path.suffix, path.read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "content_type"),
    [
        ((IMAGES / "gray.png").read_bytes(), "image/png"),
        ((IMAGES / "gray.jpg").read_bytes(), "image/jpeg"),
        ((IMAGES / "alpha.png").read_bytes(), "image/png"),
        (GIF_1X1, "image/gif"),
    ],
    ids=["gray-png", "gray-jpeg", "alpha-png", "gif"],
)
async def test_markdown_output_normalises_decodable_rasters_to_png(tmp_path, raw, content_type):
    suffix, data = await written(tmp_path, raw, content_type)
    assert suffix == ".png"
    png_size(data)


@pytest.mark.asyncio
async def test_markdown_output_tolerates_truncated_png_current(tmp_path):
    """A PNG cut off after its header still 'decodes' and is re-encoded."""
    suffix, data = await written(tmp_path, (IMAGES / "gray.png").read_bytes()[:40], "image/png")
    assert suffix == ".png"
    png_size(data)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "content_type", "suffix"),
    [
        ((RESOURCES / "supermarket-shelf.webp").read_bytes(), "image/webp", ".webp"),
        ((IMAGES / "cmyk.jpg").read_bytes(), "image/jpeg", ".jpeg"),
        (SVG, "image/svg+xml", ".svg+xml"),
    ],
    ids=["webp", "cmyk-jpeg", "svg"],
)
async def test_markdown_output_passes_through_formats_it_cannot_convert_current(tmp_path, raw, content_type, suffix):
    """WebP, CMYK JPEG and SVG are not converted: original bytes are written with an extension taken from the content type."""
    written_suffix, data = await written(tmp_path, raw, content_type)
    assert written_suffix == suffix
    assert data == raw
