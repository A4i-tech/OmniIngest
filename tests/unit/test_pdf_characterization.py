import io
import struct
from pathlib import Path
from uuid import uuid4

import pypdf
import pytest
from PIL import Image

from omni_ingest.agent.document import ImageExtractionAgent, RasterizeAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.output import MarkdownDirOutput
from omni_ingest.core.pipeline import IngestionContext

PDFS = Path(__file__).parent.parent / "resources" / "pdf"
IMGS = Path(__file__).parent.parent / "resources" / "img"
TOL = 1.5


def load(name: str) -> tuple[KnowledgeItem, bytes]:
    data = (PDFS / f"{name}.pdf").read_bytes()
    return KnowledgeItem(content=ByteContent.from_bytes(data)), data


def extract(name: str, **kwargs) -> tuple[KnowledgeItem, list[KnowledgeItem]]:
    item, data = load(name)
    return item, ImageExtractionAgent(**kwargs)._extract_pdf(item, data)


def png_size(item: KnowledgeItem) -> tuple[int, int]:
    data = item.content.data
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def boxes(items: list[KnowledgeItem]) -> list[tuple[int, list[float]]]:
    return [(i.metadata["page"], i.metadata["bbox"]) for i in items]


def assert_boxes(actual, expected):
    assert len(actual) == len(expected)
    for (page, bbox), (exp_page, exp_bbox) in zip(actual, expected, strict=True):
        assert page == exp_page
        assert bbox == pytest.approx(exp_bbox, abs=TOL)


def reader(item: KnowledgeItem) -> pypdf.PdfReader:
    return pypdf.PdfReader(io.BytesIO(item.content.data))


def marker(item: KnowledgeItem) -> str:
    return f'<image id="{item.id}"/>'


RAW = dict(include_vector=False, padding=0, dpi=72)


def test_embedded_images_have_page_bbox_and_png_pixels_at_dpi():
    _, out = extract("images", **RAW)
    assert_boxes(boxes(out), [(1, [40, 60, 360, 140]), (2, [40, 40, 200, 120]), (2, [300, 40, 360, 70])])
    assert [png_size(i) for i in out] == [(320, 80), (160, 80), (60, 30)]
    assert all(i.metadata["kind"] == "image" for i in out)


def test_dpi_scales_pixels_and_padding_grows_bbox():
    _, out = extract("images", include_vector=False, padding=6, dpi=144)
    assert_boxes(boxes(out)[:1], [(1, [34, 54, 366, 146])])
    assert png_size(out[0]) == pytest.approx((664, 184), abs=2)


def test_deduplicate_drops_same_size_copy_but_keeps_other_size():
    _, out = extract("images", **RAW)
    assert len(out) == 3
    _, out = extract("images", deduplicate=False, **RAW)
    assert len(out) == 4


def test_vector_figure_is_extracted_and_table_excluded_by_default():
    _, out = extract("vector")
    assert_boxes(boxes(out), [(1, [114, 114, 286, 286])])


def test_include_tables_extracts_table_region_too():
    _, out = extract("vector", include_tables=True)
    assert_boxes(boxes(out), [(1, [114, 114, 286, 286]), (1, [94, 444, 466, 616])])


def test_vector_disabled_extracts_nothing_from_drawings():
    _, out = extract("vector", include_vector=False)
    assert out == []


def test_rotated_page_bbox_is_in_visible_coordinates():
    _, out = extract("rotated", **RAW)
    assert_boxes(boxes(out), [(1, [60, 40, 140, 360])])


def test_hidden_layer_image_is_not_extracted():
    _, out = extract("hidden_layer", **RAW)
    assert_boxes(boxes(out), [(1, [200, 20, 300, 120])])


def test_cropbox_clips_and_offsets_bbox_to_crop_origin():
    _, out = extract("cropbox", **RAW)
    assert_boxes(boxes(out), [(1, [40, 40, 260, 120])])


def test_preserve_mode_returns_parent_bytes_unchanged():
    item, data = load("images")
    ImageExtractionAgent(**RAW)._extract_pdf(item, data)
    assert item.content.data == data


def test_redact_removes_text_and_images_and_writes_markers():
    item, out = extract("images", mode="redact", **RAW)
    pages = reader(item).pages
    assert len(out) == 3
    assert "SECRET" not in pages[0].extract_text()
    assert marker(out[0]) in pages[0].extract_text()
    assert len(pages[0].images) == 0


def test_redact_keeps_text_outside_the_image_region():
    item, _ = extract("text_outside_image", mode="redact", **RAW)
    text = reader(item).pages[0].extract_text()
    assert "KEEP" in text
    assert "SECRET" not in text


def test_redact_removes_deduplicated_copy_and_reuses_the_first_marker():
    item, out = extract("images", mode="redact", **RAW)
    page = reader(item).pages[1]
    assert len(page.images) == 0
    assert page.extract_text().count(marker(out[1])) == 2


def test_redact_marker_is_not_wrapped_in_narrow_region():
    item, out = extract("narrow_image", mode="redact", **RAW)
    assert marker(out[0]) in reader(item).pages[0].extract_text()


@pytest.mark.parametrize("name", ["rotated", "cropbox"])
@pytest.mark.parametrize("mode", ["redact", "caption"])
def test_markers_are_written_on_rotated_and_cropped_pages(name, mode):
    item, out = extract(name, mode=mode, **RAW)
    page = reader(item).pages[0]
    assert marker(out[0]) in page.extract_text()
    assert len(page.images) == (0 if mode == "redact" else 1)


def test_caption_keeps_text_and_images_and_adds_markers():
    item, out = extract("images", mode="caption", **RAW)
    pages = reader(item).pages
    assert "SECRET" in pages[0].extract_text()
    assert marker(out[0]) in pages[0].extract_text()
    assert len(pages[0].images) == 1
    assert len(pages[1].images) == 3


def test_caption_marker_for_bottom_edge_image_stays_in_the_content_stream():
    item, out = extract("image_bottom_edge", mode="caption", **RAW)
    assert marker(out[0]) in reader(item).pages[0].extract_text()


def test_damaged_pdf_with_truncated_tail_raises_value_error():
    item, data = load("images")
    with pytest.raises(ValueError, match="damaged"):
        ImageExtractionAgent(**RAW)._extract_pdf(item, data[: len(data) - 200])


def test_garbage_bytes_raise_value_error():
    item, _ = load("images")
    with pytest.raises(ValueError, match="damaged"):
        ImageExtractionAgent(**RAW)._extract_pdf(item, b"not a pdf")


def test_user_password_pdf_raises_value_error():
    item, data = load("images")
    writer = pypdf.PdfWriter(clone_from=io.BytesIO(data))
    writer.encrypt("secret")
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(ValueError, match="password-protected"):
        ImageExtractionAgent(**RAW)._extract_pdf(item, buf.getvalue())


def test_rasterize_keeps_page_count_and_sizes_and_removes_text_layer():
    item, data = load("text3")
    RasterizeAgent(dpi=72)._rasterize_pdf(item, data)
    pages = reader(item).pages
    assert [(round(float(p.mediabox.width)), round(float(p.mediabox.height))) for p in pages] == [(300, 200), (350, 200), (400, 200)]
    assert all(p.extract_text().strip() == "" for p in pages)
    assert all(len(p.images) == 1 for p in pages)


def test_rasterize_bakes_rotation_into_page_size():
    item, data = load("rotated")
    RasterizeAgent(dpi=72)._rasterize_pdf(item, data)
    page = reader(item).pages[0]
    assert (round(float(page.mediabox.width)), round(float(page.mediabox.height))) == (200, 400)


def test_rasterize_uses_crop_size_as_page_size():
    item, data = load("cropbox")
    RasterizeAgent(dpi=72)._rasterize_pdf(item, data)
    page = reader(item).pages[0]
    assert (round(float(page.mediabox.width)), round(float(page.mediabox.height))) == (280, 160)


async def write_image(tmp_path, data: bytes, content_type: str) -> list[Path]:
    ctx = IngestionContext(tenant_id=uuid4(), resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), items=[KnowledgeItem(content=ByteContent.from_bytes(data, content_type=content_type))])
    result = await MarkdownDirOutput(str(tmp_path)).write(ctx)
    assert result.status == StepStatus.SUCCESS
    return list((tmp_path / "out").iterdir())


def cmyk_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("CMYK", (8, 8)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("data,content_type", [
    ((IMGS / "gray.png").read_bytes(), "image/png"),
    ((IMGS / "rgb.jpg").read_bytes(), "image/jpeg"),
    ((IMGS / "rgba.png").read_bytes(), "image/png"),
    ((IMGS.parent / "supermarket-shelf.webp").read_bytes(), "image/webp"),
    (cmyk_jpeg(), "image/jpeg"),
], ids=["gray-png", "rgb-jpeg", "rgba-png", "webp", "cmyk-jpeg"])
async def test_markdown_output_normalizes_images_to_png(tmp_path, data, content_type):
    files = await write_image(tmp_path, data, content_type)
    assert len(files) == 1 and files[0].suffix == ".png"
    assert files[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.asyncio
async def test_markdown_output_keeps_undecodable_image_bytes_with_original_extension(tmp_path):
    files = await write_image(tmp_path, b"<svg/>", "image/svg+xml")
    assert [f.suffix for f in files] == [".svg+xml"] and files[0].read_bytes() == b"<svg/>"
