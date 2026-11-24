import io
import struct
from pathlib import Path
from uuid import uuid4

import pypdf
import pypdfium2 as pdfium
import pypdfium2.raw as raw
import pytest
from PIL import Image, ImageChops, ImageStat
from pypdf.generic import ArrayObject, ContentStream, DecodedStreamObject, DictionaryObject, NameObject, NumberObject, TextStringObject

from omni_ingest.agent.document import ImageExtractionAgent, RasterizeAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.output import MarkdownDirOutput
from omni_ingest.core.pipeline import IngestionContext
from omni_ingest.pdf import Pdf

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


def test_membership_dictionaries_decide_layer_visibility():
    _, out = extract("hidden_membership", **RAW)
    assert_boxes(boxes(out), [(1, [140, 20, 240, 120])])


def test_base_state_off_hides_every_layer_that_is_not_listed_on():
    item, data = load("hidden_layer")
    writer = pypdf.PdfWriter(clone_from=io.BytesIO(data))
    properties = writer._root_object["/OCProperties"]
    config = properties["/D"]
    hidden = {o.idnum for o in config["/OFF"]}
    config[NameObject("/ON")] = ArrayObject([o for o in properties["/OCGs"] if o.idnum not in hidden])
    config[NameObject("/BaseState")] = NameObject("/OFF")
    del config["/OFF"]
    buf = io.BytesIO()
    writer.write(buf)
    out = ImageExtractionAgent(**RAW)._extract_pdf(item, buf.getvalue())
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


@pytest.mark.parametrize("name", ["rotated", "cropbox"])
def test_caption_marker_sits_just_below_the_image_on_rotated_and_cropped_pages(name):
    item, out = extract(name, mode="caption", **RAW)
    x0, _, _, y1 = out[0].metadata["bbox"]
    with Pdf.open(item.content.data) as pdf:
        page = pdf.page(0)
        texts = [page._box(*o.get_bounds()) for o in page._objects(raw.FPDF_PAGEOBJ_TEXT)]
    assert any(abs(t[0] - x0) < 2 and y1 - 1 <= t[1] <= y1 + 8 for t in texts)


def stream_ops(data: bytes) -> list[tuple[list, bytes]]:
    r = pypdf.PdfReader(io.BytesIO(data))
    return ContentStream(r.pages[0].get_contents(), r, "bytes").operations


def redacted(data: bytes, *rects: tuple[float, float, float, float], marker: str = "~") -> bytes:
    with Pdf.open(data) as pdf:
        for rect in rects:
            pdf.page(0).redact(rect, marker)
        return pdf.to_bytes()


def redact_directly(name: str, rect: tuple[float, float, float, float], marker: str = "M") -> tuple[bytes, bytes]:
    data = (PDFS / f"{name}.pdf").read_bytes()
    return data, redacted(data, rect, marker=marker)


def one_page(content: str, widths: bool = False, setup=None) -> bytes:
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(200, 200)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    if widths:
        font[NameObject("/FirstChar")] = NumberObject(32)
        font[NameObject("/Widths")] = ArrayObject([NumberObject(500)] * 95)
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    stream.set_data(content.encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    if setup:
        setup(writer, page)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def page_lines(data: bytes) -> list[str]:
    with Pdf.open(data) as pdf:
        text = pdf.doc[0].get_textpage().get_text_range().replace("~", "")
    return [line.strip() for line in text.splitlines() if line.strip()]


def op_names(data: bytes) -> list[bytes]:
    return [op for _, op in stream_ops(data)]


def render(data: bytes, box: tuple[int, int, int, int]) -> Image.Image:
    return pdfium.PdfDocument(data)[0].render(scale=1).to_pil().convert("L").crop(box)


def char_left(data: bytes, word: str) -> float:
    with Pdf.open(data) as pdf:
        text = pdf.doc[0].get_textpage()
        return text.get_charbox(text.get_text_range().find(word))[0]


def test_redact_removes_only_the_glyphs_under_the_region_and_keeps_the_rest_in_place():
    item, out = extract("partial_text", mode="redact", **RAW)
    text = reader(item).pages[0].extract_text()
    assert all(word in text for word in ("HELLO", "ONE", "THREE"))
    assert not any(word in text for word in ("WORLD", "TWO"))
    assert marker(out[0]) in text and marker(out[1]) in text
    _, data = load("partial_text")
    assert char_left(item.content.data, "THREE") == pytest.approx(char_left(data, "THREE"), abs=0.5)


def test_redact_removes_a_shading_inside_the_region_and_keeps_the_one_outside():
    data, out = redact_directly("shading_in_region", (190, 80, 290, 140), marker=" ")
    assert op_names(out).count(b"sh") == 1
    inside, outside = (190, 80, 290, 140), (20, 20, 140, 100)
    assert ImageStat.Stat(render(data, inside)).mean[0] < 200
    assert ImageStat.Stat(render(out, inside)).mean[0] > 250
    assert ImageStat.Stat(ImageChops.difference(render(data, outside), render(out, outside))).mean[0] < 2


def test_redact_removes_no_shading_when_pdfium_and_the_stream_list_a_different_number_of_shadings():
    data = (PDFS / "shading_in_region.pdf").read_bytes().replace(b"/Sh1 sh", b"/Bad sh", 1)
    out = redacted(data, (190, 80, 290, 140))
    assert op_names(out).count(b"sh") == 2


def test_redact_keeps_a_shading_outside_the_region_and_renders_it_unchanged():
    data, out = redact_directly("shading", (190, 80, 290, 140))
    ops = [op for _, op in stream_ops(out)]
    assert ops.count(b"sh") == 1 and b"Do" not in ops
    box = (20, 20, 140, 100)
    assert ImageStat.Stat(render(out, box)).mean[0] > 20
    assert ImageStat.Stat(ImageChops.difference(render(data, box), render(out, box))).mean[0] < 2


def test_redact_removes_a_path_fully_inside_the_region_and_keeps_a_path_that_is_only_partly_covered():
    item, _ = extract("vector_paths", mode="redact", **RAW)
    ops = stream_ops(item.content.data)
    assert [list(o) for o, op in ops if op == b"re"] == [[20, 80, 40, 40], [230, 100, 60, 40], [160, 70, 30, 30]]
    assert [op for _, op in ops].count(b"f") == 2
    assert [op for _, op in ops].count(b"W") == 1


def test_redact_composite_font_keeps_narrow_table_glyphs_and_drops_default_width_glyphs_over_the_region():
    item, _ = extract("type0", mode="redact", **RAW)
    (operands,) = (o for o, op in stream_ops(item.content.data) if op == b"TJ")
    items = list(operands[0])
    assert b"".join(i for i in items if isinstance(i, bytes)) == bytes([0, 1, 0, 2, 0, 5])
    assert [i for i in items if not isinstance(i, bytes)] == [-1000, -1000]


def assert_marker_inside_region(item: KnowledgeItem, out: list[KnowledgeItem]) -> None:
    x0, y0, _, _ = out[0].metadata["bbox"]
    with Pdf.open(item.content.data) as pdf:
        page = pdf.page(0)
        texts = [page._box(*o.get_bounds()) for o in page._objects(raw.FPDF_PAGEOBJ_TEXT)]
    assert any(abs(t[0] - x0) < 2 and y0 - 1 <= t[1] and t[3] <= y0 + 9 and t[2] - t[0] > 3 * (t[3] - t[1]) for t in texts)


@pytest.mark.parametrize("name", ["rotated", "cropbox"])
def test_redact_marker_sits_inside_the_region_on_rotated_and_cropped_pages(name):
    item, out = extract(name, mode="redact", **RAW)
    assert_marker_inside_region(item, out)


@pytest.mark.parametrize("angle", [180, 270])
def test_redact_removes_text_and_images_and_places_the_marker_on_pages_rotated_180_and_270(angle):
    writer = pypdf.PdfWriter(clone_from=PDFS / "partial_text.pdf")
    writer.pages[0].rotate(angle)
    buf = io.BytesIO()
    writer.write(buf)
    data = buf.getvalue()
    item = KnowledgeItem(content=ByteContent.from_bytes(data))
    out = ImageExtractionAgent(mode="redact", **RAW)._extract_pdf(item, data)
    page = reader(item).pages[0]
    text = page.extract_text()
    assert len(out) == 2 and len(page.images) == 0
    assert all(word in text for word in ("HELLO", "ONE", "THREE")) and not any(word in text for word in ("WORLD", "TWO"))
    for image in out:
        x0, y0, x1, y1 = image.metadata["bbox"]
        assert ImageStat.Stat(render(data, (int(x0) + 2, int(y0) + 2, int(x1) - 2, int(y1) - 2))).mean[0] < 200
    assert_marker_inside_region(item, out)


def test_optional_content_membership_any_off_and_all_off_follow_the_pdf_rule():
    item, data = load("hidden_membership_off")
    out = ImageExtractionAgent(**RAW)._extract_pdf(item, data)
    drawn = [ImageStat.Stat(render(data, (x + 10, 70, x + 70, 130))).mean[0] < 250 for x in (10, 110, 210, 310)]
    assert drawn == [True, False, True, False]
    assert_boxes(boxes(out), [(1, [10, 60, 90, 140]), (1, [210, 60, 290, 140])])


def test_redact_removes_a_text_operator_in_a_font_without_widths_when_its_line_band_meets_the_region():
    data = one_page("BT /F1 6 Tf 10 150 Td (WWWWWWWWWWWWWWWWWWWWWWWW SECRETNUMBER) Tj ET")
    assert "SECRET" in " ".join(page_lines(data))
    assert page_lines(redacted(data, (150, 40, 170, 55))) == []
    assert "SECRET" in " ".join(page_lines(redacted(data, (150, 100, 170, 120))))
    assert "SECRET" in " ".join(page_lines(redacted(one_page("BT /F1 6 Tf 100 150 Td (SECRET) Tj ET"), (10, 40, 50, 55))))


def test_redact_applies_horizontal_scaling_to_the_glyph_box():
    data = one_page("BT /F1 20 Tf 200 Tz 20 100 Td (HELLO) Tj ET", widths=True)
    assert page_lines(redacted(data, (30, 85, 38, 105))) == ["ELLO"]


def test_redact_follows_text_rise():
    data = one_page("BT /F1 20 Tf 20 100 Td 20 Ts (HELLO) Tj ET", widths=True)
    assert page_lines(redacted(data, (20, 66, 120, 80))) == []
    assert page_lines(redacted(data, (20, 90, 120, 110))) == ["HELLO"]


def test_redact_uses_the_descender_below_the_baseline():
    data = one_page("BT /F1 20 Tf 20 100 Td (HELLO) Tj ET", widths=True)
    assert page_lines(redacted(data, (20, 105, 120, 106))) == []
    assert page_lines(redacted(data, (20, 107, 120, 111))) == ["HELLO"]


def test_redact_tracks_char_and_word_spacing_leading_and_line_moves_across_text_operators():
    data = one_page("BT /F1 10 Tf 5 Tc 10 Tw 12 TL 10 180 Td (AA BB) Tj 0 -30 TD (CC) Tj T* (DD) Tj (EE) ' 2 1 (FF GG) \" ET", widths=True)
    regions = {"Tc and Tw": (62, 10, 68, 25), "TD": (22, 40, 28, 55), "T*": (11, 74, 19, 82), "'": (22, 104, 28, 112), '"': (31, 134, 35, 142)}
    assert page_lines(data) == ["AA BB", "CC", "DD", "EE", "FF GG"]
    assert page_lines(redacted(data, *regions.values())) == ["AA B", "C", "D", "E", "FF G"]


def test_redact_judges_text_after_q_at_the_restored_matrix_and_after_bt_at_the_identity():
    data = one_page("q 1 0 0 1 100 0 cm BT /F1 20 Tf 10 100 Td (AB) Tj ET Q BT /F1 20 Tf 10 100 Td (CD) Tj ET", widths=True)
    assert sorted(" ".join(page_lines(redacted(data, (11, 84, 19, 106)))).split()) == ["AB", "D"]


def test_redact_survives_an_unbalanced_q():
    data = one_page("Q BT /F1 20 Tf 20 100 Td (AB) Tj 100 0 Td (CD) Tj ET", widths=True)
    assert page_lines(redacted(data, (118, 80, 142, 110))) == ["AB"]


def test_redact_survives_a_paint_operator_without_a_path():
    data = one_page("0 0 1 rg f BT /F1 20 Tf 20 100 Td (AB) Tj 100 0 Td (CD) Tj ET", widths=True)
    out = redacted(data, (118, 80, 142, 110))
    assert page_lines(out) == ["AB"] and b"f" in op_names(out)


def test_redact_survives_a_do_of_a_name_that_is_not_in_the_resources():
    data = one_page("/Nope Do BT /F1 20 Tf 20 100 Td (AB) Tj 100 0 Td (CD) Tj ET", widths=True)
    out = redacted(data, (118, 80, 142, 110))
    assert page_lines(out) == ["AB"] and b"Do" in op_names(out)


def test_redact_removes_an_inline_image_in_the_region_and_keeps_the_other():
    inline = "q 40 0 0 40 {x} 100 cm BI /W 1 /H 1 /CS /G /BPC 8 ID \x80 EI Q "
    data = one_page(inline.format(x=20) + inline.format(x=120))
    out = redacted(data, (20, 60, 60, 100), marker=" ")
    assert op_names(out).count(b"INLINE IMAGE") == 1
    assert ImageStat.Stat(render(out, (20, 60, 60, 100))).mean[0] > 250
    assert ImageStat.Stat(render(out, (120, 60, 160, 100))).mean[0] < 200


def add_annotations(writer: pypdf.PdfWriter, page) -> None:
    annots = ArrayObject()
    for rect, contents in (([30, 130, 50, 150], "INSIDE"), ([120, 20, 140, 40], "OUTSIDE")):
        annots.append(writer._add_object(DictionaryObject({
            NameObject("/Type"): NameObject("/Annot"), NameObject("/Subtype"): NameObject("/Text"),
            NameObject("/Rect"): ArrayObject([NumberObject(v) for v in rect]), NameObject("/Contents"): TextStringObject(contents)})))
    page[NameObject("/Annots")] = annots


def add_thumbnail(writer: pypdf.PdfWriter, page) -> None:
    thumb = DecodedStreamObject()
    thumb.set_data(b"\x00")
    thumb.update({NameObject("/Width"): NumberObject(1), NameObject("/Height"): NumberObject(1), NameObject("/ColorSpace"): NameObject("/DeviceGray"), NameObject("/BitsPerComponent"): NumberObject(8)})
    page[NameObject("/Thumb")] = writer._add_object(thumb)


def test_redact_drops_annotations_inside_the_region_from_the_page_and_keeps_the_others():
    out = redacted(one_page("BT /F1 20 Tf 20 100 Td (AB) Tj ET", setup=add_annotations), (20, 20, 80, 80))
    page = pypdf.PdfReader(io.BytesIO(out)).pages[0]
    assert [a.get_object()["/Contents"] for a in page["/Annots"]] == ["OUTSIDE"]


def test_redact_drops_the_page_thumbnail():
    data = one_page("BT /F1 20 Tf 20 100 Td (AB) Tj ET", setup=add_thumbnail)
    assert "/Thumb" in pypdf.PdfReader(io.BytesIO(data)).pages[0]
    assert "/Thumb" not in pypdf.PdfReader(io.BytesIO(redacted(data, (20, 20, 80, 80)))).pages[0]
