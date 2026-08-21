import io
from unittest.mock import patch

import pytest
from docx import Document
from pypdf import PdfWriter

from omni_ingest.parser import parse_audio, parse_brf, parse_document, parse_text


@pytest.mark.parametrize(
    "raw_text, strip_html, normalize_whitespace, expected_text",
    [
        ("  Hello   world.  ", True, True, "Hello world."),
        ("<html><body><p>Test</p></body></html>", True, True, "Test"),
        ("<p>Test</p>", False, True, "<p>Test</p>"),
        ("Fallback text", True, True, "Fallback text"),
    ],
)
@pytest.mark.asyncio
async def test_text_parser(raw_text, strip_html, normalize_whitespace, expected_text):
    content, _ = await parse_text(io.BytesIO(raw_text.encode()), strip_html=strip_html, normalize_whitespace=normalize_whitespace)
    assert content == expected_text


@pytest.mark.asyncio
async def test_document_parser_pdf():
    out = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(out)
    out.seek(0)

    await parse_document(out)


@pytest.mark.asyncio
async def test_document_parser_docx():
    doc = Document()
    doc.add_paragraph("Hello from DOCX")
    out = io.BytesIO()
    doc.save(out)
    out.seek(0)

    content, _ = await parse_document(out)
    assert "Hello from DOCX" in content


@pytest.mark.asyncio
async def test_audio_parser_failure():
    with pytest.raises(RuntimeError), patch("omni_ingest.core.config.settings") as mock_settings:
        mock_settings.azure_speech_key = None
        await parse_audio("/tmp/test.wav", io.BytesIO())


@pytest.mark.asyncio
async def test_brf_parser_back_translates_grade2_braille(resources_dir):
    try:
        import louis  # noqa: F401
    except OSError as e:
        pytest.skip(f"liblouis unavailable: {e}")

    with (resources_dir / "twinky-the-dinky-dog.brf").open("rb") as f:
        content, metadata = await parse_brf(f)

    assert "twinky" in content.lower()
    assert "dinky dog" in content.lower()
    assert metadata["original_length"] > 0
    assert metadata["parsed_length"] > 0
