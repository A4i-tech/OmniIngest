from pathlib import Path

import io
import pytest
from unittest.mock import AsyncMock, patch
from docx import Document
from pypdf import PdfWriter

from omni_ingest.parser import read_resources, parse, parse_document, parse_text

RESOURCES_DIR = Path(__file__).parent.parent / "resources"


@pytest.mark.asyncio
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
    content, _ = await parse_text(io.BytesIO(raw_text.encode("utf-8")), strip_html=strip_html, normalize_whitespace=normalize_whitespace)
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
        await parse("/tmp/test.wav", io.BytesIO())


@pytest.mark.parametrize("uri, expected_agent", [
    ("file://" + str((RESOURCES_DIR / "sample.mp3").resolve()), "audio"),
    ("file://" + str((RESOURCES_DIR / "when-tomorrow-was-yesterday.pdf").resolve()), "document"),
    ("data:text/plain,hello%20world", "text"),
])
@pytest.mark.asyncio
async def test_parser_routes_to_expected_agent(uri, expected_agent, monkeypatch):
    mock_run = AsyncMock(return_value=("", {}))
    monkeypatch.setattr("omni_ingest.parser.parse_audio", mock_run)
    monkeypatch.setattr("omni_ingest.parser.parse_document", mock_run)
    monkeypatch.setattr("omni_ingest.parser.parse_text", mock_run)

    async for resource in read_resources(uri):
        assert resource.metadata["selected_parser"] == expected_agent


@pytest.mark.asyncio
async def test_parser_fails_when_mime_cannot_be_determined():
    with pytest.raises(ValueError, match="Unsupported data URI MIME type: 'font/ttf'"):
        await parse("/tmp/sample.ttf", io.BytesIO())