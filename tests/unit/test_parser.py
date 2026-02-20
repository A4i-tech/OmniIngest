from pathlib import Path

import pytest

from omni_ingest.agent.parsing import TextParserAgent, UriParserAgent
from omni_ingest.core.model import IngestionContext, StepResult, StepStatus

RESOURCES_DIR = Path(__file__).parent.parent / "resources"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_text, strip_html, normalize_whitespace, expected_text, expected_status",
    [
        ("  Hello   world.  ", True, True, "Hello world.", StepStatus.SUCCESS),
        ("<html><body><p>Test</p></body></html>", True, True, "Test", StepStatus.SUCCESS),
        ("<p>Test</p>", False, True, "<p>Test</p>", StepStatus.SUCCESS),
        (None, True, True, None, StepStatus.SKIPPED),
        ("Fallback text", True, True, "Fallback text", StepStatus.SUCCESS),
    ],
)
@pytest.mark.asyncio
async def test_text_parser(raw_text, strip_html, normalize_whitespace, expected_text, expected_status):
    agent = TextParserAgent(strip_html=strip_html, normalize_whitespace=normalize_whitespace)
    metadata = {"raw_text": raw_text} if raw_text != "Fallback text" else {"text": raw_text}
    if raw_text is None:
        metadata = {}

    ctx = IngestionContext(metadata=metadata)
    result = await agent.run(ctx)

    assert result.status == expected_status
    if expected_status == StepStatus.SUCCESS:
        assert ctx.metadata["text"] == expected_text


@pytest.mark.asyncio
async def test_document_parser_pdf():
    import io

    from pypdf import PdfWriter

    from omni_ingest.agent.parsing import DocumentParserAgent

    out = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(out)
    pdf_bytes = out.getvalue()

    agent = DocumentParserAgent()
    ctx = IngestionContext(metadata={"raw_data": pdf_bytes})
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert "text" in ctx.metadata


@pytest.mark.asyncio
async def test_document_parser_docx():
    import io

    from docx import Document

    from omni_ingest.agent.parsing import DocumentParserAgent

    doc = Document()
    doc.add_paragraph("Hello from DOCX")
    out = io.BytesIO()
    doc.save(out)
    docx_bytes = out.getvalue()

    agent = DocumentParserAgent()
    ctx = IngestionContext(metadata={"raw_data": docx_bytes})
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert "Hello from DOCX" in ctx.metadata["text"]


@pytest.mark.parametrize("uri, expected_agent", [
    ("file://" + str((RESOURCES_DIR / "sample.mp3").resolve()), "audio_parser"),
    ("file://" + str((RESOURCES_DIR / "when-tomorrow-was-yesterday.pdf").resolve()), "document_parser"),
    ("data:text/plain,hello%20world", "text_parser"),
])
@pytest.mark.asyncio
async def test_uri_parser_routes_to_expected_agent(uri, expected_agent, monkeypatch):
    from unittest.mock import AsyncMock

    mock_run = AsyncMock(return_value=StepResult(status=StepStatus.SUCCESS))
    monkeypatch.setattr("omni_ingest.agent.parsing.AudioParserAgent.run", mock_run)
    monkeypatch.setattr("omni_ingest.agent.parsing.DocumentParserAgent.run", mock_run)
    monkeypatch.setattr("omni_ingest.agent.parsing.TextParserAgent.run", mock_run)

    agent = UriParserAgent()
    ctx = IngestionContext(metadata={"uri": uri})
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["selected_parser"] == expected_agent
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_uri_parser_fails_when_mime_cannot_be_determined():
    agent = UriParserAgent()
    ctx = IngestionContext(metadata={"uri": "/tmp/sample.unknownext"})
    with pytest.raises(ValueError, match="Could not determine MIME type for: /tmp/sample.unknownext"):
        await agent.run(ctx)
