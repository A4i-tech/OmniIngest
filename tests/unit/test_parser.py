import pytest

from omni_ingest.agent.parsing import TextParserAgent, UriParserAgent
from omni_ingest.core.model import IngestionContext, StepResult, StepStatus


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


@pytest.mark.asyncio
async def test_uri_parser_routes_to_document(monkeypatch):
    from unittest.mock import AsyncMock

    mock_run = AsyncMock(return_value=StepResult(status=StepStatus.SUCCESS))
    monkeypatch.setattr("omni_ingest.agent.parsing.DocumentParserAgent.run", mock_run)

    agent = UriParserAgent()
    ctx = IngestionContext(metadata={"uri": "/tmp/sample.pdf"})
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["selected_parser"] == "document_parser"
    assert ctx.metadata["file_path"] == "/tmp/sample.pdf"
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_uri_parser_routes_to_audio(monkeypatch):
    from unittest.mock import AsyncMock

    mock_run = AsyncMock(return_value=StepResult(status=StepStatus.SUCCESS))
    monkeypatch.setattr("omni_ingest.agent.parsing.AudioParserAgent.run", mock_run)

    agent = UriParserAgent()
    ctx = IngestionContext(metadata={"uri": "file:///tmp/sample.wav"})
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["selected_parser"] == "audio_parser"
    assert ctx.metadata["audio_path"] == "/tmp/sample.wav"
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_uri_parser_routes_data_text(monkeypatch):
    from unittest.mock import AsyncMock

    mock_run = AsyncMock(return_value=StepResult(status=StepStatus.SUCCESS))
    monkeypatch.setattr("omni_ingest.agent.parsing.TextParserAgent.run", mock_run)

    agent = UriParserAgent()
    ctx = IngestionContext(metadata={"uri": "data:text/plain,hello%20world"})
    result = await agent.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert result.metadata["selected_parser"] == "text_parser"
    assert ctx.metadata["raw_text"] == "hello world"
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_uri_parser_skips_when_mime_cannot_be_determined():
    agent = UriParserAgent()
    ctx = IngestionContext(metadata={"uri": "/tmp/sample.unknownext"})
    result = await agent.run(ctx)

    assert result.status == StepStatus.SKIPPED
    assert result.error is not None and "Could not determine MIME type" in result.error
