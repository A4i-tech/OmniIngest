import pytest

from omni_ingest.agent.parsing import TextParserAgent
from omni_ingest.core.model import IngestionContext, StepStatus


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
