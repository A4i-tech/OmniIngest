import html5lib
import pytest

from omni_ingest.agent.chunking import PageChunkingAgent
from omni_ingest.agent.document import OcrAgent
from omni_ingest.core.model import StepStatus
from omni_ingest.core.ocr import OcrOutputFormat
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner


@pytest.mark.parametrize(("engine", "out_fmt"), [
    # for 3p vendors, testing a structured format (e.g., HTML) and an unstructured
    # format (e.g., markdown) is sufficient.
    ("azure_di", OcrOutputFormat.MARKDOWN),  # azure di supports no structured format
    ("mistral_ocr", OcrOutputFormat.MARKDOWN),  # mistral ocr supports no structured format
    ("llm", OcrOutputFormat.HTML),
    ("llm", OcrOutputFormat.MARKDOWN),
    ("llm", OcrOutputFormat.TEXT),
    ("sarvam", OcrOutputFormat.JSON),
    ("sarvam", OcrOutputFormat.MARKDOWN),
])
async def test_ocr_engine(engine, out_fmt, metadata_store, resources_dir):
    ctx = IngestionContext(store=metadata_store, resource=resources_dir / "when-tomorrow-was-yesterday.pdf")
    results = await PipelineRunner([
        PageChunkingAgent(mode="pdf"),
        OcrAgent(engine=engine, format=out_fmt, foreach=".items[2]")
    ]).run(ctx)
    assert all(r.status == StepStatus.SUCCESS for r in results)

    item = results[-1].items[2]
    content = await item.content(ctx)
    assert await item.content_type(ctx) == out_fmt.mime_type
    assert "Theoretical Foundation" in content.decode()
    assert "shared commitment to the reversal." in content.decode()

    if out_fmt == OcrOutputFormat.HTML:
        try:
            html5lib.HTMLParser(strict=True).parse(content)
        except html5lib.html5parser.ParseError as error:
            error.add_note(f"OCR content:\n{content}")
            raise
