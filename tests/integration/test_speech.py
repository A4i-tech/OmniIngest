import pytest

from omni_ingest.agent.enrichment import SpeechAgent
from omni_ingest.core.model import StepStatus
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner
from omni_ingest.core.tts import SpeechFormat


@pytest.mark.parametrize("output_format, mime_type", [
    (SpeechFormat.MP3, "audio/mpeg"),
    (SpeechFormat.WAV, "audio/wav"),
])
@pytest.mark.asyncio
async def test_speech_agent(output_format, mime_type, metadata_store, resources_dir):
    ctx = IngestionContext(store=metadata_store, resource=resources_dir / "blueberries-are-now-green.txt")
    results = await PipelineRunner([SpeechAgent(format=output_format)]).run(ctx)
    assert all(r.status == StepStatus.SUCCESS for r in results)

    item = results[-1].items[0]
    assert await item.content_type(ctx) == mime_type
    assert len(await item.content(ctx)) > 0
