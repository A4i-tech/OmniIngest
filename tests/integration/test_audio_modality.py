from pathlib import Path

import pytest

from omni_ingest.agent.chunking import TimedChunkingAgent
from omni_ingest.agent.parsing import AudioParserAgent
from omni_ingest.core.model import IngestionContext, StepStatus
from omni_ingest.core.pipeline import PipelineRunner


@pytest.mark.asyncio
async def test_timed_chunking_from_word_timestamps():
    ctx = IngestionContext(
        metadata={
            "word_timestamps": [
                {"Word": "Hello", "Offset": 10000000, "Duration": 2000000},
                {"Word": "world", "Offset": 15000000, "Duration": 2000000},
                {"Word": "this", "Offset": 400000000, "Duration": 2000000},
                {"Word": "is", "Offset": 410000000, "Duration": 2000000},
                {"Word": "audio", "Offset": 420000000, "Duration": 2000000},
            ]
        }
    )

    chunker = TimedChunkingAgent(segment_duration="00:00:10")
    result = await chunker.run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert len(result.items) == 2
    assert result.items[0].content == "Hello world"
    assert result.items[0].metadata["start_offset_ms"] == 1000
    assert result.items[1].content == "this is audio"
    assert result.items[1].metadata["start_offset_ms"] == 40000


@pytest.mark.asyncio
async def test_audio_parser_with_real_azure(tmp_path, metadata_store):
    import struct
    import wave

    audio_file = tmp_path / "test_audio.wav"
    with wave.open(str(audio_file), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        samples = [int(32767 * 0.5 * (1 if i % 100 < 50 else -1)) for i in range(16000)]
        wav.writeframes(struct.pack(f"{len(samples)}h", *samples))

    agent = AudioParserAgent()
    ctx = IngestionContext(metadata={"audio_path": str(audio_file)}, store=metadata_store)

    res = await PipelineRunner([agent]).run(ctx)
    assert res[0].status == StepStatus.SUCCESS


@pytest.mark.asyncio
async def test_audio_parser_with_mp3(metadata_store):
    audio_file = Path(__file__).parent.parent / "resources" / "audio.mp3"

    agent = AudioParserAgent()
    ctx = IngestionContext(metadata={"audio_path": str(audio_file)}, store=metadata_store)

    res = await PipelineRunner([agent]).run(ctx)
    assert res[0].status == StepStatus.SUCCESS
    assert "interview training company" in ctx.metadata["text"]
