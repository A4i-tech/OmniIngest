from pathlib import Path

import pytest

from omni_ingest.agent.chunking import TimedChunkingAgent
from omni_ingest.core.model import ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext
from omni_ingest.parser import read_resources


@pytest.mark.asyncio
async def test_timed_chunking_from_word_timestamps():
    ctx = IngestionContext(
        resource=ResolvedResource(uri="", raw_content=b""),
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
    assert await result.items[0].decode(ctx) == "Hello world"
    assert result.items[0].metadata["start_offset_ms"] == 1000
    assert await result.items[1].decode(ctx) == "this is audio"
    assert result.items[1].metadata["start_offset_ms"] == 40000


@pytest.mark.asyncio
async def test_audio_parser_with_real_azure(tmp_path):
    import struct
    import wave

    audio_file = tmp_path / "test_audio.wav"
    with wave.open(str(audio_file), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        samples = [int(32767 * 0.5 * (1 if i % 100 < 50 else -1)) for i in range(16000)]
        wav.writeframes(struct.pack(f"{len(samples)}h", *samples))

    assert len([resource async for resource in read_resources(str(audio_file))]) == 1


@pytest.mark.asyncio
async def test_audio_parser_with_mp3():
    audio_file = Path(__file__).parent.parent / "resources" / "sample.mp3"
    resources = [resource async for resource in read_resources(str(audio_file))]

    assert len(resources) == 1
    ctx = IngestionContext(resource=resources[0])
    assert (await resources[0].content_type(ctx)).startswith("audio/")
