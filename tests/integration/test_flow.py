from pathlib import Path

import pytest

from omni_ingest.core.config import settings
from omni_ingest.core.model import IngestionContext, StepStatus
from omni_ingest.core.pipeline import create_pipeline_from_config


@pytest.fixture
def profiles_dir():
    return Path("tests/integration/profiles")


@pytest.mark.asyncio
@pytest.mark.parametrize("p", [
    "document_v1",
    "textbook_v1",
    "audio_v1",
    "health_v1",
    "education_v2",
    "audio_v2",
    "governance_v1",
    "precision_v1",
    "noisy_v1",
    "custom_prompt_v1",
])
async def test_profiles_instantiation(p, profiles_dir):
    path = profiles_dir / f"{p}.yaml"
    runner = create_pipeline_from_config(path)
    assert runner is not None
    assert len(runner.steps) > 0


@pytest.mark.asyncio
async def test_education_v2_pipeline(metadata_store, tenant, profiles_dir):
    path = profiles_dir / "education_v2.yaml"
    runner = create_pipeline_from_config(path)
    ctx = IngestionContext(
        tenant_id=tenant.id,
        store=metadata_store,
        domain_profile="education",
        metadata={"text": "Students will learn the advanced concepts of Physics."},
    )

    results = await runner.run(ctx)
    assert results[-1].status == StepStatus.SUCCESS, f"Pipeline failed: {results[-1].error}"

    assert len(ctx.items) > 0
    assert "learning_outcomes" in ctx.items[0].metadata
    assert "summary" in ctx.items[0].metadata


@pytest.mark.asyncio
async def test_audio_v2_pipeline(metadata_store, tenant, profiles_dir, tmp_path):
    if not settings.azure_speech_key or not settings.azure_speech_region:
        pytest.skip("AZURE_SPEECH_KEY or AZURE_SPEECH_REGION not set for audio pipeline")

    import struct
    import wave

    audio_file = tmp_path / "test_audio.wav"
    with wave.open(str(audio_file), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        samples = [int(32767 * 0.5 * (1 if i % 100 < 50 else -1)) for i in range(16000)]
        wav.writeframes(struct.pack(f"{len(samples)}h", *samples))

    path = profiles_dir / "audio_v2.yaml"
    runner = create_pipeline_from_config(path)

    ctx = IngestionContext(tenant_id=tenant.id, store=metadata_store, metadata={"audio_path": str(audio_file)})

    results = await runner.run(ctx)
    assert results[-1].status == StepStatus.SUCCESS, f"Pipeline failed: {results[-1].error}"


@pytest.mark.asyncio
async def test_governance_v1_pipeline(metadata_store, tenant, profiles_dir):
    path = profiles_dir / "governance_v1.yaml"
    runner = create_pipeline_from_config(path)
    ctx = IngestionContext(tenant_id=tenant.id, store=metadata_store, metadata={"text": "Audit this file for compliance."})

    results = await runner.run(ctx)
    assert results[-1].status == StepStatus.SUCCESS, f"Pipeline failed: {results[-1].error}"

    assert any(r.metadata and "timestamp" in r.metadata for r in results)


@pytest.mark.asyncio
async def test_health_v1_pipeline(metadata_store, tenant, profiles_dir):
    path = profiles_dir / "health_v1.yaml"
    runner = create_pipeline_from_config(path)
    ctx = IngestionContext(tenant_id=tenant.id, store=metadata_store, metadata={"text": "The patient was diagnosed with type 2 diabetes."})

    results = await runner.run(ctx)
    assert results[-1].status == StepStatus.SUCCESS, f"Pipeline failed: {results[-1].error}"
    assert len(ctx.items) > 0
