from pathlib import Path

import pytest

from omni_ingest.core.model import ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext, create_pipeline_from_config


@pytest.fixture
def profiles_dir():
    return Path("tests/integration/profiles")


@pytest.mark.asyncio
@pytest.mark.parametrize("p", ["audio_translation_v1", "audio_translation_v2"])
async def test_audio_translation_profiles_instantiation(p, profiles_dir):
    path = profiles_dir / f"{p}.yaml"
    runner = create_pipeline_from_config(path)
    assert runner is not None
    assert len(runner.steps) > 0


@pytest.mark.asyncio
async def test_audio_translation_v1_pipeline(metadata_store, tenant, profiles_dir, resources_dir):
    path = profiles_dir / "audio_translation_v1.yaml"
    runner = create_pipeline_from_config(path)

    audio_path = resources_dir / "sample.mp3"
    with audio_path.open("rb") as stream:
        ctx = IngestionContext(
            tenant_id=tenant.id,
            store=metadata_store,
            domain_profile="accessibility",
            resource=ResolvedResource(uri=str(audio_path), raw_content=stream.read()),
        )

    results = await runner.run(ctx)
    assert results[-1].status == StepStatus.SUCCESS, f"Pipeline failed: {results[-1].error}"
    assert len(ctx.items) > 0
    assert ctx.items[0].metadata["target_language"] == "es"


@pytest.mark.asyncio
async def test_audio_translation_v2_pipeline(metadata_store, tenant, profiles_dir, resources_dir):
    path = profiles_dir / "audio_translation_v2.yaml"
    runner = create_pipeline_from_config(path)

    audio_path = resources_dir / "sample.mp3"
    with audio_path.open("rb") as stream:
        ctx = IngestionContext(
            tenant_id=tenant.id,
            store=metadata_store,
            domain_profile="accessibility",
            resource=ResolvedResource(uri=str(audio_path), raw_content=stream.read()),
        )

    results = await runner.run(ctx)
    assert results[-1].status == StepStatus.SUCCESS, f"Pipeline failed: {results[-1].error}"
    assert "translated_transcript" in ctx.metadata
