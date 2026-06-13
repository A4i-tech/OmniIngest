from pathlib import Path

import pytest

from omni_ingest.agent.modality import TextAgent
from omni_ingest.core.model import IngestionContext, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import PipelineRunner

RESOURCES_DIR = Path(__file__).parent.parent / "resources"


@pytest.mark.asyncio
async def test_text_agent_converts_image_to_text(metadata_store):
    image_path = RESOURCES_DIR / "supermarket-shelf.webp"
    ctx = IngestionContext(resource=ResolvedResource(uri=str(image_path), content=image_path.read_bytes()), store=metadata_store)

    result = (await PipelineRunner([TextAgent()]).run(ctx))[0]

    text = ctx.resource.decode().lower()
    assert result.status == StepStatus.SUCCESS
    assert ctx.resource.metadata["selected_parser"] == "image"
    assert len(text) > 100
    assert any(word in text for word in ["shelf", "store", "supermarket", "product", "package"])
