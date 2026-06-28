import pytest

from omni_ingest.agent.modality import TextAgent
from omni_ingest.core.model import ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner


@pytest.mark.asyncio
async def test_text_agent_converts_image_to_text(metadata_store, resources_dir):
    image_path = resources_dir / "supermarket-shelf.webp"
    ctx = IngestionContext(resource=ResolvedResource(uri=str(image_path), raw_content=image_path.read_bytes()), store=metadata_store)

    result = (await PipelineRunner([TextAgent()]).run(ctx))[0]

    text = (await ctx.items[0].decode(ctx)).lower()
    assert result.status == StepStatus.SUCCESS
    assert ctx.items[0].metadata["selected_parser"] == "image"
    assert any(word in text for word in ["shelf", "store", "supermarket", "product", "package"])
