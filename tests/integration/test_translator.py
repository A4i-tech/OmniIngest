from typing import Any

import pytest

from omni_ingest.agent.enrichment import TranslateAgent
from omni_ingest.core.config import settings
from omni_ingest.core.model import ByteContent, KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext
from omni_ingest.port.translator import create_translator


@pytest.fixture
async def translator():
    async with create_translator(settings.default_translation_model) as translator:
        yield translator


async def test_azure_translator_translates_text(translator):
    assert await translator.translate(None, "fr", "Hello") == "Bonjour"


async def test_azure_translator_translates_nested_data(translator):
    data: dict[str, Any] = {"title": "Hello", "details": {"summary": "Good morning", "count": 2}, "empty": {}}
    result = await translator.translate("en", "fr", data)

    assert result["title"] == "Bonjour"
    assert result["details"]["summary"] != data["details"]["summary"]
    assert result["details"]["count"] == 2
    assert result["empty"] == {}


async def test_azure_translator_preserves_dotted_paths_and_lists(translator):
    data: dict[str, Any] = {"title.with.dot": "Tea & coffee <with milk>", "tags": ["Thank you"]}
    result = await translator.translate("en", "fr", data)

    assert all(character in result["title.with.dot"] for character in "<>")
    assert result["title.with.dot"] != data["title.with.dot"]
    assert result["tags"][0] != data["tags"][0]


async def test_azure_translator_skips_matching_fields(translator):
    data: dict[str, Any] = {"title": "Hello", "internal": {"label": "Do not translate"}, "notes": ["Keep this", "Translate this"]}

    result = await translator.translate("en", "fr", data, ["internal.*", "notes.0"])

    assert result["title"] == "Bonjour"
    assert result["internal"]["label"] == "Do not translate"
    assert result["notes"][0] == "Keep this"
    assert result["notes"][1] != "Translate this"


async def test_azure_translator_preserves_root_list(translator):
    result = await translator.translate("en", "fr", [{"title": "Hello"}, {"title": "Thank you"}])

    assert result == [{"title": "Bonjour"}, {"title": "Merci"}]


async def test_azure_translator_batches_large_structured_input(translator):
    data = {"first": "a" * 25_000, "second": "b" * 25_000}

    result = await translator.translate("en", "fr", data)

    assert result.keys() == data.keys()


async def test_translate_agent_translates_metadata_in_place():
    ctx = IngestionContext(resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")), metadata={"lesson": {"title": "Hello", "count": 2}})

    result = await TranslateAgent(src="en", dst="fr", input=".metadata.lesson", path=".metadata.translated").run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert ctx.metadata == {"lesson": {"title": "Hello", "count": 2}, "translated": {"title": "Bonjour", "count": 2}}


async def test_translate_agent_translates_item_text():
    ctx = IngestionContext(
        resource=ResolvedResource(uri="", content=ByteContent.from_bytes(b"")),
        items=[KnowledgeItem(content=ByteContent.from_bytes(b"The dog runs."))],
    )

    result = await TranslateAgent(src="en", dst="fr", input=".items[0].text", path=".metadata.translated").run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert "chien" in ctx.metadata["translated"].lower()
