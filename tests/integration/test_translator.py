from typing import Any

import pytest

from omni_ingest.agent.enrichment import TranslateAgent
from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.pipeline import IngestionContext
from omni_ingest.port.translator import create_translator


# Sarvam only translates between Indian-locale codes (en-IN, hi-IN, ...), not
# arbitrary language pairs like en->fr, so it can't share these en->fr assertions.
# It gets its own tests below, with real Indic codes.
@pytest.fixture(params=["azure", "openai"])
async def translator(request):
    async with create_translator(request.param) as translator:
        yield translator


# Azure routes structured data through the HTML round-trip in _translate_html,
# so batching and markup fidelity are meaningful for it. OpenAI's bulk path sends
# one structured-output call with no batching.
@pytest.fixture(params=["azure"])
async def html_translator(request):
    async with create_translator(request.param) as translator:
        yield translator


@pytest.fixture
async def sarvam_translator():
    async with create_translator("sarvam") as translator:
        yield translator


async def test_translator_translates_text(translator):
    assert await translator.translate(None, "fr", "Hello") == "Bonjour"


async def test_translator_translates_nested_data(translator):
    data: dict[str, Any] = {"title": "Hello", "details": {"summary": "Good morning", "count": 2}, "empty": {}}
    result = await translator.translate("en", "fr", data)

    assert result["title"] == "Bonjour"
    assert result["details"]["summary"] != data["details"]["summary"]
    assert result["details"]["count"] == 2
    assert result["empty"] == {}


async def test_translator_preserves_dotted_paths_and_lists(html_translator):
    data: dict[str, Any] = {"title.with.dot": "Tea & coffee <with milk>", "tags": ["Thank you"]}
    result = await html_translator.translate("en", "fr", data)

    assert all(character in result["title.with.dot"] for character in "<>")
    assert result["title.with.dot"] != data["title.with.dot"]
    assert result["tags"][0] != data["tags"][0]


async def test_translator_skips_matching_fields(translator):
    data: dict[str, Any] = {"title": "Hello", "internal": {"label": "Do not translate"}, "notes": ["Keep this", "Translate this"]}

    result = await translator.translate("en", "fr", data, ["internal.*", "notes.0"])

    assert result["title"] == "Bonjour"
    assert result["internal"]["label"] == "Do not translate"
    assert result["notes"][0] == "Keep this"
    assert result["notes"][1] != "Translate this"


async def test_translator_preserves_root_list(translator):
    result = await translator.translate("en", "fr", [{"title": "Hello"}, {"title": "Thank you"}])

    assert result == [{"title": "Bonjour"}, {"title": "Merci"}]


async def test_translator_batches_large_structured_input(html_translator):
    data = {"first": "a" * 25_000, "second": "b" * 25_000}

    result = await html_translator.translate("en", "fr", data)

    assert result.keys() == data.keys()


async def test_sarvam_translator_translates_text(sarvam_translator):
    result = await sarvam_translator.translate("en-IN", "hi-IN", "Good morning")

    assert result != "Good morning"
    assert result


async def test_sarvam_translator_translates_nested_data(sarvam_translator):
    data: dict[str, Any] = {"title": "Hello", "details": {"summary": "Good morning", "count": 2}, "empty": {}}
    result = await sarvam_translator.translate("en-IN", "hi-IN", data)

    assert result["title"] != data["title"]
    assert result["details"]["summary"] != data["details"]["summary"]
    assert result["details"]["count"] == 2
    assert result["empty"] == {}


async def test_sarvam_translator_skips_matching_fields(sarvam_translator):
    data: dict[str, Any] = {"title": "Hello", "internal": {"label": "Do not translate"}, "notes": ["Keep this", "Translate this"]}

    result = await sarvam_translator.translate("en-IN", "hi-IN", data, ["internal.*", "notes.0"])

    assert result["title"] != data["title"]
    assert result["internal"]["label"] == "Do not translate"
    assert result["notes"][0] == "Keep this"
    assert result["notes"][1] != "Translate this"


async def test_sarvam_translator_preserves_root_list(sarvam_translator):
    result = await sarvam_translator.translate("en-IN", "hi-IN", [{"title": "Hello"}, {"title": "Thank you"}])

    assert len(result) == 2
    assert result[0]["title"] != "Hello"
    assert result[1]["title"] != "Thank you"


async def test_translate_agent_translates_metadata_in_place():
    ctx = IngestionContext(resource=ResolvedResource(uri="", raw_content=b""), metadata={"lesson": {"title": "Hello", "count": 2}})

    result = await TranslateAgent(src="en", dst="fr", input=".metadata.lesson", path=".metadata.translated").run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert ctx.metadata == {"lesson": {"title": "Hello", "count": 2}, "translated": {"title": "Bonjour", "count": 2}}


async def test_translate_agent_translates_item_text():
    ctx = IngestionContext(
        resource=ResolvedResource(uri="", raw_content=b""),
        items=[KnowledgeItem(raw_content=b"The dog runs.")],
    )

    result = await TranslateAgent(src="en", dst="fr", input=".items[0].text", path=".metadata.translated").run(ctx)

    assert result.status == StepStatus.SUCCESS
    assert "chien" in ctx.metadata["translated"].lower()
