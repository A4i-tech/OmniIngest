import base64
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import pytest
from pydantic import SecretStr

from omni_ingest.agent.document import OcrAgent
from omni_ingest.core.model import ByteContent, KnowledgeItem
from omni_ingest.core.ocr import OcrBuilderParams, OcrOutputFormat
from omni_ingest.core.pipeline import IngestionContext
from omni_ingest.port.metadata_store import NullMetadataStore
from omni_ingest.port.ocr import default_factory


def _image(name: str, data: bytes):
    uri = f"data:image/png;base64,{base64.b64encode(data).decode()}"
    return SimpleNamespace(id=name, top_left_x=1, top_left_y=2, bottom_right_x=3, bottom_right_y=4, image_base64=uri)


def _page(index: int, markdown: str, images):
    return SimpleNamespace(index=index, markdown=markdown, images=images, dimensions=SimpleNamespace(model_dump=lambda: {"dpi": 110, "height": 10, "width": 20}), header="head", footer="foot")


class FakeClient:
    def __init__(self, pages):
        self.pages, self.calls = pages, []
        self.ocr = SimpleNamespace(process_async=self._process)

    async def _process(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(pages=self.pages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


def _ctx():
    ctx = IngestionContext(store=NullMetadataStore(), resource="x")
    ctx.items = [KnowledgeItem(content=ByteContent.from_bytes(b"pdf", "application/pdf"), metadata={"page": 7})]
    ctx.ocr_factory = default_factory()
    return ctx


async def _run(pages, **step_options):
    client = FakeClient(pages)
    ctx = _ctx()
    with patch("omni_ingest.port.ocr.settings") as s, patch("omni_ingest.port.ocr.Mistral", lambda **_: client):
        s.mistral_ocr_api_key, s.mistral_ocr_model = SecretStr("k"), "m"
        result = await OcrAgent(engine="mistral", extract_images=True, **step_options).run(ctx)
    return ctx, result, client


async def test_figures_become_items_with_markers():
    pages = [_page(0, "a ![img-0.jpeg](img-0.jpeg)", [_image("img-0.jpeg", b"one")]),
             _page(1, "b ![img-0.jpeg](img-0.jpeg) ![img-1.jpeg](img-1.jpeg)", [_image("img-0.jpeg", b"two"), _image("img-1.jpeg", b"three")])]
    ctx, _, client = await _run(pages, image_min_size=40, extract_header_footer=True)

    parent, *figures = ctx.items
    assert len(figures) == 3
    text = parent.content.data.decode()
    for figure in figures:
        assert f'<image id="{figure.id}"/>' in text
        assert figure.metadata["kind"] == "image" and figure.metadata["parent_id"] == str(parent.id)
        assert figure.metadata["page"] == 7 and figure.metadata["bbox"] == [1, 2, 3, 4]
        assert figure.content.metadata["content_type"] == "image/png"
    assert [f.content.data for f in figures] == [b"one", b"two", b"three"]
    assert client.calls[0]["include_image_base64"] and client.calls[0]["image_min_size"] == 40 and client.calls[0]["extract_header"]
    assert parent.metadata["ocr_header"] == "head\n\nhead" and isinstance(figures[0].id, UUID)


@pytest.mark.parametrize(("deduplicate", "expected"), [(True, 1), (False, 2)])
async def test_deduplicate_identical_figures(deduplicate, expected):
    pages = [_page(i, "![img-0.jpeg](img-0.jpeg)", [_image("img-0.jpeg", b"logo")]) for i in range(2)]
    ctx, _, _ = await _run(pages, deduplicate=deduplicate)

    parent, *figures = ctx.items
    assert len(figures) == expected
    assert parent.content.data.decode().count(f'<image id="{figures[0].id}"/>') == (2 if deduplicate else 1)


@pytest.mark.parametrize("engine", ["sarvam", "azure_di", "llm"])
def test_unsupported_engine_rejects_extract_images(engine):
    ctx = IngestionContext(store=NullMetadataStore(), resource="x")
    with pytest.raises(ValueError, match=f"'{engine}' does not support the 'extract_images'"):
        default_factory()(engine, OcrBuilderParams(ctx=ctx, format=OcrOutputFormat.MARKDOWN, concurrency=1, extract_images=True))


def test_azure_mistral_requires_endpoint():
    ctx = IngestionContext(store=NullMetadataStore(), resource="x")
    with pytest.raises(ValueError, match="MISTRAL_OCR_ENDPOINT"), patch("omni_ingest.port.ocr.settings") as s:
        s.mistral_ocr_api_key, s.mistral_ocr_endpoint = SecretStr("k"), None
        default_factory()("azure_mistral", OcrBuilderParams(ctx=ctx, format=OcrOutputFormat.MARKDOWN, concurrency=1))


async def test_azure_mistral_builds_client_with_endpoint_and_forwards_options():
    client, built = FakeClient([_page(0, "a ![img-0.jpeg](img-0.jpeg)", [_image("img-0.jpeg", b"one")])]), {}

    def make(**kwargs):
        built.update(kwargs)
        return client

    ctx = _ctx()
    with patch("omni_ingest.port.ocr.settings") as s, patch("omni_ingest.port.ocr.MistralAzure", make):
        s.mistral_ocr_api_key, s.mistral_ocr_endpoint, s.mistral_ocr_model = SecretStr("k"), "https://r.services.ai.azure.com/", "m"
        await OcrAgent(engine="azure_mistral", extract_images=True, image_min_size=64).run(ctx)

    assert (built["api_key"], built["server_url"]) == ("k", "https://r.services.ai.azure.com/")
    assert client.calls[0]["include_image_base64"] is True
    assert client.calls[0]["image_min_size"] == 64
