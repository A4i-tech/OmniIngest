from types import SimpleNamespace

import pytest

from omni_ingest.port.ocr import _figures


class FakeClient:
    def __init__(self, chunks: dict[str, list[bytes] | Exception]):
        self.chunks = chunks

    async def get_analyze_result_figure(self, model_id, result_id, figure_id):
        outcome = self.chunks[figure_id]
        if isinstance(outcome, Exception):
            raise outcome

        async def stream():
            for chunk in outcome:
                yield chunk
        return stream()


def make_figure(figure_id, page=1, polygon=None, caption=None):
    region = SimpleNamespace(page_number=page, polygon=polygon) if page is not None else None
    return SimpleNamespace(
        id=figure_id,
        bounding_regions=[region] if region else None,
        caption=SimpleNamespace(content=caption) if caption else None,
    )


@pytest.mark.asyncio
async def test_figures_returns_all_as_failed_when_result_id_missing():
    result = SimpleNamespace(model_id="m", figures=[make_figure("f1"), make_figure("f2")])

    collected, failed = await _figures(FakeClient({}), result, None)

    assert collected == []
    assert failed == 2


@pytest.mark.asyncio
async def test_figures_counts_missing_figure_id_as_failed():
    result = SimpleNamespace(model_id="m", figures=[make_figure(None)])

    collected, failed = await _figures(FakeClient({}), result, "result-1")

    assert collected == []
    assert failed == 1


@pytest.mark.asyncio
async def test_figures_counts_fetch_exception_as_failed():
    result = SimpleNamespace(model_id="m", figures=[make_figure("f1")])
    client = FakeClient({"f1": RuntimeError("boom")})

    collected, failed = await _figures(client, result, "result-1")

    assert collected == []
    assert failed == 1


@pytest.mark.asyncio
async def test_figures_collects_successful_crop_with_metadata():
    result = SimpleNamespace(model_id="m", figures=[make_figure("f1", page=3, polygon=[1, 2, 3, 4], caption="a chart")])
    client = FakeClient({"f1": [b"\x89PNG", b"...rest"]})

    collected, failed = await _figures(client, result, "result-1")

    assert failed == 0
    assert len(collected) == 1
    figure = collected[0]
    assert figure.caption == "a chart"
    assert figure.content.data == b"\x89PNG...rest"
    assert figure.content.metadata == {"content_type": "image/png", "id": "f1", "page": 3, "polygon": [1, 2, 3, 4]}
