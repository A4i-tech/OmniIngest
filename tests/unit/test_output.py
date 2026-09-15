import json
import pickle
from uuid import uuid4

import fsspec
import pytest

from omni_ingest.core.model import KnowledgeItem, ResolvedResource, StepStatus
from omni_ingest.core.output import BrfFileOutput, GraphStoreOutput, JsonFileOutput, MarkdownDirOutput, MongoDocumentOutput, StdoutOutput, VectorStoreOutput, parse
from omni_ingest.core.pipeline import IngestionContext
from omni_ingest.port.graph_store import InMemoryGraphStore


def create_ctx(store, content: str) -> IngestionContext:
    return IngestionContext(tenant_id=uuid4(), resource=ResolvedResource(uri="", raw_content=content.encode()), store=store)


def test_parse_detects_mongodb_scheme():
    pytest.importorskip("pymongo")
    output = parse("mongodb://localhost:27017/db/col")
    assert isinstance(output, MongoDocumentOutput)


def test_parse_detects_mongodb_srv_scheme(monkeypatch):
    monkeypatch.setattr(MongoDocumentOutput, "__init__", lambda self, uri: None)
    assert isinstance(parse("mongodb+srv://cluster.example.net/db/col"), MongoDocumentOutput)


def test_parse_rejects_mongodb_uri_missing_collection():
    with pytest.raises(ValueError, match=r"MongoDB output URI must include database and collection"):
        parse("mongodb://localhost:27017/mydb")


def test_parse_rejects_mongodb_uri_missing_db():
    with pytest.raises(ValueError, match=r"MongoDB output URI must include database and collection"):
        parse("mongodb://localhost:27017")


def _skip_if_no_liblouis():
    try:
        import louis  # noqa: F401
    except OSError as e:
        pytest.skip(f"liblouis unavailable: {e}")


def test_parse_detects_brf_format(tmp_path):
    _skip_if_no_liblouis()
    assert isinstance(parse(str(tmp_path / "output.brf")), BrfFileOutput)


def test_parse_brf_uri_selects_table(tmp_path):
    _skip_if_no_liblouis()
    output = parse(f"brf:{tmp_path / 'output.brf'}?table=hi-in-g1.utb")
    assert isinstance(output, BrfFileOutput)
    assert output.tables[1] == "hi-in-g1.utb"


def test_parse_brf_uri_selects_table_and_display(tmp_path):
    _skip_if_no_liblouis()
    output = parse(f"brf:{tmp_path / 'output.brf'}?table=en-us-g2.ctb&display=ru-brf.dis")
    assert isinstance(output, BrfFileOutput)
    assert output.tables == ("ru-brf.dis", "en-us-g2.ctb")


@pytest.mark.asyncio
async def test_json_file_output(vector_store, tmp_path):
    path = tmp_path / "items.json"
    output = JsonFileOutput(path=str(path))
    ctx = create_ctx(vector_store, "resource")
    ctx.metadata["source"] = "unit"
    ctx.items = [KnowledgeItem(raw_content=b"test", metadata={"tag": "x"}), KnowledgeItem(raw_content=b"\xff", content_encoding="binary")]

    result = await output.write(ctx)
    data = json.loads(path.read_text())

    assert result.status == StepStatus.SUCCESS
    assert result.output_paths == {"json": str(path)}
    assert data["resource"]["content"] == "resource"
    assert data["metadata"] == {"source": "unit"}
    assert data["items"][0]["content"] == "test"
    assert data["items"][0]["metadata"] == {"tag": "x"}
    assert data["items"][1]["content"] == "/w=="
    assert data["items"][1]["content_encoding"] == "base64"


@pytest.mark.asyncio
async def test_pickle_file_output(vector_store, tmp_path):
    path = tmp_path / "items.pkl"
    ctx = create_ctx(vector_store, "resource")
    ctx.items = [KnowledgeItem(raw_content=b"\xff", content_encoding="binary")]

    result = await parse(str(path)).write(ctx)
    data = pickle.loads(path.read_bytes())

    assert result.output_paths == {"pkl": str(path)}
    assert data["resource"]["content"] == b"resource"
    assert data["items"][0]["content"] == b"\xff"


@pytest.mark.asyncio
async def test_parse_output_and_stdout(vector_store, tmp_path, capsys):
    assert isinstance(parse("-"), StdoutOutput)
    assert isinstance(parse(str(tmp_path / "items.json")), JsonFileOutput)
    assert isinstance(parse("vector:default"), VectorStoreOutput)
    assert isinstance(parse(str(tmp_path / "items.yml"), "json"), JsonFileOutput)
    ctx = create_ctx(vector_store, "resource")

    result = await parse("-").write(ctx)
    data = json.loads(capsys.readouterr().out)

    assert result.status == StepStatus.SUCCESS
    assert data["resource"]["content"] == "resource"


@pytest.mark.asyncio
async def test_json_file_output_supports_fsspec_urls(vector_store):
    ctx = create_ctx(vector_store, "resource")

    await parse("memory://omni-ingest/items.txt", "json").write(ctx)

    with fsspec.open("memory://omni-ingest/items.txt", "rt") as f:
        assert json.load(f)["resource"]["content"] == "resource"


@pytest.mark.asyncio
async def test_vector_store_output(vector_store):
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(raw_content=b"test", metadata={"embedding": [0.1] * 1536})]

    result = await VectorStoreOutput(store=vector_store).write(ctx)

    assert result.status == StepStatus.SUCCESS
    assert await vector_store.count() == 1


@pytest.mark.asyncio
async def test_markdown_dir_output_omits_page_marker_when_unpaginated(vector_store, tmp_path):
    path = tmp_path / "doc.md"
    ctx = create_ctx(vector_store, "resource")
    ctx.items = [KnowledgeItem(raw_content=b"Title", metadata={"content_type": "text/plain"})]

    result = await MarkdownDirOutput(str(path)).write(ctx)

    assert result.status == StepStatus.SUCCESS
    text = path.read_text()
    assert "page" not in text
    assert "Title" in text


@pytest.mark.asyncio
async def test_markdown_dir_output_keeps_natural_traversal_order_and_writes_sidecar(vector_store, tmp_path):
    path = tmp_path / "doc.md"
    ctx = create_ctx(vector_store, "resource")
    ctx.items = [
        KnowledgeItem(raw_content=b"second", metadata={"content_type": "text/plain", "page": 2}),
        KnowledgeItem(raw_content=b"first", metadata={"content_type": "text/plain", "page": 1}),
    ]

    result = await MarkdownDirOutput(str(path)).write(ctx)

    assert result.status == StepStatus.SUCCESS
    text = path.read_text()
    assert text.index("second") < text.index("first")
    assert "<!-- page 1 -->" in text and "<!-- page 2 -->" in text
    sidecar = json.loads((tmp_path / "doc_metadata.json").read_text())
    assert sidecar["item_count"] == 2


@pytest.mark.asyncio
async def test_markdown_dir_output_carries_accessibility_metadata_for_images(vector_store, tmp_path):
    path = tmp_path / "doc.md"
    ctx = create_ctx(vector_store, "resource")
    png_1x1 = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
        "de0000000c4944415478da6360606060000000050001a5f645400000000049454e44ae426082"
    )
    ctx.items = [KnowledgeItem(raw_content=png_1x1, metadata={
        "content_type": "image/png",
        "page": 1,
        "accessibility": {"caption": "A red dot.", "long_description": "Solid red pixel.", "confidence": 0.9},
    })]

    result = await MarkdownDirOutput(str(path)).write(ctx)

    assert result.status == StepStatus.SUCCESS
    text = path.read_text()
    assert "![A red dot.]" in text
    sidecar = json.loads((tmp_path / "doc_metadata.json").read_text())
    [image] = sidecar["images"]
    assert image["caption"] == "A red dot."
    assert image["long_description"] == "Solid red pixel."
    assert image["confidence"] == 0.9
    assert image["page"] == 1


@pytest.mark.asyncio
async def test_graph_store_output(vector_store):
    store = InMemoryGraphStore()
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(raw_content=b"test", metadata={"tags": ["tag1"]})]
    ctx.metadata["kg_data"] = {"nodes": [{"id": "n1"}], "edges": []}

    result = await GraphStoreOutput(store=store).write(ctx)

    assert result.status == StepStatus.SUCCESS
    assert store.nodes == {"n1": {"id": "n1"}}


@pytest.mark.asyncio
async def test_graph_store_output_with_edges(vector_store):
    store = InMemoryGraphStore()
    ctx = create_ctx(vector_store, "")
    ctx.items = [KnowledgeItem(raw_content=b"test", metadata={"tags": ["tag1"]})]
    ctx.metadata["kg_data"] = {"nodes": [{"id": "n1"}], "edges": [{"source": "n1", "target": "n2"}]}

    result = await GraphStoreOutput(store=store).write(ctx)

    assert result.status == StepStatus.SUCCESS
    assert store.nodes == {"n1": {"id": "n1"}}
    assert store.edges == [{"source": "n1", "target": "n2"}]
