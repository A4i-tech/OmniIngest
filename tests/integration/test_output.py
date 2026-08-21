from unittest.mock import patch

import pytest

from omni_ingest.agent.modality import TextAgent
from omni_ingest.core.model import ResolvedResource, StepStatus
from omni_ingest.core.output import BrfFileOutput, parse
from omni_ingest.core.pipeline import IngestionContext, PipelineRunner


def test_parse_rejects_mongodb_uri_missing_collection():
    with pytest.raises(ValueError, match=r"MongoDB output URI must include database and collection"):
        parse("mongodb://localhost:27017/mydb")


def test_parse_rejects_mongodb_uri_missing_db():
    with pytest.raises(ValueError, match=r"MongoDB output URI must include database and collection"):
        parse("mongodb://localhost:27017")


def test_parse_raises_import_error_when_pymongo_missing():
    with patch.dict("sys.modules", {"pymongo": None, "pymongo.uri_parser": None}):
        with pytest.raises(ImportError, match=r"MongoDocumentOutput requires pymongo"):
            parse("mongodb://localhost:27017/db/col")


@pytest.mark.asyncio
async def test_brf_output_round_trips_through_pipeline(metadata_store, tmp_path):
    try:
        import louis
    except OSError as e:
        pytest.skip(f"liblouis unavailable: {e}")

    ctx = IngestionContext(store=metadata_store, resource=ResolvedResource(uri="", raw_content=b"the dog sat"))
    results = await PipelineRunner([TextAgent()]).run(ctx)
    assert results[-1].status == StepStatus.SUCCESS

    for item in ctx.items:
        item.raw_content = (await item.content(ctx)).replace(b"dog", b"cat")

    out_path = tmp_path / "output.brf"
    output = BrfFileOutput(str(out_path))
    result = await output.write(ctx)
    assert result.status == StepStatus.SUCCESS

    text = str(louis.backTranslateString(list(output.tables), out_path.read_text().lower())).lower()
    assert "dog" not in text
    assert "cat" in text
