from unittest.mock import patch

import pytest

from omni_ingest.agent.modality import TextAgent
from omni_ingest.core.model import StepStatus
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
async def test_brf_output_round_trips_through_pipeline(metadata_store, tmp_path, resources_dir):
    try:
        import louis
    except OSError as e:
        pytest.skip(f"liblouis unavailable: {e}")

    ctx = IngestionContext(store=metadata_store, resource=resources_dir / "twinky-the-dinky-dog.brf")
    results = await PipelineRunner([TextAgent()]).run(ctx)
    assert results[-1].status == StepStatus.SUCCESS

    out_path = tmp_path / "output.brf"
    output = BrfFileOutput(str(out_path))
    result = await output.write(ctx)
    assert result.status == StepStatus.SUCCESS

    # Raw file on disk should hold braille cells, not literal print text.
    raw = out_path.read_text()
    assert "Twinky" not in raw

    text = louis.backTranslateString(list(output.tables), raw)
    assert "In his heart Twinky knew he was a big dog." in text
