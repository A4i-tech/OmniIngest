from unittest.mock import patch

import pytest

from omni_ingest.core.output import BrfFileOutput, MongoDocumentOutput, parse


def test_parse_detects_mongodb_scheme():
    pytest.importorskip("pymongo")
    with patch.object(MongoDocumentOutput, "__init__", return_value=None):
        output = parse("mongodb://localhost:27017/db/col")
    assert isinstance(output, MongoDocumentOutput)


def test_parse_detects_mongodb_srv_scheme():
    pytest.importorskip("pymongo")
    with patch.object(MongoDocumentOutput, "__init__", return_value=None):
        output = parse("mongodb+srv://cluster.example.net/db/col")
    assert isinstance(output, MongoDocumentOutput)


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
