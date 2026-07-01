"""Unit tests for parse() dispatch across output formats."""

from unittest.mock import patch

import pytest

from omni_ingest.core.output import MongoDocumentOutput, parse


# ---------------------------------------------------------------------------
# URI validation — tested through the public parse() API
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# parse() scheme dispatch
# ---------------------------------------------------------------------------


def test_parse_detects_mongodb_scheme():
    output = parse("mongodb://localhost:27017/db/col")
    assert isinstance(output, MongoDocumentOutput)


def test_parse_detects_mongodb_srv_scheme():
    # mongodb+srv:// triggers real DNS SRV lookup on client creation; patch to avoid network call
    with patch("omni_ingest.core.output.MongoDocumentOutput") as mock_cls:
        parse("mongodb+srv://cluster.example.net/db/col")
        mock_cls.assert_called_once_with("mongodb+srv://cluster.example.net/db/col")
