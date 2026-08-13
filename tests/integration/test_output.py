from unittest.mock import patch

import pytest

from omni_ingest.core.output import parse


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
