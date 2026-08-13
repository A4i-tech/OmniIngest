from unittest.mock import patch

import pytest

from omni_ingest.core.output import MongoDocumentOutput, parse


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
