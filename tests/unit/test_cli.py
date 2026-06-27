import io
import sys
from types import SimpleNamespace

from omni_ingest.cli import _read_stdin


def test_read_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"stdin content")))

    resource = _read_stdin("notes.md")

    assert resource.uri == "notes.md"
    assert resource.raw_content == b"stdin content"


def test_read_stdin_uses_virtual_uri_without_name(monkeypatch):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"stdin content")))

    resource = _read_stdin(None)

    assert resource.uri == "stdin://"
