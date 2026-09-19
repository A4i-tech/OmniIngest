import io
import sys
from types import SimpleNamespace

from omni_ingest.cli import _parse_args, _read_stdin


def test_parse_nullable_string_parameter(monkeypatch, tmp_path):
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("parameters:\n  properties:\n    translate_to:\n      type: [string, null]\n      default: null\nsteps: []\n")
    monkeypatch.setattr(sys, "argv", ["omni-ingest", str(pipeline), "--input", "input.txt", "--output", "-", "--translate-to", "te"])

    assert _parse_args().params["translate_to"] == "te"


def test_read_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"stdin content")))

    resource = _read_stdin("notes.md")

    assert resource.uri == "notes.md"
    assert resource.content.data == b"stdin content"


def test_read_stdin_uses_virtual_uri_without_name(monkeypatch):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"stdin content")))

    assert _read_stdin(None).uri == "stdin://"
