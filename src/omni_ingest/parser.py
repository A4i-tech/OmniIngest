"""Resource readers for turning local or remote inputs into resolved content."""

import asyncio
import io
import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, BinaryIO

import fsspec  # type: ignore[import-untyped]
from docx import Document
from pypdf import PdfReader

from .core.model import ResolvedResource
from .core.stt import Stt


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {"script", "style", "head", "meta", "link"}
        self.current_skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.current_skip += 1

    def handle_endtag(self, tag):
        if tag in self.skip_tags and self.current_skip > 0:
            self.current_skip -= 1

    def handle_data(self, data):
        if self.current_skip == 0 and (text := data.strip()):
            self.text_parts.append(text)

    def get_text(self) -> str:
        return " ".join(self.text_parts)


async def _parse_resource(fs: Any, path: str) -> ResolvedResource:
    with fs.open(path, "rb") as stream:
        return ResolvedResource(uri=path, raw_content=stream.read())


async def read_resources(path: str, parse_limit: int | None = None, headers: dict[str, str] | None = None, **kwargs):
    fs, resolved_path = fsspec.core.url_to_fs(path, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", **(headers or {})}, **kwargs)
    paths = fs.expand_path(resolved_path)
    for coro in asyncio.as_completed([asyncio.ensure_future(_parse_resource(fs, p)) for p in paths[:parse_limit or len(paths)]]):
        yield await coro

    if parse_limit is not None and len(paths) > parse_limit:
        for path in paths[parse_limit:]:
            yield ResolvedResource(uri=path, raw_content=b"")


@asynccontextmanager
async def as_wav(path: str, stream: BinaryIO):
    """
    Return (wav_path, is_temp). If `src` is already a WAV it is returned
    unchanged. Otherwise it is converted to a temporary WAV file; the
    caller owns that file and must delete it when done.
    """
    fmt = Path(path).suffix.lstrip(".").lower() or "mp3"
    if fmt == "wav":
        yield path
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_src, tempfile.NamedTemporaryFile(suffix=".wav") as tmp_dst:
            shutil.copyfileobj(stream, tmp_src)
            tmp_src.flush()
            tmp_src.seek(0)

            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", tmp_src.name, "-ar", "16000", "-ac", "1", "-f", "wav", tmp_dst.name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg conversion failed:\n{stderr.decode()}")
            yield tmp_dst.name


async def parse_audio(path: str, stream: BinaryIO, stt: Stt) -> tuple[str, dict[str, Any]]:
    async with as_wav(path, stream) as wav_path:
        return await stt(wav_path)


async def parse_text(stream: BinaryIO, *, encoding: str = "utf-8", strip_html: bool = True, normalize_whitespace: bool = True) -> tuple[str, dict[str, Any]]:
    content = stream.read().decode(encoding)
    if strip_html and bool(re.search(r"<[a-zA-Z][^>]*>", content[:1000])):
        p = _HTMLTextExtractor()
        try:
            p.feed(content)
            content = p.get_text()
        except Exception:
            content = re.sub(r"<[^>]+>", " ", content)
    if normalize_whitespace:
        content = re.sub(r"\s+", " ", content)
        content = "\n".join(line.strip() for line in content.split("\n") if line.strip()).strip()
    return content, {"original_length": len(content), "parsed_length": len(content)}


async def parse_document(stream: BinaryIO, *, encoding: str = "utf-8") -> tuple[str, dict[str, Any]]:
    content = stream.read()
    if content.startswith(b"%PDF"):
        return "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(content)).pages), {}
    if content.startswith(b"PK\x03\x04"):
        return "\n".join(p.text for p in Document(io.BytesIO(content)).paragraphs), {}
    return content.decode(encoding), {}


async def parse_brf(stream: BinaryIO, *, encoding: str = "utf-8", tables: tuple[str, ...] | None = None) -> tuple[str, dict[str, Any]]:
    try:
        import louis
    except OSError as e:
        raise RuntimeError("Braille IO requires liblouis which is not supported for your platform.") from e
    brf_text = stream.read().decode(encoding)
    content = str(await asyncio.to_thread(louis.backTranslateString, list(tables or ("en-us-brf.dis", "en-us-g2.ctb")), brf_text))
    return content, {"original_length": len(brf_text), "parsed_length": len(content)}
