import asyncio
import io
import json
import mimetypes
import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, BinaryIO, Literal

import azure.cognitiveservices.speech as speechsdk
import fsspec  # type: ignore[import-untyped]
from docx import Document
from pypdf import PdfReader

from .core.config import settings
from .core.model import ResolvedResource


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
        return ResolvedResource(uri=path, content=stream.read())


async def read_resources(path: str, parse_limit: int | None = None, **kwargs):
    fs, resolved_path = fsspec.core.url_to_fs(path, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, **kwargs)
    paths = fs.expand_path(resolved_path)
    for coro in asyncio.as_completed(asyncio.ensure_future(_parse_resource(fs, p)) for p in paths[:parse_limit or len(paths)]):
        yield await coro

    if parse_limit is not None and len(paths) > parse_limit:
        for path in paths[parse_limit:]:
            yield ResolvedResource(uri=path, content=b"")


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


async def parse_audio(path: str, stream: BinaryIO) -> tuple[str, dict[str, Any]]:
    key = settings.azure_speech_key.get_secret_value() if settings.azure_speech_key else None
    region = settings.azure_speech_region
    if not key or not region:
        raise RuntimeError("Azure Speech credentials missing")

    loop = asyncio.get_running_loop()
    done, transcript, words = asyncio.Future[Literal[True]](), [], []

    def handle_final(evt):
        nonlocal transcript, words
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            transcript.append(evt.result.text)
            obj = json.loads(evt.result.json)
            if "NBest" in obj and obj["NBest"]:
                words.extend(obj["NBest"][0].get("Words", []))

    def stop_cb(evt):
        nonlocal done
        if not done.done():
            loop.call_soon_threadsafe(done.set_result, True)

    def cancel_cb(evt):
        nonlocal done
        if evt.result and evt.result.cancellation_details and evt.result.cancellation_details.reason != speechsdk.CancellationReason.EndOfStream:
            cancellation = evt.result.cancellation_details
            details = cancellation.error_details if cancellation.error_details else str(cancellation.reason)
            loop.call_soon_threadsafe(done.set_exception, RuntimeError(details))
        elif not done.done():
            loop.call_soon_threadsafe(done.set_result, True)

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    async with as_wav(path, stream) as wav_path:
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_RequestWordLevelTimestamps, "true")
        recognizer.recognized.connect(handle_final)
        recognizer.session_stopped.connect(stop_cb)
        recognizer.canceled.connect(cancel_cb)

        recognizer.start_continuous_recognition()
        try:
            await done
        finally:
            recognizer.stop_continuous_recognition()

    full_text = " ".join(transcript)
    return " ".join(transcript), {"text_len": len(full_text), "word_count": len(words)}


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


async def parse(path: str, stream: BinaryIO, *, encoding: str = "utf-8") -> tuple[str, dict[str, Any]]:
    content_type, _ = mimetypes.guess_type(path)
    content_type = content_type or "application/octet-stream"

    if content_type.startswith("audio/"):
        content, metadata = await parse_audio(path, stream)
        metadata["selected_parser"] = "audio"
        return content, metadata
    if content_type.startswith("text/"):
        content, metadata = await parse_text(stream, encoding=encoding)
        metadata["selected_parser"] = "text"
        return content, metadata
    if content_type.startswith("application/"):
        content, metadata = await parse_document(stream, encoding=encoding)
        metadata["selected_parser"] = "document"
        return content, metadata

    raise ValueError(f"Unsupported data URI MIME type: {content_type!r}")
