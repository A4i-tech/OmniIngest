"""Agents for extracting text from various document formats and audio streams."""

import asyncio
import io
import json
import mimetypes
import re
import tempfile
import urllib.request
from contextlib import asynccontextmanager
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

import aiohttp
import azure.cognitiveservices.speech as speechsdk
from docx import Document
from pydantic import BaseModel, Field
from pypdf import PdfReader

from ..core.config import settings
from ..core.model import IngestionContext, Step, StepResult, StepStatus


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


def _resolve_mime(uri: str) -> str:
    parsed = urlparse(uri)
    path = unquote(parsed.path) if parsed.path else uri
    return (mimetypes.guess_type(path)[0] or "").lower()


def _mime_category(mime: str) -> Literal["audio", "document"]:
    major = mime.split("/", 1)[0].lower() if mime else ""
    if major == "audio":
        return "audio"
    if major in {"text", "application"}:
        return "document"
    raise ValueError("Unsupported mime type:" + mime)


class UriParserAgent(BaseModel, Step):
    """
    Routes URI input to the appropriate parser agent.

    Inspects `ctx.metadata["uri"]`, resolves MIME type, and delegates to:
      - TextParserAgent     for `data:text/*` URIs
      - DocumentParserAgent for document/text MIME types
      - AudioParserAgent    for audio MIME types
    """

    @staticmethod
    @asynccontextmanager
    async def _resolve(uri: str, chunk_size: int = 65536):
        mime = _resolve_mime(uri)
        if not mime:
            raise ValueError(f"Could not determine MIME type for: {uri}")

        parsed = urlparse(uri)
        scheme = parsed.scheme.lower()
        path = unquote(parsed.path)
        if scheme in {"", "file"} and Path(path).is_file():  # for local fs URIs, just return local path directly
            yield mime, path
            return

        # for other URIs, read into local tmp file first
        with tempfile.NamedTemporaryFile(suffix=Path(path).suffix) as tmp:
            if scheme in {"http", "https"}:  # aiohttp to asynchronously read HTTP/HTTPs URIs
                async with aiohttp.ClientSession() as session, session.get(uri) as response:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        tmp.write(chunk)
            else:
                # urllib.parse supports several other URIs like ftp, sftp, wss, rsync, etc.
                # the only drawback is urllib.parse is blocking, so we have to run it in an executor
                # TODO: figure out whether we actually need to spin up a CPU thread here?
                def _transfer():
                    with urllib.request.urlopen(uri) as f:
                        while chunk := f.read(chunk_size):
                            tmp.write(chunk)
                await asyncio.get_running_loop().run_in_executor(None, _transfer)

            tmp.flush()
            tmp.seek(0)
            yield mime, tmp.name


    async def run(self, ctx: IngestionContext) -> StepResult:
        uri = ctx.metadata.get("uri")
        if not uri:
            return StepResult(status=StepStatus.SKIPPED, error="No uri")

        if uri.lower().startswith("data:"):
            header, _, payload = uri[5:].partition(",")
            mime = _resolve_mime(uri) or header.split(";", 1)[0].strip().lower()
            if _mime_category(mime) != "document":
                return StepResult(status=StepStatus.SKIPPED, error=f"Unsupported data URI MIME type: {mime!r}")
            ctx.metadata["raw_text"] = unquote(payload)
            result = await TextParserAgent().run(ctx)
            result.metadata["selected_parser"] = "text_parser"
            return result

        async with UriParserAgent._resolve(uri) as (mime, local_path):
            match _mime_category(mime):
                case "audio":
                    ctx.metadata["audio_path"] = local_path
                    result = await AudioParserAgent().run(ctx)
                    result.metadata["selected_parser"] = "audio_parser"
                    return result
                case "document":
                    ctx.metadata["file_path"] = local_path
                    result = await DocumentParserAgent().run(ctx)
                    result.metadata["selected_parser"] = "document_parser"
                    return result
                case _:
                    return StepResult(status=StepStatus.SKIPPED, error=f"Unsupported MIME type: {mime!r} for URI: {uri}")


class TextParserAgent(BaseModel, Step):
    """
    Normalizes raw text and extracts content from HTML documents.

    This agent processes string input by optionally removing HTML markup
    and regularizing whitespace character sequences to ensure consistent
    downstream processing.
    """

    strip_html: bool = Field(True, description="Remove HTML tags from input")
    normalize_whitespace: bool = Field(True, description="Normalize whitespace and newlines")

    async def run(self, ctx: IngestionContext) -> StepResult:
        raw = ctx.metadata.get("raw_text") or ctx.metadata.get("text")
        if not raw:
            return StepResult(status=StepStatus.SKIPPED, error="No text")
        text = raw
        if self.strip_html and bool(re.search(r"<[a-zA-Z][^>]*>", text[:1000])):
            p = _HTMLTextExtractor()
            try:
                p.feed(text)
                text = p.get_text()
            except Exception:
                text = re.sub(r"<[^>]+>", " ", text)
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text)
            text = "\n".join(line.strip() for line in text.split("\n") if line.strip()).strip()
        ctx.metadata["text"] = text
        return StepResult(status=StepStatus.SUCCESS, metadata={"original_length": len(raw), "parsed_length": len(text)})


class DocumentParserAgent(BaseModel, Step):
    """
    Extracts text content from various document file formats.

    This agent identifies and processes common file types, including PDF
    and DOCX, to retrieve the underlying text data for further analysis
    within the ingestion pipeline.
    """

    async def run(self, ctx: IngestionContext) -> StepResult:
        raw, path = ctx.metadata.get("raw_data"), ctx.metadata.get("file_path")
        if not raw and not path:
            return StepResult(status=StepStatus.SKIPPED, error="No data/path")
        if path:
            ext = path.lower()
            if ext.endswith(".pdf"):
                text = "\n".join(p.extract_text() for p in PdfReader(path).pages)
            elif ext.endswith(".docx"):
                text = "\n".join(p.text for p in Document(path).paragraphs)
            else:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
        else:
            if raw is None:
                return StepResult(status=StepStatus.SKIPPED, error="No data")
            if raw.startswith(b"%PDF"):
                text = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(raw)).pages)
            elif raw.startswith(b"PK\x03\x04"):
                text = "\n".join(p.text for p in Document(io.BytesIO(raw)).paragraphs)
            else:
                text = raw.decode("utf-8")
        ctx.metadata["text"] = text
        return StepResult(status=StepStatus.SUCCESS, metadata={"parsed_length": len(text)})


class AudioParserAgent(BaseModel, Step):
    """
    Transcribes audio data into text format using automated speech recognition.

    Utilizing Azure Speech Services, this agent converts audio input into
    synchronized text transcripts. The output includes word-level
    metadata suitable for temporal analysis of the content.

    Non-WAV inputs are automatically converted via ffmpeg (must be on PATH).
    """

    @staticmethod
    @asynccontextmanager
    async def _as_wav(src: str):
        """
        Return (wav_path, is_temp). If `src` is already a WAV it is returned
        unchanged. Otherwise it is converted to a temporary WAV file; the
        caller owns that file and must delete it when done.
        """
        fmt = Path(src).suffix.lstrip(".").lower() or "mp3"
        if fmt == "wav":
            yield src
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", "-f", "wav", tmp.name,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(f"ffmpeg conversion failed:\n{stderr.decode()}")
                yield tmp.name


    async def run(self, ctx: IngestionContext) -> StepResult:
        if not (audio_path := ctx.metadata.get("audio_path")):
            return StepResult(status=StepStatus.SKIPPED, error="No audio_path")

        key = settings.azure_speech_key.get_secret_value() if settings.azure_speech_key else None
        region = settings.azure_speech_region
        if not key or not region:
            return StepResult(status=StepStatus.FAILURE, error="Azure Speech credentials missing")

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
        async with AudioParserAgent._as_wav(audio_path) as wav_path:
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
        ctx.metadata["text"] = full_text
        ctx.metadata["word_timestamps"] = words
        return StepResult(status=StepStatus.SUCCESS, metadata={"text_len": len(full_text), "word_count": len(words)})
