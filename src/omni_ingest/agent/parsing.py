"""Agents for extracting text from various document formats and audio streams."""

import asyncio
import io
import json
import re
from html.parser import HTMLParser
from typing import Literal

import azure.cognitiveservices.speech as speechsdk
from docx import Document
from pydantic import BaseModel, Field
from pypdf import PdfReader

from ..core.config import settings
from ..core.model import IngestionContext, Step, StepResult, StepStatus
from ..core.pipeline import step


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


@step("text_parser")
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


@step("document_parser")
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


@step("audio_parser")
class AudioParserAgent(BaseModel, Step):
    """
    Transcribes audio data into text format using automated speech recognition.

    Utilizing Azure Speech Services, this agent converts audio input into
    synchronized text transcripts. The output includes word-level
    metadata suitable for temporal analysis of the content.
    """

    async def run(self, ctx: IngestionContext) -> StepResult:
        if not (audio_path := ctx.metadata.get("audio_path")):
            return StepResult(status=StepStatus.SKIPPED, error="No audio_path")

        key = settings.azure_speech_key.get_secret_value() if settings.azure_speech_key else None
        region = settings.azure_speech_region
        if not key or not region:
            return StepResult(status=StepStatus.FAILURE, error="Azure Speech credentials missing")

        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        loop = asyncio.get_event_loop()
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
