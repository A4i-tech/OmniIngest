"""Speech-to-text service implementations."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Literal

import azure.cognitiveservices.speech as speechsdk
from openai import AsyncOpenAI
from sarvamai import AsyncSarvamAI

from ..core.config import KnownSTTModelName, settings

if TYPE_CHECKING:
    from ..core.protocol import SpeechToText


class AzureSpeechToText:
    def __init__(self, key: str, region: str):
        self.key = key
        self.region = region

    async def transcribe(self, wav_path: str) -> tuple[str, dict[str, Any]]:
        loop = asyncio.get_running_loop()
        done, transcript, words = asyncio.Future[Literal[True]](), [], []

        def finish(error: RuntimeError | None = None):
            if done.done():
                return
            if error:
                done.set_exception(error)
            else:
                done.set_result(True)

        def handle_final(evt):
            nonlocal transcript, words
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                transcript.append(evt.result.text)
                obj = json.loads(evt.result.json)
                if "NBest" in obj and obj["NBest"]:
                    words.extend(obj["NBest"][0].get("Words", []))

        def stop_cb(evt):
            loop.call_soon_threadsafe(finish)

        def cancel_cb(evt):
            if evt.result and evt.result.cancellation_details and evt.result.cancellation_details.reason != speechsdk.CancellationReason.EndOfStream:
                cancellation = evt.result.cancellation_details
                details = cancellation.error_details if cancellation.error_details else str(cancellation.reason)
                loop.call_soon_threadsafe(finish, RuntimeError(details))
            else:
                loop.call_soon_threadsafe(finish)

        speech_config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_RequestWordLevelTimestamps, "true")
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        recognizer.recognized.connect(handle_final)
        recognizer.session_stopped.connect(stop_cb)
        recognizer.canceled.connect(cancel_cb)

        recognizer.start_continuous_recognition()
        try:
            await done
        finally:
            recognizer.stop_continuous_recognition()

        full_text = " ".join(transcript)
        return full_text, {"text_len": len(full_text), "word_count": len(words), "word_timestamps": words}


class OpenAISpeechToText:
    """Transcribes via an OpenAI-compatible Whisper endpoint."""

    def __init__(self, base_url: str | None = None, model: str = "whisper-1"):
        self.client = AsyncOpenAI(base_url=base_url)
        self.model = model

    async def transcribe(self, wav_path: str) -> tuple[str, dict[str, Any]]:
        with open(wav_path, "rb") as f:
            result = await self.client.audio.transcriptions.create(file=f, model=self.model)
        text = result.text
        return text, {"text_len": len(text), "word_count": len(text.split())}


class SarvamSpeechToText:
    def __init__(self):
        self.client = AsyncSarvamAI()

    async def transcribe(self, wav_path: str) -> tuple[str, dict[str, Any]]:
        with open(wav_path, "rb") as f:
            result = await self.client.speech_to_text.transcribe(file=f)
        text = result.transcript
        return text, {"text_len": len(text), "word_count": len(text.split())}


def create_stt_client(model: KnownSTTModelName = settings.default_stt_model) -> SpeechToText:
    if model == "azure":
        if not settings.azure_speech_key or not settings.azure_speech_region:
            raise ValueError(f"STT model is unsupported or not configured: {model}")
        return AzureSpeechToText(settings.azure_speech_key.get_secret_value(), settings.azure_speech_region)
    if model == "openai":
        return OpenAISpeechToText(base_url=str(settings.stt_base_url) if settings.stt_base_url else None)
    if model == "sarvam":
        return SarvamSpeechToText()
    raise ValueError(f"STT model is unsupported or not configured: {model}")
