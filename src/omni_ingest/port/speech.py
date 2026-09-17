"""Speech-to-text and text-to-speech engine implementations."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

import azure.cognitiveservices.speech as speechsdk
from openai import AsyncOpenAI
from sarvamai import AsyncSarvamAI

from ..core.config import settings
from ..core.model import ByteContent
from ..core.speech import (
    SpeechFormat,
    Stt,
    SttFactory,
    TTSBuilderParams,
    TTSFactory,
    TTSModel,
    simple_stt_factory,
    simple_tts_factory,
)

_SDK_OUTPUT_FORMAT = {
    SpeechFormat.MP3: speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3,
    SpeechFormat.WAV: speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm,
}


def azure_stt_builder() -> Stt:
    if not settings.azure_speech_key or not settings.azure_speech_region:
        raise ValueError("AZURE_SPEECH_KEY/AZURE_SPEECH_REGION are not configured. Set both environment variables to use the azure STT model.")
    key = settings.azure_speech_key.get_secret_value()
    region = settings.azure_speech_region

    async def transcribe(wav_path: str, lang: str | None = None) -> tuple[str, dict[str, Any]]:
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

        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_RequestWordLevelTimestamps, "true")
        if lang:
            speech_config.speech_recognition_language = lang
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

    return transcribe


def openai_stt_builder() -> Stt:
    """Transcribes via an OpenAI-compatible Whisper endpoint."""
    client = AsyncOpenAI(base_url=str(settings.stt_base_url) if settings.stt_base_url else None)

    async def transcribe(wav_path: str, lang: str | None = None) -> tuple[str, dict[str, Any]]:
        with open(wav_path, "rb") as f:
            kwargs = {"language": lang} if lang else {}
            result = await client.audio.transcriptions.create(file=f, model="whisper-1", **kwargs)
        text = result.text
        return text, {"text_len": len(text), "word_count": len(text.split())}

    return transcribe


def sarvam_stt_builder() -> Stt:
    client = AsyncSarvamAI()

    async def transcribe(wav_path: str, lang: str | None = None) -> tuple[str, dict[str, Any]]:
        with open(wav_path, "rb") as f:
            kwargs = {"language_code": lang} if lang else {}
            result = await client.speech_to_text.transcribe(file=f, **kwargs)
        text = result.transcript
        return text, {"text_len": len(text), "word_count": len(text.split())}

    return transcribe


def default_stt_factory() -> SttFactory:
    return simple_stt_factory({
        "azure": azure_stt_builder,
        "openai": openai_stt_builder,
        "sarvam": sarvam_stt_builder,
    })


def azure_tts_builder(params: TTSBuilderParams) -> TTSModel:
    if not settings.azure_speech_key or not settings.azure_speech_region:
        raise ValueError("Azure Speech model is not configured")
    key = settings.azure_speech_key.get_secret_value()
    region = settings.azure_speech_region

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    if params.voice:
        speech_config.speech_synthesis_voice_name = params.voice

    async def model(content: ByteContent, format: SpeechFormat) -> ByteContent:
        speech_config.set_speech_synthesis_output_format(_SDK_OUTPUT_FORMAT[format])
        text = (await content.content(params.ctx)).decode(content.content_encoding)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result = await asyncio.to_thread(synthesizer.speak_text_async(text).get)
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            cancellation = result.cancellation_details
            details = cancellation.error_details if cancellation and cancellation.error_details else str(result.reason)
            raise RuntimeError(f"Speech synthesis failed: {details}")
        return ByteContent.from_bytes(bytes(result.audio_data), content_type=format.mime_type)

    return model


def default_tts_factory() -> TTSFactory:
    return simple_tts_factory({"azure": azure_tts_builder})
