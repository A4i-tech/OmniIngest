"""Speech synthesis service implementations."""

from __future__ import annotations

import asyncio
from enum import StrEnum

import azure.cognitiveservices.speech as speechsdk

from ..core.config import KnownTTSModelName, settings


class TtsOutputFormat(StrEnum):
    MP3 = "mp3", "audio/mpeg"
    WAV = "wav", "audio/wav"

    mime_type: str

    def __new__(cls, value: str, mime_type: str):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.mime_type = mime_type
        return obj


_SDK_OUTPUT_FORMAT = {
    TtsOutputFormat.MP3: speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3,
    TtsOutputFormat.WAV: speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm,
}


class AzureSpeechSynthesizer:
    def __init__(self, key: str, region: str):
        self._key = key
        self._region = region

    async def synthesize(self, text: str, *, voice: str | None = None, format: TtsOutputFormat = TtsOutputFormat.MP3) -> bytes:
        return await asyncio.to_thread(self._synthesize_sync, text, voice, format)

    def _synthesize_sync(self, text: str, voice: str | None, format: TtsOutputFormat) -> bytes:
        speech_config = speechsdk.SpeechConfig(subscription=self._key, region=self._region)
        speech_config.set_speech_synthesis_output_format(_SDK_OUTPUT_FORMAT[format])
        if voice:
            speech_config.speech_synthesis_voice_name = voice

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result = synthesizer.speak_text_async(text).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            cancellation = result.cancellation_details
            details = cancellation.error_details if cancellation and cancellation.error_details else str(result.reason)
            raise RuntimeError(f"Speech synthesis failed: {details}")
        return bytes(result.audio_data)


def create_synthesizer(model: KnownTTSModelName = settings.default_tts_model) -> AzureSpeechSynthesizer:
    if model != "azure" or not settings.azure_speech_key or not settings.azure_speech_region:
        raise ValueError(f"Text-to-speech model is unsupported or not configured: {model}")
    return AzureSpeechSynthesizer(settings.azure_speech_key.get_secret_value(), settings.azure_speech_region)
