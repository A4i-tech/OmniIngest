"""Azure Speech synthesis service binding."""

from __future__ import annotations

import asyncio

import azure.cognitiveservices.speech as speechsdk

from ..core.config import settings
from ..core.model import ByteContent
from ..core.tts import SpeechBuilderParams, SpeechFactory, SpeechFormat, SpeechModel, simple

_SDK_OUTPUT_FORMAT = {
    SpeechFormat.MP3: speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3,
    SpeechFormat.WAV: speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm,
}


def _synthesize_sync(key: str, region: str, text: str, voice: str | None, format: SpeechFormat) -> bytes:
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
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


def azure_speech_builder(params: SpeechBuilderParams) -> SpeechModel:
    if not settings.azure_speech_key or not settings.azure_speech_region:
        raise ValueError("Azure Speech model is not configured")
    key = settings.azure_speech_key.get_secret_value()
    region = settings.azure_speech_region

    async def model(content: ByteContent, format: SpeechFormat) -> ByteContent:
        text = (await content.content(params.ctx)).decode(content.content_encoding)
        audio = await asyncio.to_thread(_synthesize_sync, key, region, text, params.voice, format)
        return ByteContent.from_bytes(audio, content_type=format.mime_type)

    return model


def default_factory() -> SpeechFactory:
    return simple({"azure": azure_speech_builder})
