"""Translation service implementations."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from fnmatch import fnmatch
from random import sample
from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload
from xml.etree import ElementTree

from azure.ai.translation.text.aio import TextTranslationClient
from azure.ai.translation.text.models import TranslateInputItem, TranslationTarget
from azure.core.credentials import AzureKeyCredential
from openai import AsyncOpenAI
from sarvamai import AsyncSarvamAI

from ..core.config import KnownTranslationModelName, settings

if TYPE_CHECKING:
    from ..core.protocol import Translator

Path: TypeAlias = tuple[str | int, ...]


def flatten(data: dict[str, Any] | list[Any]) -> tuple[dict[str, Any], dict[str, Path]]:
    leaves: list[tuple[Path, Any]] = []
    stack: list[tuple[Path, Any]] = [((), data)]
    while stack:
        path, value = stack.pop()
        if isinstance(value, (dict, list)) and value:
            items = value.items() if isinstance(value, dict) else enumerate(value)
            stack.extend(((*path, key), child) for key, child in reversed(list(items)))
        else:
            leaves.append((path, value))
    width = max(1, ((len(leaves) - 1).bit_length() + 3) // 4)
    keys = [f"{value:0{width}x}" for value in sample(range(len(leaves)), len(leaves))]
    return ({key: value for key, (_, value) in zip(keys, leaves, strict=True)}, {key: path for key, (path, _) in zip(keys, leaves, strict=True)})


def unflatten(flat: dict[str, Any], mapping: dict[str, Path]) -> dict[str, Any] | list[Any]:
    first_path = next(iter(mapping.values()))
    root: list[Any] | dict[str, Any] = [] if first_path and isinstance(first_path[0], int) else {}
    for key, path in mapping.items():
        if not path:
            return cast(dict[str, Any] | list[Any], flat[key])
        parent: list | dict = root
        for index, part in enumerate(path):
            value = flat[key] if index == len(path) - 1 else ([] if isinstance(path[index + 1], int) else {})
            if isinstance(parent, list):
                part = cast(int, part)
                parent.extend([None] * (part - len(parent) + 1))
                parent[part] = parent[part] if parent[part] is not None else value
                parent = parent[part]
            else:
                parent = parent.setdefault(part, value)
    return root


class AzureTranslator:
    def __init__(self, key: str, region: str):
        self.client = TextTranslationClient(credential=AzureKeyCredential(key), region=region)

    async def __aenter__(self) -> AzureTranslator:
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.client.__aexit__(exc_type, exc, tb)

    @overload
    async def translate(self, src: str | None, dst: str, data: str) -> str: ...

    @overload
    async def translate(self, src: str | None, dst: str, data: dict[str, Any], skip_fields: Iterable[str] = ()) -> dict[str, Any]: ...

    @overload
    async def translate(self, src: str | None, dst: str, data: list[Any], skip_fields: Iterable[str] = ()) -> list[Any]: ...

    async def translate(self, src: str | None, dst: str, data: str | dict[str, Any] | list[Any], skip_fields: Iterable[str] = ()) -> str | dict[str, Any] | list[Any]:
        if isinstance(data, str):
            return await self._translate(src, dst, data, "Plain")

        flat, mapping = flatten(data)
        patterns = tuple(skip_fields)
        strings = {key: value for key, value in flat.items() if isinstance(value, str) and not any(fnmatch(".".join(map(str, mapping[key])), pattern) for pattern in patterns)}
        if not strings:
            return data

        batches = [ElementTree.Element("div")]
        size = 11
        for key, value in strings.items():
            element = ElementTree.Element("div", id=key)
            element.text = value
            length = len(ElementTree.tostring(element, encoding="unicode"))
            if size + length > 49_000:
                batches.append(ElementTree.Element("div"))
                size = 11
            batches[-1].append(element)
            size += length
        for html in batches:
            root = ElementTree.fromstring(await self._translate(src, dst, ElementTree.tostring(html, encoding="unicode"), "Html"))
            flat.update({element.attrib["id"]: "".join(element.itertext()) for element in root})
        return unflatten(flat, mapping)

    async def _translate(self, src: str | None, dst: str, text: str, text_type: str) -> str:
        item = TranslateInputItem(text=text, language=src, text_type=text_type, targets=[TranslationTarget(language=dst)])
        return (await self.client.translate(body=[item]))[0].translations[0].text


async def _translate_leaves(data: dict[str, Any] | list[Any], skip_fields: Iterable[str], translate_one) -> dict[str, Any] | list[Any]:
    flat, mapping = flatten(data)
    patterns = tuple(skip_fields)
    strings = {key: value for key, value in flat.items() if isinstance(value, str) and not any(fnmatch(".".join(map(str, mapping[key])), pattern) for pattern in patterns)}
    if not strings:
        return data
    translated = await asyncio.gather(*(translate_one(value) for value in strings.values()))
    flat.update(dict(zip(strings.keys(), translated, strict=True)))
    return unflatten(flat, mapping)


class OpenAICompatibleTranslator:
    """Translates via chat completion against any OpenAI-compatible endpoint."""

    def __init__(self, model: str, base_url: str | None):
        self.client = AsyncOpenAI(base_url=base_url)
        self.model = model

    async def __aenter__(self) -> OpenAICompatibleTranslator:
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.client.__aexit__(exc_type, exc, tb)

    @overload
    async def translate(self, src: str | None, dst: str, data: str) -> str: ...

    @overload
    async def translate(self, src: str | None, dst: str, data: dict[str, Any], skip_fields: Iterable[str] = ()) -> dict[str, Any]: ...

    @overload
    async def translate(self, src: str | None, dst: str, data: list[Any], skip_fields: Iterable[str] = ()) -> list[Any]: ...

    async def translate(self, src: str | None, dst: str, data: str | dict[str, Any] | list[Any], skip_fields: Iterable[str] = ()) -> str | dict[str, Any] | list[Any]:
        if isinstance(data, str):
            return await self._translate(src, dst, data)
        return await _translate_leaves(data, skip_fields, lambda text: self._translate(src, dst, text))

    async def _translate(self, src: str | None, dst: str, text: str) -> str:
        instructions = f"Translate the user's text{f' from {src}' if src else ''} into {dst}. Return only the translation, with no commentary, quotes, or code fences."
        response = await self.client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": instructions}, {"role": "user", "content": text}])
        return response.choices[0].message.content or ""


class SarvamTranslator:
    """Translates via Sarvam's dedicated text-translation API.

    `src`/`dst` must already be Sarvam language codes (e.g. `"en-IN"`, `"hi-IN"`), or `"auto"` for
    automatic source-language detection. Sarvam's codes do not match Azure's or arbitrary ISO codes.
    """

    def __init__(self):
        self.client = AsyncSarvamAI()

    async def __aenter__(self) -> SarvamTranslator:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        pass

    @overload
    async def translate(self, src: str | None, dst: str, data: str) -> str: ...

    @overload
    async def translate(self, src: str | None, dst: str, data: dict[str, Any], skip_fields: Iterable[str] = ()) -> dict[str, Any]: ...

    @overload
    async def translate(self, src: str | None, dst: str, data: list[Any], skip_fields: Iterable[str] = ()) -> list[Any]: ...

    async def translate(self, src: str | None, dst: str, data: str | dict[str, Any] | list[Any], skip_fields: Iterable[str] = ()) -> str | dict[str, Any] | list[Any]:
        if isinstance(data, str):
            return await self._translate(src, dst, data)
        return await _translate_leaves(data, skip_fields, lambda text: self._translate(src, dst, text))

    async def _translate(self, src: str | None, dst: str, text: str) -> str:
        response = await self.client.text.translate(input=text, source_language_code=src or "auto", target_language_code=dst)
        return cast(str, response.translated_text)


def create_translator(model: KnownTranslationModelName = settings.default_translation_model) -> Translator:
    if model == "azure":
        if not settings.azure_translation_key or not settings.azure_translation_region:
            raise ValueError("AZURE_TRANSLATION_KEY/AZURE_TRANSLATION_REGION are not configured. Set both environment variables to use the azure translation model.")
        return AzureTranslator(settings.azure_translation_key.get_secret_value(), settings.azure_translation_region)
    if model == "openai":
        return OpenAICompatibleTranslator(settings.openai_translation_model, str(settings.openai_translation_base_url) if settings.openai_translation_base_url else None)
    if model == "sarvam":
        return SarvamTranslator()
    raise ValueError(f"Translation model is unsupported or not configured: {model}")
