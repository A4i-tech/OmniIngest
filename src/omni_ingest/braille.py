"""BRF (Braille Ready Format) translation, built on the vendored liblouis bindings.

See `omni_ingest._liblouis_bindings` for why this vendors liblouis's official
ctypes bindings instead of depending on a pip package: no usable one exists.
"""

from __future__ import annotations

from . import _liblouis_bindings as _lib

DEFAULT_TRANSLATION_TABLE = "en-us-g2.ctb"
DEFAULT_DISPLAY_TABLE = "en-us-brf.dis"


def check_tables(tables: list[str]) -> None:
    """Validate that liblouis can compile the given tables.

    :raise RuntimeError: If any table cannot be found or compiled.
    """
    _lib.checkTable(tables)


def back_translate(braille_text: str, tables: list[str]) -> str:
    """Back-translate BRF (braille-ASCII) text into print text.

    :param braille_text: The BRF content, e.g. read from a `.brf` file.
    :param tables: liblouis translation tables to apply, e.g. `["en-us-g2.ctb"]`.
    :raise RuntimeError: If the given tables could not back-translate the content.
    """
    # liblouis's braille-ASCII cell mapping treats lowercase a-z as the base
    # alphabet dot patterns; real BRF files are conventionally all-uppercase
    # (NABCC/BANA convention), so they must be lowercased before back-translation.
    return _lib.backTranslateString(tables, braille_text.lower())


def translate(print_text: str, tables: list[str]) -> str:
    """Translate print text into BRF (braille-ASCII) text.

    :param print_text: The print text to translate.
    :param tables: liblouis tables to apply, e.g. `["en-us-brf.dis", "en-us-g2.ctb"]`.
    :raise RuntimeError: If the given tables could not translate the content.
    """
    return _lib.translateString(tables, print_text)
