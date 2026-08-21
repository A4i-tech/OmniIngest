"""BRF braille translation, built on liblouis's official Python bindings.

Uses the `liblouis` PyPI package (https://github.com/liblouis/liblouis/issues/1700),
a self-contained wheel bundling the shared library and all tables — no system
liblouis install required. Only Linux and Windows wheels exist upstream so far
(pinned in `[tool.uv.sources]`); there is no macOS build yet.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_TRANSLATION_TABLE = "en-us-g2.ctb"
DEFAULT_DISPLAY_TABLE = "en-us-brf.dis"


def check_tables(tables: Sequence[str]) -> None:
    """Validate that liblouis can compile the given tables.

    :raise RuntimeError: If any table cannot be found or compiled.
    """
    try:
        import louis
    except OSError as e:
        raise RuntimeError("Braille IO requires liblouis which is not supported for your platform.") from e
    louis.checkTable(list(tables))


def back_translate(braille_text: str, tables: Sequence[str], *, ascii_braille: bool = True) -> str:
    """Back-translate braille text into print text.

    :param braille_text: The braille content, e.g. read from a `.brf` file.
    :param tables: liblouis tables to apply, e.g. `["en-us-g2.ctb"]`.
    :param ascii_braille: Whether `braille_text` uses liblouis's braille-ASCII
        convention (lowercase a-z as the base alphabet dot patterns). Real BRF
        files are conventionally all-uppercase (NABCC/BANA convention), so they
        must be lowercased before back-translation.
    :raise RuntimeError: If the given tables could not back-translate the content.
    """
    try:
        import louis
    except OSError as e:
        raise RuntimeError("Braille IO requires liblouis which is not supported for your platform.") from e
    return str(louis.backTranslateString(list(tables), braille_text.lower() if ascii_braille else braille_text))


def translate(print_text: str, tables: Sequence[str]) -> str:
    """Translate print text into BRF (braille-ASCII) text.

    :param print_text: The print text to translate.
    :param tables: liblouis tables to apply, e.g. `["en-us-brf.dis", "en-us-g2.ctb"]`.
    :raise RuntimeError: If the given tables could not translate the content.
    """
    try:
        import louis
    except OSError as e:
        raise RuntimeError("Braille IO requires liblouis which is not supported for your platform.") from e
    return str(louis.translateString(list(tables), print_text))
