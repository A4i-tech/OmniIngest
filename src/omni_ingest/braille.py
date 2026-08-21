"""BRF/PEF braille translation, built on liblouis's official Python bindings.

Uses the `liblouis` PyPI package (https://github.com/liblouis/liblouis/issues/1700),
a self-contained wheel bundling the shared library and all tables — no system
liblouis install required. Only Linux and Windows wheels exist upstream so far
(pinned in `[tool.uv.sources]`); there is no macOS build yet.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Sequence
from types import ModuleType

DEFAULT_TRANSLATION_TABLE = "en-us-g2.ctb"
DEFAULT_DISPLAY_TABLE = "en-us-brf.dis"
PEF_DISPLAY_TABLE = "unicode.dis"

_lib: ModuleType | None = None
_lib_lock = threading.Lock()


def _get_lib() -> ModuleType:
    global _lib
    if _lib is not None:
        return _lib
    with _lib_lock:
        if _lib is None:
            try:
                import louis
            except ImportError as exc:
                if sys.platform == "darwin":
                    raise RuntimeError(
                        "BRF/PEF braille conversion needs the 'liblouis' package, which has no macOS build yet "
                        "(https://github.com/liblouis/liblouis/issues/1700). Run this on Linux or Windows instead."
                    ) from exc
                raise RuntimeError(
                    "BRF/PEF braille conversion needs the 'liblouis' package, but it isn't installed in this "
                    "environment. Run `uv sync` (or `pip install liblouis`) and try again."
                ) from exc
            _lib = louis
        return _lib


def check_tables(tables: Sequence[str]) -> None:
    """Validate that liblouis can compile the given tables.

    :raise RuntimeError: If any table cannot be found or compiled.
    """
    _get_lib().checkTable(list(tables))


def back_translate(braille_text: str, tables: Sequence[str], *, ascii_braille: bool = True) -> str:
    """Back-translate braille text into print text.

    :param braille_text: The braille content, e.g. read from a `.brf` or `.pef` file.
    :param tables: liblouis tables to apply, e.g. `["en-us-g2.ctb"]` for BRF or
        `["unicode.dis", "en-us-g2.ctb"]` for PEF's Unicode braille cells.
    :param ascii_braille: Whether `braille_text` uses liblouis's braille-ASCII
        convention (lowercase a-z as the base alphabet dot patterns). Real BRF
        files are conventionally all-uppercase (NABCC/BANA convention), so they
        must be lowercased before back-translation. PEF and other Unicode
        braille (U+2800-U+28FF) formats have no case and must be left as-is.
    :raise RuntimeError: If the given tables could not back-translate the content.
    """
    return str(_get_lib().backTranslateString(list(tables), braille_text.lower() if ascii_braille else braille_text))


def translate(print_text: str, tables: Sequence[str]) -> str:
    """Translate print text into BRF (braille-ASCII) text.

    :param print_text: The print text to translate.
    :param tables: liblouis tables to apply, e.g. `["en-us-brf.dis", "en-us-g2.ctb"]`.
    :raise RuntimeError: If the given tables could not translate the content.
    """
    return str(_get_lib().translateString(list(tables), print_text))
