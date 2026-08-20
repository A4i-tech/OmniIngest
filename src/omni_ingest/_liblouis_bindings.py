# Liblouis Python ctypes bindings (trimmed)
#
# Copyright (C) 2009-2010 James Teh <jamie@jantrid.net>
# Copyright (C) 2010-2012 Michael Whapples <mwhapples@gmail.com>
# Copyright (C) 2012-2024 Christian Egli <christian.egli@sbs.ch>
# Copyright (C) 2015-2016 Davy Kager <mail@davykager.nl>
# Copyright (C) 2018-2025 Leonard de Ruijter <alderuijter@gmail.com>
#
# This file is part of liblouis.
#
# liblouis is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 2.1 of the License, or
# (at your option) any later version.
#
# liblouis is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with liblouis. If not, see <http://www.gnu.org/licenses/>.
#
# Vendored and trimmed (translateString/backTranslateString/checkTable/version
# only) from https://github.com/liblouis/liblouis/blob/master/python/louis/__init__.py.in
# because that file is a build-time template (the ###LIBLOUIS_SONAME### and
# @VERSION@ placeholders are only substituted by liblouis's autotools build),
# so it cannot be installed as a pip/uv dependency directly from the liblouis
# repository. This module resolves the shared library by name at import time
# instead, so it works from a plain `liblouis` OS package install.

from __future__ import annotations

import sys
import threading
from ctypes import CDLL, POINTER, c_char, c_char_p, c_int, c_ushort, create_string_buffer
from ctypes.util import find_library

INSTALL_HINT = (
    "Could not load the liblouis shared library, which is required for BRF (Braille "
    "Ready Format) translation. Install liblouis for your OS (e.g. `apt install "
    "liblouis-bin`, `brew install liblouis`, or the installer at https://liblouis.io/), "
    "then try again."
)

_lib: CDLL | None = None
_lib_lock = threading.Lock()


def _get_lib() -> CDLL:
    global _lib
    if _lib is not None:
        return _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        return _load_lib()


def _load_lib() -> CDLL:
    global _lib
    for name in filter(None, (find_library("louis"), "liblouis.dll", "liblouis.so.20", "liblouis.dylib")):
        try:
            lib = CDLL(name)
            break
        except OSError:
            continue
    else:
        raise RuntimeError(INSTALL_HINT)

    lib.lou_version.restype = c_char_p
    lib.lou_version.argtypes = ()
    lib.lou_charSize.restype = c_int
    lib.lou_charSize.argtypes = ()
    lib.lou_translateString.restype = c_int
    lib.lou_translateString.argtypes = (
        c_char_p, POINTER(c_char), POINTER(c_int), POINTER(c_char), POINTER(c_int),
        POINTER(c_ushort), POINTER(c_char), c_int,
    )
    lib.lou_backTranslateString.restype = c_int
    lib.lou_backTranslateString.argtypes = (
        c_char_p, POINTER(c_char), POINTER(c_int), POINTER(c_char), POINTER(c_int),
        POINTER(c_ushort), POINTER(c_char), c_int,
    )
    lib.lou_checkTable.restype = c_int
    lib.lou_checkTable.argtypes = (c_char_p,)
    _lib = lib
    return lib


def _tableBuf(tableList: list[str]) -> bytes:
    return b",".join(t.encode("utf-8") for t in tableList)


def _conversionEncoding(lib: CDLL) -> tuple[int, str]:
    wideCharBytes = lib.lou_charSize()
    return wideCharBytes, f"utf_{wideCharBytes * 8}_{'be' if sys.byteorder == 'big' else 'le'}"


def version() -> str:
    return str(_get_lib().lou_version().decode("ASCII"))


def _translate(fn_name: str, verb: str, tableList: list[str], inbuf: str) -> str:
    lib = _get_lib()
    wideCharBytes, encoding = _conversionEncoding(lib)
    tableBuf = create_string_buffer(_tableBuf(tableList))
    _inbuf = create_string_buffer(inbuf.encode(encoding, errors="surrogatepass"))
    inlen = c_int(len(inbuf))
    outlen = c_int(inlen.value * (4 + wideCharBytes * 2))
    outbuf = create_string_buffer(outlen.value * wideCharBytes)
    if not getattr(lib, fn_name)(tableBuf, _inbuf, inlen, outbuf, outlen, None, None, 0):
        raise RuntimeError(f"Can't {verb}: tables {tableList}, inbuf {inbuf!r}")
    return outbuf.raw[: outlen.value * wideCharBytes].decode(encoding, errors="surrogatepass")


def translateString(tableList: list[str], inbuf: str) -> str:
    return _translate("lou_translateString", "translate", tableList, inbuf)


def backTranslateString(tableList: list[str], inbuf: str) -> str:
    return _translate("lou_backTranslateString", "back translate", tableList, inbuf)


def checkTable(tableList: list[str]) -> None:
    lib = _get_lib()
    tableBuf = create_string_buffer(_tableBuf(tableList))
    if not lib.lou_checkTable(tableBuf):
        raise RuntimeError(f"Can't compile: tables {tableList}")
