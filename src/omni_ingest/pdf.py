from __future__ import annotations

import ctypes
import io
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import Any

import pypdf
import pypdfium2 as pdfium
import pypdfium2.raw as raw
from PIL import Image
from pypdf.generic import ArrayObject, ByteStringObject, ContentStream, DictionaryObject, FloatObject, IndirectObject, NameObject

Rect = tuple[float, float, float, float]
Matrix = tuple[float, float, float, float, float, float]

PDFIUM = ThreadPoolExecutor(max_workers=1)  # PDFium is not thread-safe, even across documents

_PNG_MODES = {"1", "L", "LA", "P", "RGB", "RGBA"}


def _near(a: Rect, b: Rect, gap: float) -> bool:
    return a[0] <= b[2] + gap and b[0] <= a[2] + gap and a[1] <= b[3] + gap and b[1] <= a[3] + gap


def _merge(boxes: list[Rect], gap: float) -> list[Rect]:
    merged = True
    while merged:
        merged, out = False, list[Rect]()
        for b in boxes:
            for i, o in enumerate(out):
                if _near(b, o, gap):
                    out[i] = (min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3]))
                    merged = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def _overlaps(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _oc_hidden(oc: Any, off: set[int], on: set[int], base_off: bool) -> bool:
    def hidden(ref: IndirectObject) -> bool:
        return ref.idnum not in on if base_off else ref.idnum in off

    node: Any = oc.get_object()
    if node.get("/Type") != "/OCMD":
        return isinstance(oc, IndirectObject) and hidden(oc)
    ocgs = node.get("/OCGs")
    target = ocgs.get_object() if isinstance(ocgs, IndirectObject) else ocgs
    shown = [not hidden(r) for r in (target if isinstance(target, list) else [ocgs]) if isinstance(r, IndirectObject)]
    visible = {"/AnyOn": any(shown), "/AllOn": all(shown), "/AnyOff": not all(shown), "/AllOff": not any(shown)}.get(node.get("/P", "/AnyOn"), True)
    return bool(shown) and not visible


def _within(e: Rect, box: Rect) -> bool:
    return box[0] - 0.5 <= e[0] and box[1] - 0.5 <= e[1] and e[2] <= box[2] + 0.5 and e[3] <= box[3] + 0.5


def _mul(m: Matrix, n: Matrix) -> Matrix:
    return (m[0] * n[0] + m[1] * n[2], m[0] * n[1] + m[1] * n[3], m[2] * n[0] + m[3] * n[2], m[2] * n[1] + m[3] * n[3],
            m[4] * n[0] + m[5] * n[2] + n[4], m[4] * n[1] + m[5] * n[3] + n[5])


def _inv(m: Matrix) -> Matrix:
    a, b, c, d, e, f = m
    k = a * d - b * c
    return (d / k, -b / k, -c / k, a / k, (c * f - d * e) / k, (b * e - a * f) / k)


def _apply(m: Matrix, x: float, y: float) -> tuple[float, float]:
    return m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]


def _widths(font: Any) -> tuple[int, Callable[[int], float] | None] | None:
    kind = font["/Subtype"]
    if kind == "/Type3" or (kind == "/Type0" and font["/Encoding"] != "/Identity-H"):
        return None
    if kind == "/Type0":
        cid = font["/DescendantFonts"][0]
        default, table, w, i = cid.get("/DW", 1000), {}, [x.get_object() for x in cid.get("/W", [])], 0
        while i < len(w):
            if isinstance(w[i + 1], list):
                table.update({w[i] + k: v.get_object() for k, v in enumerate(w[i + 1])})
                i += 2
            else:
                table.update(dict.fromkeys(range(w[i], w[i + 1] + 1), w[i + 2]))
                i += 3
        return 2, lambda code: table.get(code, default)
    if "/Widths" not in font:
        return 1, None
    first, widths = font.get("/FirstChar", 0), [x.get_object() for x in font["/Widths"]]
    missing = font["/FontDescriptor"].get("/MissingWidth", 500) if "/FontDescriptor" in font else 500
    return 1, lambda code: widths[code - first] if 0 <= code - first < len(widths) else missing


_IDENTITY: Matrix = (1, 0, 0, 1, 0, 0)
_UNIT = [(0, 0), (1, 0), (0, 1), (1, 1)]
_PATH = {b"m", b"l", b"c", b"v", b"y", b"re", b"h", b"W", b"W*"}
_PAINT = {b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*"}
_SHOW = {b"Tj", b"TJ", b"'", b'"'}
_TEXT_STATE = {b"Tc": "tc", b"Tw": "tw", b"Tz": "tz", b"TL": "tl", b"Ts": "ts"}
_FAR = 1e6


class _Redactor:
    def __init__(self, page: Page, fonts: Any, xobjects: Any, regions: list[Rect], shadings: list[Rect]):
        self.page, self.fonts, self.xobjects, self.regions, self.shadings = page, fonts, xobjects, regions, shadings
        self.models: dict[str, tuple[int, Callable[[int], float] | None] | None] = {}
        self.s: dict[str, Any] = {"ctm": _IDENTITY, "font": None, "size": 0.0, "tc": 0.0, "tw": 0.0, "tz": 100.0, "tl": 0.0, "ts": 0.0}
        self.stack: list[dict[str, Any]] = []
        self.dropped: set[str] = set()
        self.tm = self.tlm = _IDENTITY

    def box(self, points: Any, m: Matrix) -> Rect:
        xs, ys = zip(*(_apply(m, x, y) for x, y in points), strict=True)
        return self.page._box(min(xs), min(ys), max(xs), max(ys))

    def hit(self, box: Rect) -> bool:
        return any(_overlaps(box, r) for r in self.regions)

    def move(self, tx: float, ty: float) -> None:
        self.tm = self.tlm = _mul((1, 0, 0, 1, tx, ty), self.tlm)

    def run(self, operations: list[Any]) -> list[Any]:
        out: list[Any] = []
        path: list[Any] = []
        points: list[tuple[float, float]] = []
        clip = False
        shadings = iter(self.shadings)
        for operands, op in operations:
            s = self.s
            if op == b"q":
                self.stack.append(dict(s))
            elif op == b"Q" and self.stack:  # real files carry extra Q
                self.s = self.stack.pop()
            elif op == b"cm":
                s["ctm"] = _mul(tuple(operands), s["ctm"])
            elif op == b"BT":
                self.tm = self.tlm = _IDENTITY
            elif op == b"Tf":
                s["font"], s["size"] = operands
            elif op in _TEXT_STATE:
                s[_TEXT_STATE[op]] = operands[0]
            elif op in (b"Td", b"TD"):
                if op == b"TD":
                    s["tl"] = -operands[1]
                self.move(*operands)
            elif op == b"Tm":
                self.tm = self.tlm = tuple(operands)
            elif op == b"T*":
                self.move(0, -s["tl"])
            elif op in _SHOW:
                out += self.show(op, operands)
                continue
            elif op == b"INLINE IMAGE" or (op == b"Do" and operands[0] in self.xobjects and self.xobjects[operands[0]]["/Subtype"] == "/Image"):
                if self.hit(self.box(_UNIT, s["ctm"])):
                    if op == b"Do":
                        self.dropped.add(operands[0])
                    continue
            elif op == b"sh":
                box = next(shadings, None)
                if box is not None and self.hit(box):
                    continue
            elif op in _PATH:
                path.append((operands, op))
                clip = clip or op in (b"W", b"W*")
                if op == b"re":
                    x, y, w, h = operands
                    points += [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
                else:
                    points += zip(operands[::2], operands[1::2], strict=False)
                continue
            elif op in _PAINT or op == b"n":
                if op == b"n" or clip or not points or not any(_within(self.box(points, s["ctm"]), r) for r in self.regions):
                    out += [*path, (operands, op)]
                path, points, clip = [], [], False
                continue
            out.append((operands, op))
        return out

    def show(self, op: bytes, operands: list[Any]) -> list[Any]:
        s = self.s
        prefix: list[Any] = []
        if op == b'"':
            s["tw"], s["tc"] = operands[0], operands[1]
            prefix = [([operands[0]], b"Tw"), ([operands[1]], b"Tc")]
        if op in (b"'", b'"'):
            self.move(0, -s["tl"])
            prefix.append(([], b"T*"))
        kept = self.glyphs(list(operands[0]) if op == b"TJ" else [operands[-1]])
        return [(operands, op)] if kept is None else [*prefix, ([ArrayObject(kept)], b"TJ")]

    def glyphs(self, items: list[Any]) -> list[Any] | None:
        s = self.s
        size, th = s["size"], s["tz"] / 100
        if not size:
            return None
        if s["font"] not in self.models:
            self.models[s["font"]] = _widths(self.fonts[s["font"]])
        model = self.models[s["font"]]
        y0, y1 = s["ts"] - 0.3 * size, s["ts"] + 0.8 * size
        if model is None:
            return [] if self.hit(self.box([(0, s["ts"])], _mul(self.tm, s["ctm"]))) else None
        step, width = model
        if width is None:
            return [] if self.hit(self.box([(0, y0), (_FAR, y0), (0, y1), (_FAR, y1)], _mul(self.tm, s["ctm"]))) else None
        out: list[Any] = []
        buf, removed = bytearray(), False

        def flush() -> None:
            if buf:
                out.append(ByteStringObject(bytes(buf)))
                buf.clear()

        for item in items:
            if not isinstance(item, bytes):
                flush()
                out.append(item)
                self.tm = _mul((1, 0, 0, 1, -item / 1000 * size * th, 0), self.tm)
                continue
            for i in range(0, len(item), step):
                code = item[i : i + step]
                w = width(int.from_bytes(code, "big"))
                gap = s["tc"] + (s["tw"] if step == 1 and code == b" " else 0)
                x1 = w / 1000 * size
                if self.hit(self.box([(0, y0), (x1 * th, y0), (0, y1), (x1 * th, y1)], _mul(self.tm, s["ctm"]))):
                    flush()
                    out.append(FloatObject(-(w + gap * 1000 / size)))
                    removed = True
                else:
                    buf += code
                self.tm = _mul((1, 0, 0, 1, (x1 + gap) * th, 0), self.tm)
        flush()
        return out if removed else None


def _is_encrypted(data: bytes) -> bool:
    # Linux PDFium reports a zero-page PDF as a password error
    try:
        return pypdf.PdfReader(io.BytesIO(data)).is_encrypted
    except pypdf.errors.PdfReadError:
        return True


class Pdf:
    def __init__(self, data: bytes, doc: pdfium.PdfDocument):
        self.data, self.doc = data, doc
        self.redactions: dict[int, list[tuple[Rect, str]]] = {}

    @classmethod
    def open(cls, data: bytes) -> Pdf:
        try:
            return cls(data, pdfium.PdfDocument(data))
        except pdfium.PdfiumError as e:
            if e.err_code == raw.FPDF_ERR_PASSWORD and _is_encrypted(data):
                raise ValueError("The PDF is password-protected. Remove the password and try again.") from e
            raise ValueError("The PDF is damaged and cannot be opened. Re-export the file or remove it from the batch.") from e

    def __enter__(self) -> Pdf:
        return self

    def __exit__(self, *exc: object) -> None:
        self.doc.close()

    def __len__(self) -> int:
        return len(self.doc)

    def page(self, index: int) -> Page:
        return Page(self, index)

    def to_bytes(self) -> bytes:
        out = io.BytesIO()
        if not self.redactions:
            self.doc.save(out)
            return out.getvalue()
        writer = pypdf.PdfWriter(clone_from=self.reader)
        for index, regions in self.redactions.items():
            self.page(index).apply_redactions(writer, regions)
        writer.write(out)
        return out.getvalue()

    @cached_property
    def reader(self) -> pypdf.PdfReader:
        try:
            return pypdf.PdfReader(io.BytesIO(self.data))
        except pypdf.errors.PdfReadError as e:
            raise ValueError("The PDF has a broken cross-reference table and cannot be processed. Re-export the file or remove it from the batch.") from e

    @cached_property
    def font(self) -> raw.FPDF_FONT:
        return raw.FPDFText_LoadStandardFont(self.doc.raw, b"Helvetica")


class Page:
    def __init__(self, pdf: Pdf, index: int):
        self._pdf, self._index, self._page = pdf, index, pdf.doc[index]
        self._renders: dict[int, Image.Image] = {}
        w, h = self._page.get_size()
        self.rect: Rect = (0.0, 0.0, w, h)
        rotation = self._page.get_rotation()
        w0, h0 = (w, h) if rotation in (0, 180) else (h, w)
        mb, cb = self._page.get_mediabox(), self._page.get_cropbox()
        self._origin = (max(mb[0], cb[0]), max(mb[1], cb[1]))
        self._m: Matrix = {0: (1, 0, 0, -1, 0, h0), 90: (0, 1, 1, 0, 0, 0), 180: (-1, 0, 0, 1, w0, 0), 270: (0, -1, -1, 0, h0, w0)}[rotation]

    def _point(self, x: float, y: float) -> tuple[float, float]:
        a, b, c, d, e, f = self._m
        u, v = x - self._origin[0], y - self._origin[1]
        return a * u + c * v + e, b * u + d * v + f

    def _box(self, x0: float, y0: float, x1: float, y1: float) -> Rect:
        xs, ys = zip(*(self._point(x, y) for x in (x0, x1) for y in (y0, y1)), strict=True)
        return min(xs), min(ys), max(xs), max(ys)

    def _objects(self, kind: int) -> list[pdfium.PdfObject]:
        return list(self._page.get_objects(filter=[kind], max_depth=0))

    def _path_box(self, obj: pdfium.PdfObject) -> Rect:
        width, fill, stroke = ctypes.c_float(), ctypes.c_int(), ctypes.c_int()
        raw.FPDFPageObj_GetStrokeWidth(obj, ctypes.byref(width))
        raw.FPDFPath_GetDrawMode(obj, ctypes.byref(fill), ctypes.byref(stroke))
        x0, y0, x1, y1 = obj.get_bounds()
        s = width.value if stroke.value else 0  # PDFium inflates stroked bounds by the full width
        return self._box(x0 + s, y0 + s, x1 - s, y1 - s)

    def _hidden_images(self, count: int) -> set[int]:
        root: Any = self._pdf.reader.trailer["/Root"]
        group: Any = root["/OCProperties"]["/D"] if "/OCProperties" in root else {}
        off = {o.idnum for o in group["/OFF"] if isinstance(o, IndirectObject)} if "/OFF" in group else set()
        on = {o.idnum for o in group["/ON"] if isinstance(o, IndirectObject)} if "/ON" in group else set()
        base_off = group.get("/BaseState") == "/OFF"
        if not off and not base_off:
            return set()
        page: Any = self._pdf.reader.pages[self._index]
        resources: Any = page["/Resources"] if "/Resources" in page else {}
        xobjects: Any = resources["/XObject"] if "/XObject" in resources else {}
        hidden, n = set(), 0
        for operands, op in ContentStream(page.get_contents(), self._pdf.reader).operations:
            if op == b"Do":
                xobject = xobjects[operands[0]] if operands[0] in xobjects else {}
                if xobject.get("/Subtype") == "/Image":
                    oc = xobject.get("/OC")
                    if oc is not None and _oc_hidden(oc, off, on, base_off):
                        hidden.add(n)
                    n += 1
            elif op == b"INLINE IMAGE":
                n += 1
        return hidden if n == count else set()  # the indexes line up only when pdfium lists the same images

    def image_boxes(self) -> list[Rect]:
        images = self._objects(raw.FPDF_PAGEOBJ_IMAGE)
        hidden = self._hidden_images(len(images))
        # PDFium types some image-painted shapes as shading objects, so count them as images
        return [self._box(*o.get_bounds()) for o in (*(o for i, o in enumerate(images) if i not in hidden), *self._objects(raw.FPDF_PAGEOBJ_SHADING))]

    def drawing_clusters(self) -> list[Rect]:
        return _merge([self._path_box(o) for o in self._objects(raw.FPDF_PAGEOBJ_PATH)], 3)

    def table_boxes(self) -> list[Rect]:
        h: list[Rect] = []
        v: list[Rect] = []
        for o in self._objects(raw.FPDF_PAGEOBJ_PATH):
            if any(raw.FPDFPathSegment_GetType(raw.FPDFPath_GetPathSegment(o, i)) == raw.FPDF_SEGMENT_BEZIERTO for i in range(raw.FPDFPath_CountSegments(o))):
                continue
            x0, y0, x1, y1 = self._path_box(o)
            if min(x1 - x0, y1 - y0) <= 2:
                (h if x1 - x0 >= y1 - y0 else v).append((x0, y0, x1, y1))
            elif raw.FPDFPath_CountSegments(o) <= 5:
                h += [(x0, y0, x1, y0), (x0, y1, x1, y1)]
                v += [(x0, y0, x0, y1), (x1, y0, x1, y1)]
        tables = []
        for box in _merge(h + v, 1):
            rows, cols = {round(e[1]) for e in h if _within(e, box)}, {round(e[0]) for e in v if _within(e, box)}
            if len(rows) >= 3 and len(cols) >= 3:
                tables.append(box)
        return tables

    def render_png(self, clip: Rect, dpi: int) -> bytes:
        s = dpi / 72
        # Cached per page so edits made after the first render do not change later clips.
        if dpi not in self._renders:
            self._renders[dpi] = self._page.render(scale=s).to_pil()
        eps = 1e-3
        box = (math.floor(clip[0] * s + eps), math.floor(clip[1] * s + eps), math.ceil(clip[2] * s - eps), math.ceil(clip[3] * s - eps))
        out = io.BytesIO()
        self._renders[dpi].crop(box).save(out, "PNG")
        return out.getvalue()

    def _write(self, x: float, y: float, text: str) -> None:
        obj = raw.FPDFPageObj_CreateTextObj(self._pdf.doc.raw, self._pdf.font, 6)
        buf = (ctypes.c_ushort * (len(text) + 1))(*map(ord, text), 0)
        raw.FPDFText_SetText(obj, ctypes.cast(buf, ctypes.POINTER(ctypes.c_ushort)))
        m = _mul((1, 0, 0, -1, x, y), _inv(self._m))
        raw.FPDFPageObj_Transform(obj, *m[:4], m[4] + self._origin[0], m[5] + self._origin[1])
        raw.FPDFPage_InsertObject(self._page.raw, obj)
        self._page.gen_content()

    def redact(self, rect: Rect, marker: str) -> None:
        self._pdf.redactions.setdefault(self._index, []).append((rect, marker))

    def apply_redactions(self, writer: pypdf.PdfWriter, regions: list[tuple[Rect, str]]) -> None:
        page = writer.pages[self._index]
        stream = ContentStream(page.get_contents(), writer, "bytes")
        shadings = [self._box(*o.get_bounds()) for o in self._objects(raw.FPDF_PAGEOBJ_SHADING)]
        original: Any = page["/Resources"]
        fonts, xobjects = (DictionaryObject(original.get(key, {})) for key in ("/Font", "/XObject"))
        resources = DictionaryObject(original)
        resources[NameObject("/Font")], resources[NameObject("/XObject")] = fonts, xobjects
        page[NameObject("/Resources")] = resources
        rects = [rect for rect, _ in regions]
        redactor = _Redactor(self, fonts, xobjects, rects, shadings if len(shadings) == sum(op == b"sh" for _, op in stream.operations) else [])
        ops = redactor.run(stream.operations) + [([], b"Q")] * len(redactor.stack)
        for name in redactor.dropped - {operands[0] for operands, op in ops if op == b"Do"}:
            del xobjects[name]
        for rect, marker in regions:
            m = _mul((1, 0, 0, -1, rect[0], rect[1] + 6), _inv(self._m))
            cm = [FloatObject(v) for v in (*m[:4], m[4] + self._origin[0], m[5] + self._origin[1])]
            ops += [([], b"q"), (cm, b"cm"), ([], b"BT"), ([NameObject("/FMark"), FloatObject(6)], b"Tf"), ([FloatObject(0), FloatObject(0)], b"Td"),
                    ([ByteStringObject(marker.encode())], b"Tj"), ([], b"ET"), ([], b"Q")]
        stream.operations = ops
        page.pop("/Thumb", None)
        annots: Any = page.get("/Annots")
        if annots is not None:
            page[NameObject("/Annots")] = ArrayObject(a for a in annots.get_object() if not any(_overlaps(self._box(*a.get_object()["/Rect"]), r) for r in rects))
        page.replace_contents(stream)
        page.compress_content_streams()
        fonts[NameObject("/FMark")] = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                                                                    NameObject("/BaseFont"): NameObject("/Helvetica"), NameObject("/Encoding"): NameObject("/WinAnsiEncoding")})

    def caption(self, rect: Rect, marker: str) -> None:
        self._write(rect[0], rect[3] + 7, marker)


def rasterize(data: bytes, dpi: int) -> bytes:
    with Pdf.open(data) as source, pdfium.PdfDocument.new() as target:
        for i in range(len(source)):
            page = source.doc[i]
            w, h = page.get_size()
            out = target.new_page(w, h)
            image = pdfium.PdfImage.new(target)
            image.set_bitmap(page.render(scale=dpi / 72))
            image.set_matrix(pdfium.PdfMatrix(w, 0, 0, h, 0, 0))
            out.insert_obj(image)
            out.gen_content()
        buf = io.BytesIO()
        target.save(buf)
        return buf.getvalue()


def to_png(raw_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(raw_bytes)) as image:
        out = io.BytesIO()
        (image if image.mode in _PNG_MODES else image.convert("RGB")).save(out, "PNG")
        return out.getvalue()
