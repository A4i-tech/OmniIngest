from __future__ import annotations

import ctypes
import io
import math
from functools import cached_property
from typing import Any

import pypdf
import pypdfium2 as pdfium
import pypdfium2.raw as raw
from PIL import Image
from pypdf.generic import ContentStream, IndirectObject

Rect = tuple[float, float, float, float]
Matrix = tuple[float, float, float, float, float, float]

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


def _within(e: Rect, box: Rect) -> bool:
    return box[0] - 0.5 <= e[0] and box[1] - 0.5 <= e[1] and e[2] <= box[2] + 0.5 and e[3] <= box[3] + 0.5


def _mul(m: Matrix, n: Matrix) -> Matrix:
    return (m[0] * n[0] + m[1] * n[2], m[0] * n[1] + m[1] * n[3], m[2] * n[0] + m[3] * n[2], m[2] * n[1] + m[3] * n[3],
            m[4] * n[0] + m[5] * n[2] + n[4], m[4] * n[1] + m[5] * n[3] + n[5])


def _inv(m: Matrix) -> Matrix:
    a, b, c, d, e, f = m
    k = a * d - b * c
    return (d / k, -b / k, -c / k, a / k, (c * f - d * e) / k, (b * e - a * f) / k)


class Pdf:
    def __init__(self, data: bytes, doc: pdfium.PdfDocument):
        self.data, self.doc = data, doc

    @classmethod
    def open(cls, data: bytes) -> Pdf:
        try:
            return cls(data, pdfium.PdfDocument(data))
        except pdfium.PdfiumError as e:
            if e.err_code == raw.FPDF_ERR_PASSWORD:
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
        self.doc.save(out)
        return out.getvalue()

    @cached_property
    def reader(self) -> pypdf.PdfReader:
        return pypdf.PdfReader(io.BytesIO(self.data))

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
        if not off:
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
                    if isinstance(oc, IndirectObject) and oc.idnum in off:
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
        for kind in (raw.FPDF_PAGEOBJ_IMAGE, raw.FPDF_PAGEOBJ_SHADING, raw.FPDF_PAGEOBJ_TEXT):
            for o in self._objects(kind):
                if _overlaps(self._box(*o.get_bounds()), rect):
                    self._page.remove_obj(o)
        self._write(rect[0], rect[1] + 6, marker)

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
