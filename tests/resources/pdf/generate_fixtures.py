"""Regenerates the synthetic PDFs in this directory.

One-off developer script. It uses PyMuPDF only to author the fixtures; the tests
that consume them must not import PyMuPDF. Run: python tests/resources/pdf/generate_fixtures.py
"""

import math
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent


def png(color: int, size: int = 16) -> bytes:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, size, size), False)
    pixmap.clear_with(color)
    return pixmap.tobytes("png")


def one_image() -> None:
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=200)
        page.insert_image(pymupdf.Rect(40, 60, 360, 140), stream=png(0x336699))
        pdf.save(HERE / "one_image.pdf")


def multi_page() -> None:
    """p1: two different images. p2: two identical images. p3: one tiny and one near-full-page image."""
    with pymupdf.open() as pdf:
        p1 = pdf.new_page(width=400, height=300)
        p1.insert_image(pymupdf.Rect(20, 20, 180, 120), stream=png(0xCC3333))
        p1.insert_image(pymupdf.Rect(220, 150, 380, 270), stream=png(0x33CC33))
        p2 = pdf.new_page(width=400, height=300)
        same = png(0x3333CC)
        p2.insert_image(pymupdf.Rect(20, 20, 180, 120), stream=same)
        p2.insert_image(pymupdf.Rect(220, 150, 380, 250), stream=same)  # same size as the first so rendered bytes match
        p3 = pdf.new_page(width=400, height=300)
        p3.insert_image(pymupdf.Rect(10, 10, 20, 20), stream=png(0x999999))  # ~0.08% of page: below 0.1% floor
        p3.insert_image(pymupdf.Rect(5, 5, 395, 295), stream=png(0x666666))  # ~95% of page: above 80% ceiling
        pdf.save(HERE / "multi_page.pdf")


def vector_shapes() -> None:
    """Vector-only figure (rect + circle) plus a ruled table, on one page, with body text."""
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=500, height=500)
        page.insert_text((50, 40), "Quarterly report", fontsize=14)
        page.draw_rect(pymupdf.Rect(50, 60, 200, 160), color=(0, 0, 0), fill=(0.8, 0.9, 1), width=1)
        page.draw_circle(pymupdf.Point(125, 110), 30, color=(1, 0, 0), fill=(1, 0.8, 0.8), width=1)
        top, left, rows, cols, cw, rh = 240, 50, 3, 3, 120, 30
        for r in range(rows + 1):
            page.draw_line((left, top + r * rh), (left + cols * cw, top + r * rh), color=(0, 0, 0), width=1)
        for c in range(cols + 1):
            page.draw_line((left + c * cw, top), (left + c * cw, top + rows * rh), color=(0, 0, 0), width=1)
        for r in range(rows):
            for c in range(cols):
                page.insert_text((left + c * cw + 6, top + r * rh + 20), f"r{r}c{c}", fontsize=10)
        pdf.save(HERE / "vector_shapes.pdf")


def math_chem() -> None:
    """Maths and chemistry as PDFs really carry them: glyph text plus thin vector strokes, no MathML.

    - radical sign (polyline + overline) and matrix brackets: vector strokes that get extracted as figures
    - fraction bar: a hairline whose area is below the 0.1% floor, so it is skipped
    - summation and integral signs: glyph text from the Symbol font, never extracted
    - benzene ring with a bond to an OH label, and a reaction arrow: vector figures
    - formulas typed as text ("2 H2 + O2 -> 2 H2O"): never extracted
    """
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=500, height=500)
        page.insert_text((50, 60), "E = mc", fontsize=16)
        page.insert_text((50, 120), "a+b", fontsize=14)
        page.draw_line((45, 125), (90, 125), width=0.8)
        page.insert_text((55, 140), "c", fontsize=14)
        page.draw_polyline([(150, 130), (155, 138), (162, 112), (190, 112)], width=0.8)
        page.insert_text((165, 130), "x+1", fontsize=14)
        page.insert_text((250, 130), "\u2211", fontname="symb", fontsize=24)
        page.insert_text((300, 130), "\u222b", fontname="symb", fontsize=30)
        page.draw_polyline([(360, 100), (355, 100), (355, 160), (360, 160)], width=1)
        page.draw_polyline([(420, 100), (425, 100), (425, 160), (420, 160)], width=1)
        for i, (x, y) in enumerate([(365, 125), (395, 125), (365, 148), (395, 148)]):
            page.insert_text((x, y), str(i), fontsize=12)

        # chemistry, lower on the page so nothing merges with the maths above
        cx, cy, r = 120, 320, 40
        outer = [(cx + r * math.cos(math.radians(60 * i + 30)), cy + r * math.sin(math.radians(60 * i + 30))) for i in range(6)]
        inner = [(cx + (r - 8) * math.cos(math.radians(60 * i + 30)), cy + (r - 8) * math.sin(math.radians(60 * i + 30))) for i in range(6)]
        page.draw_polyline([*outer, outer[0]], width=1.2)
        for i in (0, 2, 4):
            page.draw_line(inner[i], inner[(i + 1) % 6], width=1)
        page.draw_line(outer[0], (outer[0][0] + 25, outer[0][1] + 15), width=1.2)
        page.insert_text((outer[0][0] + 27, outer[0][1] + 22), "OH", fontsize=12)
        page.draw_line((250, 320), (320, 320), width=1.5)
        page.draw_polyline([(312, 314), (320, 320), (312, 326)], width=1.5)
        page.insert_text((255, 308), "heat", fontsize=9)
        page.insert_text((50, 440), "2 H2 + O2  ->  2 H2O", fontsize=14)
        page.insert_text((60, 470), "CH3-CH2-OH  +  O2", fontsize=12)
        pdf.save(HERE / "math_chem.pdf")


def metadata_rich() -> None:
    """3 pages with outline, metadata, link, attachment (MathML), text annotation and a form field. Page 1 holds one image."""
    with pymupdf.open() as pdf:
        for i in range(3):
            pdf.new_page(width=400, height=300).insert_text((50, 50), f"Page {i}", fontsize=14)
        pdf[0].insert_image(pymupdf.Rect(40, 100, 200, 200), stream=png(0xFF0000))
        pdf.set_toc([[1, "Intro", 1], [1, "Two", 2]])
        pdf.set_metadata({"title": "Title", "author": "Author"})
        pdf[0].insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(50, 40, 120, 55), "uri": "https://example.org"})
        pdf.embfile_add("equation.mathml", b"<math><mi>x</mi></math>", filename="equation.mathml")
        pdf[1].add_text_annot((60, 60), "note")
        widget = pymupdf.Widget()
        widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
        widget.rect = pymupdf.Rect(50, 100, 200, 120)
        widget.field_name = "f"
        widget.field_value = "val"
        pdf[1].add_widget(widget)
        pdf.save(HERE / "metadata_rich.pdf")


def edge_pages() -> None:
    """Small single-purpose PDFs for page-geometry and layering edge cases."""
    # page rotated 90 degrees (stored 400x200, displayed 200x400)
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=200)
        page.insert_image(pymupdf.Rect(40, 60, 360, 140), stream=png(0xFF0000))
        page.set_rotation(90)
        pdf.save(HERE / "edge_rotated.pdf")
    # CropBox smaller than MediaBox: one image bigger than the crop, one inside it
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=400)
        page.insert_image(pymupdf.Rect(20, 20, 380, 380), stream=png(0x00FF00))
        page.insert_image(pymupdf.Rect(100, 100, 200, 200), stream=png(0x0000FF))
        page.set_cropbox(pymupdf.Rect(50, 50, 300, 300))
        pdf.save(HERE / "edge_cropbox.pdf")
    # image on an optional-content layer that is switched off
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=200)
        hidden = pdf.add_ocg("hidden", on=False)
        page.insert_image(pymupdf.Rect(40, 60, 360, 140), stream=png(0xFF0000), oc=hidden)
        pdf.save(HERE / "edge_hidden_layer.pdf")
    # PNG with an alpha channel
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=200)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 16, 16), True)
        pixmap.clear_with(0x80)
        page.insert_image(pymupdf.Rect(40, 60, 360, 140), stream=pixmap.tobytes("png"))
        pdf.save(HERE / "edge_alpha_image.pdf")
    # image placed with a 90 degree rotation
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=400)
        page.insert_image(pymupdf.Rect(100, 100, 300, 200), stream=png(0xFF0000), rotate=90)
        pdf.save(HERE / "edge_rotated_image.pdf")
    # text and a ruled line crossing the image, plus text well outside it (redact side effects)
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=200)
        page.insert_image(pymupdf.Rect(40, 60, 360, 140), stream=png(0xFF0000))
        page.insert_text((100, 100), "OVERLAPTEXT", fontsize=12)
        page.draw_line((50, 70), (350, 70), width=2)
        page.insert_text((100, 30), "OUTSIDE", fontsize=12)
        pdf.save(HERE / "edge_overlap_text.pdf")
    # highlight annotation over text
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=200)
        page.insert_text((50, 50), "highlighted", fontsize=14)
        page.add_highlight_annot(pymupdf.Rect(40, 35, 120, 55))
        pdf.save(HERE / "edge_annotated.pdf")
    # encryption: user password required, and owner-password-only (opens without a password)
    for name, user_pw in (("edge_encrypted_user_pw.pdf", "secret"), ("edge_encrypted_owner_only.pdf", "")):
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=400, height=200)
            page.insert_image(pymupdf.Rect(40, 60, 360, 140), stream=png(0xFF0000))
            (HERE / name).write_bytes(pdf.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=user_pw, owner_pw="owner"))


def image_formats() -> None:
    """Raster files for the image-output conversion tests (WebP lives in tests/resources already)."""
    out = HERE.parent / "images"
    out.mkdir(exist_ok=True)
    cmyk = pymupdf.Pixmap(pymupdf.csCMYK, pymupdf.IRect(0, 0, 8, 8), False)
    cmyk.clear_with(100)
    (out / "cmyk.jpg").write_bytes(cmyk.tobytes("jpg"))
    gray = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 8, 8), False)
    gray.clear_with(100)
    (out / "gray.png").write_bytes(gray.tobytes("png"))
    (out / "gray.jpg").write_bytes(gray.tobytes("jpg"))
    alpha = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), True)
    alpha.clear_with(100)
    (out / "alpha.png").write_bytes(alpha.tobytes("png"))


if __name__ == "__main__":
    one_image()
    multi_page()
    vector_shapes()
    math_chem()
    metadata_rich()
    edge_pages()
    image_formats()
