"""Mesin tata letak laporan PDF (ReportLab Platypus)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

# --------------------------------------------------------------------- warna
NAVY = colors.HexColor("#1F3864")
BLUE = colors.HexColor("#2E75B6")
RED = colors.HexColor("#C00000")
GRAY = colors.HexColor("#7F7F7F")
LIGHT = colors.HexColor("#D9E2F3")
ROW_ALT = colors.HexColor("#F3F6FB")
GRID = colors.HexColor("#C9D3E2")
CODE_BG = colors.HexColor("#F6F8FB")

# --------------------------------------------------------------------- ukuran
PAGE_W, PAGE_H = A4
MARGIN_L, MARGIN_R = 3.0 * cm, 2.3 * cm
MARGIN_T, MARGIN_B = 2.4 * cm, 2.2 * cm
W = PAGE_W - MARGIN_L - MARGIN_R

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "artifacts" / "figures"
ART_DIR = ROOT / "artifacts"
EQ_DIR = FIG_DIR / "persamaan"
EQ_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------- font
def _register() -> tuple[str, str]:
    """Daftarkan Times (serif) dan Arial (sans) dari font sistem Windows."""
    serif, sans = "Times-Roman", "Helvetica"
    fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    pairs = {
        "Serif": ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"),
        "Sans": ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"),
    }
    for family, (normal, bold, italic, bold_italic) in pairs.items():
        try:
            pdfmetrics.registerFont(TTFont(family, str(fonts / normal)))
            pdfmetrics.registerFont(TTFont(f"{family}-Bold", str(fonts / bold)))
            pdfmetrics.registerFont(TTFont(f"{family}-Italic", str(fonts / italic)))
            pdfmetrics.registerFont(TTFont(f"{family}-BoldItalic", str(fonts / bold_italic)))
            pdfmetrics.registerFontFamily(
                family,
                normal=family,
                bold=f"{family}-Bold",
                italic=f"{family}-Italic",
                boldItalic=f"{family}-BoldItalic",
            )
            if family == "Serif":
                serif = family
            else:
                sans = family
        except Exception:  # pragma: no cover - fallback bila font tidak ada
            pass
    return serif, sans


SERIF, SANS = _register()


def _style(name: str, **kw) -> ParagraphStyle:
    base = dict(fontName=SERIF, fontSize=11, leading=17.6, alignment=TA_JUSTIFY,
                textColor=colors.HexColor("#1A1A1A"), spaceBefore=0, spaceAfter=0)
    base.update(kw)
    return ParagraphStyle(name, **base)


STYLES = {
    "Body": _style("Body", firstLineIndent=1.0 * cm, spaceAfter=2),
    "BodyTight": _style("BodyTight", firstLineIndent=0, spaceAfter=6),
    "Bullet": _style("Bullet", leftIndent=0.95 * cm, bulletIndent=0.25 * cm,
                     firstLineIndent=0, spaceAfter=3),
    "H1": _style("H1", fontName=f"{SANS}-Bold", fontSize=15, leading=20,
                 alignment=TA_CENTER, spaceBefore=2, spaceAfter=4),
    "ChapterNum": _style("ChapterNum", fontName=f"{SANS}-Bold", fontSize=12.5,
                         leading=16, alignment=TA_CENTER, textColor=NAVY,
                         spaceAfter=2),
    "H2": _style("H2", fontName=f"{SANS}-Bold", fontSize=11.5, leading=15.5,
                 alignment=TA_LEFT, spaceBefore=13, spaceAfter=5,
                 textColor=colors.HexColor("#16233F")),
    "H3": _style("H3", fontName=f"{SANS}-Bold", fontSize=10.5, leading=14,
                 alignment=TA_LEFT, spaceBefore=9, spaceAfter=3,
                 textColor=colors.HexColor("#243B63")),
    "CapTable": _style("CapTable", fontName=SANS, fontSize=8.8, leading=12,
                       alignment=TA_CENTER, spaceBefore=8, spaceAfter=4,
                       textColor=colors.HexColor("#333333")),
    "CapFigure": _style("CapFigure", fontName=SANS, fontSize=8.8, leading=12,
                        alignment=TA_CENTER, spaceBefore=3, spaceAfter=9,
                        textColor=colors.HexColor("#333333")),
    "TableText": _style("TableText", fontSize=8.7, leading=11.2, alignment=TA_LEFT),
    "TableHead": _style("TableHead", fontName=f"{SANS}-Bold", fontSize=8.2,
                        leading=10.6, alignment=TA_CENTER, textColor=colors.white),
    "Code": _style("Code", fontName="Courier", fontSize=7.6, leading=9.8,
                   alignment=TA_LEFT),
    "EqNum": _style("EqNum", fontSize=10, alignment=TA_RIGHT),
    "CoverKicker": _style("CoverKicker", fontName=f"{SANS}-Bold", fontSize=10.5,
                          leading=14, alignment=TA_CENTER, textColor=NAVY),
    "CoverTitle": _style("CoverTitle", fontName=f"{SANS}-Bold", fontSize=19,
                         leading=26, alignment=TA_CENTER,
                         textColor=colors.HexColor("#16233F")),
    "CoverSub": _style("CoverSub", fontSize=12.5, leading=19, alignment=TA_CENTER,
                       textColor=colors.HexColor("#404040")),
    "CoverSmall": _style("CoverSmall", fontName=SANS, fontSize=10.5, leading=15,
                         alignment=TA_CENTER, textColor=colors.HexColor("#333333")),
    "Note": _style("Note", fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY,
                   textColor=colors.HexColor("#333333")),
}

TOC_STYLES = [
    _style("TOC0", fontName=f"{SANS}-Bold", fontSize=10, leading=15,
           alignment=TA_LEFT, spaceBefore=5, textColor=NAVY),
    _style("TOC1", fontSize=10.5, leading=15, alignment=TA_LEFT, leftIndent=0.6 * cm),
    _style("TOC2", fontSize=10, leading=14, alignment=TA_LEFT,
           leftIndent=1.2 * cm, textColor=colors.HexColor("#333333")),
]


# ------------------------------------------------------------------- helper
class Ctx:
    """State yang dibagi antar modul konten."""

    def __init__(self) -> None:
        self.figures: list[tuple[str, str]] = []      # (label, judul)
        self.tables: list[tuple[str, str]] = []
        self.caption_pages: dict[tuple[str, str], int] = {}
        self.metrics = json.loads((ART_DIR / "metrics.json").read_text(encoding="utf-8"))
        self.eq_counter = 0
        self.flow: list[Flowable] = []

    def add(self, *flowables) -> None:
        for f in flowables:
            if f is not None:
                self.flow.append(f)


CTX = Ctx()


def num(value: float, digits: int = 2) -> str:
    """Format angka gaya Indonesia (koma desimal, titik ribuan)."""
    return f"{value:,.{digits}f}".replace(",", "\u00a0").replace(".", ",").replace("\u00a0", ".")


def pct(value: float, digits: int = 2) -> str:
    return f"{num(value * 100, digits)}%"


def p(text: str, style: str = "Body"):
    return Paragraph(text, STYLES[style])


def bullet(text: str):
    return Paragraph(text, STYLES["Bullet"], bulletText="\u2022")


def numbered(items: list[str], start: int = 1):
    return [
        Paragraph(item, STYLES["Bullet"], bulletText=f"{i}.")
        for i, item in enumerate(items, start=start)
    ]


def h2(text: str):
    return Paragraph(text, STYLES["H2"])


def h3(text: str):
    return Paragraph(text, STYLES["H3"])


def quote(text: str):
    return Paragraph(text, STYLES["Note"])


def chapter(kicker: str, title: str):
    """Judul bab + garis pemisah; entri TOC memakai satu baris utuh."""
    heading = Paragraph(title.upper(), STYLES["H1"])
    heading._toc_text = f"{kicker} {title.upper()}"
    return [
        Spacer(1, 6),
        Paragraph(kicker, STYLES["ChapterNum"]),
        heading,
        HRFlowable(width="45%", thickness=1.4, color=NAVY,
                   spaceBefore=4, spaceAfter=16, hAlign="CENTER"),
    ]


def img_box(path: Path, width: float | None = None, dpi: float = 220) -> Image:
    with PILImage.open(path) as im:
        px_w, px_h = im.size
    width = width or W
    natural = px_w / dpi * 2.54 * cm
    if natural <= width:
        width = natural
    ratio = px_h / px_w
    return Image(str(path), width=width, height=width * ratio)


def figure(filename: str, caption: str, label: str, width: float | None = None):
    """Gambar + keterangan di bawah; otomatis masuk daftar gambar."""
    CTX.figures.append((label, caption))
    cap = Paragraph(f"<b>{label}</b>  {caption}", STYLES["CapFigure"])
    cap._caption_key = ("gambar", label)
    img = img_box(FIG_DIR / filename, width)
    return KeepTogether([Spacer(1, 6), img, cap])


def _cell(value, header: bool = False, align: str = "LEFT"):
    alignment = {"LEFT": TA_LEFT, "CENTER": TA_CENTER, "RIGHT": TA_RIGHT}[align]
    style = STYLES["TableHead"] if header else STYLES["TableText"]
    style = ParagraphStyle(
        f"{style.name}-{align}", parent=style, alignment=alignment
    )
    return Paragraph(str(value), style)


def table_block(
    header: list[str],
    rows: list[list],
    widths: list[float],
    caption: str,
    label: str,
    align: list[str] | None = None,
    font_size: float = 8.7,
    keep: bool = True,
    head_bg=NAVY,
    padding: float = 4,
):
    """Tabel bergaris + keterangan di atas; otomatis masuk daftar tabel."""
    align = align or ["LEFT"] * len(header)
    data = [[_cell(h, header=True, align="CENTER") for h in header]]
    for row in rows:
        data.append([_cell(v, align=a) for v, a in zip(row, align)])

    table = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), head_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("BOX", (0, 0), (-1, -1), 0.7, NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))

    if font_size != 8.7:
        for i, row in enumerate(data):
            for j, cell in enumerate(row):
                parent = STYLES["TableHead"] if i == 0 else STYLES["TableText"]
                cell.style = ParagraphStyle(
                    f"{parent.name}-{font_size}-{align[j]}",
                    parent=parent,
                    fontSize=font_size,
                    leading=font_size * 1.30,
                    alignment={"LEFT": TA_LEFT, "CENTER": TA_CENTER, "RIGHT": TA_RIGHT}[align[j]],
                )

    table.setStyle(TableStyle(style))

    CTX.tables.append((label, caption))
    cap = Paragraph(f"<b>{label}</b>  {caption}", STYLES["CapTable"])
    cap._caption_key = ("tabel", label)

    if keep and len(rows) <= 18:
        return [KeepTogether([cap, table, Spacer(1, 9)])]
    return [cap, table, Spacer(1, 9)]


def code_block(text: str, caption: str | None = None):
    body = Preformatted(text.strip("\n"), STYLES["Code"])
    table = Table([[body]], colWidths=[W], hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    out = [Spacer(1, 3), table]
    if caption:
        out.append(Paragraph(caption, STYLES["CapTable"]))
    out.append(Spacer(1, 9))
    return out


def equation(tex: str, label: str):
    """Render persamaan dengan mathtext lalu tempel sebagai gambar."""
    safe = "".join(c if c.isalnum() else "_" for c in label)
    path = EQ_DIR / f"eq_{safe}.png"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=12)
    fig.savefig(path, dpi=420, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)

    with PILImage.open(path) as im:
        px_w, px_h = im.size
    width = px_w / 420 * 2.54 * cm
    height = px_h / 420 * 2.54 * cm
    width = min(width, W - 3.0 * cm)

    img = Image(str(path), width=width, height=height * (width / (px_w / 420 * 2.54 * cm)))
    number = Paragraph(f"({label})", STYLES["EqNum"])
    row = Table([[img, number]], colWidths=[W - 2.2 * cm, 2.2 * cm], hAlign="CENTER")
    row.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return [Spacer(1, 2), row, Spacer(1, 4)]


class CaptionIndex(Flowable):
    """Daftar tabel/gambar dengan dot leader; halaman diambil saat render."""

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self.entries = list(CTX.tables if kind == "tabel" else CTX.figures)
        self.line_height = 15
        self.pad = 0.15 * cm

    def wrap(self, availWidth, availHeight):
        self.width = availWidth
        self.height = self.line_height * len(self.entries) + 4
        return self.width, self.height

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        canvas.setFont(SANS, 9.6)
        y = self.height - self.line_height
        for label, title in self.entries:
            text = f"{label}  {title}"
            while canvas.stringWidth(text, SANS, 9.6) > self.width - 1.6 * cm and len(text) > 12:
                text = text[:-2]
            if text != f"{label}  {title}":
                text = text.rstrip(". ") + "..."

            canvas.setFillColor(colors.HexColor("#1A1A1A"))
            canvas.drawString(0, y, text)
            text_w = canvas.stringWidth(text, SANS, 9.6)

            page = CTX.caption_pages.get((self.kind, label), "")
            page_text = str(page) if page else "\u2013"
            page_w = canvas.stringWidth(page_text, SANS, 9.6)
            canvas.drawRightString(self.width, y, page_text)

            dot_start = text_w + 3
            dot_end = self.width - page_w - 3
            if dot_end > dot_start:
                canvas.setFillColor(colors.HexColor("#B4BCC8"))
                x = dot_start
                while x < dot_end:
                    canvas.circle(x, y + 1.6, 0.42, stroke=0, fill=1)
                    x += 4.2
            y -= self.line_height
        canvas.restoreState()


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str, **kw):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=MARGIN_L,
            rightMargin=MARGIN_R,
            topMargin=MARGIN_T,
            bottomMargin=MARGIN_B,
            title="Laporan Tugas P9 - Kelompok 1",
            author="Kelompok 1",
            subject="Klasifikasi Logistic Regression - Customer Churn",
            **kw,
        )
        frame = Frame(
            MARGIN_L, MARGIN_B, W, PAGE_H - MARGIN_T - MARGIN_B,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )
        self.addPageTemplates(
            [PageTemplate(id="main", frames=[frame], onPage=self._decorate)]
        )

    @staticmethod
    def _decorate(canvas, doc):
        page = canvas.getPageNumber()
        canvas.saveState()
        if page > 1:
            # header
            canvas.setStrokeColor(colors.HexColor("#C9D3E2"))
            canvas.setLineWidth(0.6)
            canvas.line(MARGIN_L, PAGE_H - 1.75 * cm, PAGE_W - MARGIN_R, PAGE_H - 1.75 * cm)
            canvas.setFont(SANS, 7.8)
            canvas.setFillColor(GRAY)
            canvas.drawString(MARGIN_L, PAGE_H - 1.55 * cm,
                              "Sistem Prediksi Customer Churn \u2014 Logistic Regression")
            canvas.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 1.55 * cm,
                                   "Kelompok 1")
            # footer
            canvas.setFont(SANS, 8.6)
            canvas.setFillColor(colors.HexColor("#404040"))
            canvas.drawCentredString(PAGE_W / 2, 1.35 * cm, str(page))
            canvas.setStrokeColor(colors.HexColor("#DCE3EC"))
            canvas.line(PAGE_W / 2 - 2.2 * cm, 1.72 * cm, PAGE_W / 2 + 2.2 * cm, 1.72 * cm)
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        name = flowable.style.name
        if name in ("H1", "H2", "H3"):
            level = {"H1": 0, "H2": 1, "H3": 2}[name]
            text = getattr(flowable, "_toc_text", None) or flowable.getPlainText()
            self.notify("TOCEntry", (level, text, self.page))
        key = getattr(flowable, "_caption_key", None)
        if key:
            CTX.caption_pages[key] = self.page


def toc_flowable() -> TableOfContents:
    toc = TableOfContents()
    toc.levelStyles = TOC_STYLES
    toc.dotsMinLevel = 0
    return toc


def build_pdf(output: Path) -> Path:
    """Bangun PDF dua putaran agar nomor halaman daftar tabel/gambar akurat."""
    output.parent.mkdir(parents=True, exist_ok=True)
    from scripts.content import assemble

    for _ in range(2):
        CTX.flow = []
        assemble(CTX)
        doc = ReportDoc(str(output))
        doc.multiBuild(CTX.flow)
    return output


def hr(color=GRID, thickness=0.8, space_before=4, space_after=4, width="100%"):
    return HRFlowable(width=width, thickness=thickness, color=color,
                      spaceBefore=space_before, spaceAfter=space_after)


def page_break():
    return PageBreak()