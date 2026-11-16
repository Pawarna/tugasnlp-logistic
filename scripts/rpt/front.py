"""Halaman depan: cover, abstrak, kata pengantar, daftar isi/tabel/gambar."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, Spacer, Table, TableStyle

from src import config

from . import engine
from .engine import W, CTX, CaptionIndex, bullet, hr, p, toc_flowable


def _front_title(text: str) -> Paragraph:
    style = ParagraphStyle("FrontTitle", parent=engine.STYLES["H1"], spaceAfter=14)
    return Paragraph(text, style)


def cover(ctx) -> None:
    s = engine.STYLES
    ctx.add(Spacer(1, 1.1 * cm))
    ctx.add(Paragraph("TUGAS P9  ·  STUDI KASUS KLASIFIKASI", s["CoverKicker"]))
    ctx.add(hr(color=engine.NAVY, thickness=1.6, width="28%", space_before=8, space_after=26))
    ctx.add(Paragraph(config.REPORT_TITLE, s["CoverTitle"]))
    ctx.add(Spacer(1, 0.5 * cm))
    ctx.add(Paragraph(
        "Studi kasus: prediksi pelanggan berhenti berlangganan (<i>customer churn</i>) "
        "pada perusahaan telekomunikasi", s["CoverSub"]))
    ctx.add(Spacer(1, 1.5 * cm))
    ctx.add(hr(color=engine.GRID, thickness=0.8, width="60%", space_before=4, space_after=16))
    ctx.add(Paragraph("Disusun oleh:", s["CoverSmall"]))
    ctx.add(Spacer(1, 0.35 * cm))

    rows = [[Paragraph(f"<b>{nama}</b>", engine.STYLES["Note"]),
             Paragraph(nim, engine.STYLES["Note"])] for nama, nim, _ in config.MEMBERS]
    t = Table(rows, colWidths=[8.4 * cm, 3.2 * cm], hAlign="CENTER")
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, engine.GRID),
    ]))
    ctx.add(t)
    ctx.add(Spacer(1, 2.1 * cm))
    ctx.add(hr(color=engine.NAVY, thickness=1.6, width="28%", space_before=4, space_after=14))
    ctx.add(Paragraph(config.PROGRAM.upper(), s["CoverSmall"]))
    ctx.add(Paragraph(config.FACULTY.upper(), s["CoverSmall"]))
    ctx.add(Paragraph(config.INSTITUTION.upper(), s["CoverSmall"]))
    ctx.add(Spacer(1, 0.5 * cm))
    ctx.add(Paragraph(f"<b>{config.YEAR}</b>", s["CoverSmall"]))
    ctx.add(PageBreak())


def abstract(ctx) -> None:
    m = CTX.metrics
    md = m["metrics_default"]
    ctx.add(_front_title("ABSTRAK"))
    ctx.add(p(
        "Penelitian ini membangun sistem prediksi <i>customer churn</i> pada industri "
        "telekomunikasi menggunakan algoritma klasifikasi <i>Logistic Regression</i>. "
        f"Data yang digunakan adalah <i>Telco Customer Churn</i> yang terdiri atas {engine.num(m['dataset']['n_rows'], 0)} "
        "pelanggan dengan 19 fitur demografis, layanan, kontrak, dan tagihan, serta satu "
        "variabel target biner. Proporsi pelanggan yang berhenti berlangganan sebesar "
        f"{engine.num(m['dataset']['churn_rate'])}%, sehingga data tergolong tidak seimbang. "
        "Tahapan penelitian meliputi pembersihan data, analisis data eksploratif, "
        "<i>one-hot encoding</i>, standardisasi, pembagian data latih dan uji secara "
        "<i>stratified</i> dengan perbandingan 80:20, pelatihan model, evaluasi, serta "
        "penentuan ambang keputusan yang selaras dengan kebutuhan bisnis."
    ))
    ctx.add(p(
        f"Model menghasilkan <i>accuracy</i> {engine.pct(md['accuracy'])}, "
        f"<i>precision</i> {engine.pct(md['precision'])}, <i>recall</i> {engine.pct(md['recall'])}, "
        f"<i>F1-score</i> {engine.pct(md['f1'])}, dan <i>ROC-AUC</i> {engine.num(md['roc_auc'], 4)} "
        "pada ambang 0,50. Nilai ROC-AUC tersebut menunjukkan kemampuan diskriminasi yang baik, "
        f"dan hasil validasi silang lima lipatan sebesar {engine.num(m['cv']['mean'], 4)} ± "
        f"{engine.num(m['cv']['std'], 4)} mengindikasikan model tidak mengalami <i>overfitting</i>. "
        "Dengan menurunkan ambang keputusan menjadi 0,30, <i>recall</i> meningkat menjadi "
        f"{engine.pct(m['metrics_recommended']['recall'])}, sehingga proporsi pelanggan berisiko yang "
        "berhasil diidentifikasi menjadi lebih besar. Faktor paling dominan adalah tipe kontrak, "
        "masa berlangganan, dan jenis layanan internet. Model juga diterjemahkan menjadi sistem "
        "yang dapat dijalankan ulang (<i>pipeline</i>), antarmuka <i>command line</i> untuk "
        "memberi skor pada pelanggan baru, dan <i>dashboard</i> pemantauan risiko."
    ))
    ctx.add(Spacer(1, 0.4 * cm))
    ctx.add(Paragraph(
        "<b>Kata kunci:</b> <i>customer churn</i>, klasifikasi, <i>logistic regression</i>, "
        "<i>odds ratio</i>, <i>threshold tuning</i>, ROC-AUC", engine.STYLES["Note"]))
    ctx.add(PageBreak())


def preface(ctx) -> None:
    ctx.add(_front_title("KATA PENGANTAR"))
    ctx.add(p(
        "Puji syukur kami panjatkan kepada Tuhan Yang Maha Esa karena atas rahmat-Nya "
        "laporan Tugas P9 dengan topik <b>Klasifikasi Logistic Regression</b> ini dapat "
        "diselesaikan dengan baik. Laporan ini disusun untuk memenuhi salah satu tugas "
        "mata kuliah pada pertemuan kesembilan, sekaligus menjadi bahan presentasi kelompok "
        "pada pertemuan berikutnya."))
    ctx.add(p(
        "Topik yang kami pilih adalah prediksi <i>customer churn</i> pada perusahaan "
        "telekomunikasi. Kami tidak berhenti pada pembahasan teori, tetapi juga membangun "
        "sistem lengkap: <i>pipeline</i> pelatihan model yang dapat dijalankan ulang, evaluasi "
        "dengan beberapa metrik, penentuan ambang keputusan, hingga <i>dashboard</i> yang "
        "menampilkan pelanggan paling berisiko. Seluruh angka yang disajikan dalam laporan ini "
        "berasal dari eksekusi kode yang dilampirkan, bukan nilai perkiraan."))
    ctx.add(p(
        "Kami menyadari laporan ini masih memiliki kekurangan, terutama pada cakupan data "
        "yang bersifat <i>cross-sectional</i> dan belum dilakukannya pencarian hiperparameter "
        "secara menyeluruh. Oleh karena itu, kritik dan saran yang membangun sangat kami "
        "harapkan demi perbaikan pada tugas-tugas berikutnya."))
    ctx.add(p("Terima kasih kepada dosen pengampu atas arahan yang diberikan, serta kepada "
              "seluruh anggota kelompok yang telah berkontribusi sesuai pembagian tugas."))
    ctx.add(Spacer(1, 1.4 * cm))
    ctx.add(Paragraph(f"{config.CITY}, 16 November {config.YEAR}",
                      engine.STYLES["EqNum"]))
    ctx.add(Spacer(1, 0.2 * cm))
    ctx.add(Paragraph("<b>Kelompok 1</b>", engine.STYLES["EqNum"]))
    ctx.add(PageBreak())


def toc(ctx) -> None:
    ctx.add(_front_title("DAFTAR ISI"))
    ctx.add(toc_flowable())
    ctx.add(PageBreak())


def list_of_tables(ctx) -> None:
    ctx.add(_front_title("DAFTAR TABEL"))
    ctx.add(Paragraph(
        "Seluruh tabel berikut dihasilkan langsung oleh <font name='Courier'>run_pipeline.py</font> "
        "sehingga konsisten dengan angka pada Bab IV.",
        ParagraphStyle("NoteSmall", parent=engine.STYLES["Note"], fontSize=9.2, spaceAfter=12)))
    ctx.add(CaptionIndex("tabel"))
    ctx.add(PageBreak())


def list_of_figures(ctx) -> None:
    ctx.add(_front_title("DAFTAR GAMBAR"))
    ctx.add(CaptionIndex("gambar"))
    ctx.add(PageBreak())


def build(ctx) -> None:
    cover(ctx)
    abstract(ctx)
    preface(ctx)
    toc(ctx)
    list_of_tables(ctx)
    list_of_figures(ctx)