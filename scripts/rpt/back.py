"""Daftar pustaka dan lampiran."""
from __future__ import annotations

import pandas as pd
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, Spacer

from src import config

from . import engine
from .engine import CTX, bullet, code_block, h2, h3, p, table_block

BIB = [
    ("Ascarza, E. (2018). Retention futility: Targeting high-risk customers might be ineffective. "
     "<i>Journal of Marketing Research</i>, 55(1), 80-98."),
    ("Chawla, N. V., Bowyer, K. W., Hall, L. O., &amp; Kegelmeyer, W. P. (2002). SMOTE: Synthetic "
     "minority over-sampling technique. <i>Journal of Artificial Intelligence Research</i>, 16, "
     "321-357."),
    ("Fawcett, T. (2006). An introduction to ROC analysis. <i>Pattern Recognition Letters</i>, "
     "27(8), 861-874."),
    ("Gupta, S., &amp; Lehmann, D. R. (2006). Customer lifetime value and firm valuation. "
     "<i>Journal of Relationship Marketing</i>, 5(2-3), 87-110."),
    ("Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical "
     "Learning</i> (2nd ed.). Springer."),
    ("Hosmer, D. W., Lemeshow, S., &amp; Sturdivant, R. X. (2013). <i>Applied Logistic "
     "Regression</i> (3rd ed.). John Wiley &amp; Sons."),
    ("IBM. (2019). <i>Telco Customer Churn</i> [Dataset]. IBM Sample Data Sets. Diakses dari "
     "https://github.com/IBM/telco-customer-churn-on-icp4d"),
    ("James, G., Witten, D., Hastie, T., &amp; Tibshirani, R. (2021). <i>An Introduction to "
     "Statistical Learning with Applications in R</i> (2nd ed.). Springer."),
    ("Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... "
     "&amp; Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. <i>Journal of "
     "Machine Learning Research</i>, 12, 2825-2830."),
    ("Provost, F., &amp; Fawcett, T. (2013). <i>Data Science for Business</i>. O'Reilly Media."),
    ("Reichheld, F. F., &amp; Sasser, W. E. (1990). Zero defections: Quality comes to services. "
     "<i>Harvard Business Review</i>, 68(5), 105-111."),
    ("Verbeke, W., Dejaeger, K., Martens, D., Hur, J., &amp; Baesens, B. (2012). New insights into "
     "churn prediction in the telecommunication sector: A profit driven data mining approach. "
     "<i>European Journal of Operational Research</i>, 218(1), 211-229."),
]


def _bib_style() -> ParagraphStyle:
    return ParagraphStyle(
        "Bib", fontName=engine.SERIF, fontSize=10.5, leading=15.5,
        alignment=TA_JUSTIFY, leftIndent=0.9 * cm, firstLineIndent=-0.9 * cm,
        spaceAfter=7, textColor=engine.colors.HexColor("#1A1A1A"),
    )


def bibliography(ctx) -> None:
    ctx.add(PageBreak())
    ctx.add(Paragraph("DAFTAR PUSTAKA", ParagraphStyle(
        "BackTitle", parent=engine.STYLES["H1"], spaceAfter=16)))
    style = _bib_style()
    for item in sorted(BIB):
        ctx.add(Paragraph(item, style))


def appendix(ctx) -> None:
    ctx.add(PageBreak())
    ctx.add(Paragraph("LAMPIRAN A", ParagraphStyle(
        "BackTitle2", parent=engine.STYLES["H1"], spaceAfter=4)))
    ctx.add(Paragraph("Artefak yang Dihasilkan Sistem", engine.STYLES["ChapterNum"]))
    ctx.add(Spacer(1, 0.4 * cm))

    ctx.add(p(
        "Setiap eksekusi <font name='Courier'>run_pipeline.py</font> menghasilkan berkas berikut. "
        "Seluruh angka di dalam laporan ini dibaca langsung dari berkas-berkas tersebut, sehingga "
        "laporan tidak dapat memuat nilai yang berbeda dari hasil eksekusi."
    ))
    ctx.add(*table_block(
        ["Berkas", "Isi", "Dipakai pada"],
        [
            ["artifacts/model.joblib", "Pipeline terlatih (praproses + model)", "predict.py, dashboard"],
            ["artifacts/metrics.json", "Seluruh metrik, confusion matrix, hasil validasi silang",
             "Bab IV, dashboard"],
            ["artifacts/coefficients.csv", "Koefisien dan odds ratio 30 fitur", "Subbab 4.5, Lampiran D"],
            ["artifacts/threshold_tuning.csv", "Metrik pada 10 nilai ambang", "Subbab 4.3"],
            ["artifacts/high_risk_customers.csv", "Skor probabilitas data uji", "Dashboard"],
            ["artifacts/dashboard.html", "Dashboard interaktif statis", "Presentasi"],
            ["artifacts/figures/*.png", "11 gambar untuk laporan dan slide", "Bab III-IV"],
        ],
        widths=[5.6 * cm, 6.4 * cm, 3.6 * cm],
        caption="Artefak keluaran sistem",
        label="Tabel A.1",
        font_size=8.2,
    ))

    # --------------------------------------------------------------- B ----
    ctx.add(PageBreak())
    ctx.add(Paragraph("LAMPIRAN B", ParagraphStyle(
        "BackTitle3", parent=engine.STYLES["H1"], spaceAfter=4)))
    ctx.add(Paragraph("Kode Inti Sistem", engine.STYLES["ChapterNum"]))
    ctx.add(Spacer(1, 0.4 * cm))

    ctx.add(h2("B.1  Pipeline Praproses dan Model"))
    ctx.add(*code_block(
        '''def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(drop="first",
                                  handle_unknown="ignore",
                                  sparse_output=False), CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

def build_pipeline(C=1.0, max_iter=1000) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(C=C, max_iter=max_iter,
                                          solver="lbfgs",
                                          random_state=SEED)),
    ])''',
        caption="Kode B.1  src/features.py",
    ))

    ctx.add(h2("B.2  Pelatihan dan Evaluasi"))
    ctx.add(*code_block(
        '''X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = build_pipeline()
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

metrics = compute_metrics(y_test, proba, threshold=0.5)
table   = threshold_table(y_test, proba)
cv      = cross_validate(model, X, y, folds=5)''',
        caption="Kode B.2  Alur pelatihan dan evaluasi",
    ))

    ctx.add(h2("B.3  Pemberian Skor Pelanggan Baru"))
    ctx.add(*code_block(
        '''model = joblib.load("artifacts/model.joblib")
proba = model.predict_proba(frame[FEATURES])[:, 1]

hasil = frame.copy()
hasil["churn_probability"] = proba.round(4)
hasil["risk_band"] = ["Rendah" if p < 0.30 else
                      "Sedang" if p <= 0.60 else "Tinggi"
                      for p in proba]''',
        caption="Kode B.3  predict.py (ringkas)",
    ))

    # --------------------------------------------------------------- C ----
    ctx.add(PageBreak())
    ctx.add(Paragraph("LAMPIRAN C", ParagraphStyle(
        "BackTitle4", parent=engine.STYLES["H1"], spaceAfter=4)))
    ctx.add(Paragraph("Contoh Keluaran Sistem", engine.STYLES["ChapterNum"]))
    ctx.add(Spacer(1, 0.4 * cm))

    ctx.add(p(
        "Perintah <font name='Courier'>python predict.py --demo</font> memberi skor pada tiga "
        "profil pelanggan sintetis. Hasilnya memperlihatkan bahwa model tidak sekadar menghafal "
        "pola data, tetapi merespons kombinasi kontrak, masa berlangganan, dan jenis layanan "
        "secara masuk akal."
    ))
    ctx.add(*table_block(
        ["Profil pelanggan", "Kontrak", "Tenure", "Risiko"],
        [
            ["Pelanggan baru, fiber optic, kontrak bulanan", "Month-to-month", "3 bulan",
             "82,04% (tinggi)"],
            ["Pelanggan setia, DSL, kontrak dua tahun", "Two year", "60 bulan",
             "0,47% (rendah)"],
            ["Pelanggan menengah, fiber optic, kontrak satu tahun", "One year", "18 bulan",
             "28,43% (rendah)"],
        ],
        widths=[7.2 * cm, 3.6 * cm, 2.2 * cm, 3.0 * cm],
        caption="Hasil scoring tiga profil pelanggan contoh",
        label="Tabel C.1",
        align=["LEFT", "LEFT", "CENTER", "RIGHT"],
        font_size=8.3,
    ))
    ctx.add(p(
        "Keluaran tersebut juga menunjukkan ambang risiko yang dipakai sistem: di bawah 30% "
        "tergolong rendah, 30-60% sedang, dan di atas 60% tinggi. Ambang ini terpisah dari ambang "
        "keputusan klasifikasi 0,30 dan berfungsi sebagai alat komunikasi bagi tim operasional."
    ))

    # --------------------------------------------------------------- D ----
    ctx.add(PageBreak())
    ctx.add(Paragraph("LAMPIRAN D", ParagraphStyle(
        "BackTitle5", parent=engine.STYLES["H1"], spaceAfter=4)))
    ctx.add(Paragraph("Tabel Lengkap Koefisien Model", engine.STYLES["ChapterNum"]))
    ctx.add(Spacer(1, 0.4 * cm))

    ctx.add(p(
        "Seluruh 30 fitur hasil encoding disajikan berikut, diurutkan berdasarkan nilai koefisien. "
        "Fitur numerik bertanda keterangan \u201cper 1 SD\u201d karena telah distandardisasi, "
        "sedangkan fitur kategorikal dibandingkan terhadap kategori acuannya."
    ))
    ctx.add(p(
        "<i>Catatan:</i> nilai koefisien dipengaruhi korelasi antar fitur (lihat Subbab 4.6.2), "
        "sehingga tanda pada fitur tagihan tidak dapat ditafsirkan secara terpisah tanpa koreksi "
        "multikolinearitas."
    ))

    coef = pd.read_csv(engine.ART_DIR / "coefficients.csv")
    numeric = {"tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"}
    rows = []
    for i, row in coef.iterrows():
        nama = row["label"]
        if row["feature"] in numeric:
            nama += " (per 1 SD)"
        rows.append([
            str(i + 1), nama,
            engine.num(row["coefficient"]),
            engine.num(row["odds_ratio"], 3),
            "Naik" if row["coefficient"] > 0 else "Turun",
        ])
    ctx.add(*table_block(
        ["No", "Fitur", "Koefisien", "Odds ratio", "Arah"],
        rows,
        widths=[1.2 * cm, 7.6 * cm, 2.6 * cm, 2.8 * cm, 1.8 * cm],
        caption="Koefisien dan odds ratio seluruh fitur",
        label="Tabel D.1",
        align=["CENTER", "LEFT", "RIGHT", "RIGHT", "CENTER"],
        font_size=7.6,
        keep=False,
        padding=2.6,
    ))


def build(ctx) -> None:
    bibliography(ctx)
    appendix(ctx)