"""Bab IV Hasil dan Pembahasan."""
from __future__ import annotations

import pandas as pd
from reportlab.lib.units import cm

from src import data

from . import engine
from .engine import (
    CTX, bullet, chapter, figure, h2, h3, numbered, p, table_block,
)

M = CTX.metrics
D = M["dataset"]
MD = M["metrics_default"]
MR = M["metrics_recommended"]
CV = M["cv"]

_DF: pd.DataFrame | None = None


def df() -> pd.DataFrame:
    global _DF
    if _DF is None:
        _DF = data.clean(data.load_raw())
    return _DF


def _threshold(level: float) -> dict:
    for row in M["threshold_table"]:
        if abs(row["threshold"] - level) < 1e-9:
            return row
    raise KeyError(level)


def build(ctx) -> None:
    d = df()
    total = len(d)
    churn_rate = D["churn_rate"]

    ctx.add(*chapter("BAB IV", "Hasil dan Pembahasan"))

    # ------------------------------------------------------------- 4.1 ----
    ctx.add(h2("4.1  Analisis Data Eksploratif"))
    ctx.add(p(
        "Analisis eksploratif dilakukan sebelum pemodelan untuk memastikan kualitas data dan "
        "membentuk hipotesis awal mengenai faktor pemicu churn. Bagian ini menyajikan empat "
        "aspek yang paling informatif: distribusi kelas target, kaitan churn dengan tipe kontrak, "
        "kaitan churn dengan masa berlangganan, serta struktur korelasi antar fitur numerik."
    ))

    ctx.add(h3("4.1.1  Distribusi Kelas Target"))
    counts = d["Churn"].value_counts().sort_index()
    ctx.add(p(
        f"Dari {engine.num(total, 0)} pelanggan, sebanyak {engine.num(int(counts[1]), 0)} "
        f"({engine.num(churn_rate)}%) berhenti berlangganan dan {engine.num(int(counts[0]), 0)} "
        f"({engine.num(100 - churn_rate)}%) bertahan. Rasio sekitar 1:2,8 ini termasuk kategori "
        "tidak seimbang moderat. Dampaknya bukan pada kemampuan model belajar, melainkan pada "
        "cara kita menilai keberhasilan: sebuah model yang selalu memprediksi \u201ctidak churn\u201d "
        f"sudah memperoleh <i>accuracy</i> {engine.pct(M['baseline']['accuracy'])}, sehingga angka "
        "<i>accuracy</i> saja tidak cukup untuk menyimpulkan bahwa model bermanfaat."
    ))
    ctx.add(*table_block(
        ["Kelas", "Jumlah pelanggan", "Persentase"],
        [
            ["Tidak churn (0)", engine.num(int(counts[0]), 0), engine.num(100 - churn_rate) + "%"],
            ["Churn (1)", engine.num(int(counts[1]), 0), engine.num(churn_rate) + "%"],
            ["Total", engine.num(total, 0), "100,00%"],
        ],
        widths=[5.6 * cm, 5.0 * cm, 4.2 * cm],
        caption="Distribusi kelas target pada dataset",
        label="Tabel 4.1",
        align=["LEFT", "RIGHT", "RIGHT"],
    ))
    ctx.add(figure("01_distribusi_kelas.png", "Distribusi kelas target (Churn)",
                   "Gambar 4.1", width=11.0 * cm))

    ctx.add(h3("4.1.2  Churn Berdasarkan Tipe Kontrak"))
    contract = (
        d.groupby("Contract", observed=True)["Churn"]
        .agg(jumlah="size", churn="sum")
        .reindex(["Month-to-month", "One year", "Two year"])
    )
    contract["rate"] = contract["churn"] / contract["jumlah"] * 100
    mtm = contract.loc["Month-to-month"]
    two = contract.loc["Two year"]
    ratio = mtm["rate"] / two["rate"]

    ctx.add(p(
        "Tipe kontrak memperlihatkan pola yang sangat tajam. Pelanggan dengan kontrak bulanan "
        f"memiliki churn rate {engine.num(mtm['rate'])}%, sedangkan pelanggan kontrak dua tahun "
        f"hanya {engine.num(two['rate'])}%. Selisihnya mencapai sekitar {engine.num(ratio, 1)} kali. "
        "Pola ini mudah dijelaskan secara bisnis: kontrak jangka panjang mengikat pelanggan, "
        "menyertakan konsekuensi biaya bila berhenti lebih awal, dan umumnya disertai harga "
        "per bulan yang lebih rendah. Sebaliknya, pelanggan prabayar bulanan dapat berpindah "
        "penyedia tanpa hambatan apa pun."
    ))
    ctx.add(*table_block(
        ["Tipe kontrak", "Jumlah", "Churn", "Churn rate"],
        [
            [name, engine.num(int(row["jumlah"]), 0), engine.num(int(row["churn"]), 0),
             engine.num(row["rate"]) + "%"]
            for name, row in contract.iterrows()
        ],
        widths=[5.6 * cm, 4.0 * cm, 3.0 * cm, 3.0 * cm],
        caption="Jumlah pelanggan dan churn rate per tipe kontrak",
        label="Tabel 4.2",
        align=["LEFT", "RIGHT", "RIGHT", "RIGHT"],
    ))
    ctx.add(figure("02_churn_per_kontrak.png", "Churn rate berdasarkan tipe kontrak",
                   "Gambar 4.2", width=11.6 * cm))

    ctx.add(h3("4.1.3  Churn Berdasarkan Masa Berlangganan"))
    bins = [0, 12, 24, 48, 72]
    labels = ["0-12 bulan", "13-24 bulan", "25-48 bulan", "49-72 bulan"]
    groups = pd.cut(d["tenure"], bins=bins, labels=labels, include_lowest=True)
    tenure = d.groupby(groups, observed=True)["Churn"].agg(jumlah="size", churn="sum")
    tenure["rate"] = tenure["churn"] / tenure["jumlah"] * 100

    ctx.add(p(
        "Masa berlangganan memperlihatkan pola yang konsisten menurun. Hampir separuh pelanggan "
        f"pada tahun pertama berhenti berlangganan ({engine.num(tenure.loc[labels[0], 'rate'])}%), "
        f"sedangkan pada kelompok di atas empat tahun angkanya turun menjadi "
        f"{engine.num(tenure.loc[labels[3], 'rate'])}%. Temuan ini sejalan dengan gagasan "
        "<i>early tenure risk</i>: keputusan pelanggan untuk bertahan atau pergi paling banyak "
        "ditentukan pada masa awal, ketika pengalaman pertama terhadap kualitas jaringan dan "
        "layanan masih membentuk penilaian. Secara statistik, pola menurun ini menjelaskan mengapa "
        "variabel tenure menjadi salah satu prediktor terkuat pada model."
    ))
    ctx.add(*table_block(
        ["Kelompok tenure", "Jumlah", "Churn", "Churn rate"],
        [
            [name, engine.num(int(row["jumlah"]), 0), engine.num(int(row["churn"]), 0),
             engine.num(row["rate"]) + "%"]
            for name, row in tenure.iterrows()
        ],
        widths=[5.6 * cm, 4.0 * cm, 3.0 * cm, 3.0 * cm],
        caption="Churn rate per kelompok masa berlangganan",
        label="Tabel 4.3",
        align=["LEFT", "RIGHT", "RIGHT", "RIGHT"],
    ))
    ctx.add(figure("03_churn_per_tenure.png", "Churn rate menurun seiring masa berlangganan",
                   "Gambar 4.3", width=11.6 * cm))

    ctx.add(h3("4.1.4  Sebaran Tagihan Bulanan"))
    ctx.add(p(
        f"Tagihan bulanan memiliki rata-rata {engine.num(D['monthly_mean'])} dan median "
        f"{engine.num(D['monthly_median'])} dengan rentang {engine.num(D['monthly_min'])} sampai "
        f"{engine.num(D['monthly_max'])}. Distribusinya cenderung menumpuk pada nilai tinggi, "
        "berbeda dengan kelompok pelanggan yang bertahan yang lebih menyebar pada nilai menengah "
        "ke bawah. Perbedaan bentuk sebaran ini menandakan bahwa besaran tagihan membawa "
        "informasi, meskipun arah pengaruhnya tidak sesederhana \u201csemakin mahal semakin "
        "berisiko\u201d, sebagaimana dibahas pada Subbab 4.6.2."
    ))
    ctx.add(figure("04_distribusi_tagihan.png",
                   "Sebaran tagihan bulanan pada dua kelas",
                   "Gambar 4.4", width=11.6 * cm))

    ctx.add(h3("4.1.5  Korelasi Antar Fitur Numerik"))
    corr = d[["tenure", "MonthlyCharges", "TotalCharges"]].corr()
    r_ten_total = corr.loc["tenure", "TotalCharges"]
    r_month_total = corr.loc["MonthlyCharges", "TotalCharges"]

    ctx.add(p(
        f"Terdapat korelasi kuat antara <i>tenure</i> dan <i>TotalCharges</i> sebesar "
        f"{engine.num(r_ten_total, 3)}, serta korelasi moderat antara <i>MonthlyCharges</i> dan "
        f"<i>TotalCharges</i> sebesar {engine.num(r_month_total, 3)}. Hubungan ini wajar secara "
        "aritmetika: total tagihan pada dasarnya adalah akumulasi tagihan bulanan sepanjang masa "
        "berlangganan. Namun dari sudut pemodelan, korelasi sebesar itu berpotensi menimbulkan "
        "multikolinearitas, yaitu kondisi ketika dua fitur membawa informasi yang hampir sama "
        "sehingga estimasi koefisien menjadi tidak stabil dan tandanya dapat berbalik dari "
        "dugaan intuitif."
    ))
    ctx.add(figure("05_korelasi.png", "Matriks korelasi fitur numerik",
                   "Gambar 4.5", width=9.4 * cm))
    ctx.add(p(
        "Peringatan ini bukan sekadar teoretis; dampaknya terlihat langsung pada Subbab 4.5, di "
        "mana koefisien <i>MonthlyCharges</i> bertanda negatif meskipun secara bisnis tagihan "
        "tinggi biasanya dikaitkan dengan risiko churn yang lebih besar. Pembahasan mendalam "
        "mengenai hal tersebut disajikan pada Subbab 4.6.2."
    ))

    # ------------------------------------------------------------- 4.2 ----
    ctx.add(h2("4.2  Kinerja Model pada Ambang 0,50"))
    ctx.add(p(
        f"Model dilatih pada {engine.num(M['split']['train_size'], 0)} baris data latih dan diuji "
        f"pada {engine.num(M['split']['test_size'], 0)} baris data uji yang belum pernah dilihat "
        "sebelumnya. Seluruh angka pada subbab ini dihitung pada data uji tersebut, sehingga "
        "mencerminkan performa model terhadap data baru dan bukan hafalan terhadap data latih."
    ))

    ctx.add(h3("4.2.1  Confusion Matrix"))
    tn, fp = MD["tn"], MD["fp"]
    fn, tp = MD["fn"], MD["tp"]
    ctx.add(p(
        f"Dengan ambang 0,50, model menghasilkan {engine.num(tp, 0)} <i>true positive</i>, "
        f"{engine.num(tn, 0)} <i>true negative</i>, {engine.num(fp, 0)} <i>false positive</i>, dan "
        f"{engine.num(fn, 0)} <i>false negative</i>. Artinya, dari {engine.num(tp + fn, 0)} "
        f"pelanggan yang benar-benar berhenti berlangganan, sebanyak {engine.num(fn, 0)} orang "
        "tidak terdeteksi. Kelompok inilah yang paling merugikan, karena mereka tidak menerima "
        "intervensi apa pun dan langsung hilang."
    ))
    ctx.add(*table_block(
        ["", "Prediksi: tidak churn", "Prediksi: churn", "Total aktual"],
        [
            ["Aktual: tidak churn", f"TN = {engine.num(tn, 0)}", f"FP = {engine.num(fp, 0)}",
             engine.num(tn + fp, 0)],
            ["Aktual: churn", f"FN = {engine.num(fn, 0)}", f"TP = {engine.num(tp, 0)}",
             engine.num(tp + fn, 0)],
        ],
        widths=[4.4 * cm, 4.0 * cm, 3.6 * cm, 3.2 * cm],
        caption="Confusion matrix model pada data uji (ambang 0,50)",
        label="Tabel 4.4",
        align=["LEFT", "CENTER", "CENTER", "CENTER"],
    ))
    ctx.add(figure("06_confusion_matrix.png", "Visualisasi confusion matrix pada data uji",
                   "Gambar 4.6", width=9.8 * cm))

    ctx.add(h3("4.2.2  Ringkasan Metrik"))
    ctx.add(p(
        f"Secara ringkas, model memperoleh <i>accuracy</i> {engine.pct(MD['accuracy'])}, "
        f"<i>precision</i> {engine.pct(MD['precision'])}, <i>recall</i> {engine.pct(MD['recall'])}, "
        f"<i>F1-score</i> {engine.pct(MD['f1'])}, dan <i>ROC-AUC</i> "
        f"{engine.num(MD['roc_auc'], 4)}. Dibandingkan baseline naif yang selalu menjawab "
        f"\u201ctidak churn\u201d dengan <i>accuracy</i> {engine.pct(M['baseline']['accuracy'])}, "
        f"model unggul sekitar {engine.num((MD['accuracy'] - M['baseline']['accuracy']) * 100)} "
        "poin persentase. Kenaikan ini tampak sederhana, tetapi perlu diingat bahwa baseline "
        "tersebut sama sekali tidak mampu mengidentifikasi pelanggan berisiko."
    ))
    ctx.add(*table_block(
        ["Metrik", "Nilai", "Makna praktis"],
        [
            ["Accuracy", engine.pct(MD["accuracy"]),
             "Proporsi keseluruhan prediksi yang benar"],
            ["Precision", engine.pct(MD["precision"]),
             "Dari pelanggan yang ditandai berisiko, sekitar dua pertiganya benar-benar churn"],
            ["Recall", engine.pct(MD["recall"]),
             "Hanya sekitar separuh pelanggan yang benar-benar churn berhasil ditangkap"],
            ["F1-score", engine.pct(MD["f1"]),
             "Keseimbangan precision dan recall"],
            ["ROC-AUC", engine.num(MD["roc_auc"], 4),
             "Kemampuan diskriminasi tergolong baik (0,8-0,9)"],
            ["Baseline accuracy", engine.pct(M["baseline"]["accuracy"]),
             "Acuan minimal yang wajib dilampaui"],
        ],
        widths=[3.2 * cm, 2.6 * cm, 9.8 * cm],
        caption="Ringkasan metrik evaluasi pada data uji",
        label="Tabel 4.5",
        align=["LEFT", "RIGHT", "LEFT"],
    ))
    ctx.add(p(
        f"Pola yang menonjol adalah ketidakseimbangan antara <i>precision</i> "
        f"({engine.pct(MD['precision'])}) dan <i>recall</i> ({engine.pct(MD['recall'])}). Model "
        "cenderung konservatif: ketika menyatakan seorang pelanggan berisiko, pernyataan itu "
        "relatif dapat dipercaya, tetapi banyak pelanggan berisiko justru tidak ditandai. Pada "
        "konteks retensi, sikap konservatif ini merugikan karena kesalahan yang mahal adalah "
        "kegagalan menangkap pelanggan yang akan pergi, bukan kesalahan memberi promo kepada "
        "pelanggan yang sebenarnya loyal."
    ))

    # ------------------------------------------------------------- 4.3 ----
    ctx.add(h2("4.3  Analisis Ambang Keputusan"))
    r30 = _threshold(0.30)
    r20 = _threshold(0.20)
    r45 = _threshold(0.45)

    ctx.add(p(
        "Karena keluaran model berupa probabilitas, ambang keputusan dapat digeser tanpa melatih "
        "ulang model. Tabel 4.6 memperlihatkan konsekuensi pergeseran tersebut. Ketika ambang "
        f"diturunkan menjadi 0,30, jumlah pelanggan churn yang tertangkap naik dari "
        f"{engine.num(tp, 0)} menjadi {engine.num(r30['churn_tertangkap'], 0)}, sehingga yang "
        f"terlewat turun dari {engine.num(fn, 0)} menjadi {engine.num(r30['churn_terlewat'], 0)}. "
        f"Sebagai konsekuensinya, <i>precision</i> turun menjadi {engine.num(r30['precision_pct'])}% "
        f"dan <i>accuracy</i> menjadi {engine.num(r30['accuracy_pct'])}%."
    ))
    ctx.add(p(
        "Penurunan <i>accuracy</i> sering disalahartikan sebagai kemunduran model. Padahal, "
        "<i>accuracy</i> yang lebih rendah di sini berasal dari bertambahnya <i>false positive</i>, "
        "yaitu pelanggan loyal yang ikut ditawari program retensi. Biaya tambahan tersebut "
        "bersifat terbatas dan terlihat, sedangkan kerugian akibat pelanggan churn yang lolos "
        "bersifat berulang sepanjang masa berlangganan yang seharusnya masih bisa diselamatkan. "
        "Karena itu, ambang 0,30 lebih masuk akal secara ekonomi daripada 0,50, sepanjang "
        "kapasitas tim retensi mencukupi."
    ))

    rows = []
    for level in (0.20, 0.30, 0.40, 0.45, 0.50, 0.60):
        r = _threshold(level)
        rows.append([
            engine.num(level), engine.num(r["accuracy_pct"]) + "%",
            engine.num(r["precision_pct"]) + "%", engine.num(r["recall_pct"]) + "%",
            engine.num(r["f1_pct"]) + "%", engine.num(r["churn_tertangkap"], 0),
            engine.num(r["churn_terlewat"], 0),
        ])
    ctx.add(*table_block(
        ["Ambang", "Accuracy", "Precision", "Recall", "F1", "Tertangkap", "Terlewat"],
        rows,
        widths=[2.0 * cm, 2.4 * cm, 2.5 * cm, 2.2 * cm, 2.0 * cm, 2.4 * cm, 2.2 * cm],
        caption="Pergeseran metrik pada berbagai ambang keputusan",
        label="Tabel 4.6",
        align=["CENTER", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"],
        font_size=8.3,
    ))
    ctx.add(figure("08_threshold_tuning.png",
                   "Trade-off precision, recall, dan F1-score terhadap ambang keputusan",
                   "Gambar 4.7", width=13.8 * cm))
    ctx.add(p(
        f"Apabila kapasitas tim retensi sangat terbatas sehingga hanya mampu menghubungi sebagian "
        f"kecil pelanggan, ambang {engine.num(0.45)} dapat menjadi kompromi: <i>recall</i> "
        f"{engine.num(r45['recall_pct'])}% dengan <i>precision</i> "
        f"{engine.num(r45['precision_pct'])}%. Sebaliknya, bila tujuan utamanya adalah meminimalkan "
        f"pelanggan yang lolos, ambang {engine.num(0.20)} menaikkan <i>recall</i> hingga "
        f"{engine.num(r20['recall_pct'])}% dengan konsekuensi <i>precision</i> turun ke "
        f"{engine.num(r20['precision_pct'])}%. Tidak ada pilihan yang benar secara universal; "
        "yang ada adalah pilihan yang paling sesuai dengan anggaran dan tujuan program retensi."
    ))

    # ------------------------------------------------------------- 4.4 ----
    ctx.add(h2("4.4  Kurva ROC dan Validasi Silang"))
    ctx.add(p(
        f"Nilai ROC-AUC sebesar {engine.num(MD['roc_auc'], 4)} menunjukkan bahwa jika diambil satu "
        "pelanggan yang benar-benar churn dan satu pelanggan yang bertahan secara acak, model "
        "memberi skor risiko lebih tinggi kepada pelanggan churn dengan peluang sekitar "
        f"{engine.num(MD['roc_auc'] * 100)}%. Angka ini termasuk kategori diskriminasi baik "
        "(0,8-0,9). Kurva ROC pada Gambar 4.8 memperlihatkan bahwa model konsisten berada jauh di "
        "atas garis diagonal, terutama pada daerah <i>false positive rate</i> rendah, yaitu daerah "
        "yang paling relevan ketika tim retensi hanya sanggup menghubungi sebagian kecil pelanggan."
    ))
    ctx.add(figure("07_roc_curve.png", "Kurva ROC model pada data uji",
                   "Gambar 4.8", width=10.4 * cm))

    ctx.add(p(
        "Untuk memastikan hasil bukan kebetulan satu kali pembagian data, dilakukan validasi silang "
        f"lima lipatan pada seluruh dataset. Nilai AUC tiap lipatan berkisar antara "
        f"{engine.num(min(CV['scores']), 4)} sampai {engine.num(max(CV['scores']), 4)} dengan "
        f"rata-rata {engine.num(CV['mean'], 4)} dan simpangan baku {engine.num(CV['std'], 4)}. "
        "Simpangan baku yang kecil menunjukkan performa stabil, sementara kesesuaian antara rata-rata "
        "validasi silang dan hasil pada data uji mengindikasikan tidak adanya <i>overfitting</i> "
        "yang berarti."
    ))
    ctx.add(*table_block(
        ["Lipatan", "ROC-AUC"],
        [[f"Fold {i + 1}", engine.num(s, 4)] for i, s in enumerate(CV["scores"])]
        + [["Rata-rata", engine.num(CV["mean"], 4)], ["Simpangan baku", engine.num(CV["std"], 4)]],
        widths=[5.2 * cm, 4.4 * cm],
        caption="Hasil validasi silang lima lipatan",
        label="Tabel 4.7",
        align=["LEFT", "RIGHT"],
        keep=True,
    ))
    ctx.add(figure("10_cross_validation.png", "ROC-AUC pada lima lipatan validasi silang",
                   "Gambar 4.9", width=11.0 * cm))

    # ------------------------------------------------------------- 4.5 ----
    ctx.add(h2("4.5  Interpretasi Koefisien dan Odds Ratio"))
    ctx.add(p(
        "Karena fitur numerik telah distandardisasi, koefisien dapat dibandingkan secara langsung "
        "antar fitur. Tabel 4.8 dan Tabel 4.9 menyajikan enam fitur dengan pengaruh terbesar pada "
        "masing-masing arah. Nilai <i>odds ratio</i> lebih mudah dibaca daripada koefisien mentah: "
        "OR 0,265 berarti odds churn turun sekitar 73,5%, sedangkan OR 3,262 berarti odds churn "
        "naik lebih dari tiga kali."
    ))

    neg = M["top_negative"][:6]
    pos = M["top_positive"][:6]
    ctx.add(*table_block(
        ["Fitur", "Koefisien", "Odds ratio", "Tafsir"],
        [[row["label"], engine.num(row["coefficient"], 3), engine.num(row["odds_ratio"], 3),
          f"Odds churn {engine.num((1 - row['odds_ratio']) * 100, 1)}% lebih rendah"]
         for row in neg],
        widths=[5.0 * cm, 2.6 * cm, 2.6 * cm, 5.4 * cm],
        caption="Enam fitur yang paling menurunkan risiko churn",
        label="Tabel 4.8",
        align=["LEFT", "RIGHT", "RIGHT", "LEFT"],
        font_size=8.4,
    ))
    ctx.add(*table_block(
        ["Fitur", "Koefisien", "Odds ratio", "Tafsir"],
        [[row["label"], engine.num(row["coefficient"], 3), engine.num(row["odds_ratio"], 3),
          f"Odds churn {engine.num((row['odds_ratio'] - 1) * 100, 1)}% lebih tinggi"]
         for row in pos],
        widths=[5.0 * cm, 2.6 * cm, 2.6 * cm, 5.4 * cm],
        caption="Enam fitur yang paling menaikkan risiko churn",
        label="Tabel 4.9",
        align=["LEFT", "RIGHT", "RIGHT", "LEFT"],
        font_size=8.4,
    ))
    ctx.add(figure("09_koefisien.png", "Dua belas fitur dengan koefisien terbesar",
                   "Gambar 4.10", width=13.2 * cm))
    ctx.add(p(
        "Tiga temuan utama dapat ditarik dari kedua tabel tersebut. Pertama, <b>tipe kontrak</b> "
        "adalah faktor protektif terkuat: pelanggan kontrak dua tahun memiliki odds churn hanya "
        "26,5% dibandingkan pelanggan kontrak bulanan. Kedua, <b>masa berlangganan</b> memiliki "
        "pengaruh hampir sebesar itu; setiap kenaikan satu simpangan baku (sekitar 24,6 bulan) "
        "menurunkan odds churn menjadi 28,6%. Ketiga, <b>jenis layanan internet</b> merupakan "
        "faktor risiko utama: pengguna fiber optic memiliki odds churn 3,26 kali lebih tinggi "
        "daripada pengguna DSL, bahkan setelah faktor lain diperhitungkan."
    ))
    ctx.add(p(
        "Faktor lain yang perlu dicatat adalah metode pembayaran <i>electronic check</i> "
        "(OR 1,468), <i>paperless billing</i> (OR 1,450), serta langganan layanan hiburan "
        "<i>streaming</i> (OR sekitar 1,46). Ketiganya berkaitan dengan profil pelanggan yang "
        "cenderung aktif secara digital dan lebih mudah membandingkan penawaran pesaing. Di sisi "
        "protektif, kepemilikan layanan tambahan seperti <i>online security</i> (OR 0,705) "
        "menurunkan risiko, sejalan dengan gagasan bahwa semakin banyak layanan yang terikat pada "
        "satu akun, semakin tinggi biaya perpindahan (<i>switching cost</i>)."
    ))

    # ------------------------------------------------------------- 4.6 ----
    ctx.add(h2("4.6  Pembahasan"))
    ctx.add(h3("4.6.1  Mengapa Kontrak dan Masa Berlangganan Dominan"))
    ctx.add(p(
        "Dominasi tipe kontrak dan masa berlangganan masuk akal secara teoretis maupun praktis. "
        "Keduanya merupakan proksi dari <i>switching cost</i> dan <i>relationship strength</i>. "
        "Pelanggan yang telah bertahan lama memiliki pengalaman positif kumulatif, hubungan "
        "fungsional dengan layanan, dan biasanya telah melewati masa kritis awal. Sebaliknya, "
        "pelanggan baru pada kontrak bulanan belum memiliki ikatan apa pun. Kombinasi keduanya "
        "menghasilkan segmen paling rentan: pelanggan bulanan dengan masa berlangganan di bawah "
        "satu tahun, yang pada data ini memiliki churn rate hingga hampir separuh."
    ))
    ctx.add(h3("4.6.2  Anomali Tanda Koefisien Tagihan Bulanan"))
    ctx.add(p(
        "Salah satu hasil yang menuntut kehati-hatian adalah koefisien <i>MonthlyCharges</i> yang "
        "bertanda negatif (OR 0,625). Secara intuitif, tagihan bulanan yang lebih tinggi "
        "seharusnya berkaitan dengan risiko churn yang lebih besar, karena pelanggan membayar lebih "
        "mahal. Namun setelah <i>TotalCharges</i> dan <i>tenure</i> ikut diperhitungkan, koefisien "
        "tersebut berubah makna menjadi pengaruh <i>parsial</i>: pada pelanggan dengan total "
        "tagihan dan masa berlangganan yang sama, tagihan bulanan yang lebih tinggi justru berarti "
        "pelanggan tersebut memperoleh layanan lebih lengkap."
    ))
    ctx.add(p(
        "Gejala ini adalah tanda klasik multikolinearitas, yang telah diperkirakan dari korelasi "
        f"{engine.num(r_ten_total, 3)} antara <i>tenure</i> dan <i>TotalCharges</i>. Ketika dua "
        "fitur hampir saling menggantikan, model membagi pengaruhnya secara arbitrer dan tanda "
        "koefisien dapat terbalik. Konsekuensi praktisnya penting: koefisien <i>MonthlyCharges</i> "
        "tidak boleh dipakai untuk menyimpulkan bahwa menaikkan harga akan menurunkan churn. "
        "Langkah perbaikan yang disarankan adalah menghitung <i>variance inflation factor</i>, "
        "memilih salah satu dari kedua fitur, atau membentuk fitur baru seperti rasio "
        "<i>MonthlyCharges</i> terhadap rata-rata tagihan."
    ))
    ctx.add(h3("4.6.3  Konsekuensi Asimetri Kesalahan"))
    ctx.add(p(
        "Perbandingan confusion matrix pada ambang 0,50 memperlihatkan bahwa kesalahan model tidak "
        f"simetris: {engine.num(fn, 0)} <i>false negative</i> berbanding {engine.num(fp, 0)} "
        "<i>false positive</i>. Dalam konteks retensi, kedua jenis kesalahan memiliki konsekuensi "
        "yang sangat berbeda. <i>False positive</i> berarti perusahaan memberikan insentif kepada "
        "pelanggan yang sebenarnya tidak akan pergi; kerugiannya terbatas pada biaya kampanye dan "
        "potensi penurunan margin. <i>False negative</i> berarti perusahaan kehilangan pelanggan "
        "beserta seluruh pendapatan masa depannya. Karena itu, menurunkan ambang keputusan sampai "
        "0,30 adalah keputusan yang dapat dipertanggungjawabkan, dengan catatan tim retensi mampu "
        "menangani tambahan kontak yang muncul."
    ))
    ctx.add(h3("4.6.4  Perbandingan dengan Baseline"))
    ctx.add(p(
        "Model mengungguli baseline naif sebesar "
        f"{engine.num((MD['accuracy'] - M['baseline']['accuracy']) * 100)} poin persentase "
        "<i>accuracy</i>, tetapi keunggulan sebenarnya jauh lebih besar daripada angka itu. "
        "Baseline tidak memiliki kemampuan diskriminasi sama sekali (AUC setara 0,5) dan tidak "
        "dapat menghasilkan peringkat risiko, sehingga tidak dapat dipakai untuk menyusun prioritas "
        "kontak. Model ini, meskipun sederhana, mampu mengurutkan pelanggan berdasarkan probabilitas "
        f"churn dengan AUC {engine.num(MD['roc_auc'], 4)} dan menyediakan penjelasan arah pengaruh "
        "setiap fitur. Nilai tambahnya terletak pada kemampuan pengambilan keputusan, bukan semata "
        "pada akurasi."
    ))

    # ------------------------------------------------------------- 4.7 ----
    ctx.add(h2("4.7  Rekomendasi Strategi Retensi"))
    ctx.add(*numbered([
        "<b>Fokus pada segmen kontrak bulanan dengan masa berlangganan di bawah 12 bulan.</b> "
        "Segmen ini memiliki churn rate tertinggi dan paling responsif terhadap intervensi awal. "
        "Program <i>onboarding</i> pada 90 hari pertama, misalnya verifikasi kualitas jaringan dan "
        "edukasi penggunaan layanan, layak diprioritaskan.",
        "<b>Dorong migrasi ke kontrak satu atau dua tahun.</b> Karena kontrak dua tahun menurunkan "
        "odds churn hingga 73,5%, insentif berupa potongan harga atau tambahan kuota dapat "
        "diarahkan khusus kepada pelanggan bulanan yang telah bertahan lebih dari enam bulan.",
        "<b>Audit layanan fiber optic.</b> Risiko churn pengguna fiber optic 3,26 kali lebih tinggi "
        "menjadi sinyal bahwa ada persoalan pada kualitas layanan, ekspektasi harga, atau "
        "komunikasi manfaat. Perlu dipisahkan apakah penyebabnya gangguan teknis atau persepsi "
        "nilai.",
        "<b>Migrasikan pelanggan <i>electronic check</i> ke pembayaran otomatis.</b> Kelompok ini "
        "berisiko 1,47 kali lebih tinggi, dan migrasi ke autopay sekaligus mengurangi friksi "
        "administratif bulanan yang sering memicu pemutusan.",
        "<b>Bundling layanan protektif.</b> <i>Online security</i> dan <i>tech support</i> "
        "berkaitan dengan penurunan risiko, sehingga penawarannya sebagai paket dapat memperkuat "
        "ikatan pelanggan sekaligus menaikkan pendapatan per pengguna.",
        "<b>Gunakan skor risiko sebagai daftar kerja bulanan.</b> Ambang 0,30 direkomendasikan "
        "untuk operasi normal, dengan peninjauan ulang bila kapasitas tim berubah.",
    ]))

    # ------------------------------------------------------------- 4.8 ----
    ctx.add(h2("4.8  Keterbatasan Model"))
    ctx.add(*[
        bullet("<b>Multikolinearitas.</b> Korelasi tinggi antara tenure, TotalCharges, dan "
               "MonthlyCharges membuat koefisien sebagian fitur tidak stabil dan sulit "
               "diinterpretasi secara terpisah."),
        bullet("<b>Penanganan kelas tidak seimbang masih terbatas.</b> Model utama belum "
               "menggunakan <i>class weight</i> maupun SMOTE; penyesuaian hanya dilakukan pada "
               "ambang keputusan."),
        bullet("<b>Belum ada pencarian hiperparameter.</b> Nilai C = 1,0 dan solver lbfgs dipakai "
               "apa adanya, sehingga belum dapat diklaim optimal."),
        bullet("<b>Tidak ada pembanding model lain.</b> Tanpa <i>random forest</i> atau "
               "<i>gradient boosting</i>, klaim mengenai kecukupan performa bersifat terbatas."),
        bullet("<b>Data cross-sectional.</b> Model tidak menangkap perubahan perilaku dari waktu "
               "ke waktu, sehingga validitasnya menurun bila diterapkan jauh di masa depan tanpa "
               "pelatihan ulang."),
        bullet("<b>Evaluasi baru pada satu pembagian data.</b> Validasi silang dilakukan untuk "
               "AUC, belum untuk seluruh metrik."),
        bullet("<b>Tidak ada uji signifikansi koefisien.</b> Nilai p dan interval kepercayaan "
               "belum dihitung, sehingga kekuatan bukti tiap fitur belum dinilai secara formal."),
    ])