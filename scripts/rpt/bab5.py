"""Bab V Kesimpulan dan Saran."""
from __future__ import annotations

from .engine import CTX, chapter, h2, numbered, p, bullet

M = CTX.metrics
MD = M["metrics_default"]
MR = M["metrics_recommended"]


def build(ctx) -> None:
    ctx.add(*chapter("BAB V", "Kesimpulan dan Saran"))

    ctx.add(h2("5.1  Kesimpulan"))
    ctx.add(p(
        "Berdasarkan hasil pemodelan dan pembahasan pada Bab IV, dapat ditarik empat kesimpulan "
        "yang menjawab rumusan masalah pada Subbab 1.2."
    ))
    ctx.add(*numbered([
        "<b>Faktor paling berpengaruh.</b> Tipe kontrak merupakan faktor protektif terkuat "
        "(kontrak dua tahun memiliki odds churn 0,265 kali kontrak bulanan), diikuti oleh masa "
        "berlangganan (odds turun menjadi 0,286 setiap kenaikan satu simpangan baku) dan jenis "
        "layanan internet (fiber optic menaikkan odds 3,262 kali). Dengan demikian, pelanggan "
        "kontrak bulanan yang baru berlangganan dan menggunakan fiber optic merupakan segmen "
        "paling rentan.",

        f"<b>Kinerja model.</b> Regresi logistik menghasilkan <i>accuracy</i> "
        f"{_pct(MD['accuracy'])}, <i>precision</i> {_pct(MD['precision'])}, <i>recall</i> "
        f"{_pct(MD['recall'])}, <i>F1-score</i> {_pct(MD['f1'])}, dan <i>ROC-AUC</i> "
        f"{_num(MD['roc_auc'], 4)} pada ambang 0,50. Nilai tersebut mengungguli baseline naif "
        f"({_pct(M['baseline']['accuracy'])}) dan tergolong diskriminasi baik. Validasi silang "
        f"lima lipatan memberi AUC {_num(M['cv']['mean'], 4)} ± {_num(M['cv']['std'], 4)}, "
        "sehingga model dinilai stabil dan tidak mengalami <i>overfitting</i>.",

        "<b>Ambang keputusan.</b> Ambang bawaan 0,50 terlalu konservatif untuk konteks retensi "
        f"karena hanya menangkap {_pct(MD['recall'])} pelanggan yang benar-benar churn. Ambang "
        f"0,30 menaikkan <i>recall</i> menjadi {_pct(MR['recall'])} dengan <i>precision</i> "
        f"{_pct(MR['precision'])}, sehingga lebih sesuai dengan asimetri biaya kesalahan pada "
        "program retensi. Ambang ini bersifat rekomendasi operasional dan perlu ditinjau ulang "
        "apabila kapasitas tim berubah.",

        "<b>Penerjemahan menjadi sistem.</b> Model diwujudkan sebagai <i>pipeline</i> yang dapat "
        "dijalankan ulang, dilengkapi antarmuka baris perintah untuk memberi skor pada pelanggan "
        "baru serta <i>dashboard</i> yang menampilkan pelanggan berisiko tertinggi. Sistem ini "
        "memungkinkan hasil analisis dipakai langsung oleh tim retensi, bukan berhenti sebagai "
        "eksperimen di dalam buku kerja.",
    ]))

    ctx.add(h2("5.2  Saran"))
    ctx.add(p("Untuk pengembangan selanjutnya, kami mengusulkan beberapa arah perbaikan:"))
    ctx.add(*[
        bullet("<b>Penanganan multikolinearitas.</b> Hitung <i>variance inflation factor</i> dan "
               "pertimbangkan untuk menghapus salah satu fitur tagihan, atau bentuk fitur turunan "
               "seperti rasio tagihan bulanan terhadap rata-rata tagihan pelanggan."),
        bullet("<b>Penanganan ketidakseimbangan kelas.</b> Bandingkan ambang keputusan dengan "
               "pendekatan <i>class weight</i> dan SMOTE untuk melihat mana yang memberi "
               "keseimbangan terbaik antara <i>recall</i> dan <i>precision</i>."),
        bullet("<b>Pencarian hiperparameter.</b> Lakukan <i>grid search</i> atau "
               "<i>random search</i> pada parameter C dan jenis penalti dengan validasi silang "
               "sebagai dasar pemilihan."),
        bullet("<b>Perbandingan model.</b> Bandingkan regresi logistik dengan <i>random forest</i>, "
               "<i>gradient boosting</i>, dan jaringan saraf sederhana, lalu gunakan regresi "
               "logistik sebagai model acuan yang interpretable."),
        bullet("<b>Evaluasi berbasis biaya.</b> Susun matriks biaya atau hitung "
               "<i>expected maximum profit</i> agar pemilihan ambang tidak hanya berdasarkan "
               "metrik statistik, tetapi juga nilai ekonomi."),
        bullet("<b>Validasi temporal.</b> Apabila tersedia data beberapa periode, gunakan "
               "pembagian berdasarkan waktu untuk menguji apakah model tetap akurat ketika "
               "diterapkan pada pelanggan di masa depan."),
        bullet("<b>Pengayaan fitur.</b> Tambahkan data interaksi layanan pelanggan, jumlah keluhan, "
               "pemakaian kuota, dan riwayat pembayaran agar model menangkap penyebab churn yang "
               "lebih dekat dengan pengalaman pelanggan."),
    ])
    ctx.add(p(
        "Dengan memperhatikan saran-saran tersebut, model prediksi churn berbasis regresi logistik "
        "ini diharapkan dapat berkembang dari prototipe akademik menjadi komponen yang benar-benar "
        "digunakan dalam pengambilan keputusan retensi."
    ))


def _pct(value: float, digits: int = 2) -> str:
    from .engine import pct
    return pct(value, digits)


def _num(value: float, digits: int = 2) -> str:
    from .engine import num
    return num(value, digits)