"""Bab III Metodologi Penelitian."""
from __future__ import annotations

from reportlab.lib.units import cm
from reportlab.platypus import Spacer

from . import engine
from .engine import (
    CTX, bullet, chapter, code_block, figure, h2, h3, numbered, p, table_block,
)

M = CTX.metrics
D = M["dataset"]


def build(ctx) -> None:
    ctx.add(*chapter("BAB III", "Metodologi Penelitian"))

    ctx.add(h2("3.1  Gambaran Umum Sistem"))
    ctx.add(p(
        "Sistem yang dibangun menerjemahkan persoalan bisnis menjadi alur pemrosesan data yang "
        "dapat dijalankan ulang. Alur tersebut terdiri atas delapan tahap: pemuatan dataset, "
        "pembersihan data, praproses, pembagian data, pelatihan model, evaluasi, interpretasi "
        "koefisien, dan penyajian hasil melalui antarmuka. Seluruh tahap diorkestrasi oleh satu "
        "berkas <font name='Courier'>run_pipeline.py</font>, sehingga dosen maupun pengguna lain "
        "dapat mereproduksi hasil laporan ini tanpa menyalin kode secara manual."
    ))
    ctx.add(figure("11_arsitektur.png", "Arsitektur sistem prediksi churn yang dibangun",
                   "Gambar 3.1", width=15.2 * cm))
    ctx.add(p(
        "Pemisahan tahap bukan sekadar kerapian teknis. Dengan memisahkan praproses dari "
        "pelatihan, objek <i>pipeline</i> yang disimpan ke disk sudah membawa seluruh aturan "
        "transformasi, sehingga saat menerima data pelanggan baru sistem tidak perlu mengetahui "
        "detail encoding yang dipakai. Inilah yang membuat model dapat dipakai di luar notebook "
        "eksperimen."
    ))

    ctx.add(h2("3.2  Sumber Data"))
    ctx.add(p(
        "Data yang digunakan adalah dataset publik <i>Telco Customer Churn</i> yang diterbitkan "
        "IBM sebagai contoh penerapan analitik pelanggan. Dataset ini lazim dipakai pada "
        "literatur prediksi churn karena memuat campuran fitur demografis, layanan, kontrak, dan "
        "tagihan dalam satu tabel yang bersih. Tabel 3.1 merangkum spesifikasi dataset."
    ))
    ctx.add(*table_block(
        ["Atribut", "Keterangan"],
        [
            ["Nama dataset", "Telco Customer Churn (IBM Sample Data Sets)"],
            ["Jumlah baris", f"{D['n_rows']:,}".replace(",", ".") + " pelanggan"],
            ["Jumlah kolom", f"{D['n_cols']} kolom (1 identifier, 19 fitur, 1 target)"],
            ["Unit observasi", "Satu baris merepresentasikan satu pelanggan"],
            ["Target", "Churn: Yes (1) / No (0)"],
            ["Proporsi target", f"Churn {engine.num(D['churn_rate'])}% · "
                                f"tidak churn {engine.num(100 - D['churn_rate'])}%"],
            ["Nilai hilang", f"{D['n_missing_totalcharges']} sel pada kolom TotalCharges"],
            ["Duplikasi ID", f"{D['n_duplicate_id']} baris"],
            ["Sifat data", "Cross-sectional (satu potret waktu)"],
        ],
        widths=[4.0 * cm, 11.6 * cm],
        caption="Spesifikasi dataset yang digunakan",
        label="Tabel 3.1",
    ))

    ctx.add(h2("3.3  Variabel Penelitian"))
    ctx.add(p(
        "Kedua puluh variabel pada dataset dikelompokkan menjadi lima kelompok seperti pada "
        "Tabel 3.2. Kolom <font name='Courier'>customerID</font> tidak dipakai sebagai fitur "
        "karena hanya berfungsi sebagai penanda unik; memasukkannya berisiko membuat model "
        "menghafal identitas dan bukan pola perilaku."
    ))
    ctx.add(*table_block(
        ["Kelompok", "Variabel", "Tipe"],
        [
            ["Demografi", "gender, SeniorCitizen, Partner, Dependents", "Kategorikal &amp; biner"],
            ["Layanan telepon", "PhoneService, MultipleLines", "Kategorikal"],
            ["Layanan internet", "InternetService, OnlineSecurity, OnlineBackup, "
                                 "DeviceProtection, TechSupport, StreamingTV, StreamingMovies",
             "Kategorikal"],
            ["Kontrak &amp; pembayaran", "tenure, Contract, PaperlessBilling, PaymentMethod",
             "Numerik &amp; kategorikal"],
            ["Tagihan", "MonthlyCharges, TotalCharges", "Numerik kontinu"],
            ["Target", "Churn", "Biner"],
        ],
        widths=[3.4 * cm, 9.4 * cm, 2.8 * cm],
        caption="Kelompok variabel pada dataset",
        label="Tabel 3.2",
    ))

    ctx.add(h2("3.4  Tahapan Penelitian"))
    ctx.add(*numbered([
        "<b>Pemuatan data.</b> Dataset dibaca dari berkas CSV lokal; apabila berkas belum tersedia, "
        "sistem mengunduhnya dari repositori publik secara otomatis.",
        "<b>Pembersihan data.</b> Kolom TotalCharges dikonversi menjadi numerik, nilai kosong diisi "
        "nol, dan label target dipetakan menjadi 0 dan 1.",
        "<b>Analisis data eksploratif.</b> Distribusi kelas, churn rate per kategori, sebaran "
        "tagihan, dan korelasi antar fitur numerik divisualisasikan untuk membentuk hipotesis awal.",
        "<b>Praproses.</b> Fitur kategorikal di-<i>one-hot encode</i>, fitur numerik "
        "distandardisasi, dan seluruh transformasi dibungkus dalam <i>pipeline</i>.",
        "<b>Pembagian data.</b> Data dibagi 80% latih dan 20% uji secara <i>stratified</i> dengan "
        "<i>random state</i> 42 agar hasil dapat direproduksi.",
        "<b>Pelatihan model.</b> Regresi logistik dengan regularisasi L2 dilatih pada data latih.",
        "<b>Evaluasi.</b> Confusion matrix, <i>accuracy</i>, <i>precision</i>, <i>recall</i>, "
        "<i>F1-score</i>, ROC-AUC, dan validasi silang lima lipatan dihitung.",
        "<b>Penentuan ambang dan penyajian.</b> Ambang keputusan dipilih berdasarkan tujuan retensi, "
        "lalu hasil disajikan dalam <i>dashboard</i> dan antarmuka baris perintah.",
    ]))

    ctx.add(h2("3.5  Pembersihan Data"))
    ctx.add(p(
        "Dataset ini relatif bersih, namun terdapat satu jebakan yang sering terlewat: kolom "
        "TotalCharges tersimpan sebagai teks karena memuat spasi kosong, bukan angka nol. Jika "
        "kolom tersebut dipaksa menjadi numerik tanpa penanganan, seluruh kolom akan berubah "
        "menjadi tipe objek dan model gagal dilatih. Tabel 3.3 merangkum temuan dan penanganannya."
    ))
    ctx.add(*table_block(
        ["Temuan", "Jumlah", "Penanganan", "Alasan"],
        [
            ["TotalCharges bertipe teks dengan sel kosong", "11 sel",
             "Konversi paksa ke numerik, nilai kosong diisi 0",
             "Seluruh baris tersebut memiliki tenure = 0, artinya pelanggan baru yang belum pernah ditagih"],
            ["customerID unik", "0 duplikat", "Kolom dihapus dari fitur",
             "Identifier tidak memiliki daya prediksi"],
            ["Tidak ada nilai hilang lain", "0", "Tidak ada tindakan",
             "Seluruh fitur lain lengkap"],
            ["Rentang nilai ekstrem", "tenure 0-72 bulan", "Dipertahankan",
             "Masih wajar secara bisnis, bukan kesalahan input"],
            ["Ketidakseimbangan kelas", "26,54% churn", "Ditangani pada tahap evaluasi",
             "Lihat Subbab 2.4.4"],
        ],
        widths=[4.6 * cm, 2.2 * cm, 4.4 * cm, 4.4 * cm],
        caption="Temuan pembersihan data dan penanganannya",
        label="Tabel 3.3",
        align=["LEFT", "CENTER", "LEFT", "LEFT"],
        font_size=8.2,
    ))

    ctx.add(h2("3.6  Praproses Data"))
    ctx.add(p(
        "Seluruh transformasi dirangkai menggunakan <i>ColumnTransformer</i> dan <i>Pipeline</i> "
        "dari scikit-learn. Kedua objek tersebut memastikan bahwa aturan transformasi yang "
        "dipelajari dari data latih diterapkan secara identik pada data uji maupun data baru. "
        "Rincian perlakuan setiap kelompok fitur disajikan pada Tabel 3.4."
    ))
    ctx.add(*table_block(
        ["Kelompok fitur", "Perlakuan", "Jumlah kolom hasil"],
        [
            ["Numerik (tenure, MonthlyCharges, TotalCharges, SeniorCitizen)",
             "Standardisasi z-score", "4"],
            ["Kategorikal (15 variabel)",
             "One-hot encoding, drop kategori pertama, abaikan kategori tak dikenal", "26"],
            ["Total fitur setelah transformasi", "—", "30"],
        ],
        widths=[6.4 * cm, 6.8 * cm, 2.4 * cm],
        caption="Perlakuan praproses per kelompok fitur",
        label="Tabel 3.4",
        align=["LEFT", "LEFT", "CENTER"],
    ))
    ctx.add(p(
        "Pembagian data menghasilkan 5.634 baris latih dan 1.409 baris uji. Karena pemisahan "
        "dilakukan secara <i>stratified</i>, proporsi churn pada kedua bagian tetap "
        f"{engine.num(M['split']['train_churn_rate'])}% dan "
        f"{engine.num(M['split']['test_churn_rate'])}%, sama dengan proporsi pada dataset penuh. "
        "Dengan demikian, performa pada data uji dapat dianggap mewakili populasi secara wajar."
    ))

    ctx.add(h2("3.7  Konfigurasi Model"))
    ctx.add(*table_block(
        ["Parameter", "Nilai", "Keterangan"],
        [
            ["Algoritma", "Logistic Regression", "Regresi logistik biner"],
            ["Solver", "lbfgs", "Kuasi-Newton, cocok untuk data berdimensi sedang"],
            ["Regularisasi", "L2", "Menekan koefisien berlebih"],
            ["C (kebalikan kekuatan penalti)", "1,0", "Nilai bawaan scikit-learn"],
            ["max_iter", "1.000", "Batas iterasi optimasi"],
            ["class_weight", "None", "Belum digunakan pada model utama"],
            ["random_state", "42", "Menjamin reproduksibilitas"],
            ["Ambang evaluasi utama", "0,50", "Ambang bawaan"],
            ["Ambang rekomendasi operasional", "0,30", "Hasil analisis tuning pada Subbab 4.3"],
        ],
        widths=[5.4 * cm, 4.2 * cm, 6.0 * cm],
        caption="Konfigurasi model yang digunakan",
        label="Tabel 3.5",
        align=["LEFT", "CENTER", "LEFT"],
    ))
    ctx.add(p(
        "Nilai C = 1,0 dipilih karena merupakan nilai bawaan yang telah terbukti memadai pada "
        "banyak kasus, bukan hasil pencarian menyeluruh. Konsekuensinya, laporan ini tidak "
        "mengklaim bahwa konfigurasi tersebut optimal; pencarian hiperparameter dicantumkan "
        "sebagai saran pengembangan pada Bab V."
    ))

    ctx.add(h2("3.8  Lingkungan Implementasi"))
    ctx.add(*table_block(
        ["Komponen", "Versi / Keterangan"],
        [
            ["Bahasa pemrograman", "Python 3.12"],
            ["Manipulasi data", "pandas 3.1"],
            ["Komputasi numerik", "NumPy"],
            ["Pembelajaran mesin", "scikit-learn 1.9"],
            ["Visualisasi", "Matplotlib"],
            ["Penyimpanan model", "joblib"],
            ["Pembuatan laporan", "ReportLab"],
            ["Sistem operasi pengujian", "Windows"],
        ],
        widths=[5.0 * cm, 10.6 * cm],
        caption="Lingkungan implementasi dan pustaka yang digunakan",
        label="Tabel 3.6",
        align=["LEFT", "LEFT"],
    ))
    ctx.add(p(
        "Seluruh pustaka tersebut bersifat sumber terbuka dan tersedia gratis, sehingga proyek "
        "dapat dijalankan ulang tanpa biaya lisensi. Berkas <font name='Courier'>requirements.txt</font> "
        "mencatat dependensi beserta batas versi minimumnya."
    ))

    ctx.add(h2("3.9  Struktur Proyek dan Cara Menjalankan"))
    ctx.add(p(
        "Kode disusun dalam paket <font name='Courier'>src</font> agar setiap tanggung jawab "
        "terpisah: pemuatan data, rekayasa fitur, pelatihan, evaluasi, visualisasi, dan penyajian. "
        "Struktur direktori ditunjukkan pada Kode 3.1."
    ))
    ctx.add(*code_block(
        """Tugas-P9-Kelompok-1/
|-- run_pipeline.py          # orkestrasi end-to-end
|-- predict.py               # CLI scoring pelanggan
|-- requirements.txt
|-- data/
|   `-- Telco-Customer-Churn.csv
|-- src/
|   |-- config.py            # path, konstanta, identitas kelompok
|   |-- data.py              # load + cleaning
|   |-- features.py          # preprocessing + pipeline
|   |-- train.py             # split, training, CV, koefisien
|   |-- evaluate.py          # metrik + tabel threshold
|   |-- plots.py             # seluruh visualisasi
|   `-- dashboard.py         # generator dashboard HTML
|-- scripts/
|   `-- build_report.py      # generator laporan PDF ini
|-- artifacts/
|   |-- model.joblib         # model siap pakai
|   |-- metrics.json         # seluruh angka pada laporan
|   |-- coefficients.csv
|   |-- threshold_tuning.csv
|   |-- high_risk_customers.csv
|   |-- dashboard.html
|   `-- figures/
`-- reports/
    `-- Laporan_Tugas_P9_Kelompok_1.pdf""",
        caption="Kode 3.1  Struktur direktori proyek",
    ))
    ctx.add(p("Untuk menjalankan sistem dari awal, cukup dua perintah berikut:"))
    ctx.add(*code_block(
        """pip install -r requirements.txt
python run_pipeline.py

# memberi skor pada pelanggan baru
python predict.py --demo
python predict.py --input data/pelanggan_baru.csv

# membangun ulang laporan PDF ini
python -m scripts.build_report""",
        caption="Kode 3.2  Perintah utama sistem",
    ))
    ctx.add(p(
        "Eksekusi <font name='Courier'>run_pipeline.py</font> pada mesin pengujian memerlukan waktu "
        "sekitar lima detik, karena model regresi logistik memang ringan. Seluruh keluaran "
        "tersimpan di direktori <font name='Courier'>artifacts</font>, sehingga laporan dan "
        "dashboard selalu dapat dibangun ulang dari angka yang sama."
    ))
    ctx.add(Spacer(1, 0.2 * cm))