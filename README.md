# Tugas P9 — Klasifikasi Logistic Regression

**Sistem Prediksi Customer Churn pada Perusahaan Telekomunikasi**

Kelompok 1:

| Nama | NIM |
|---|---|
| Muhammad Besttan Dhya Ulhaq | 2023018039 |
| Ulul Hikam | 2023018037 |
| Riski Intan Pertiwi | 2023018048 |
| Pawarna | 2023018061 |

---

## Ringkasan Hasil

Dataset: **Telco Customer Churn (IBM)** — 7.043 pelanggan, 19 fitur, 26,54% churn.
Model: regresi logistik dengan regularisasi L2, 30 fitur setelah one-hot encoding.

| Metrik | Threshold 0,50 | Threshold 0,30 |
|---|---|---|
| Accuracy | 80,70% | 74,95% |
| Precision | 66,04% | 51,93% |
| Recall | 56,15% | **75,40%** |
| F1-score | 60,69% | 61,50% |
| ROC-AUC | **0,8422** | 0,8422 |

- Baseline naif (selalu "tidak churn"): 73,46%
- Validasi silang 5-fold: AUC **0,8451 ± 0,0134**
- Confusion matrix @0,50: TN 927 · FP 108 · FN 164 · TP 210

---

## Struktur Proyek

```
Tugas-P9-Kelompok-1/
|-- run_pipeline.py          # jalankan seluruh pipeline (data -> model -> dashboard)
|-- predict.py               # CLI scoring pelanggan baru
|-- requirements.txt
|-- data/                    # dataset (diunduh otomatis bila kosong)
|-- src/                     # kode inti (config, data, features, train, evaluate, plots)
|-- scripts/                 # generator laporan PDF & notebook
|-- notebook/                # notebook yang sudah tereksekusi
|-- artifacts/               # model, metrik, gambar, dashboard.html
`-- reports/                 # laporan PDF
```

## Cara Menjalankan

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

python run_pipeline.py            # latih model + buat artefak + dashboard
python predict.py --demo          # coba scoring 3 profil pelanggan
python predict.py --top 10        # 10 pelanggan paling berisiko
python -m scripts.build_report    # bangun ulang laporan PDF
```

Keluaran utama:

- `reports/Laporan_Tugas_P9_Kelompok_1.pdf` — laporan lengkap (39 halaman)
- `artifacts/dashboard.html` — dashboard statis (buka di browser)
- `notebook/Tugas_P9_Logistic_Regression.ipynb` — notebook tereksekusi

## Yang Perlu Disesuaikan Sebelum Dikumpulkan

Edit `src/config.py` pada bagian identitas:

```python
INSTITUTION = "Universitas ...."
FACULTY     = "Fakultas ...."
PROGRAM     = "Program Studi ...."
CITY        = "...."
```

Setelah diubah, jalankan `python -m scripts.build_report` agar cover PDF ikut berubah.

## Catatan Metodologis

- Seluruh transformasi dibungkus `Pipeline`, di-*fit* hanya pada data latih
  (mencegah *data leakage*).
- Pembagian data *stratified* 80:20 dengan `random_state=42` agar dapat direproduksi.
- Terdapat multikolinearitas antara `tenure`, `TotalCharges`, dan `MonthlyCharges`
  (r = 0,826 dan 0,651) yang menyebabkan koefisien tagihan bulanan bertanda negatif.
  Hal ini dibahas pada Subbab 4.6.2 laporan dan menjadi salah satu keterbatasan model.

## Sumber Data

IBM Sample Data Sets — *Telco Customer Churn*
<https://github.com/IBM/telco-customer-churn-on-icp4d># tugasnlp-logistic
