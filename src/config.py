"""Konfigurasi global proyek Tugas P9 - Klasifikasi Logistic Regression.

Semua path diturunkan dari lokasi file ini sehingga proyek tetap portabel
(tinggal pindahkan foldernya, tidak ada path absolut yang di-hardcode).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Path
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
FIGURE_DIR = ARTIFACT_DIR / "figures"
NOTEBOOK_DIR = ROOT / "notebook"
REPORT_DIR = ROOT / "reports"

for _d in (DATA_DIR, ARTIFACT_DIR, FIGURE_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "Telco-Customer-Churn.csv"
MODEL_FILE = ARTIFACT_DIR / "model.joblib"
METRICS_FILE = ARTIFACT_DIR / "metrics.json"
COEF_FILE = ARTIFACT_DIR / "coefficients.csv"
THRESHOLD_FILE = ARTIFACT_DIR / "threshold_tuning.csv"
HIGHRISK_FILE = ARTIFACT_DIR / "high_risk_customers.csv"
REPORT_PDF = REPORT_DIR / "Laporan_Tugas_P9_Kelompok_1.pdf"
DASHBOARD_FILE = ARTIFACT_DIR / "dashboard.html"

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

# --------------------------------------------------------------------------
# Eksperimen
# --------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DECISION_THRESHOLD = 0.50
RECOMMENDED_THRESHOLD = 0.30
RISK_BANDS = {"rendah": 0.30, "sedang": 0.60}  # <0.30 rendah, 0.30-0.60 sedang, >0.60 tinggi

# --------------------------------------------------------------------------
# Identitas kelompok (dipakai di laporan & dashboard)
# --------------------------------------------------------------------------
GROUP_NAME = "Kelompok 1"
TOPIC = "Klasifikasi Logistic Regression"
REPORT_TITLE = (
    "Sistem Prediksi Customer Churn pada Perusahaan Telekomunikasi "
    "Menggunakan Klasifikasi Logistic Regression"
)
COURSE = "Pengantar Data Science / Penambangan Data"
LECTURER = "Dosen Pengampu"
INSTITUTION = "Universitas ................"
FACULTY = "Fakultas ................"
PROGRAM = "Program Studi ................"
CITY = "................"
YEAR = "2026"
MEMBERS = [
    ("Muhammad Besttan Dhya Ulhaq", "2023018039", "Studi literatur, landasan teori, penyusunan laporan"),
    ("Ulul Hikam", "2023018037", "Preprocessing, implementasi pipeline, training model"),
    ("Riski Intan Pertiwi", "2023018048", "Evaluasi model, visualisasi data, interpretasi hasil"),
    ("Pawarna", "2023018061", "Desain sistem, dashboard, penyusunan slide presentasi"),
]
