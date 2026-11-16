"""Pemuatan dan pembersihan dataset Telco Customer Churn."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from . import config


def ensure_dataset(path: Path | None = None) -> Path:
    """Unduh dataset bila belum tersedia secara lokal."""
    path = Path(path or config.DATA_FILE)
    if path.exists() and path.stat().st_size > 100_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] mengunduh dataset -> {path}")
    urllib.request.urlretrieve(config.DATA_URL, path)
    return path


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Muat data mentah apa adanya (tanpa transformasi)."""
    path = ensure_dataset(path)
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data mentah.

    Langkah:
      1. ``TotalCharges`` dikonversi ke numerik; nilai kosong (11 baris,
         pelanggan dengan ``tenure = 0``) diisi 0.
      2. ``Churn`` dipetakan menjadi 1/0.
      3. Baris duplikat dibuang (tidak ada pada dataset ini, tetap dijaga).
    """
    out = df.copy()

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0)

    if not pd.api.types.is_numeric_dtype(out[config.TARGET]):
        out[config.TARGET] = out[config.TARGET].map({"No": 0, "Yes": 1}).astype(int)

    out = out.drop_duplicates(subset=[config.ID_COLUMN]).reset_index(drop=True)
    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Pisahkan fitur (tanpa identifier) dan target."""
    X = df.drop(columns=[config.ID_COLUMN, config.TARGET])
    y = df[config.TARGET].astype(int)
    return X, y


def describe_dataset(df: pd.DataFrame) -> dict:
    """Ringkasan statistik yang dipakai di laporan."""
    y = df[config.TARGET]
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_features": int(df.shape[1] - 2),
        "n_churn": int(y.sum()),
        "n_no_churn": int((1 - y).sum()),
        "churn_rate": round(float(y.mean()) * 100, 2),
        "n_missing_totalcharges": int(
            pd.to_numeric(df["TotalCharges"], errors="coerce").isna().sum()
        ),
        "n_duplicate_id": int(df[config.ID_COLUMN].duplicated().sum()),
        "tenure_mean": round(float(df["tenure"].mean()), 2),
        "tenure_median": float(df["tenure"].median()),
        "tenure_min": float(df["tenure"].min()),
        "tenure_max": float(df["tenure"].max()),
        "tenure_sd": round(float(df["tenure"].std()), 2),
        "monthly_mean": round(float(df["MonthlyCharges"].mean()), 2),
        "monthly_median": round(float(df["MonthlyCharges"].median()), 2),
        "monthly_min": round(float(df["MonthlyCharges"].min()), 2),
        "monthly_max": round(float(df["MonthlyCharges"].max()), 2),
        "monthly_sd": round(float(df["MonthlyCharges"].std()), 2),
        "total_sd": round(float(df["TotalCharges"].std()), 2),
    }
