"""CLI untuk melakukan scoring pelanggan dengan model yang sudah dilatih.

Contoh:
    python predict.py --demo
    python predict.py --input data/pelanggan_baru.csv --output hasil_skor.csv
    python predict.py --top 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src import train as trainer

DEMO_CUSTOMERS = [
    {
        "nama": "Pelanggan baru - fiber, kontrak bulanan",
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 3, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 286.5,
    },
    {
        "nama": "Pelanggan setia - DSL, kontrak 2 tahun",
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 55.0, "TotalCharges": 3300.0,
    },
    {
        "nama": "Pelanggan menengah - kontrak 1 tahun",
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 18, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check", "MonthlyCharges": 78.0, "TotalCharges": 1404.0,
    },
]

FEATURES = config.NUMERIC_COLS + config.CATEGORICAL_COLS


def load_model():
    if not config.MODEL_FILE.exists():
        sys.exit(
            f"Model belum ada di {config.MODEL_FILE}.\n"
            "Jalankan dulu: python run_pipeline.py"
        )
    return joblib.load(config.MODEL_FILE)


def score(model, frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        sys.exit(f"Kolom berikut wajib ada tetapi tidak ditemukan: {', '.join(missing)}")

    proba = model.predict_proba(frame[FEATURES])[:, 1]
    out = frame.copy()
    out["churn_probability"] = proba.round(4)
    out["risk_band"] = [trainer.risk_band(p) for p in proba]
    out["rekomendasi"] = [
        "Prioritaskan retensi (kontak + penawaran khusus)" if p >= 0.6
        else "Pantau & kirim program loyalitas" if p >= 0.3
        else "Tidak perlu intervensi khusus"
        for p in proba
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Scoring risiko churn pelanggan")
    parser.add_argument("--demo", action="store_true", help="jalankan 3 contoh pelanggan")
    parser.add_argument("--input", type=Path, help="CSV berisi data pelanggan")
    parser.add_argument("--output", type=Path, help="simpan hasil scoring ke CSV")
    parser.add_argument("--top", type=int, help="tampilkan N pelanggan paling berisiko dari dataset")
    args = parser.parse_args()

    model = load_model()

    if args.top:
        from src import data

        frame = data.clean(data.load_raw())
        scored = score(model, frame)
        cols = [config.ID_COLUMN, "Contract", "tenure", "MonthlyCharges",
                "churn_probability", "risk_band", "rekomendasi"]
        top = scored.nlargest(args.top, "churn_probability")[cols]
        print(top.to_string(index=False))
        if args.output:
            top.to_csv(args.output, index=False)
            print(f"\nDisimpan: {args.output}")
        return

    if args.input:
        frame = pd.read_csv(args.input)
        scored = score(model, frame)
        path = args.output or args.input.with_name(args.input.stem + "_scored.csv")
        scored.to_csv(path, index=False)
        print(scored.to_string(index=False))
        print(f"\nDisimpan: {path}")
        return

    # default: demo
    frame = pd.DataFrame(DEMO_CUSTOMERS)
    scored = score(model, frame)
    display = scored[["nama", "Contract", "tenure", "MonthlyCharges",
                      "churn_probability", "risk_band", "rekomendasi"]]
    print("\nHasil scoring contoh pelanggan\n" + "-" * 100)
    print(display.to_string(index=False))
    print("-" * 100)
    if args.output:
        scored.to_csv(args.output, index=False)
        print(f"Disimpan: {args.output}")
    print(json.dumps(
        [{"nama": r["nama"], "probabilitas": round(float(r["churn_probability"]), 4),
          "risiko": r["risk_band"]} for _, r in scored.iterrows()],
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
