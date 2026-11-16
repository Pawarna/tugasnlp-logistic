"""Orkestrasi end-to-end: data -> model -> evaluasi -> artifact -> dashboard.

Jalankan:
    python run_pipeline.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src import config, data, evaluate, plots
from src import train as trainer
from src.dashboard import build_dashboard


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    started = time.time()

    # ---------------------------------------------------------- 1. Data ----
    print("[1/8] Memuat dan membersihkan data ...")
    raw = data.load_raw()
    df = data.clean(raw)
    stats = data.describe_dataset(df)
    print(f"      {stats['n_rows']} baris x {stats['n_cols']} kolom | "
          f"churn rate {stats['churn_rate']}%")

    # ------------------------------------------------------ 2. EDA plot ----
    print("[2/8] Membuat visualisasi EDA ...")
    plots.plot_class_balance(df)
    plots.plot_churn_by_contract(df)
    plots.plot_churn_by_tenure(df)
    plots.plot_monthly_distribution(df)
    plots.plot_correlation(df)
    plots.plot_architecture()

    # --------------------------------------------------------- 3. Split ----
    print("[3/8] Split stratified 80:20 ...")
    X_train, X_test, y_train, y_test = trainer.split_data(df)
    X_all, y_all = data.split_xy(df)
    print(f"      train={X_train.shape[0]} test={X_test.shape[0]}")

    # -------------------------------------------------------- 4. Train ----
    print("[4/8] Melatih Logistic Regression ...")
    model = trainer.fit_model(X_train, y_train)
    proba = trainer.score_customers(model, X_test)
    print(f"      konvergen dalam {model.named_steps['classifier'].n_iter_[0]} iterasi")

    # ------------------------------------------------------- 5. Eval ----
    print("[5/8] Evaluasi model ...")
    metrics_default = evaluate.compute_metrics(y_test, proba, config.DECISION_THRESHOLD)
    metrics_recommended = evaluate.compute_metrics(y_test, proba, config.RECOMMENDED_THRESHOLD)
    threshold_df = evaluate.threshold_table(y_test, proba)
    threshold_df.to_csv(config.THRESHOLD_FILE, index=False)

    cv_result = trainer.cross_validate(model, X_all, y_all)
    coef_df = trainer.coefficients_table(model)
    coef_df.to_csv(config.COEF_FILE, index=False)

    fpr, tpr, _ = roc_curve(y_test, proba)
    plots.plot_confusion(
        np.array([
            [metrics_default["tn"], metrics_default["fp"]],
            [metrics_default["fn"], metrics_default["tp"]],
        ])
    )
    plots.plot_roc(fpr, tpr, metrics_default["roc_auc"])
    plots.plot_threshold_curve(threshold_df)
    plots.plot_coefficients(coef_df)
    plots.plot_cv(cv_result)

    # -------------------------------------------------- 6. High risk ----
    print("[6/8] Menyusun daftar pelanggan berisiko ...")
    risk = df.loc[X_test.index, [
        config.ID_COLUMN, "Contract", "tenure", "MonthlyCharges",
        "InternetService", "PaymentMethod",
    ]].copy()
    risk["churn_probability"] = np.round(proba, 4)
    risk["risk_band"] = [trainer.risk_band(p) for p in proba]
    risk = risk.sort_values("churn_probability", ascending=False)
    risk.to_csv(config.HIGHRISK_FILE, index=False)

    metrics = {
        "dataset": stats,
        "split": {
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "train_churn_rate": round(float(y_train.mean()) * 100, 2),
            "test_churn_rate": round(float(y_test.mean()) * 100, 2),
        },
        "model": {
            "algorithm": "Logistic Regression (L2, lbfgs)",
            "n_features": int(len(coef_df)),
            "C": 1.0,
            "max_iter": 1000,
            "n_iter": int(model.named_steps["classifier"].n_iter_[0]),
            "seed": config.SEED,
        },
        "baseline": {
            "accuracy": round(float((y_test == 0).mean()), 4),
        },
        "metrics_default": metrics_default,
        "metrics_recommended": metrics_recommended,
        "confusion": [
            [metrics_default["tn"], metrics_default["fp"]],
            [metrics_default["fn"], metrics_default["tp"]],
        ],
        "cv": cv_result,
        "threshold_table": threshold_df.to_dict(orient="records"),
        "top_positive": coef_df.sort_values("odds_ratio", ascending=False)
        .head(8)[["label", "coefficient", "odds_ratio"]].to_dict(orient="records"),
        "top_negative": coef_df.sort_values("odds_ratio")
        .head(8)[["label", "coefficient", "odds_ratio"]].to_dict(orient="records"),
    }
    config.METRICS_FILE.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------- 7. Simpan model ----
    print("[7/8] Menyimpan model & artifact ...")
    joblib.dump(model, config.MODEL_FILE)

    # ------------------------------------------------------ 8. Dashboard ----
    print("[8/8] Membuat dashboard HTML ...")
    build_dashboard(metrics, coef_df, risk)

    # ---------------------------------------------------------- Ringkasan ----
    elapsed = time.time() - started
    print("\n" + "=" * 62)
    print("RINGKASAN HASIL")
    print("=" * 62)
    print(f"Accuracy  : {_pct(metrics_default['accuracy'])}  "
          f"(baseline naif {_pct(metrics['baseline']['accuracy'])})")
    print(f"Precision : {_pct(metrics_default['precision'])}")
    print(f"Recall    : {_pct(metrics_default['recall'])}")
    print(f"F1-Score  : {_pct(metrics_default['f1'])}")
    print(f"ROC-AUC   : {metrics_default['roc_auc']:.4f}")
    print(f"CV AUC    : {cv_result['mean']:.4f} +/- {cv_result['std']:.4f}")
    print(f"Confusion : TN={metrics_default['tn']} FP={metrics_default['fp']} "
          f"FN={metrics_default['fn']} TP={metrics_default['tp']}")
    print("-" * 62)
    print(f"Threshold {config.RECOMMENDED_THRESHOLD:.2f} -> recall "
          f"{_pct(metrics_recommended['recall'])}, precision "
          f"{_pct(metrics_recommended['precision'])}")
    print(f"Selesai dalam {elapsed:.1f} detik")
    print(f"Artifact  : {config.ARTIFACT_DIR}")
    print(f"Dashboard : {config.DASHBOARD_FILE}")


if __name__ == "__main__":
    main()
