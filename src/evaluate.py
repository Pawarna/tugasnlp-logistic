"""Metrik evaluasi dan tabel threshold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Hitung metrik klasifikasi pada threshold tertentu."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 2),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_table(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """Tabel trade-off precision/recall untuk berbagai threshold."""
    if thresholds is None:
        thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, y_proba, t)
        rows.append(
            {
                "threshold": round(float(t), 2),
                "accuracy_pct": round(m["accuracy"] * 100, 2),
                "precision_pct": round(m["precision"] * 100, 2),
                "recall_pct": round(m["recall"] * 100, 2),
                "f1_pct": round(m["f1"] * 100, 2),
                "churn_tertangkap": m["tp"],
                "churn_terlewat": m["fn"],
                "false_alarm": m["fp"],
            }
        )
    return pd.DataFrame(rows)


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate (%) per kategori sebuah kolom."""
    table = (
        df.groupby(column, observed=True)["Churn"]
        .agg(jumlah="size", churn="sum")
        .assign(churn_rate=lambda d: (d["churn"] / d["jumlah"] * 100).round(2))
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )
    return table
