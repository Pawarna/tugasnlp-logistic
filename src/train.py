"""Prosedur training, validasi silang, dan ekstraksi koefisien."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import config, data, features


def split_data(df: pd.DataFrame, test_size: float = config.TEST_SIZE, seed: int = config.SEED):
    """Split stratified 80:20 agar proporsi kelas churn terjaga."""
    X, y = data.split_xy(df)
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def fit_model(X_train, y_train, **kwargs):
    """Latih pipeline Logistic Regression."""
    model = features.build_pipeline(**kwargs)
    model.fit(X_train, y_train)
    return model


def cross_validate(model, X, y, folds: int = config.CV_FOLDS, seed: int = config.SEED) -> dict:
    """K-fold stratified cross-validation dengan metrik ROC-AUC."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return {
        "scores": [round(float(s), 4) for s in scores],
        "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
        "folds": folds,
    }


def coefficients_table(model) -> pd.DataFrame:
    """Koefisien, odds ratio, dan arah pengaruh tiap fitur."""
    names = features.feature_names(model)
    coefs = model.named_steps["classifier"].coef_[0]
    df = pd.DataFrame(
        {
            "feature": names,
            "label": [features.humanize(n) for n in names],
            "coefficient": np.round(coefs, 4),
            "odds_ratio": np.round(np.exp(coefs), 4),
        }
    )
    df["arah"] = np.where(df["coefficient"] > 0, "menaikkan risiko", "menurunkan risiko")
    return df.sort_values("coefficient").reset_index(drop=True)


def score_customers(model, X) -> np.ndarray:
    """Probabilitas churn untuk setiap baris."""
    return model.predict_proba(X)[:, 1]


def risk_band(probability: float) -> str:
    """Kategorisasi risiko untuk kebutuhan operasional."""
    if probability < config.RISK_BANDS["rendah"]:
        return "Rendah"
    if probability <= config.RISK_BANDS["sedang"]:
        return "Sedang"
    return "Tinggi"
