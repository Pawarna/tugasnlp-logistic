"""Visualisasi untuk laporan dan dashboard (matplotlib, backend Agg)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from . import config, features

NAVY = "#1F3864"
BLUE = "#2E75B6"
RED = "#C00000"
GRAY = "#7F7F7F"
LIGHT = "#D9E2F3"
GREEN = "#548235"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#BFBFBF",
        "axes.labelcolor": "#404040",
        "xtick.color": "#404040",
        "ytick.color": "#404040",
        "legend.frameon": False,
        "figure.autolayout": False,
    }
)


def _save(fig, name: str) -> Path:
    path = config.FIGURE_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- EDA ----
def plot_class_balance(df: pd.DataFrame) -> Path:
    counts = df["Churn"].value_counts().sort_index()
    labels = ["Tidak churn (0)", "Churn (1)"]
    colors = [BLUE, RED]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bars = ax.bar(labels, counts.values, color=colors, width=0.55)
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 90,
            f"{value:,}\n({value / len(df) * 100:.2f}%)".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#262626",
        )
    ax.set_ylabel("Jumlah pelanggan")
    ax.set_title("Distribusi kelas target (Churn)")
    ax.set_ylim(0, counts.max() * 1.25)
    return _save(fig, "01_distribusi_kelas")


def plot_churn_by_contract(df: pd.DataFrame) -> Path:
    order = ["Month-to-month", "One year", "Two year"]
    rates = (
        df.groupby("Contract")["Churn"].mean().reindex(order) * 100
    )
    counts = df["Contract"].value_counts().reindex(order)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    bars = ax.bar(
        ["Month-to-month", "One year", "Two year"],
        rates.values,
        color=[RED, BLUE, NAVY],
        width=0.55,
    )
    for bar, rate, n in zip(bars, rates.values, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.9,
            f"{rate:.1f}%\n(n={n:,})".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Churn rate")
    ax.set_title("Churn rate berdasarkan tipe kontrak")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylim(0, max(rates) * 1.28)
    return _save(fig, "02_churn_per_kontrak")


def plot_churn_by_tenure(df: pd.DataFrame) -> Path:
    bins = [0, 12, 24, 48, 72]
    labels = ["0-12 bln", "13-24 bln", "25-48 bln", "49-72 bln"]
    groups = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
    rates = df.groupby(groups, observed=True)["Churn"].mean() * 100
    counts = df.groupby(groups, observed=True)["Churn"].size()

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    bars = ax.bar(labels, rates.values, color=[RED, "#E07B39", BLUE, NAVY], width=0.55)
    for bar, rate, n in zip(bars, rates.values, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.8,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_ylabel("Churn rate")
    ax.set_xlabel("Kelompok masa berlangganan")
    ax.set_title("Churn rate menurun seiring masa berlangganan")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylim(0, max(rates) * 1.22)
    return _save(fig, "03_churn_per_tenure")


def plot_monthly_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    bins = np.arange(15, 125, 5)
    ax.hist(
        df.loc[df["Churn"] == 0, "MonthlyCharges"],
        bins=bins, alpha=0.6, color=BLUE, label="Tidak churn", density=True,
    )
    ax.hist(
        df.loc[df["Churn"] == 1, "MonthlyCharges"],
        bins=bins, alpha=0.55, color=RED, label="Churn", density=True,
    )
    ax.set_xlabel("Tagihan bulanan")
    ax.set_ylabel("Kepadatan")
    ax.set_title("Sebaran tagihan bulanan per kelas")
    ax.legend(loc="upper right")
    return _save(fig, "04_distribusi_tagihan")


def plot_correlation(df: pd.DataFrame) -> Path:
    cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(corr.values, cmap="Blues", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(["tenure", "Monthly", "Total"], fontsize=8)
    ax.set_yticklabels(["tenure", "Monthly", "Total"], fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr.values[i, j]
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center",
                color="white" if abs(value) > 0.6 else "#262626", fontsize=8.5,
            )
    ax.set_title("Korelasi antar fitur numerik")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, "05_korelasi")


# ------------------------------------------------------------ EVALUASI ----
def plot_confusion(cm: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.imshow(cm, cmap="Blues")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            ax.text(
                j, i - 0.12, f"{value:,}".replace(",", "."),
                ha="center", va="center", fontsize=15, fontweight="bold",
                color="white" if value > cm.max() * 0.6 else "#1F3864",
            )
            ax.text(
                j, i + 0.22, labels[i][j],
                ha="center", va="center", fontsize=8,
                color="white" if value > cm.max() * 0.6 else "#404040",
            )
    ax.set_xticks([0, 1], ["Prediksi: tidak churn", "Prediksi: churn"])
    ax.set_yticks([0, 1], ["Aktual:\ntidak churn", "Aktual:\nchurn"])
    ax.set_title("Confusion matrix - data uji (threshold 0,50)")
    ax.set_xticks(np.arange(-0.5, 2), minor=True)
    ax.set_yticks(np.arange(-0.5, 2), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    return _save(fig, "06_confusion_matrix")


def plot_roc(fpr, tpr, auc_value: float) -> Path:
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.plot(fpr, tpr, color=NAVY, linewidth=2, label=f"Logistic Regression (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, linewidth=1, label="Acak (AUC = 0,5)")
    ax.fill_between(fpr, tpr, alpha=0.10, color=NAVY)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Kurva ROC - data uji")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    return _save(fig, "07_roc_curve")


def plot_threshold_curve(table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(table["threshold"], table["precision_pct"], marker="o", ms=3.5,
            color=BLUE, linewidth=1.6, label="Precision")
    ax.plot(table["threshold"], table["recall_pct"], marker="s", ms=3.5,
            color=RED, linewidth=1.6, label="Recall")
    ax.plot(table["threshold"], table["f1_pct"], marker="^", ms=3.5,
            color=GREEN, linewidth=1.4, linestyle="--", label="F1-score")
    ax.axvline(0.5, color=GRAY, linestyle=":", linewidth=1)
    ax.axvline(0.3, color=NAVY, linestyle=":", linewidth=1)
    ax.text(0.505, 92, "default 0,50", fontsize=7, color=GRAY)
    ax.text(0.255, 92, "rekomendasi 0,30", fontsize=7, color=NAVY, ha="right")
    ax.set_xlabel("Threshold probabilitas")
    ax.set_ylabel("Persentase")
    ax.set_title("Trade-off precision, recall, dan F1 terhadap threshold")
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.55), fontsize=7.5)
    ax.set_ylim(35, 100)
    ax.grid(axis="y", linestyle=":", color="#DDDDDD")
    ax.set_axisbelow(True)
    return _save(fig, "08_threshold_tuning")


def plot_coefficients(coef_df: pd.DataFrame, top_n: int = 12) -> Path:
    subset = coef_df.reindex(coef_df["coefficient"].abs().sort_values(ascending=False).index).head(top_n)
    subset = subset.sort_values("coefficient")
    colors = [RED if c > 0 else BLUE for c in subset["coefficient"]]

    fig, ax = plt.subplots(figsize=(6.0, 0.34 * top_n + 1.4))
    bars = ax.barh(subset["label"], subset["coefficient"], color=colors, height=0.66)
    ax.axvline(0, color="#404040", linewidth=0.8)
    for bar, value in zip(bars, subset["coefficient"]):
        offset = 0.03 if value >= 0 else -0.03
        ax.text(
            value + offset, bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}", va="center",
            ha="left" if value >= 0 else "right", fontsize=7.5,
        )
    ax.set_xlabel("Koefisien (log-odds)")
    ax.set_title(f"{top_n} fitur paling berpengaruh")
    ax.set_xlim(subset["coefficient"].min() - 0.35, subset["coefficient"].max() + 0.35)
    handles = [
        mpatches.Patch(color=RED, label="Menaikkan risiko churn"),
        mpatches.Patch(color=BLUE, label="Menurunkan risiko churn"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5)
    return _save(fig, "09_koefisien")


def plot_cv(cv_result: dict) -> Path:
    scores = cv_result["scores"]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    bars = ax.bar([f"Fold {i + 1}" for i in range(len(scores))], scores, color=BLUE, width=0.55)
    for bar, value in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.4f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.axhline(cv_result["mean"], color=RED, linestyle="--", linewidth=1.2)
    ax.text(0.015, 0.955,
            f"Rata-rata = {cv_result['mean']:.4f}   SD = {cv_result['std']:.4f}",
            transform=ax.transAxes, color=RED, fontsize=7.6, va="top")
    ax.set_ylim(0.78, 0.90)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Validasi silang {cv_result['folds']}-fold (AUC)")
    return _save(fig, "10_cross_validation")


# --------------------------------------------------------------- SISTEM ----
def plot_architecture() -> Path:
    """Diagram alur sistem (pola snake: atas kiri->kanan, bawah kanan->kiri)."""
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    width, height = 2.05, 0.92
    xs = [0.18, 2.60, 5.02, 7.44]

    top = [
        "1. Dataset\n7.043 pelanggan",
        "2. Data cleaning\n11 missing value",
        "3. Preprocessing\nencoding + scaling",
        "4. Split 80:20\nstratified",
    ]
    # baris bawah dibaca dari kanan ke kiri
    bottom = [
        "8. Deployment\nCLI + dashboard",
        "7. Interpretasi\nkoefisien & odds ratio",
        "6. Evaluasi\nAUC, CM, threshold",
        "5. Training\nLogistic Regression",
    ]

    for x, label in zip(xs, top):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 2.55), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            linewidth=1.0, edgecolor=NAVY, facecolor=LIGHT))
        ax.text(x + width / 2, 2.55 + height / 2, label,
                ha="center", va="center", fontsize=7.4, color="#1F1F1F")

    for x, label in zip(xs, bottom):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.32), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            linewidth=1.0, edgecolor=NAVY, facecolor="#EDF2FA"))
        ax.text(x + width / 2, 0.32 + height / 2, label,
                ha="center", va="center", fontsize=7.4, color="#1F1F1F")

    # panah horizontal baris atas
    for i in range(3):
        x0 = xs[i] + width + 0.05
        x1 = xs[i + 1] - 0.05
        ax.annotate("", xy=(x1, 3.01), xytext=(x0, 3.01),
                    arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2))

    # panah turun dari kotak 4 ke kotak 5
    ax.annotate("", xy=(8.465, 1.31), xytext=(8.465, 2.48),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2))

    # panah horizontal baris bawah (kanan -> kiri)
    for i in range(3, 0, -1):
        x0 = xs[i] - 0.05
        x1 = xs[i - 1] + width + 0.05
        ax.annotate("", xy=(x1, 0.78), xytext=(x0, 0.78),
                    arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2))

    ax.text(0.18, 3.62, "Alur sistem prediksi churn", fontsize=9.5,
            fontweight="bold", color=NAVY)
    ax.text(0.18, 0.03,
            "Seluruh alur dijalankan ulang dengan satu perintah:  python run_pipeline.py",
            fontsize=6.9, color=GRAY)
    return _save(fig, "11_arsitektur")
