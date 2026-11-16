"""Feature engineering: encoding, scaling, dan pipeline model."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    """One-hot encoding untuk kategorikal + standardisasi untuk numerik.

    ``drop='first'`` mencegah dummy variable trap (kolinearitas sempurna),
    ``handle_unknown='ignore'`` membuat model tetap aman saat menerima
    kategori baru pada data produksi.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                config.CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(
    C: float = 1.0,
    class_weight: str | dict | None = None,
    max_iter: int = 1000,
) -> Pipeline:
    """Pipeline lengkap: preprocessing -> Logistic Regression."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=config.SEED,
                ),
            ),
        ]
    )


def feature_names(pipeline: Pipeline) -> list[str]:
    """Nama fitur hasil transformasi tanpa prefix ``num__``/``cat__``."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return [str(n).split("__", 1)[-1] for n in names]


_LABELS = {
    "tenure": "Masa berlangganan (tenure)",
    "MonthlyCharges": "Tagihan bulanan",
    "TotalCharges": "Total tagihan",
    "SeniorCitizen": "Pelanggan senior",
    "Contract_Two year": "Kontrak: 2 tahun",
    "Contract_One year": "Kontrak: 1 tahun",
    "InternetService_Fiber optic": "Internet: Fiber optic",
    "InternetService_No": "Tanpa layanan internet",
    "PaymentMethod_Electronic check": "Bayar: electronic check",
    "PaymentMethod_Mailed check": "Bayar: mailed check",
    "PaymentMethod_Credit card (automatic)": "Bayar: kartu kredit (auto)",
    "PaperlessBilling_Yes": "Paperless billing",
    "StreamingTV_Yes": "Streaming TV",
    "StreamingMovies_Yes": "Streaming Movies",
    "OnlineSecurity_Yes": "Online Security",
    "OnlineBackup_Yes": "Online Backup",
    "DeviceProtection_Yes": "Device Protection",
    "TechSupport_Yes": "Tech Support",
    "PhoneService_Yes": "Punya layanan telepon",
    "MultipleLines_Yes": "Multi-line",
    "MultipleLines_No phone service": "Tanpa layanan telepon",
    "gender_Male": "Gender: laki-laki",
    "Partner_Yes": "Punya pasangan",
    "Dependents_Yes": "Punya tanggungan",
}

_INTERNET = {
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
}

for _base, _label in _INTERNET.items():
    _LABELS[f"{_base}_No internet service"] = f"{_label}: tanpa internet"


def humanize(name: str) -> str:
    """Ubah nama fitur teknis menjadi label ramah presentasi."""
    if name in _LABELS:
        return _LABELS[name]
    if name.endswith("_No phone service"):
        return name[: -len("_No phone service")] + ": tanpa layanan telepon"
    return name.replace("_", " ")
