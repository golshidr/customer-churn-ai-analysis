"""Loading, validating and cleaning the IBM Telco Customer Churn data."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ID_COLUMN = "customerID"
TARGET = "Churn"

#: Allowed values of every categorical column, as they appear in the cleaned data.
CATEGORY_LEVELS: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
CATEGORICAL_COLUMNS: list[str] = list(CATEGORY_LEVELS)
NUMERIC_COLUMNS: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]
FEATURE_COLUMNS: list[str] = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
RAW_COLUMNS: list[str] = [ID_COLUMN, *FEATURE_COLUMNS, TARGET]

KAGGLE_DATASET = "blastchar/telco-customer-churn"
KAGGLE_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
IBM_MIRROR_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
#: SHA-256 of the CSV after normalising line endings to "\n". The Kaggle copy uses CRLF
#: and the IBM GitHub copy uses LF; their contents are otherwise identical.
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


def normalized_sha256(path: str | Path) -> str:
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def download_dataset(dest: str | Path, source: str = "auto", verify: bool = True) -> Path:
    """Download the dataset to ``dest``.

    ``source`` is ``"kaggle"`` (via ``kagglehub``), ``"ibm"`` (IBM's public GitHub copy) or
    ``"auto"`` (Kaggle first, IBM as a fallback). The file is checked against
    :data:`EXPECTED_SHA256` so both sources are known to give the same data.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    sources = ["kaggle", "ibm"] if source == "auto" else [source]
    for src in sources:
        try:
            if src == "kaggle":
                import kagglehub  # optional dependency

                folder = Path(kagglehub.dataset_download(KAGGLE_DATASET))
                shutil.copyfile(folder / KAGGLE_FILENAME, dest)
            elif src == "ibm":
                import requests

                resp = requests.get(IBM_MIRROR_URL, timeout=60)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            else:
                raise ValueError(f"unknown source {src!r}")
        except Exception as exc:  # noqa: BLE001 - try the next source
            errors.append(f"{src}: {exc}")
            continue
        if verify and normalized_sha256(dest) != EXPECTED_SHA256:
            errors.append(f"{src}: checksum mismatch")
            dest.unlink(missing_ok=True)
            continue
        return dest
    raise RuntimeError("could not download the dataset: " + "; ".join(errors))


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if required columns are missing or categories are unexpected."""
    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for col, levels in CATEGORY_LEVELS.items():
        values = df[col]
        if col == "SeniorCitizen" and pd.api.types.is_numeric_dtype(values):
            values = values.map({0: "No", 1: "Yes"})
        unknown = set(values.dropna().astype(str).unique()) - set(levels)
        if unknown:
            raise ValueError(f"unexpected values in {col}: {sorted(unknown)}")


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def clean(df: pd.DataFrame, drop_missing_total_charges: bool = True) -> pd.DataFrame:
    """Return a cleaned copy of the raw data.

    * ``TotalCharges`` is stored as text with blanks for tenure-0 customers; coerce it to a
      number and either drop those rows (original behaviour) or set them to 0.
    * ``SeniorCitizen`` 0/1 becomes "No"/"Yes" so all categorical columns are strings.
    * ``Churn`` "Yes"/"No" becomes 1/0.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    if drop_missing_total_charges:
        out = out.dropna(subset=["TotalCharges"])
    else:
        out["TotalCharges"] = out["TotalCharges"].fillna(0.0)
    if pd.api.types.is_numeric_dtype(out["SeniorCitizen"]):
        out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"})
    if not pd.api.types.is_numeric_dtype(out[TARGET]):
        out[TARGET] = out[TARGET].map({"Yes": 1, "No": 0})
    if out[TARGET].isna().any():
        raise ValueError("Churn must contain only 'Yes'/'No' or 1/0")
    out[TARGET] = out[TARGET].astype(int)
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].astype(str)
    out["tenure"] = out["tenure"].astype(int)
    out["MonthlyCharges"] = out["MonthlyCharges"].astype(float)
    if ID_COLUMN not in out.columns:
        out[ID_COLUMN] = [f"C{i:05d}" for i in range(len(out))]
    return out.reset_index(drop=True)


def load_dataset(path: str | Path, drop_missing_total_charges: bool = True) -> pd.DataFrame:
    return clean(load_raw(path), drop_missing_total_charges=drop_missing_total_charges)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS].copy(), df[TARGET].copy()


def train_test(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, stratify: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split of the cleaned frame (keeps ID and target columns)."""
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET] if stratify else None
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
