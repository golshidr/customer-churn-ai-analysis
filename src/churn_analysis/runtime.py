"""I/O helpers shared by the pipeline stages (data access, artifacts, JSON/CSV)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import resolve_path
from .data import clean, load_dataset, load_raw, train_test
from .synthetic import make_synthetic_telco

log = logging.getLogger("churn_analysis")


def load_raw_frame(cfg: dict) -> pd.DataFrame:
    """Raw (uncleaned) data: the downloaded CSV, or synthetic rows if ``cfg['synthetic']``."""
    syn = cfg.get("synthetic")
    if syn:
        return make_synthetic_telco(n=int(syn.get("n", 800)), seed=int(syn.get("seed", 0)))
    path = resolve_path(cfg, "raw_data")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `make data` (or `churn download`) first")
    return load_raw(path)


def load_data(cfg: dict) -> pd.DataFrame:
    drop = cfg["data"]["drop_missing_total_charges"]
    if cfg.get("synthetic"):
        return clean(load_raw_frame(cfg), drop_missing_total_charges=drop)
    return load_dataset(resolve_path(cfg, "raw_data"), drop_missing_total_charges=drop)


def get_split(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_data(cfg)
    return train_test(
        df, test_size=cfg["split"]["test_size"], seed=cfg["seed"], stratify=cfg["split"]["stratify"]
    )


def _default(o: Any):
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default) + "\n")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def results_path(cfg: dict, name: str) -> Path:
    p = resolve_path(cfg, "results") / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def figure_path(cfg: dict, name: str) -> Path:
    return resolve_path(cfg, "figures") / name


def artifact_path(cfg: dict, name: str) -> Path:
    p = resolve_path(cfg, "artifacts") / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def model_path(cfg: dict, name: str) -> Path:
    p = resolve_path(cfg, "models") / f"{name}.joblib"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_model(model: Any, cfg: dict, name: str) -> Path:
    path = model_path(cfg, name)
    joblib.dump(model, path)
    return path


def load_model(cfg: dict, name: str) -> Any:
    return joblib.load(model_path(cfg, name))


def selected_model_name(cfg: dict) -> str:
    return load_json(results_path(cfg, "selected_model.json"))["selected_model"]


def best_params(cfg: dict) -> dict[str, dict]:
    return load_json(results_path(cfg, "best_params.json"))


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Small dependency-free Markdown table renderer (avoids requiring ``tabulate``)."""

    def fmt(v):
        if isinstance(v, float | np.floating):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows]) + "\n"
