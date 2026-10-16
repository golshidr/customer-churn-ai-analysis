"""Descriptive churn rates (no modelling)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import plotting as P
from .runtime import figure_path, load_data, results_path, save_json

TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]


def churn_rates(df: pd.DataFrame, column: str) -> pd.DataFrame:
    g = df.groupby(column, observed=True)["Churn"].agg(["mean", "size"])
    return g.rename(columns={"mean": "churn_rate", "size": "n"}).reset_index()


def run_eda(cfg: dict) -> dict:
    df = load_data(cfg)
    df = df.assign(tenure_months=pd.cut(df.tenure, TENURE_BINS, labels=TENURE_LABELS))
    panels = [
        ("Contract", "Contract"),
        ("InternetService", "Internet service"),
        ("PaymentMethod", "Payment method"),
        ("tenure_months", "Tenure (months)"),
    ]
    summary = {"n_customers": int(len(df)), "churn_rate": float(df.Churn.mean())}
    P.apply_style()
    fig, axes = P.plt.subplots(2, 2, figsize=(11, 6.8))
    for ax, (col, title) in zip(axes.ravel(), panels):
        r = churn_rates(df, col)
        r = r.sort_values("churn_rate") if col != "tenure_months" else r.iloc[::-1]
        y = np.arange(len(r))
        ax.barh(y, r.churn_rate, height=0.55, color=P.BLUE)
        for i, (rate, n) in enumerate(zip(r.churn_rate, r.n)):
            ax.text(
                rate + 0.008, i, f"{rate:.0%}  (n={n:,})", va="center", fontsize=9, color=P.INK_2
            )
        ax.axvline(summary["churn_rate"], color=P.INK_2, lw=1)
        ax.set_yticks(y, r[col].astype(str))
        ax.set_xlim(0, max(0.62, r.churn_rate.max() + 0.17))
        ax.xaxis.set_major_formatter(P.plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        summary[col] = dict(zip(r[col].astype(str), r.churn_rate.round(4)))
    axes[1, 0].set_xlabel("Churn rate")
    axes[1, 1].set_xlabel("Churn rate")
    fig.suptitle(
        f"Observed churn rate by customer segment (vertical line: overall {summary['churn_rate']:.1%})",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
        color=P.INK,
    )
    fig.tight_layout()
    P.savefig(fig, figure_path(cfg, "eda_churn_rates.png"))
    save_json(summary, results_path(cfg, "eda_summary.json"))
    return summary
