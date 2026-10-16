"""Command-line interface: ``churn <stage>`` or ``python -m churn_analysis <stage>``."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import time

from .config import ensure_dirs, load_config, resolve_path

#: stage name -> (module, function, help text). Stages run in this order for ``all``.
STAGES = {
    "reproduce-original": (
        "benchmark",
        "reproduce_original",
        "re-run the original notebook pipeline",
    ),
    "eda": ("eda", "run_eda", "descriptive churn-rate figure"),
    "benchmark": ("benchmark", "run_benchmark", "tune and compare models with CV + bootstrap CIs"),
    "decide": ("decision", "run_decision", "calibration and cost-sensitive targeting"),
    "simulate": ("campaign", "run_campaign", "Monte Carlo retention-campaign simulation"),
    "survival": ("survival", "run_survival", "Kaplan-Meier and Cox models of time to churn"),
    "explain": ("explain", "run_explain", "SHAP global, interaction and per-customer explanations"),
    "fairness": ("fairness", "run_fairness", "subgroup performance audit"),
    "export": ("serving", "export_bundle", "package the selected model for the API"),
}


def run_stage(stage: str, cfg: dict):
    module, func, _ = STAGES[stage]
    fn = getattr(importlib.import_module(f"churn_analysis.{module}"), func)
    t0 = time.perf_counter()
    out = fn(cfg)
    logging.getLogger("churn_analysis").info(
        "stage %s finished in %.1fs", stage, time.perf_counter() - t0
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="churn", description=__doc__)
    parser.add_argument("--config", default=None, help="YAML file merged over configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="download the Telco dataset (Kaggle, IBM fallback)")
    dl.add_argument("--source", default="auto", choices=["auto", "kaggle", "ibm"])
    for name, (_, _, help_text) in STAGES.items():
        sub.add_parser(name, help=help_text)
    sub.add_parser("all", help="run every analysis stage in order")
    srv = sub.add_parser("serve", help="start the FastAPI scoring service")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    if args.command == "download":
        from .data import download_dataset

        path = download_dataset(resolve_path(cfg, "raw_data"), source=args.source)
        print(f"saved {path}")
    elif args.command == "serve":
        import uvicorn

        os.environ.setdefault(
            "CHURN_MODEL_PATH", str(resolve_path(cfg, "models") / cfg["serving"]["bundle_name"])
        )
        uvicorn.run("churn_analysis.api:app", host=args.host, port=args.port)
    elif args.command == "all":
        for stage in STAGES:
            run_stage(stage, cfg)
    else:
        run_stage(args.command, cfg)


if __name__ == "__main__":
    main()
