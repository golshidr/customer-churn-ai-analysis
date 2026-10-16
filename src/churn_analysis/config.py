"""Configuration loading.

The configuration is a plain nested dictionary read from YAML. ``load_config`` merges a
user file over ``configs/default.yaml`` so a small override file (for example the CI
config) only needs the keys it changes.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _find_repo_root() -> Path:
    """Repository root: the source checkout for editable installs, else the working dir."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "configs" / "default.yaml").exists():
        return candidate
    return Path.cwd()


REPO_ROOT = _find_repo_root()
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` with ``override`` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Load the default config, then merge ``path`` and ``overrides`` on top."""
    with open(DEFAULT_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path) as fh:
            cfg = deep_update(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve ``cfg['paths'][key]`` against the repository root (absolute paths kept)."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else REPO_ROOT / p


def ensure_dirs(cfg: dict) -> None:
    for key in ("results", "figures", "models", "artifacts"):
        resolve_path(cfg, key).mkdir(parents=True, exist_ok=True)
