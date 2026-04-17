"""
utils/helpers.py
----------------
Shared utility functions used across the entire pipeline.
"""

import yaml
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np


# ── Logger setup ────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Config loader ────────────────────────────────────────────────────────────

_CONFIG_CACHE: Optional[Dict] = None

def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load and cache the YAML config."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        with open(path, "r") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


# ── Column normalisation ─────────────────────────────────────────────────────

PARTY_ALIASES: Dict[str, str] = {
    # Tamil Nadu
    "All India Anna Dravida Munnetra Kazhagam": "AIADMK",
    "Dravida Munnetra Kazhagam": "DMK",
    "Indian National Congress": "INC",
    "Bharatiya Janata Party": "BJP",
    "Viduthalai Chiruthaigal Katchi": "VCK",
    "Pattali Makkal Katchi": "PMK",
    "Marumalarchi Dravida Munnetra Kazhagam": "MDMK",
    "Communist Party of India  (Marxist)": "CPI(M)",
    "Communist Party of India": "CPI",
    "Indian Union Muslim League": "IUML",
    "Naam Tamilar Katchi": "NTK",
    "Tamilaga Vettri Kazhagam": "TVK",
    # West Bengal
    "All India Trinamool Congress": "TMC",
    "Indian National Congress": "INC",
    # Kerala / generic
    "Communist Party of India  (Marxist)": "CPI(M)",
    "Kerala Congress": "KC",
    # Independent
    "Independent": "IND",
    "NOTA": "NOTA",
}

def normalise_party(name: str) -> str:
    """Map verbose party names to short canonical aliases."""
    if pd.isna(name):
        return "UNKNOWN"
    name = str(name).strip()
    return PARTY_ALIASES.get(name, name)


def normalise_constituency(name: str) -> str:
    """Lowercase + strip for consistent constituency matching."""
    if pd.isna(name):
        return ""
    return str(name).strip().lower().replace("  ", " ")


# ── Vote share helpers ───────────────────────────────────────────────────────

def compute_vote_share(votes: pd.Series, total_valid: pd.Series) -> pd.Series:
    """Safe vote share computation, returns 0.0 on division by zero."""
    return np.where(total_valid > 0, votes / total_valid * 100, 0.0)


def compute_margin_pct(
    first_votes: float, second_votes: float, total_valid: float
) -> float:
    """Margin of victory as % of total valid votes."""
    if total_valid <= 0:
        return 0.0
    return (first_votes - second_votes) / total_valid * 100


# ── DataFrame helpers ────────────────────────────────────────────────────────

def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: list,
    how: str = "left",
    label: str = "",
) -> pd.DataFrame:
    """Merge with row-count logging to catch data leaks early."""
    logger = get_logger("safe_merge")
    before = len(left)
    result = pd.merge(left, right, on=on, how=how)
    after = len(result)
    if before != after:
        logger.warning(
            "%s merge: %d rows → %d rows (Δ %+d)", label, before, after, after - before
        )
    else:
        logger.debug("%s merge: %d rows — stable", label, after)
    return result


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ── Validation ───────────────────────────────────────────────────────────────

def validate_required_columns(df: pd.DataFrame, required: list, label: str) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Missing columns: {missing}")


def report_nulls(df: pd.DataFrame, label: str = "") -> None:
    """Log columns with null values — useful after each pipeline step."""
    logger = get_logger("null_report")
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0]
    if null_cols.empty:
        logger.info("%s — No nulls found ✓", label)
    else:
        logger.warning("%s — Nulls detected:\n%s", label, null_cols.to_string())
