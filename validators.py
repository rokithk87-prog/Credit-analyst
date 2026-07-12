"""
utils/validators.py
────────────────────
Input validation utilities.  Keeps business-logic modules clean by
centralising all guard clauses here.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd

from config.settings import ALLOWED_EXTENSIONS


# ── File Validation ───────────────────────────────────────────────────────────

def validate_file_extension(filename: str) -> bool:
    """Return True if the file's extension is in the allowed set."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def validate_file_size(file_bytes: bytes, max_mb: int = 50) -> bool:
    """Return True if file is under *max_mb* megabytes."""
    return len(file_bytes) <= max_mb * 1024 * 1024


# ── DataFrame Validation ──────────────────────────────────────────────────────

def validate_dataframe_not_empty(df: pd.DataFrame, context: str = "") -> None:
    """Raise ValueError if DataFrame is empty."""
    if df is None or df.empty:
        label = f" ({context})" if context else ""
        raise ValueError(f"Extracted DataFrame is empty{label}. "
                         "Check that the uploaded file contains financial tables.")


def validate_numeric_columns(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame has at least one numeric column."""
    return not df.select_dtypes(include="number").empty


# ── Financial Value Validation ────────────────────────────────────────────────

def is_valid_financial_value(value: Any) -> bool:
    """Return True if *value* is a finite, non-null number."""
    if value is None:
        return False
    try:
        f = float(value)
        import math
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Division that returns *fallback* instead of raising on zero denominator."""
    if denominator == 0 or not is_valid_financial_value(denominator):
        return fallback
    return numerator / denominator
