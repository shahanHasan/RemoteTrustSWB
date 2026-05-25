"""Data validation and feature preparation utilities."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .constants import MODEL_FEATURE_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN, TEXT_COLUMNS


def required_columns(include_target: bool = True) -> list[str]:
    """Return required dataframe columns for training or inference."""
    cols = list(MODEL_FEATURE_COLUMNS)
    if include_target:
        cols.append(TARGET_COLUMN)
    return cols


def validate_required_columns(df: pd.DataFrame, include_target: bool = True) -> None:
    """Raise ValueError if required columns are missing."""
    missing = [col for col in required_columns(include_target=include_target) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def prepare_features(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Prepare model-ready dataframe with combined text and numeric passthrough columns."""
    validate_required_columns(df, include_target=include_target)
    work = df.copy()

    for col in TEXT_COLUMNS:
        work[col] = _clean_text_series(work[col])

    for col in NUMERIC_COLUMNS:
        work[col] = _coerce_numeric_series(work[col])

    work["combined_text"] = (
        work[TEXT_COLUMNS]
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    features = work[["combined_text", *NUMERIC_COLUMNS]].copy()

    if include_target:
        features[TARGET_COLUMN] = _coerce_numeric_series(work[TARGET_COLUMN])

    return features


def posting_to_frame(posting: Mapping[str, Any]) -> pd.DataFrame:
    """Convert one posting dict into the raw model-schema dataframe."""
    row: dict[str, Any] = {}
    for col in MODEL_FEATURE_COLUMNS:
        if col in NUMERIC_COLUMNS:
            row[col] = posting.get(col, 0)
        else:
            row[col] = posting.get(col, "")
    return pd.DataFrame([row])


def prepare_single_posting(posting: Mapping[str, Any]) -> pd.DataFrame:
    """Prepare one posting dict for model inference."""
    raw_df = posting_to_frame(posting)
    return prepare_features(raw_df, include_target=False)
