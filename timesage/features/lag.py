"""Lag feature engineering for time series."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def create_lag_features(
    df: pd.DataFrame,
    target: str,
    lags: Optional[list[int]] = None,
    columns: Optional[list[str]] = None,
    drop_na: bool = False,
) -> pd.DataFrame:
    """Create lag features for specified columns.

    Args:
        df: Input DataFrame with DatetimeIndex.
        target: Target column name.
        lags: List of lag values (e.g., [1, 2, 3, 7, 14]).
        columns: Columns to create lags for. Defaults to target only.
        drop_na: Whether to drop rows with NaN from lagging.

    Returns:
        DataFrame with lag features appended.
    """
    if lags is None:
        lags = [1, 2, 3, 7, 14]

    if columns is None:
        columns = [target]

    result = df.copy()

    for col in columns:
        if col not in result.columns:
            continue
        for lag in lags:
            result[f"{col}_lag_{lag}"] = result[col].shift(lag)

    if drop_na:
        result = result.dropna()

    return result
