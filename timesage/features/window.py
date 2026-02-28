"""Window (rolling/expanding) feature engineering."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def create_window_features(
    df: pd.DataFrame,
    target: str,
    windows: Optional[list[int]] = None,
    functions: Optional[list[str]] = None,
    expanding: bool = False,
    ewm_spans: Optional[list[int]] = None,
    drop_na: bool = False,
) -> pd.DataFrame:
    """Create rolling/expanding window features.

    Args:
        df: Input DataFrame with DatetimeIndex.
        target: Target column name.
        windows: Rolling window sizes (e.g., [3, 7, 14, 30]).
        functions: Aggregation functions ('mean', 'std', 'min', 'max', 'median', 'sum').
        expanding: If True, create expanding window features instead of rolling.
        ewm_spans: Spans for exponentially weighted features.
        drop_na: Whether to drop rows with NaN.

    Returns:
        DataFrame with window features appended.
    """
    if windows is None:
        windows = [3, 7, 14, 30]

    if functions is None:
        functions = ["mean", "std", "min", "max"]

    result = df.copy()
    col = result[target]

    if not expanding:
        for w in windows:
            roller = col.shift(1).rolling(window=w, min_periods=1)
            for func_name in functions:
                result[f"{target}_roll_{func_name}_{w}"] = getattr(roller, func_name)()
    else:
        expander = col.shift(1).expanding(min_periods=1)
        for func_name in functions:
            result[f"{target}_expand_{func_name}"] = getattr(expander, func_name)()

    if ewm_spans is not None:
        for span in ewm_spans:
            ewm = col.shift(1).ewm(span=span, min_periods=1)
            result[f"{target}_ewm_mean_{span}"] = ewm.mean()
            result[f"{target}_ewm_std_{span}"] = ewm.std()

    if drop_na:
        result = result.dropna()

    return result
