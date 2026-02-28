"""Utility functions for TimeSage."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def infer_frequency(index: pd.DatetimeIndex) -> str:
    """Infer the frequency of a DatetimeIndex, defaulting to D."""
    freq = pd.infer_freq(index)
    return freq or "D"


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Detect outliers using the IQR method.

    Parameters
    ----------
    series : pd.Series
        Input data.
    factor : float
        IQR multiplier for outlier bounds (default 1.5).

    Returns
    -------
    pd.Series
        Boolean mask where True indicates an outlier.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - factor * iqr) | (series > q3 + factor * iqr)


def safe_import(module_name: str):
    """Safely import a module, returning None if unavailable."""
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        return None


def format_number(n: float, decimals: int = 2) -> str:
    """Format a number with K/M suffixes for readability."""
    if abs(n) >= 1e6:
        return "%.*fM" % (decimals, n / 1e6)
    elif abs(n) >= 1e3:
        return "%.*fK" % (decimals, n / 1e3)
    return "%.*f" % (decimals, n)


def train_test_split_ts(
    series: pd.Series, test_size: float = 0.2
) -> Tuple[pd.Series, pd.Series]:
    """Split a time series into train and test sets preserving temporal order.

    Parameters
    ----------
    series : pd.Series
        Time series to split.
    test_size : float
        Fraction of data for the test set.

    Returns
    -------
    tuple of pd.Series
        (train, test) series.
    """
    split = int(len(series) * (1 - test_size))
    return series[:split], series[split:]
