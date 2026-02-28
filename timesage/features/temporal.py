"""Temporal feature engineering from datetime index."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_temporal_features(
    df: pd.DataFrame,
    cyclical: bool = True,
    holidays: bool = False,
) -> pd.DataFrame:
    """Extract temporal features from the DatetimeIndex.

    Args:
        df: DataFrame with DatetimeIndex.
        cyclical: If True, encode periodic features as sin/cos pairs.
        holidays: If True, add US holiday indicators.

    Returns:
        DataFrame with temporal features appended.
    """
    result = df.copy()
    idx = result.index

    result["hour"] = idx.hour
    result["day_of_week"] = idx.dayofweek
    result["day_of_month"] = idx.day
    result["day_of_year"] = idx.dayofyear
    result["week_of_year"] = idx.isocalendar().week.values.astype(int)
    result["month"] = idx.month
    result["quarter"] = idx.quarter
    result["year"] = idx.year
    result["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    result["is_month_start"] = idx.is_month_start.astype(int)
    result["is_month_end"] = idx.is_month_end.astype(int)

    if cyclical:
        for col, period in [
            ("hour", 24), ("day_of_week", 7), ("day_of_month", 31),
            ("month", 12), ("day_of_year", 365),
        ]:
            if col in result.columns:
                result[f"{col}_sin"] = np.sin(2 * np.pi * result[col] / period)
                result[f"{col}_cos"] = np.cos(2 * np.pi * result[col] / period)

    if holidays:
        try:
            from pandas.tseries.holiday import USFederalHolidayCalendar
            cal = USFederalHolidayCalendar()
            holidays_range = cal.holidays(start=idx.min(), end=idx.max())
            result["is_holiday"] = idx.isin(holidays_range).astype(int)
        except Exception:
            pass

    return result
