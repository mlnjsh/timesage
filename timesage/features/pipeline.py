"""Feature engineering pipeline for time series ML models."""

from typing import Optional, List

import numpy as np
import pandas as pd


class FeaturePipeline:
    """Create ML-ready features from time series data.

    Parameters
    ----------
    target : str
        Name of the target column.
    lags : list of int, optional
        Lag values. Default: [1, 2, 3, 7, 14, 28].
    windows : list of int, optional
        Rolling window sizes. Default: [7, 14, 30].
    temporal : bool
        Include calendar/temporal features.
    """

    def __init__(
        self,
        target: str,
        lags: Optional[List[int]] = None,
        windows: Optional[List[int]] = None,
        temporal: bool = True,
    ):
        self.target = target
        self.lags = lags or [1, 2, 3, 7, 14, 28]
        self.windows = windows or [7, 14, 30]
        self.temporal = temporal

    def transform(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """Transform a DataFrame by adding engineered features.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with DatetimeIndex.
        drop_na : bool
            Drop rows with NaN from lagging/rolling.

        Returns
        -------
        pd.DataFrame
            DataFrame with new feature columns.
        """
        result = df.copy()
        series = result[self.target]

        # Lag features
        for lag in self.lags:
            result["lag_%d" % lag] = series.shift(lag)

        # Rolling window features
        for window in self.windows:
            result["roll_mean_%d" % window] = series.rolling(window=window).mean()
            result["roll_std_%d" % window] = series.rolling(window=window).std()
            result["roll_min_%d" % window] = series.rolling(window=window).min()
            result["roll_max_%d" % window] = series.rolling(window=window).max()

        # Exponential weighted features
        for span in [7, 14, 30]:
            result["ewm_%d" % span] = series.ewm(span=span).mean()

        # Diff features
        result["diff_1"] = series.diff(1)
        result["diff_7"] = series.diff(7)
        result["pct_change_1"] = series.pct_change(1)
        result["pct_change_7"] = series.pct_change(7)

        # Temporal features
        if self.temporal and isinstance(result.index, pd.DatetimeIndex):
            result["day_of_week"] = result.index.dayofweek
            result["day_of_month"] = result.index.day
            result["month"] = result.index.month
            result["quarter"] = result.index.quarter
            result["week_of_year"] = result.index.isocalendar().week.astype(int)
            result["is_weekend"] = (result.index.dayofweek >= 5).astype(int)
            result["is_month_start"] = result.index.is_month_start.astype(int)
            result["is_month_end"] = result.index.is_month_end.astype(int)

        if drop_na:
            result = result.dropna()

        return result
