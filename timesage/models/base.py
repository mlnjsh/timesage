"""Base forecaster class — unified interface for all models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for all TimeSage forecasters.

    All models implement the same interface:
        fit(series) -> self
        predict(horizon) -> pd.Series
        confidence_intervals(horizon, alpha) -> dict
        residuals() -> pd.Series
        feature_importance() -> pd.Series or None
    """

    name: str = "BaseForecaster"

    def __init__(self, **kwargs):
        self._fitted = False
        self._train_series: Optional[pd.Series] = None
        self._residuals: Optional[pd.Series] = None
        self._model: Any = None
        self._params = kwargs

    @abstractmethod
    def fit(self, series: pd.Series) -> "BaseForecaster":
        """Fit the model to a time series."""
        ...

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        """Generate forecasts for `horizon` steps ahead."""
        ...

    def confidence_intervals(
        self, horizon: int, alpha: float = 0.05
    ) -> Optional[dict[str, pd.Series]]:
        """Return prediction intervals. Override in subclasses for model-specific CIs."""
        forecast = self.predict(horizon)
        if self._residuals is not None:
            std = self._residuals.std()
        elif self._train_series is not None:
            std = self._train_series.std() * 0.1
        else:
            return None

        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        steps = np.arange(1, horizon + 1)
        widths = z * std * np.sqrt(steps / steps.mean())

        return {
            "lower": pd.Series(forecast.values - widths, index=forecast.index),
            "upper": pd.Series(forecast.values + widths, index=forecast.index),
        }

    def residuals(self) -> Optional[pd.Series]:
        """Return in-sample residuals."""
        return self._residuals

    def feature_importance(self) -> Optional[pd.Series]:
        """Return feature importance scores. None for non-ML models."""
        return None

    def get_params(self) -> dict:
        """Return model parameters."""
        return self._params

    def _generate_future_index(self, horizon: int) -> pd.DatetimeIndex:
        """Generate future datetime index based on training data frequency."""
        if self._train_series is None:
            return pd.RangeIndex(horizon)

        last_date = self._train_series.index[-1]
        freq = self._train_series.index.freq
        if freq is None:
            freq = pd.infer_freq(self._train_series.index)
            if freq is None:
                diffs = self._train_series.index.to_series().diff().dropna()
                median_diff = diffs.median()
                return pd.date_range(start=last_date + median_diff, periods=horizon, freq=median_diff)

        return pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
