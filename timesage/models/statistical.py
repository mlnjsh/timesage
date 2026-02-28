"""Statistical forecasting models: ARIMA, ETS, Theta."""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple


class _BaseForecaster:
    """Base class for all forecasters."""

    def __init__(self):
        self._model = None
        self._fitted = None
        self._series = None

    def fit(self, series: pd.Series, full_df: Optional[pd.DataFrame] = None):
        raise NotImplementedError

    def predict(self, horizon: int) -> pd.Series:
        raise NotImplementedError

    def confidence_intervals(self, horizon: int, confidence: float = 0.95):
        return None

    def feature_importance(self) -> Optional[Dict[str, float]]:
        return None

    def residuals(self) -> Optional[pd.Series]:
        if self._fitted is not None and hasattr(self._fitted, "resid"):
            return self._fitted.resid
        return None


class ARIMAForecaster(_BaseForecaster):
    """Auto ARIMA forecaster using statsmodels."""

    def fit(self, series: pd.Series, full_df: Optional[pd.DataFrame] = None):
        from statsmodels.tsa.arima.model import ARIMA
        self._series = series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Try common orders
            best_aic = float("inf")
            best_order = (1, 1, 1)
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                        except Exception:
                            continue
            self._model = ARIMA(series, order=best_order)
            self._fitted = self._model.fit()

    def predict(self, horizon: int) -> pd.Series:
        forecast = self._fitted.forecast(steps=horizon)
        return forecast

    def confidence_intervals(self, horizon: int, confidence: float = 0.95):
        pred = self._fitted.get_forecast(steps=horizon)
        ci = pred.conf_int(alpha=1 - confidence)
        return ci.iloc[:, 0], ci.iloc[:, 1]


class ETSForecaster(_BaseForecaster):
    """Exponential Smoothing (ETS) forecaster."""

    def fit(self, series: pd.Series, full_df: Optional[pd.DataFrame] = None):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self._series = series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self._model = ExponentialSmoothing(
                    series, trend="add", seasonal=None, damped_trend=True,
                )
                self._fitted = self._model.fit(optimized=True)
            except Exception:
                self._model = ExponentialSmoothing(series, trend="add")
                self._fitted = self._model.fit(optimized=True)

    def predict(self, horizon: int) -> pd.Series:
        return self._fitted.forecast(steps=horizon)

    def confidence_intervals(self, horizon: int, confidence: float = 0.95):
        # ETS does not natively provide CIs, use residual-based estimate
        if self._fitted is None:
            return None
        resid_std = np.std(self._fitted.resid.dropna())
        from scipy import stats as sp_stats
        z = sp_stats.norm.ppf((1 + confidence) / 2)
        forecast = self.predict(horizon)
        lower = forecast - z * resid_std
        upper = forecast + z * resid_std
        return lower, upper


class ThetaForecaster(_BaseForecaster):
    """Theta method forecaster."""

    def fit(self, series: pd.Series, full_df: Optional[pd.DataFrame] = None):
        from statsmodels.tsa.forecasting.theta import ThetaModel
        self._series = series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = ThetaModel(series)
            self._fitted = self._model.fit()

    def predict(self, horizon: int) -> pd.Series:
        return self._fitted.forecast(steps=horizon)

    def confidence_intervals(self, horizon: int, confidence: float = 0.95):
        pred = self._fitted.forecast(steps=horizon)
        resid_std = np.std(self._fitted.resid.dropna()) if hasattr(self._fitted, "resid") else 0
        from scipy import stats as sp_stats
        z = sp_stats.norm.ppf((1 + confidence) / 2)
        lower = pred - z * resid_std
        upper = pred + z * resid_std
        return lower, upper
