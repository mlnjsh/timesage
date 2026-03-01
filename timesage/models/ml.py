"""Machine Learning forecasting models: RF, XGBoost, LightGBM."""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict


class _BaseMLForecaster:
    """Base class for ML-based forecasters."""

    def __init__(self):
        self._model = None
        self._features = None
        self._target_name = None
        self._last_row = None
        self._series = None
        self._full_df = None
        self._feature_names = None
        self._importance = None
        self._residuals = None

    def _create_features(self, series: pd.Series) -> pd.DataFrame:
        """Create lag and rolling features for ML models."""
        df = pd.DataFrame({"value": series})

        # Lag features
        for lag in [1, 2, 3, 7, 14]:
            df["lag_%d" % lag] = df["value"].shift(lag)

        # Rolling features
        for w in [7, 14]:
            df["roll_mean_%d" % w] = df["value"].rolling(w).mean()
            df["roll_std_%d" % w] = df["value"].rolling(w).std()

        # Temporal
        if isinstance(df.index, pd.DatetimeIndex):
            df["dayofweek"] = df.index.dayofweek
            df["month"] = df.index.month

        df = df.dropna()
        return df

    def fit(self, series: pd.Series, full_df: Optional[pd.DataFrame] = None):
        """Fit the ML model on engineered features."""
        self._series = series
        self._full_df = full_df

        df = self._create_features(series)
        X = df.drop(columns=["value"])
        y = df["value"]

        self._feature_names = list(X.columns)
        self._last_row = df.iloc[-1:]
        self._fit_model(X, y)

        # Compute residuals
        preds = self._model.predict(X)
        self._residuals = pd.Series(y.values - preds, index=y.index)

        # Feature importance
        if hasattr(self._model, "feature_importances_"):
            total = sum(self._model.feature_importances_)
            if total > 0:
                self._importance = {
                    name: float(imp / total)
                    for name, imp in zip(self._feature_names, self._model.feature_importances_)
                }

    def _fit_model(self, X, y):
        raise NotImplementedError

    def predict(self, horizon: int) -> pd.Series:
        """Generate recursive multi-step forecasts."""
        series = self._series.copy()
        predictions = []

        freq = pd.infer_freq(series.index) or "D"
        last_date = series.index[-1]

        for step in range(horizon):
            df = self._create_features(series)
            X = df.drop(columns=["value"]).iloc[-1:]
            pred = self._model.predict(X)[0]
            predictions.append(pred)

            next_date = last_date + pd.tseries.frequencies.to_offset(freq) * (step + 1)
            new_point = pd.Series([pred], index=[next_date], name="value")
            series = pd.concat([series, new_point])

        future_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )
        return pd.Series(predictions, index=future_dates)

    def confidence_intervals(self, horizon: int, confidence: float = 0.95):
        """Estimate confidence intervals from residuals."""
        if self._residuals is None:
            return None
        from scipy import stats as sp_stats
        resid_std = np.std(self._residuals.dropna())
        z = sp_stats.norm.ppf((1 + confidence) / 2)
        forecast = self.predict(horizon)
        lower = forecast - z * resid_std
        upper = forecast + z * resid_std
        return lower, upper

    def feature_importance(self) -> Optional[Dict[str, float]]:
        return self._importance

    def residuals(self) -> Optional[pd.Series]:
        return self._residuals

    def model_summary(self) -> Dict:
        """Return structured summary for ML model."""
        from timesage.models.statistical import _compute_residual_diagnostics

        info = {
            "model_type": self.__class__.__name__.replace("Forecaster", ""),
            "nobs": int(len(self._series)) if self._series is not None else 0,
            "n_features": len(self._feature_names) if self._feature_names else 0,
        }

        # Hyperparameters
        if self._model is not None and hasattr(self._model, "get_params"):
            info["hyperparameters"] = {
                k: v for k, v in self._model.get_params().items()
                if v is not None and k not in ("random_state", "verbose", "verbosity", "n_jobs")
            }

        # Feature importance as coefficients table (sorted by importance)
        coefficients = []
        if self._importance:
            sorted_imp = sorted(self._importance.items(), key=lambda x: x[1], reverse=True)
            for name, imp in sorted_imp:
                coefficients.append({
                    "name": name,
                    "coef": float(imp),
                    "std_err": None, "z": None, "p_value": None,
                    "ci_lower": None, "ci_upper": None,
                })
        info["coefficients"] = coefficients

        # Residual diagnostics
        info["diagnostics"] = _compute_residual_diagnostics(self._residuals)
        info["raw_summary"] = None
        return info


class RandomForestForecaster(_BaseMLForecaster):
    """Random Forest time series forecaster."""

    def _fit_model(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        self._model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self._model.fit(X, y)


class XGBoostForecaster(_BaseMLForecaster):
    """XGBoost time series forecaster."""

    def _fit_model(self, X, y):
        import xgboost as xgb
        self._model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0,
        )
        self._model.fit(X, y)


class LightGBMForecaster(_BaseMLForecaster):
    """LightGBM time series forecaster."""

    def _fit_model(self, X, y):
        import lightgbm as lgb
        self._model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1,
        )
        self._model.fit(X, y)
