"""AutoForecaster -- automatic model selection."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple


class AutoForecaster:
    """Automatically selects the best forecasting model.

    Tries multiple models and picks the one with the lowest MAPE
    on a holdout validation set.
    """

    def __init__(self):
        self._best_model = None
        self._fitted_series = None
        self._residuals = None

    def fit(self, series: pd.Series, full_df: Optional[pd.DataFrame] = None):
        """Fit by trying multiple models and selecting the best one."""
        from timesage.models.statistical import ARIMAForecaster, ETSForecaster, ThetaForecaster

        candidates = [
            ("ARIMA", ARIMAForecaster),
            ("ETS", ETSForecaster),
            ("Theta", ThetaForecaster),
        ]

        # Try ML models if enough data
        if len(series) > 50:
            from timesage.models.ml import RandomForestForecaster
            candidates.append(("RandomForest", RandomForestForecaster))

        # Validation split
        val_size = max(int(len(series) * 0.15), 5)
        train = series[:-val_size]
        val = series[-val_size:]

        best_score = float("inf")
        best_model = None

        for name, ModelClass in candidates:
            try:
                model = ModelClass()
                model.fit(train, full_df)
                pred = model.predict(val_size)
                pred.index = val.index

                # Score: MAPE
                mask = val.values != 0
                if mask.any():
                    mape = 100 * np.mean(np.abs((val.values[mask] - pred.values[mask]) / val.values[mask]))
                else:
                    mape = float("inf")

                if mape < best_score:
                    best_score = mape
                    best_model = ModelClass
            except Exception:
                continue

        if best_model is None:
            from timesage.models.statistical import ETSForecaster
            best_model = ETSForecaster

        # Refit best model on full series
        self._best_model = best_model()
        self._best_model.fit(series, full_df)
        self._fitted_series = series

    def predict(self, horizon: int) -> pd.Series:
        """Generate forecasts using the best model."""
        return self._best_model.predict(horizon)

    def confidence_intervals(self, horizon: int, confidence: float = 0.95):
        """Return confidence intervals from the best model."""
        return self._best_model.confidence_intervals(horizon, confidence)

    def feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importance if available."""
        return self._best_model.feature_importance()

    def residuals(self) -> Optional[pd.Series]:
        """Return residuals from the best model."""
        return self._best_model.residuals()

    def model_summary(self) -> Dict:
        """Return model summary from the best model."""
        if self._best_model is not None and hasattr(self._best_model, "model_summary"):
            return self._best_model.model_summary()
        return {"model_type": "Auto", "raw_summary": None}
