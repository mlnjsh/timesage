"""Statistical forecasting models: ARIMA, ETS, Theta."""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from scipy import stats as sp_stats


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

    def model_summary(self) -> Dict:
        """Return a structured dict of model info, coefficients, and diagnostics."""
        return {"model_type": self.__class__.__name__, "raw_summary": None}


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

    def model_summary(self) -> Dict:
        """Extract full ARIMA model summary with coefficients and diagnostics."""
        f = self._fitted
        if f is None:
            return {"model_type": "ARIMA", "error": "Model not fitted"}

        order = self._model.order if hasattr(self._model, "order") else None
        info = {
            "model_type": "ARIMA%s" % str(order) if order else "ARIMA",
            "nobs": int(f.nobs),
        }

        # Information criteria
        for attr in ("aic", "bic", "hqic"):
            if hasattr(f, attr):
                info[attr] = float(getattr(f, attr))
        if hasattr(f, "llf"):
            info["log_likelihood"] = float(f.llf)

        # Sigma2 (residual variance)
        if hasattr(f, "params") and "sigma2" in f.params.index:
            info["sigma2"] = float(f.params["sigma2"])
        elif hasattr(f, "resid"):
            info["sigma2"] = float(f.resid.var())

        # Coefficients table
        coefficients = []
        if hasattr(f, "params"):
            names = list(f.params.index)
            values = f.params.values
            std_errs = f.bse.values if hasattr(f, "bse") else [None] * len(names)
            zvals = f.tvalues.values if hasattr(f, "tvalues") else [None] * len(names)
            pvals = f.pvalues.values if hasattr(f, "pvalues") else [None] * len(names)
            try:
                ci = f.conf_int()
                ci_lo = ci.iloc[:, 0].values
                ci_hi = ci.iloc[:, 1].values
            except Exception:
                ci_lo = [None] * len(names)
                ci_hi = [None] * len(names)

            for i, name in enumerate(names):
                coefficients.append({
                    "name": name,
                    "coef": float(values[i]) if values[i] is not None else None,
                    "std_err": float(std_errs[i]) if std_errs[i] is not None else None,
                    "z": float(zvals[i]) if zvals[i] is not None else None,
                    "p_value": float(pvals[i]) if pvals[i] is not None else None,
                    "ci_lower": float(ci_lo[i]) if ci_lo[i] is not None else None,
                    "ci_upper": float(ci_hi[i]) if ci_hi[i] is not None else None,
                })
        info["coefficients"] = coefficients

        # Diagnostic tests on residuals
        info["diagnostics"] = _compute_residual_diagnostics(f.resid if hasattr(f, "resid") else None)

        # Raw summary string
        try:
            info["raw_summary"] = str(f.summary())
        except Exception:
            info["raw_summary"] = None

        return info


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
        z = sp_stats.norm.ppf((1 + confidence) / 2)
        forecast = self.predict(horizon)
        lower = forecast - z * resid_std
        upper = forecast + z * resid_std
        return lower, upper

    def model_summary(self) -> Dict:
        """Extract ETS model summary with smoothing parameters."""
        f = self._fitted
        if f is None:
            return {"model_type": "ETS", "error": "Model not fitted"}

        info = {
            "model_type": "ExponentialSmoothing",
            "nobs": int(len(self._series)) if self._series is not None else 0,
        }

        # Information criteria
        for attr in ("aic", "bic", "aicc"):
            if hasattr(f, attr):
                try:
                    info[attr] = float(getattr(f, attr))
                except Exception:
                    pass
        if hasattr(f, "sse"):
            info["sse"] = float(f.sse)

        # Smoothing parameters as coefficients
        coefficients = []
        param_names = {
            "smoothing_level": "alpha (level)",
            "smoothing_trend": "beta (trend)",
            "smoothing_seasonal": "gamma (seasonal)",
            "damping_trend": "phi (damping)",
            "initial_level": "l0 (initial level)",
            "initial_trend": "b0 (initial trend)",
        }
        for attr, label in param_names.items():
            if hasattr(f, attr) and getattr(f, attr) is not None:
                val = getattr(f, attr)
                if not np.isnan(val):
                    coefficients.append({
                        "name": label,
                        "coef": float(val),
                        "std_err": None, "z": None, "p_value": None,
                        "ci_lower": None, "ci_upper": None,
                    })
        info["coefficients"] = coefficients

        # Diagnostics
        info["diagnostics"] = _compute_residual_diagnostics(f.resid if hasattr(f, "resid") else None)
        info["raw_summary"] = None
        return info


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
        z = sp_stats.norm.ppf((1 + confidence) / 2)
        lower = pred - z * resid_std
        upper = pred + z * resid_std
        return lower, upper

    def model_summary(self) -> Dict:
        """Extract Theta model summary."""
        f = self._fitted
        if f is None:
            return {"model_type": "Theta", "error": "Model not fitted"}

        info = {
            "model_type": "Theta",
            "nobs": int(len(self._series)) if self._series is not None else 0,
        }

        # Theta-specific params
        coefficients = []
        for attr in ("alpha", "b0"):
            if hasattr(f, attr):
                try:
                    coefficients.append({
                        "name": attr,
                        "coef": float(getattr(f, attr)),
                        "std_err": None, "z": None, "p_value": None,
                        "ci_lower": None, "ci_upper": None,
                    })
                except Exception:
                    pass
        info["coefficients"] = coefficients
        info["diagnostics"] = _compute_residual_diagnostics(f.resid if hasattr(f, "resid") else None)
        info["raw_summary"] = None

        try:
            info["raw_summary"] = str(f.summary())
        except Exception:
            pass

        return info


# ── Shared diagnostic helper ────────────────────────────────────────────

def _compute_residual_diagnostics(resid) -> Dict:
    """Run Ljung-Box, Jarque-Bera, heteroskedasticity, skew/kurtosis on residuals."""
    if resid is None:
        return {}

    r = resid.dropna()
    if len(r) < 10:
        return {}

    diag = {}

    # Ljung-Box
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(r, lags=[10], return_df=True)
        diag["ljung_box_stat"] = float(lb["lb_stat"].iloc[0])
        diag["ljung_box_pvalue"] = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        pass

    # Jarque-Bera
    try:
        jb_stat, jb_p = sp_stats.jarque_bera(r)
        diag["jarque_bera_stat"] = float(jb_stat)
        diag["jarque_bera_pvalue"] = float(jb_p)
    except Exception:
        pass

    # Heteroskedasticity (variance ratio)
    try:
        mid = len(r) // 2
        v1, v2 = r.iloc[:mid].var(), r.iloc[mid:].var()
        if min(v1, v2) > 0:
            h_stat = float(max(v1, v2) / min(v1, v2))
        else:
            h_stat = 1.0
        diag["het_stat"] = h_stat
        # F-test p-value
        df1, df2 = mid - 1, len(r) - mid - 1
        if df1 > 0 and df2 > 0:
            diag["het_pvalue"] = float(2 * min(
                sp_stats.f.cdf(h_stat, df1, df2),
                1 - sp_stats.f.cdf(h_stat, df1, df2),
            ))
    except Exception:
        pass

    # Skew and Kurtosis
    try:
        diag["skew"] = float(r.skew())
        diag["kurtosis"] = float(r.kurtosis())
    except Exception:
        pass

    return diag
