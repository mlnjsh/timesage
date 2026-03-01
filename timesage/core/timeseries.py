"""Core TimeSeries class -- the heart of TimeSage."""

import warnings
from typing import Optional, Union, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)


class TimeSeries:
    """A smart time series container with built-in analysis and forecasting.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Time series data.
    target : str, optional
        Column name of the target variable (for DataFrames).
    time : str, optional
        Column name of the time/date column. If None, uses the index.
    freq : str, optional
        Frequency of the data. Auto-detected if None.
    name : str, optional
        Name for the time series.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, pd.Series],
        target: Optional[str] = None,
        time: Optional[str] = None,
        freq: Optional[str] = None,
        name: Optional[str] = None,
    ):
        # Handle Series input
        if isinstance(data, pd.Series):
            self._df = data.to_frame(name=data.name or "value")
            self._target = data.name or "value"
        elif isinstance(data, pd.DataFrame):
            if time is not None and time in data.columns:
                data = data.set_index(time)
            self._df = data.copy()
            self._target = target or data.columns[0]
        else:
            raise TypeError("data must be a pandas DataFrame or Series")

        # Ensure datetime index
        if not isinstance(self._df.index, pd.DatetimeIndex):
            try:
                self._df.index = pd.to_datetime(self._df.index)
            except Exception:
                raise ValueError("Could not convert index to DatetimeIndex")

        self._df = self._df.sort_index()
        self._name = name or self._target
        self._freq = freq or pd.infer_freq(self._df.index) or "D"

    @property
    def values(self) -> pd.Series:
        """Return the target column as a Series."""
        return self._df[self._target]

    @property
    def target(self) -> str:
        """Return the name of the target column."""
        return self._target

    @property
    def frequency(self) -> str:
        """Return a human-readable frequency string."""
        freq_map = {
            "D": "daily", "W": "weekly", "M": "monthly",
            "MS": "monthly", "Q": "quarterly", "QS": "quarterly",
            "Y": "yearly", "YS": "yearly", "H": "hourly",
            "T": "minutely", "min": "minutely", "B": "business_daily",
        }
        return freq_map.get(self._freq, self._freq)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        n = self._name
        l = len(self)
        f = self.frequency
        d0 = self._df.index[0].date()
        d1 = self._df.index[-1].date()
        return "TimeSeries(name=%r, length=%d, freq=%r, range=[%s -> %s])" % (n, l, f, d0, d1)

    def describe(self) -> pd.Series:
        """Enhanced descriptive statistics beyond pandas .describe()."""
        s = self.values
        desc = s.describe()
        desc["skewness"] = s.skew()
        desc["kurtosis"] = s.kurtosis()
        desc["missing"] = s.isna().sum()
        desc["missing_pct"] = 100 * s.isna().mean()
        desc["cv"] = s.std() / s.mean() if s.mean() != 0 else np.nan
        return desc

    def test_stationarity(self, verbose: bool = True) -> Dict[str, Any]:
        """Run ADF and KPSS stationarity tests with plain-English interpretation.

        Parameters
        ----------
        verbose : bool
            If True, print a pretty table with results.

        Returns
        -------
        dict
            Dictionary with adf, kpss, and conclusion keys.
        """
        from statsmodels.tsa.stattools import adfuller, kpss

        s = self.values.dropna()

        # ADF test (H0: unit root exists, i.e. non-stationary)
        adf_result = adfuller(s, autolag="AIC")
        adf_stat, adf_pval = adf_result[0], adf_result[1]
        adf_stationary = adf_pval < 0.05

        # KPSS test (H0: stationary)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(s, regression="c", nlags="auto")
        kpss_stat, kpss_pval = kpss_result[0], kpss_result[1]
        kpss_stationary = kpss_pval >= 0.05

        # Conclusion combining both tests
        if adf_stationary and kpss_stationary:
            conclusion = "STATIONARY -- Both tests agree the series is stationary."
        elif not adf_stationary and not kpss_stationary:
            conclusion = "NON-STATIONARY -- Both tests agree the series has a unit root."
        elif adf_stationary and not kpss_stationary:
            conclusion = "TREND-STATIONARY -- Stationary around a deterministic trend."
        else:
            conclusion = "DIFFERENCE-STATIONARY -- Has a unit root but no deterministic trend."

        results = {
            "adf": {"statistic": adf_stat, "p_value": adf_pval, "stationary": adf_stationary},
            "kpss": {"statistic": kpss_stat, "p_value": kpss_pval, "stationary": kpss_stationary},
            "conclusion": conclusion,
        }

        if verbose:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Stationarity Tests", show_header=True, header_style="bold cyan")
            table.add_column("Test", style="bold")
            table.add_column("Statistic", justify="right")
            table.add_column("P-value", justify="right")
            table.add_column("Stationary?", justify="center")
            table.add_row(
                "ADF (H0: non-stat)",
                "%.4f" % adf_stat, "%.4f" % adf_pval,
                "[green]Yes[/green]" if adf_stationary else "[red]No[/red]",
            )
            table.add_row(
                "KPSS (H0: stationary)",
                "%.4f" % kpss_stat, "%.4f" % kpss_pval,
                "[green]Yes[/green]" if kpss_stationary else "[red]No[/red]",
            )
            console.print(table)
            console.print("\n  [bold]%s[/bold]\n" % conclusion)

        return results

    def detect_seasonality(self, max_lag: int = 90, verbose: bool = True) -> Dict[str, Any]:
        """Detect seasonality using ACF analysis.

        Parameters
        ----------
        max_lag : int
            Maximum lag to examine for seasonal patterns.
        verbose : bool
            If True, print detection results.

        Returns
        -------
        dict
            Dictionary with seasonal, period, peaks, acf_values.
        """
        from statsmodels.tsa.stattools import acf

        s = self.values.dropna()
        max_lag = min(max_lag, len(s) // 3)
        acf_values = acf(s, nlags=max_lag, fft=True)

        # Find significant peaks in autocorrelation
        peaks = []
        for i in range(2, len(acf_values) - 1):
            if acf_values[i] > acf_values[i - 1] and acf_values[i] > acf_values[i + 1]:
                if acf_values[i] > 1.96 / np.sqrt(len(s)):
                    peaks.append((i, acf_values[i]))

        seasonal = len(peaks) > 0
        period = peaks[0][0] if peaks else None

        results = {
            "seasonal": seasonal,
            "period": period,
            "peaks": peaks[:5],
            "acf_values": acf_values,
        }

        if verbose:
            from rich.console import Console
            console = Console()
            if seasonal:
                console.print("  [green]Seasonality detected![/green] Primary period: [bold]%s[/bold]" % period)
            else:
                console.print("  [yellow]No significant seasonality detected.[/yellow]")

        return results

    def decompose(self, model: str = "additive", period: Optional[int] = None):
        """Decompose the series into trend, seasonal, and residual components."""
        from statsmodels.tsa.seasonal import seasonal_decompose

        s = self.values.dropna()
        if period is None:
            det = self.detect_seasonality(verbose=False)
            period = det.get("period") or 7

        return seasonal_decompose(s, model=model, period=period)

    def eda(self, show_plots: bool = True):
        """Run full automated Exploratory Data Analysis."""
        from timesage.eda.profiler import profile
        return profile(self, show_plots=show_plots)

    def plot(self, show_trend: bool = False, show_outliers: bool = False,
             figsize=(14, 5), **kwargs):
        """Plot the time series with TimeSage styling."""
        from timesage.plot.theme import sage_theme
        from timesage.plot.timeplots import plot_series
        sage_theme()
        return plot_series(
            self.values, show_trend=show_trend, show_outliers=show_outliers,
            title=self._name, figsize=figsize, **kwargs,
        )

    def plot_acf(self, lags: int = 40, figsize=(14, 4)):
        """Plot ACF and PACF side by side."""
        from timesage.plot.theme import sage_theme
        from timesage.plot.diagnostic import plot_acf_pacf
        sage_theme()
        return plot_acf_pacf(self.values, lags=lags, figsize=figsize)

    def interpret_acf(self, lags: int = 40, verbose: bool = True) -> dict:
        """Analyze ACF/PACF and return plain-English interpretation with model suggestions."""
        from timesage.plot.diagnostic import interpret_acf_pacf
        return interpret_acf_pacf(self.values, lags=lags, verbose=verbose)

    def plot_decomposition(self, model: str = "additive",
                           period: Optional[int] = None, figsize=(14, 10)):
        """Plot seasonal decomposition."""
        from timesage.plot.theme import sage_theme
        from timesage.plot.timeplots import plot_components
        sage_theme()
        result = self.decompose(model=model, period=period)
        return plot_components(result, title=self._name, figsize=figsize)

    def create_features(self, lags: Optional[List[int]] = None,
                        windows: Optional[List[int]] = None,
                        temporal: bool = True, drop_na: bool = True) -> pd.DataFrame:
        """Create features for ML models."""
        from timesage.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline(
            target=self._target, lags=lags, windows=windows, temporal=temporal
        )
        return pipeline.transform(self._df, drop_na=drop_na)

    def forecast(self, horizon: int = 30, model: str = "auto",
                 test_size: float = 0.2, confidence: float = 0.95,
                 verbose: bool = True) -> "ForecastResult":
        """Forecast future values.

        Parameters
        ----------
        horizon : int
            Number of steps to forecast.
        model : str
            Model to use: auto, arima, ets, theta, rf, xgboost, lightgbm.
        test_size : float
            Fraction of data for testing (used for metrics).
        confidence : float
            Confidence level for prediction intervals.
        verbose : bool
            Print progress and results.

        Returns
        -------
        ForecastResult
            Object with forecast, metrics, interpretation, and plotting.
        """
        from timesage.core.result import ForecastResult

        s = self.values.dropna()
        split = int(len(s) * (1 - test_size))
        train, test = s[:split], s[split:]

        model_map = {
            "arima": "timesage.models.statistical:ARIMAForecaster",
            "ets": "timesage.models.statistical:ETSForecaster",
            "theta": "timesage.models.statistical:ThetaForecaster",
            "rf": "timesage.models.ml:RandomForestForecaster",
            "xgboost": "timesage.models.ml:XGBoostForecaster",
            "lightgbm": "timesage.models.ml:LightGBMForecaster",
            "auto": "timesage.models.auto:AutoForecaster",
        }

        model_key = model.lower()
        if model_key not in model_map:
            raise ValueError("Unknown model. Choose from: %s" % list(model_map.keys()))

        module_path, class_name = model_map[model_key].rsplit(":", 1)
        import importlib
        mod = importlib.import_module(module_path)
        forecaster_class = getattr(mod, class_name)

        forecaster = forecaster_class()
        forecaster.fit(train, self._df)

        # Predict test period for metrics
        test_pred = forecaster.predict(len(test))
        test_pred.index = test.index

        # Refit on full data and forecast
        forecaster.fit(s, self._df)
        forecast_vals = forecaster.predict(horizon)

        # Confidence intervals
        ci = forecaster.confidence_intervals(horizon, confidence)

        result = ForecastResult(
            forecast=forecast_vals,
            actual=test,
            train=train,
            test_predictions=test_pred,
            confidence_lower=ci[0] if ci else None,
            confidence_upper=ci[1] if ci else None,
            model_name=forecaster.__class__.__name__.replace("Forecaster", ""),
            feature_importance=forecaster.feature_importance(),
            _residuals_raw=forecaster.residuals(),
        )

        if verbose:
            result.interpret()

        return result

    def compare_models(self, test_size: float = 0.2,
                       models: Optional[List[str]] = None,
                       verbose: bool = True) -> pd.DataFrame:
        """Compare multiple forecasting models side by side.

        Parameters
        ----------
        test_size : float
            Fraction of data for testing.
        models : list of str, optional
            Models to compare. Auto-detects available models if None.
        verbose : bool
            Print comparison table.

        Returns
        -------
        pd.DataFrame
            Comparison table sorted by MAPE.
        """
        import time
        from rich.console import Console
        from rich.table import Table

        if models is None:
            models = ["arima", "ets", "theta", "rf"]
            try:
                import xgboost
                models.append("xgboost")
            except ImportError:
                pass
            try:
                import lightgbm
                models.append("lightgbm")
            except ImportError:
                pass

        results = []
        for m in models:
            try:
                t0 = time.time()
                result = self.forecast(horizon=1, model=m, test_size=test_size, verbose=False)
                elapsed = time.time() - t0
                metrics = result.metrics
                results.append({
                    "Model": result.model_name,
                    "MAE": metrics.get("MAE", np.nan),
                    "RMSE": metrics.get("RMSE", np.nan),
                    "MAPE": metrics.get("MAPE", np.nan),
                    "Time (s)": elapsed,
                })
            except Exception as e:
                results.append({
                    "Model": m, "MAE": np.nan, "RMSE": np.nan,
                    "MAPE": np.nan, "Time (s)": 0, "Error": str(e),
                })

        df = pd.DataFrame(results).sort_values("MAPE").reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "Rank"

        if verbose:
            console = Console()
            table = Table(title="Model Comparison", show_header=True, header_style="bold cyan")
            table.add_column("Rank", justify="center", style="bold")
            table.add_column("Model", style="bold yellow")
            table.add_column("MAE", justify="right")
            table.add_column("RMSE", justify="right")
            table.add_column("MAPE", justify="right")
            table.add_column("Time (s)", justify="right")
            for i, row in df.iterrows():
                style = "bold green" if i == 1 else ""
                table.add_row(
                    str(i), row["Model"],
                    "%.4f" % row["MAE"], "%.4f" % row["RMSE"],
                    "%.2f%%" % row["MAPE"], "%.2f" % row["Time (s)"],
                    style=style,
                )
            console.print(table)
            if not df.empty:
                winner = df.iloc[0]
                console.print(
                    "\n  [bold green]Winner: %s (MAPE: %.2f%%)[/bold green]\n" % (
                        winner["Model"], winner["MAPE"]
                    )
                )

        return df
