"""ForecastResult -- container for forecast output with built-in interpretation."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """Container for forecast results with metrics, interpretation, and plotting.

    Attributes
    ----------
    forecast : pd.Series
        The forecasted values for the future horizon.
    actual : pd.Series, optional
        Actual values from the test set.
    train : pd.Series, optional
        Training data used for fitting.
    test_predictions : pd.Series, optional
        Model predictions on the test set (for computing metrics).
    confidence_lower : pd.Series, optional
        Lower bound of the confidence interval.
    confidence_upper : pd.Series, optional
        Upper bound of the confidence interval.
    model_name : str
        Name of the model used.
    feature_importance : dict, optional
        Feature importance scores (for ML models).
    residuals : pd.Series, optional
        Model residuals from the fit.
    """

    forecast: pd.Series
    actual: Optional[pd.Series] = None
    train: Optional[pd.Series] = None
    test_predictions: Optional[pd.Series] = None
    confidence_lower: Optional[pd.Series] = None
    confidence_upper: Optional[pd.Series] = None
    model_name: str = "Unknown"
    feature_importance: Optional[Dict[str, float]] = None
    _residuals_raw: Optional[pd.Series] = None
    _metrics: Optional[Dict[str, float]] = field(default=None, repr=False)


    @property
    def residuals(self) -> Optional[pd.Series]:
        """Return residuals, computing from actual - test_predictions as fallback."""
        if self._residuals_raw is not None:
            return self._residuals_raw
        if self.actual is not None and self.test_predictions is not None:
            return self.actual - self.test_predictions
        return None
    @property
    def mae(self) -> Optional[float]:
        """Mean Absolute Error."""
        return self.metrics.get('MAE')

    @property
    def rmse(self) -> Optional[float]:
        """Root Mean Squared Error."""
        return self.metrics.get('RMSE')

    @property
    def mape(self) -> Optional[float]:
        """Mean Absolute Percentage Error."""
        return self.metrics.get('MAPE')

    @property
    def r2(self) -> Optional[float]:
        """R-squared score."""
        return self.metrics.get('R2')

    @property
    def mase(self) -> Optional[float]:
        """Mean Absolute Scaled Error."""
        return self.metrics.get('MASE')

    @property
    def metrics(self) -> Dict[str, float]:
        """Compute and cache forecast accuracy metrics."""
        if self._metrics is None:
            self._metrics = self._compute_metrics()
        return self._metrics

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute MAE, RMSE, MSE, MedAE, MAPE, R2, and MASE."""
        if self.actual is None or self.test_predictions is None:
            return {}

        a = self.actual.values
        p = self.test_predictions.values
        min_len = min(len(a), len(p))
        a, p = a[:min_len], p[:min_len]

        errors = a - p
        abs_errors = np.abs(errors)

        metrics = {
            "MAE": np.mean(abs_errors),
            "RMSE": np.sqrt(np.mean(errors ** 2)),
            "MSE": np.mean(errors ** 2),
            "MedAE": np.median(abs_errors),
        }

        # MAPE (avoid division by zero)
        mask = a != 0
        if mask.any():
            metrics["MAPE"] = 100 * np.mean(np.abs(errors[mask] / a[mask]))

        # R-squared
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        metrics["R2"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # MASE (vs naive forecast)
        if self.train is not None and len(self.train) > 1:
            naive_errors = np.abs(np.diff(self.train.values))
            mae_naive = np.mean(naive_errors)
            if mae_naive > 0:
                metrics["MASE"] = metrics["MAE"] / mae_naive

        return metrics

    def _build_interpretation(self) -> Dict[str, str]:
        """Build plain-English interpretations of every metric."""
        m = self.metrics
        interp = {}

        # Accuracy assessment based on MAPE
        mape = m.get("MAPE", None)
        if mape is not None:
            if mape < 5:
                level = "Excellent"
            elif mape < 10:
                level = "Good"
            elif mape < 20:
                level = "Moderate"
            else:
                level = "Poor"
            interp["accuracy"] = (
                "[%s] The model achieves %s accuracy with an "
                "average error of %.1f%%. For every 100 units of actual value, "
                "the prediction is typically off by about %.1f units."
            ) % (level, level.lower(), mape, mape)

        # Error pattern (consistency of errors)
        mae = m.get("MAE", 0)
        rmse = m.get("RMSE", 0)
        if mae > 0:
            ratio = rmse / mae
            if ratio < 1.2:
                pattern = "very CONSISTENT"
            elif ratio < 1.5:
                pattern = "CONSISTENT"
            else:
                pattern = "INCONSISTENT (has outlier errors)"
            interp["error_pattern"] = (
                "The model has %s errors (RMSE/MAE ratio = %.2f)." % (pattern, ratio)
            )

        # Naive benchmark comparison
        mase = m.get("MASE", None)
        if mase is not None:
            pct = (1 - mase) * 100
            if mase < 1:
                interp["benchmark"] = (
                    "The model OUTPERFORMS a naive forecast by %.0f%% "
                    "(MASE = %.2f). It adds significant value beyond simple approaches."
                ) % (pct, mase)
            else:
                interp["benchmark"] = (
                    "The model UNDERPERFORMS a naive forecast "
                    "(MASE = %.2f). "
                    "A simple repeat-last-value strategy would be better."
                ) % mase

        # Feature importance summary
        if self.feature_importance:
            top = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            parts = []
            for k, v in top:
                parts.append("%r (%.1f%%)" % (k, 100 * v))
            interp["drivers"] = "Top predictive features: %s." % ", ".join(parts)

        # Forecast direction
        if len(self.forecast) > 1:
            first = self.forecast.iloc[0]
            last = self.forecast.iloc[-1]
            if first > 0:
                change_pct = (last - first) / first * 100
                direction = "UPWARD" if change_pct > 0 else "DOWNWARD"
                interp["direction"] = (
                    "The forecast shows an %s trend of %+.1f%% "
                    "over the prediction horizon. Peak: %.2f."
                ) % (direction, change_pct, self.forecast.max())

        return interp

    def interpret(self):
        """Print plain-English interpretation of forecast results."""
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        interp = self._build_interpretation()

        sections = []
        labels = {
            "accuracy": "Overall Accuracy",
            "error_pattern": "Error Pattern",
            "benchmark": "Benchmark",
            "drivers": "Key Drivers",
            "direction": "Forecast Direction",
        }

        for key, label in labels.items():
            if key in interp:
                sections.append("  [bold]%s:[/bold]\n  %s" % (label, interp[key]))

        content = "\n\n".join(sections)
        console.print(Panel(
            content,
            title="[bold]%s Forecast Interpretation[/bold]" % self.model_name,
            border_style="green",
            padding=(1, 2),
        ))

    def summary(self) -> pd.DataFrame:
        """Return metrics as a single-row DataFrame."""
        return pd.DataFrame([self.metrics], index=[self.model_name])

    def interpret_metrics(self, verbose: bool = True) -> dict:
        """Explain every metric in plain English with context.

        Returns a dict mapping metric names to interpretation strings.
        """
        m = self.metrics
        if not m:
            if verbose:
                print("No metrics available (missing actual or test_predictions).")
            return {}

        explanations = {}

        # MAE
        if "MAE" in m:
            v = m["MAE"]
            explanations["MAE"] = (
                "Mean Absolute Error = %.4f. "
                "On average, predictions are off by %.4f units from the actual values. "
                "Lower is better." % (v, v)
            )

        # RMSE
        if "RMSE" in m:
            v = m["RMSE"]
            mae = m.get("MAE", v)
            ratio = v / mae if mae > 0 else 1
            if ratio < 1.2:
                pattern = "Errors are very consistent (few large outlier errors)."
            elif ratio < 1.5:
                pattern = "Errors are fairly consistent."
            else:
                pattern = "Some predictions have large errors (outlier spikes present)."
            explanations["RMSE"] = (
                "Root Mean Squared Error = %.4f. "
                "Like MAE but penalizes large errors more heavily. "
                "RMSE/MAE ratio = %.2f. %s" % (v, ratio, pattern)
            )

        # MSE
        if "MSE" in m:
            explanations["MSE"] = (
                "Mean Squared Error = %.4f. "
                "The squared version of RMSE. Useful for optimization but harder to interpret directly."
                % m["MSE"]
            )

        # MedAE
        if "MedAE" in m:
            v = m["MedAE"]
            mae = m.get("MAE", v)
            if v < mae * 0.8:
                note = "MedAE is much lower than MAE, confirming a few large errors skew the average."
            else:
                note = "MedAE is close to MAE, indicating errors are evenly distributed."
            explanations["MedAE"] = (
                "Median Absolute Error = %.4f. "
                "The typical (middle) error, robust to outliers. %s" % (v, note)
            )

        # MAPE
        if "MAPE" in m:
            v = m["MAPE"]
            if v < 3:
                level, desc = "Exceptional", "near-perfect predictions"
            elif v < 5:
                level, desc = "Excellent", "highly accurate predictions"
            elif v < 10:
                level, desc = "Very Good", "strong predictive performance"
            elif v < 15:
                level, desc = "Good", "solid predictions for most use cases"
            elif v < 25:
                level, desc = "Fair", "useful but has room for improvement"
            elif v < 40:
                level, desc = "Needs Work", "consider trying other models or features"
            else:
                level, desc = "Poor", "the model struggles with this data"
            explanations["MAPE"] = (
                "Mean Absolute Percentage Error = %.2f%%. "
                "[%s] For every 100 units of actual value, the prediction is off by ~%.1f units. "
                "Rating: %s."
                % (v, level, v, desc)
            )

        # R2
        if "R2" in m:
            v = m["R2"]
            pct = v * 100
            if v > 0.95:
                quality = "The model explains almost all variance -- excellent fit."
            elif v > 0.85:
                quality = "The model captures most patterns well."
            elif v > 0.70:
                quality = "Reasonable fit, but significant unexplained variance remains."
            elif v > 0.50:
                quality = "Moderate fit. The model misses many patterns."
            else:
                quality = "Weak fit. The model explains less than half the variance."
            explanations["R2"] = (
                "R-squared = %.4f. "
                "The model explains %.1f%% of the variance in the data. "
                "%s" % (v, pct, quality)
            )

        # MASE
        if "MASE" in m:
            v = m["MASE"]
            if v < 0.5:
                verdict = "Dramatically outperforms the naive baseline."
            elif v < 1.0:
                pct_better = (1 - v) * 100
                verdict = "Outperforms naive forecast by %.0f%%. The model adds real value." % pct_better
            elif v == 1.0:
                verdict = "Performs exactly like a naive repeat-last-value forecast."
            else:
                verdict = "Underperforms naive forecast. A simple baseline would be better."
            explanations["MASE"] = (
                "Mean Absolute Scaled Error = %.4f. "
                "Compares the model against a naive (repeat last value) forecast. "
                "MASE < 1 means better than naive. %s" % (v, verdict)
            )

        if verbose:
            _print_metric_explanations(explanations, self.model_name)

        return explanations

    def plot(self, figsize=(14, 5)):
        """Plot forecast results with actual vs predicted and confidence intervals."""
        from timesage.plot.theme import sage_theme
        from timesage.plot.timeplots import plot_forecast
        sage_theme()
        return plot_forecast(self, figsize=figsize)



def _print_metric_explanations(explanations: dict, model_name: str = "Model"):
    """Pretty-print metric explanations."""
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        lines = []
        for metric, explanation in explanations.items():
            lines.append("[bold cyan]%s:[/bold cyan]  %s" % (metric, explanation))

        content = "\n\n".join(lines)
        console.print(Panel(
            content,
            title="[bold]%s -- Metrics Explained[/bold]" % model_name,
            border_style="blue",
            padding=(1, 2),
        ))
    except ImportError:
        print("=== %s -- Metrics Explained ===" % model_name)
        for metric, explanation in explanations.items():
            print("  %s: %s" % (metric, explanation))
