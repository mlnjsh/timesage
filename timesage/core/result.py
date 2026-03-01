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
    _forecaster: Optional[Any] = field(default=None, repr=False)


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

    # ── Model Summary & Interpretation ──────────────────────────────

    def model_summary(self, verbose: bool = True) -> dict:
        """Show full model summary: coefficients, info criteria, diagnostics.

        For statistical models (ARIMA, ETS, Theta): shows coefficient table,
        AIC/BIC, log-likelihood, and diagnostic tests.
        For ML models: shows feature importances and hyperparameters.

        Returns a dict with all extracted model information.
        """
        if self._forecaster is None or not hasattr(self._forecaster, "model_summary"):
            if verbose:
                print("No model summary available (forecaster not stored).")
            return {}

        info = self._forecaster.model_summary()
        if verbose:
            _print_model_summary(info, self.model_name)
        return info

    def interpret_summary(self, verbose: bool = True) -> dict:
        """Explain every number in the model summary in plain English.

        Covers: coefficients, sigma, AIC/BIC, log-likelihood, Ljung-Box,
        Jarque-Bera, heteroskedasticity, skew, kurtosis, and feature importance.
        """
        info = self.model_summary(verbose=False)
        if not info:
            if verbose:
                print("No model summary available to interpret.")
            return {}

        explanations = {}

        # Model type
        mt = info.get("model_type", "Unknown")
        nobs = info.get("nobs", 0)
        explanations["model"] = "Model: %s fitted on %d observations." % (mt, nobs)

        # Information criteria
        aic = info.get("aic")
        bic = info.get("bic")
        if aic is not None:
            explanations["aic"] = (
                "AIC (Akaike Information Criterion) = %.2f. "
                "Balances model fit against complexity. Lower AIC = better model. "
                "Use AIC to compare different model orders (e.g. ARIMA(1,1,0) vs ARIMA(2,1,1))."
                % aic
            )
        if bic is not None:
            explanations["bic"] = (
                "BIC (Bayesian Information Criterion) = %.2f. "
                "Like AIC but penalizes complexity more heavily. "
                "Prefers simpler models. Lower BIC = better." % bic
            )
        hqic = info.get("hqic")
        if hqic is not None:
            explanations["hqic"] = (
                "HQIC (Hannan-Quinn) = %.2f. "
                "Another information criterion, between AIC and BIC in penalty strength."
                % hqic
            )

        # Log-likelihood
        ll = info.get("log_likelihood")
        if ll is not None:
            explanations["log_likelihood"] = (
                "Log-Likelihood = %.2f. "
                "Measures how well the model fits the observed data. "
                "Higher (less negative) = better fit. AIC and BIC are derived from this."
                % ll
            )

        # Sigma2
        sigma2 = info.get("sigma2")
        if sigma2 is not None:
            explanations["sigma2"] = (
                "sigma2 (Residual Variance) = %.4f. "
                "The average squared residual. Represents unexplained noise. "
                "Residual Std Dev = %.4f -- this is roughly how much "
                "predictions deviate due to random error."
                % (sigma2, np.sqrt(sigma2))
            )

        # SSE for ETS
        sse = info.get("sse")
        if sse is not None:
            explanations["sse"] = (
                "SSE (Sum of Squared Errors) = %.4f. "
                "Total squared prediction error. Lower means better fit." % sse
            )

        # Coefficients
        coefficients = info.get("coefficients", [])
        is_ml = "hyperparameters" in info
        coeff_explanations = []
        for c in coefficients:
            name = c["name"]
            coef = c["coef"]
            if coef is None:
                continue

            if is_ml:
                # ML feature importance
                coeff_explanations.append(
                    "[bold]%s[/bold]: importance = %.4f (%.1f%%). "
                    "This feature contributes %.1f%% of the model's predictive power."
                    % (name, coef, coef * 100, coef * 100)
                )
            else:
                # Statistical model coefficient
                pval = c.get("p_value")
                stderr = c.get("std_err")
                z = c.get("z")
                ci_lo = c.get("ci_lower")
                ci_hi = c.get("ci_upper")

                parts = ["[bold]%s[/bold]: coef = %.4f" % (name, coef)]

                if name == "sigma2":
                    parts.append(
                        "-- Residual variance. Std dev of noise = %.4f."
                        % np.sqrt(abs(coef))
                    )
                elif name.startswith("ar.L"):
                    lag_num = name.replace("ar.L", "")
                    parts.append(
                        "-- AR(%s) coefficient. Each unit increase in the value "
                        "%s step(s) ago changes the current prediction by %.4f."
                        % (lag_num, lag_num, coef)
                    )
                elif name.startswith("ma.L"):
                    lag_num = name.replace("ma.L", "")
                    parts.append(
                        "-- MA(%s) coefficient. Each unit of forecast error "
                        "%s step(s) ago adjusts the current prediction by %.4f."
                        % (lag_num, lag_num, coef)
                    )
                elif "alpha" in name.lower() or "level" in name.lower():
                    parts.append(
                        "-- Level smoothing. Higher alpha (closer to 1) means "
                        "the model reacts faster to recent changes."
                    )
                elif "beta" in name.lower() or "trend" in name.lower():
                    parts.append(
                        "-- Trend smoothing. Controls how quickly the trend "
                        "estimate adapts."
                    )
                elif "phi" in name.lower() or "damp" in name.lower():
                    parts.append(
                        "-- Damping factor. Values < 1 dampen the trend over time "
                        "(prevents over-extrapolation)."
                    )

                if pval is not None:
                    if pval < 0.001:
                        sig = "Highly significant (p < 0.001)"
                    elif pval < 0.01:
                        sig = "Significant (p < 0.01)"
                    elif pval < 0.05:
                        sig = "Significant (p < 0.05)"
                    else:
                        sig = "NOT significant (p = %.3f) -- consider removing" % pval
                    parts.append("[%s]" % sig)

                if ci_lo is not None and ci_hi is not None:
                    parts.append("95%% CI: [%.4f, %.4f]" % (ci_lo, ci_hi))

                coeff_explanations.append(" ".join(parts))

        if coeff_explanations:
            if is_ml:
                explanations["feature_importance"] = "\n".join(coeff_explanations)
            else:
                explanations["coefficients"] = "\n".join(coeff_explanations)

        # Hyperparameters (ML only)
        hp = info.get("hyperparameters", {})
        if hp:
            hp_parts = []
            for k, v in hp.items():
                hp_parts.append("  %s = %s" % (k, v))
            explanations["hyperparameters"] = (
                "Model hyperparameters:\n" + "\n".join(hp_parts)
            )

        # Diagnostic tests
        diag = info.get("diagnostics", {})
        if diag:
            diag_explanations = []

            # Ljung-Box
            lb_stat = diag.get("ljung_box_stat")
            lb_p = diag.get("ljung_box_pvalue")
            if lb_stat is not None and lb_p is not None:
                if lb_p > 0.05:
                    verdict = "PASS -- residuals appear independent (no leftover patterns)."
                else:
                    verdict = (
                        "FAIL -- residuals are autocorrelated. The model is missing "
                        "temporal patterns. Try increasing model order or adding features."
                    )
                diag_explanations.append(
                    "[bold]Ljung-Box Q[/bold] = %.2f (p = %.4f). "
                    "Tests whether residuals are independent (no autocorrelation). "
                    "%s" % (lb_stat, lb_p, verdict)
                )

            # Jarque-Bera
            jb_stat = diag.get("jarque_bera_stat")
            jb_p = diag.get("jarque_bera_pvalue")
            if jb_stat is not None and jb_p is not None:
                if jb_p > 0.05:
                    verdict = "PASS -- residuals are approximately normal. Confidence intervals are reliable."
                else:
                    verdict = (
                        "FAIL -- residuals are NOT normally distributed. "
                        "Confidence intervals may be too narrow or wide. "
                        "Consider log transformation or robust methods."
                    )
                diag_explanations.append(
                    "[bold]Jarque-Bera[/bold] = %.2f (p = %.4f). "
                    "Tests whether residuals follow a normal distribution. "
                    "%s" % (jb_stat, jb_p, verdict)
                )

            # Heteroskedasticity
            h_stat = diag.get("het_stat")
            h_p = diag.get("het_pvalue")
            if h_stat is not None:
                if h_stat < 2.0:
                    verdict = "PASS -- error variance is stable over time."
                else:
                    verdict = (
                        "FAIL -- error variance changes over time. "
                        "The model is more accurate in some periods than others. "
                        "Consider log transform or GARCH modeling."
                    )
                p_str = " (p = %.4f)" % h_p if h_p is not None else ""
                diag_explanations.append(
                    "[bold]Heteroskedasticity (H)[/bold] = %.2f%s. "
                    "Tests whether residual variance is constant. "
                    "%s" % (h_stat, p_str, verdict)
                )

            # Skew
            skew = diag.get("skew")
            if skew is not None:
                if abs(skew) < 0.5:
                    verdict = "approximately symmetric (good)."
                elif skew > 0:
                    verdict = "right-skewed (occasional large positive errors)."
                else:
                    verdict = "left-skewed (occasional large negative errors)."
                diag_explanations.append(
                    "[bold]Skew[/bold] = %.2f. "
                    "Measures asymmetry of residuals. 0 = perfectly symmetric. "
                    "Residuals are %s" % (skew, verdict)
                )

            # Kurtosis
            kurt = diag.get("kurtosis")
            if kurt is not None:
                if abs(kurt) < 1:
                    verdict = "close to normal (good)."
                elif kurt > 0:
                    verdict = (
                        "heavy-tailed (more extreme errors than expected). "
                        "Outlier-sensitive models may struggle."
                    )
                else:
                    verdict = "light-tailed (fewer extreme errors than a normal distribution)."
                diag_explanations.append(
                    "[bold]Kurtosis[/bold] = %.2f. "
                    "Measures tail heaviness. 0 = normal distribution. "
                    "Residuals are %s" % (kurt, verdict)
                )

            if diag_explanations:
                explanations["diagnostics"] = "\n\n".join(diag_explanations)

        if verbose:
            _print_summary_interpretation(explanations, self.model_name)

        return explanations

    def plot(self, figsize=(14, 5)):
        """Plot forecast results with actual vs predicted and confidence intervals."""
        from timesage.plot.theme import sage_theme
        from timesage.plot.timeplots import plot_forecast
        sage_theme()
        return plot_forecast(self, figsize=figsize)


# ── Pretty-print helpers ────────────────────────────────────────────────

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


def _print_model_summary(info: dict, model_name: str = "Model"):
    """Pretty-print the model summary with coefficient table and diagnostics."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        # Header info
        mt = info.get("model_type", "Unknown")
        nobs = info.get("nobs", "?")
        header_parts = ["[bold]%s[/bold]  |  Observations: %s" % (mt, nobs)]
        for key in ("aic", "bic", "hqic", "log_likelihood"):
            if key in info:
                label = key.upper().replace("_", " ").replace("LOG LIKELIHOOD", "Log-Likelihood")
                header_parts.append("%s: %.2f" % (label, info[key]))
        if "sigma2" in info:
            header_parts.append("sigma2: %.4f" % info["sigma2"])
        if "sse" in info:
            header_parts.append("SSE: %.4f" % info["sse"])

        console.print(Panel(
            "  ".join(header_parts),
            title="[bold]%s -- Model Summary[/bold]" % model_name,
            border_style="cyan",
            padding=(0, 2),
        ))

        # Coefficients table
        coefficients = info.get("coefficients", [])
        is_ml = "hyperparameters" in info
        if coefficients:
            if is_ml:
                table = Table(title="Feature Importance", show_header=True, header_style="bold cyan")
                table.add_column("Feature", style="bold")
                table.add_column("Importance", justify="right")
                table.add_column("Pct", justify="right")
                for c in coefficients:
                    table.add_row(
                        c["name"],
                        "%.4f" % c["coef"] if c["coef"] is not None else "N/A",
                        "%.1f%%" % (c["coef"] * 100) if c["coef"] is not None else "N/A",
                    )
            else:
                table = Table(title="Coefficients", show_header=True, header_style="bold cyan")
                table.add_column("", style="bold")
                table.add_column("coef", justify="right")
                table.add_column("std err", justify="right")
                table.add_column("z", justify="right")
                table.add_column("P>|z|", justify="right")
                table.add_column("[0.025", justify="right")
                table.add_column("0.975]", justify="right")
                for c in coefficients:
                    pval = c.get("p_value")
                    sig_style = ""
                    if pval is not None and pval < 0.05:
                        sig_style = "bold green"
                    elif pval is not None and pval >= 0.05:
                        sig_style = "dim"
                    table.add_row(
                        c["name"],
                        "%.4f" % c["coef"] if c["coef"] is not None else "N/A",
                        "%.4f" % c["std_err"] if c["std_err"] is not None else "N/A",
                        "%.3f" % c["z"] if c["z"] is not None else "N/A",
                        "%.3f" % c["p_value"] if c["p_value"] is not None else "N/A",
                        "%.4f" % c["ci_lower"] if c["ci_lower"] is not None else "N/A",
                        "%.4f" % c["ci_upper"] if c["ci_upper"] is not None else "N/A",
                        style=sig_style,
                    )
            console.print(table)

        # Hyperparameters (ML)
        hp = info.get("hyperparameters", {})
        if hp:
            hp_lines = []
            for k, v in hp.items():
                hp_lines.append("  %s = %s" % (k, v))
            console.print(Panel(
                "\n".join(hp_lines),
                title="Hyperparameters",
                border_style="yellow",
                padding=(0, 2),
            ))

        # Diagnostics
        diag = info.get("diagnostics", {})
        if diag:
            diag_table = Table(title="Diagnostic Tests", show_header=True, header_style="bold cyan")
            diag_table.add_column("Test", style="bold")
            diag_table.add_column("Statistic", justify="right")
            diag_table.add_column("P-value", justify="right")
            diag_table.add_column("Result", justify="center")

            # Ljung-Box
            if "ljung_box_stat" in diag:
                p = diag["ljung_box_pvalue"]
                diag_table.add_row(
                    "Ljung-Box (Q)",
                    "%.2f" % diag["ljung_box_stat"],
                    "%.4f" % p,
                    "[green]Pass[/green]" if p > 0.05 else "[red]Fail[/red]",
                )
            # Jarque-Bera
            if "jarque_bera_stat" in diag:
                p = diag["jarque_bera_pvalue"]
                diag_table.add_row(
                    "Jarque-Bera",
                    "%.2f" % diag["jarque_bera_stat"],
                    "%.4f" % p,
                    "[green]Pass[/green]" if p > 0.05 else "[red]Fail[/red]",
                )
            # Heteroskedasticity
            if "het_stat" in diag:
                p = diag.get("het_pvalue")
                diag_table.add_row(
                    "Heteroskedasticity (H)",
                    "%.2f" % diag["het_stat"],
                    "%.4f" % p if p is not None else "N/A",
                    "[green]Pass[/green]" if diag["het_stat"] < 2.0 else "[red]Fail[/red]",
                )
            console.print(diag_table)

            # Skew + Kurtosis
            skew = diag.get("skew")
            kurt = diag.get("kurtosis")
            if skew is not None or kurt is not None:
                extra = []
                if skew is not None:
                    extra.append("Skew: %.2f" % skew)
                if kurt is not None:
                    extra.append("Kurtosis: %.2f" % kurt)
                console.print("  " + "  |  ".join(extra))

        # Raw summary (statsmodels text)
        raw = info.get("raw_summary")
        if raw:
            console.print()
            console.print(Panel(raw, title="Raw statsmodels Summary", border_style="dim"))

    except ImportError:
        # Fallback without rich
        print("=== %s -- Model Summary ===" % model_name)
        for k, v in info.items():
            if k not in ("coefficients", "diagnostics", "raw_summary"):
                print("  %s: %s" % (k, v))
        for c in info.get("coefficients", []):
            print("  %s: coef=%.4f" % (c["name"], c["coef"]) if c["coef"] else "")


def _print_summary_interpretation(explanations: dict, model_name: str = "Model"):
    """Pretty-print the interpret_summary output."""
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        sections = []
        section_order = [
            ("model", "Model"),
            ("aic", "AIC"),
            ("bic", "BIC"),
            ("hqic", "HQIC"),
            ("log_likelihood", "Log-Likelihood"),
            ("sigma2", "Residual Variance"),
            ("sse", "Sum of Squared Errors"),
            ("coefficients", "Coefficients"),
            ("feature_importance", "Feature Importance"),
            ("hyperparameters", "Hyperparameters"),
            ("diagnostics", "Diagnostic Tests"),
        ]

        for key, label in section_order:
            if key in explanations:
                sections.append("[bold cyan]%s:[/bold cyan]\n%s" % (label, explanations[key]))

        content = "\n\n".join(sections)
        console.print(Panel(
            content,
            title="[bold]%s -- Summary Interpreted[/bold]" % model_name,
            border_style="green",
            padding=(1, 2),
        ))
    except ImportError:
        print("=== %s -- Summary Interpreted ===" % model_name)
        for key, text in explanations.items():
            print("\n%s:\n%s" % (key.upper(), text))
