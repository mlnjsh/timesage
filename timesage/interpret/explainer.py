"""Plain-English interpretation engine — the soul of TimeSage."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from timesage.core.result import ForecastResult


def explain_forecast(result: "ForecastResult", verbose: bool = True) -> dict[str, str]:
    """Generate comprehensive plain-English interpretation of forecast results.

    This is what makes TimeSage unique — turning complex statistical output
    into actionable insights that anyone can understand.
    """
    explanations = {}

    # 1. Overall accuracy assessment
    if "MAPE" in result.metrics:
        mape = result.metrics["MAPE"]
        if mape < 3:
            level, emoji_word = "exceptional", "Outstanding"
        elif mape < 5:
            level, emoji_word = "excellent", "Excellent"
        elif mape < 10:
            level, emoji_word = "very good", "Very Good"
        elif mape < 15:
            level, emoji_word = "good", "Good"
        elif mape < 25:
            level, emoji_word = "moderate", "Fair"
        elif mape < 40:
            level, emoji_word = "poor", "Needs Work"
        else:
            level, emoji_word = "very poor", "Poor"

        explanations["overall_accuracy"] = (
            f"[{emoji_word}] The {result.model_name} model achieves {level} accuracy "
            f"with an average error of {mape:.1f}%. "
            f"In practical terms, for every 100 units of actual value, "
            f"the model's prediction is typically off by about {mape:.1f} units."
        )

    # 2. Error pattern analysis
    if "RMSE" in result.metrics and "MAE" in result.metrics:
        rmse = result.metrics["RMSE"]
        mae = result.metrics["MAE"]

        if mae > 0:
            ratio = rmse / mae
            if ratio > 1.5:
                explanations["error_pattern"] = (
                    f"The model has INCONSISTENT errors (RMSE/MAE ratio = {ratio:.2f}). "
                    f"While most predictions are reasonably close (MAE = {mae:.2f}), "
                    f"some predictions are significantly off (RMSE = {rmse:.2f}). "
                    f"This suggests the model struggles with certain patterns — "
                    f"possibly outliers, regime changes, or sudden spikes."
                )
            else:
                explanations["error_pattern"] = (
                    f"The model has CONSISTENT errors (RMSE/MAE ratio = {ratio:.2f}). "
                    f"Errors are relatively uniform — the model doesn't have blind spots. "
                    f"Average absolute error: {mae:.2f}, root mean square error: {rmse:.2f}."
                )

    # 3. Naive benchmark comparison
    if "MASE" in result.metrics:
        mase = result.metrics["MASE"]
        if mase < 0.5:
            explanations["benchmark"] = (
                f"The model is MUCH BETTER than a naive forecast (MASE = {mase:.2f}). "
                f"It captures meaningful patterns that simple approaches miss. "
                f"The model reduces error by {(1-mase)*100:.0f}% vs. using yesterday's value as today's prediction."
            )
        elif mase < 1.0:
            pct = (1 - mase) * 100
            explanations["benchmark"] = (
                f"The model OUTPERFORMS a naive forecast by {pct:.0f}% (MASE = {mase:.2f}). "
                f"It adds value beyond simply using the most recent observation."
            )
        elif mase < 1.2:
            explanations["benchmark"] = (
                f"The model is ROUGHLY EQUAL to a naive forecast (MASE = {mase:.2f}). "
                f"It may not be capturing enough signal to justify its complexity. "
                f"Consider: Is a simple approach sufficient? Are there missing features?"
            )
        else:
            explanations["benchmark"] = (
                f"The model UNDERPERFORMS a naive forecast (MASE = {mase:.2f}). "
                f"A simpler approach (e.g., using yesterday's value) would be more accurate. "
                f"Action items: Check for data issues, try different features, "
                f"or consider if this series is inherently unpredictable."
            )

    # 4. Variance explanation
    if "R2" in result.metrics:
        r2 = result.metrics["R2"]
        pct = r2 * 100
        if r2 > 0.95:
            explanations["predictability"] = (
                f"The model explains {pct:.1f}% of the variance — nearly perfect. "
                f"The time series is highly predictable with the current features."
            )
        elif r2 > 0.8:
            explanations["predictability"] = (
                f"The model explains {pct:.1f}% of the variance — strong performance. "
                f"Most of the variation is captured, with {100-pct:.1f}% unexplained "
                f"(likely noise, external factors, or rare events)."
            )
        elif r2 > 0.5:
            explanations["predictability"] = (
                f"The model explains {pct:.1f}% of the variance — moderate. "
                f"About half the variation remains unexplained. "
                f"Adding more features or trying different models could help."
            )
        else:
            explanations["predictability"] = (
                f"The model explains only {pct:.1f}% of the variance — weak. "
                f"The series may have high randomness, or important "
                f"predictive signals are missing from the features."
            )

    # 5. Feature importance insights
    if result.feature_importance is not None and len(result.feature_importance) > 0:
        top5 = result.feature_importance.nlargest(5)
        total_imp = result.feature_importance.sum()

        top_names = []
        for name, imp in top5.items():
            pct = imp / total_imp * 100 if total_imp > 0 else 0
            top_names.append(f"'{name}' ({pct:.1f}%)")

        explanations["key_drivers"] = (
            f"The top predictive features are: {', '.join(top_names)}. "
            f"Together they account for {top5.sum()/total_imp*100:.0f}% "
            f"of the model's predictive power."
        )

        # Identify lag vs temporal features
        lag_features = [n for n in top5.index if "lag" in str(n).lower()]
        temporal_features = [n for n in top5.index if any(t in str(n).lower() for t in ["month", "day", "hour", "week", "year"])]

        if lag_features and not temporal_features:
            explanations["pattern_type"] = (
                "The model relies primarily on AUTOREGRESSIVE patterns "
                "(past values predicting future values). This suggests "
                "the series has strong momentum or persistence."
            )
        elif temporal_features and not lag_features:
            explanations["pattern_type"] = (
                "The model relies primarily on CALENDAR patterns "
                "(time-of-year, day-of-week effects). This suggests "
                "strong seasonal or cyclical behavior."
            )
        elif lag_features and temporal_features:
            explanations["pattern_type"] = (
                "The model uses BOTH autoregressive and calendar patterns. "
                "This is typical of series with both momentum and seasonality."
            )

    # 6. Forecast trend
    if len(result.forecast) > 1:
        vals = result.forecast.values
        first_val = vals[0]
        last_val = vals[-1]
        change_pct = (last_val - first_val) / abs(first_val) * 100 if first_val != 0 else 0

        if abs(change_pct) < 2:
            trend = "FLAT"
            explanations["forecast_direction"] = (
                f"The forecast is relatively FLAT over the prediction horizon, "
                f"changing by only {change_pct:+.1f}%. "
                f"Expected range: {vals.min():.2f} to {vals.max():.2f}."
            )
        elif change_pct > 0:
            explanations["forecast_direction"] = (
                f"The forecast shows an UPWARD trend of {change_pct:+.1f}% "
                f"from {first_val:.2f} to {last_val:.2f} over the horizon. "
                f"Peak predicted value: {vals.max():.2f}."
            )
        else:
            explanations["forecast_direction"] = (
                f"The forecast shows a DOWNWARD trend of {change_pct:+.1f}% "
                f"from {first_val:.2f} to {last_val:.2f} over the horizon. "
                f"Minimum predicted value: {vals.min():.2f}."
            )

    if verbose:
        _print_explanation(result.model_name, explanations)

    return explanations


def explain_model(model_name: str, params: dict, verbose: bool = True) -> dict[str, str]:
    """Explain what a model does in plain English."""
    explanations = {}

    model_descriptions = {
        "ARIMA": (
            "ARIMA (AutoRegressive Integrated Moving Average) models the time series "
            "by combining three components: autoregression (using past values), "
            "differencing (removing trends), and moving averages (smoothing errors). "
            "It's one of the most widely-used statistical forecasting methods."
        ),
        "ETS": (
            "ETS (Exponential Smoothing) forecasts by giving exponentially decreasing "
            "weights to older observations. Recent data influences the forecast more "
            "than distant data. It handles trend and seasonality through separate components."
        ),
        "Theta": (
            "The Theta method decomposes the time series into two 'theta lines' — "
            "one capturing long-term trend and another capturing short-term dynamics. "
            "It's simple but has won multiple forecasting competitions."
        ),
        "XGBoost": (
            "XGBoost is a gradient boosting algorithm that builds an ensemble of decision trees. "
            "Each tree corrects the errors of previous trees. It excels at capturing "
            "complex non-linear patterns and interactions between features."
        ),
        "LightGBM": (
            "LightGBM is a fast gradient boosting framework that uses histogram-based "
            "learning. It's highly efficient for large datasets and can capture "
            "complex patterns while being resistant to overfitting."
        ),
        "Random Forest": (
            "Random Forest builds many independent decision trees and averages their "
            "predictions. This 'wisdom of crowds' approach is robust to outliers "
            "and provides natural feature importance rankings."
        ),
    }

    for key, desc in model_descriptions.items():
        if key.lower() in model_name.lower():
            explanations["model_description"] = desc
            break

    if "order" in params:
        p, d, q = params["order"]
        explanations["arima_order"] = (
            f"Order (p={p}, d={d}, q={q}): Uses {p} past values (AR), "
            f"{d} level(s) of differencing, and {q} past errors (MA). "
            f"{'No differencing needed — series is already stationary.' if d == 0 else f'Applied {d}x differencing to remove trend.'}"
        )

    if verbose:
        _print_explanation(model_name, explanations)

    return explanations


def _print_explanation(title: str, explanations: dict[str, str]):
    """Pretty-print explanations."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown

        console = Console()
        console.print()
        console.print(Panel(
            f"[bold cyan]{title}[/bold cyan] — Detailed Interpretation",
            style="bold green", expand=False,
        ))

        for key, text in explanations.items():
            label = key.replace("_", " ").upper()
            console.print(f"\n  [bold yellow]{label}[/bold yellow]")
            console.print(f"  {text}")

        console.print()

    except ImportError:
        print(f"\n{'='*60}")
        print(f"  {title} — Detailed Interpretation")
        print(f"{'='*60}")
        for key, text in explanations.items():
            label = key.replace("_", " ").upper()
            print(f"\n  {label}")
            print(f"  {text}")
        print()
