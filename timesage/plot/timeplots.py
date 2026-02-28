"""Time series plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from timesage.plot.theme import COLORS


def plot_series(series, show_trend=False, show_outliers=False,
                title="Time Series", figsize=(14, 5), **kwargs):
    """Plot a time series with optional trend and outlier highlighting."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series.index, series.values, color=COLORS["primary"],
            linewidth=1.5, alpha=0.9, label=title)

    if show_trend:
        window = max(len(series) // 20, 3)
        trend = series.rolling(window=window, center=True).mean()
        ax.plot(trend.index, trend.values, color=COLORS["accent"],
                linewidth=2.5, alpha=0.8, label="Trend")

    if show_outliers:
        from timesage.utils.helpers import detect_outliers_iqr
        mask = detect_outliers_iqr(series)
        if mask.any():
            ax.scatter(series.index[mask], series.values[mask],
                       color=COLORS["danger"], s=40, zorder=5,
                       label="Outliers", alpha=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    plt.show()
    return fig


def plot_components(decomposition, title="Decomposition", figsize=(14, 10)):
    """Plot seasonal decomposition components."""
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    components = [
        ("Observed", decomposition.observed, COLORS["primary"]),
        ("Trend", decomposition.trend, COLORS["accent"]),
        ("Seasonal", decomposition.seasonal, COLORS["secondary"]),
        ("Residual", decomposition.resid, COLORS["neutral"]),
    ]

    for ax, (name, data, color) in zip(axes, components):
        ax.plot(data, color=color, linewidth=1.2)
        ax.set_ylabel(name, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle("%s -- Seasonal Decomposition" % title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return fig


def plot_forecast(result, figsize=(14, 5)):
    """Plot forecast results with actuals, predictions, and confidence intervals."""
    fig, ax = plt.subplots(figsize=figsize)

    # Plot training data
    if result.train is not None:
        ax.plot(result.train.index, result.train.values,
                color=COLORS["neutral"], linewidth=1, alpha=0.5, label="Training")

    # Plot actual test data
    if result.actual is not None:
        ax.plot(result.actual.index, result.actual.values,
                color=COLORS["primary"], linewidth=1.5, label="Actual")

    # Plot test predictions
    if result.test_predictions is not None:
        ax.plot(result.test_predictions.index, result.test_predictions.values,
                color=COLORS["accent"], linewidth=1.5, linestyle="--", label="Predicted (test)")

    # Plot forecast
    ax.plot(result.forecast.index, result.forecast.values,
            color=COLORS["danger"], linewidth=2, label="Forecast")

    # Confidence intervals
    if result.confidence_lower is not None and result.confidence_upper is not None:
        ax.fill_between(
            result.forecast.index,
            result.confidence_lower.values,
            result.confidence_upper.values,
            alpha=0.15, color=COLORS["danger"], label="Confidence Interval",
        )

    ax.set_title("%s Forecast" % result.model_name, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    plt.show()
    return fig
