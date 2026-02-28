"""Diagnostic plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
from timesage.plot.theme import COLORS


def plot_acf_pacf(series, lags=40, figsize=(14, 4)):
    """Plot ACF and PACF side by side."""
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf as sm_plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf as sm_plot_pacf

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    s = series.dropna()
    nlags = min(lags, len(s) // 2 - 1)

    sm_plot_acf(s, lags=nlags, ax=ax1, color=COLORS["primary"])
    ax1.set_title("Autocorrelation (ACF)", fontsize=12, fontweight="bold")

    sm_plot_pacf(s, lags=nlags, ax=ax2, color=COLORS["secondary"], method="ywm")
    ax2.set_title("Partial Autocorrelation (PACF)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()
    return fig
