"""Diagnostic plotting functions with interpretation."""

import matplotlib.pyplot as plt
import numpy as np
from timesage.plot.theme import COLORS


def plot_acf_pacf(series, lags=40, figsize=(14, 4)):
    """Plot ACF and PACF side by side."""
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
    return fig


def interpret_acf_pacf(series, lags=40, verbose=True):
    """Analyze ACF and PACF values and return plain-English interpretation with model suggestions.

    Parameters
    ----------
    series : pd.Series
        The time series to analyze.
    lags : int
        Number of lags to compute.
    verbose : bool
        If True, print the interpretation using rich.

    Returns
    -------
    dict
        Dictionary with ACF/PACF analysis results and suggestions.
    """
    from statsmodels.tsa.stattools import acf, pacf

    s = series.dropna()
    nlags = min(lags, len(s) // 2 - 1)
    n = len(s)
    ci = 1.96 / np.sqrt(n)  # 95% confidence interval

    acf_vals = acf(s, nlags=nlags, fft=True)
    pacf_vals = pacf(s, nlags=nlags, method="ywm")

    results = {}

    # Significant ACF lags (excluding lag 0)
    sig_acf = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > ci]
    sig_pacf = [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > ci]

    results["significant_acf_lags"] = sig_acf
    results["significant_pacf_lags"] = sig_pacf
    results["confidence_interval"] = float(ci)

    # --- ACF Pattern Analysis ---
    interpretations = []
    suggestions = []

    # Check for slow decay in ACF (non-stationarity)
    if len(sig_acf) > nlags * 0.6:
        interpretations.append(
            "ACF decays SLOWLY with many significant lags (%d out of %d). "
            "This indicates the series is likely NON-STATIONARY." % (len(sig_acf), nlags)
        )
        suggestions.append("Apply differencing (d=1) before modeling, or use ARIMA with d>=1.")
    elif len(sig_acf) > 5:
        interpretations.append(
            "ACF shows GRADUAL decay with %d significant lags. "
            "This suggests an autoregressive (AR) process." % len(sig_acf)
        )
    else:
        interpretations.append(
            "ACF cuts off quickly with only %d significant lags. "
            "This suggests a moving average (MA) process." % len(sig_acf)
        )

    # --- PACF Pattern Analysis ---
    if len(sig_pacf) <= 3 and len(sig_acf) > 5:
        p_order = len(sig_pacf)
        interpretations.append(
            "PACF cuts off after lag %d while ACF decays gradually. "
            "This is a classic AR(%d) signature." % (p_order, p_order)
        )
        suggestions.append("Try ARIMA(p=%d, d=_, q=0)." % p_order)
    elif len(sig_acf) <= 3 and len(sig_pacf) > 5:
        q_order = len(sig_acf)
        interpretations.append(
            "ACF cuts off after lag %d while PACF decays gradually. "
            "This is a classic MA(%d) signature." % (q_order, q_order)
        )
        suggestions.append("Try ARIMA(p=0, d=_, q=%d)." % q_order)
    elif len(sig_acf) > 3 and len(sig_pacf) > 3:
        interpretations.append(
            "Both ACF and PACF show gradual decay. "
            "This suggests a mixed ARMA process."
        )
        suggestions.append("Try ARIMA with both p and q > 0 (e.g., ARIMA(1,d,1) or ARIMA(2,d,2)).")

    # --- Seasonality Detection ---
    seasonal_period = None
    for period in [12, 7, 4, 24, 52]:
        if period < len(acf_vals) and abs(acf_vals[period]) > ci:
            # Check if it's a repeating pattern
            multiples = [p for p in range(period, len(acf_vals), period)
                         if abs(acf_vals[p]) > ci]
            if len(multiples) >= 2:
                seasonal_period = period
                break

    if seasonal_period:
        interpretations.append(
            "SEASONAL pattern detected at period %d (significant ACF spikes at multiples of %d). "
            "This series has recurring cycles every %d observations."
            % (seasonal_period, seasonal_period, seasonal_period)
        )
        suggestions.append(
            "Use seasonal models: SARIMA(p,d,q)(P,D,Q,%d) or seasonal decomposition before modeling."
            % seasonal_period
        )
        results["seasonal_period"] = seasonal_period

    # --- Negative ACF at lag 1 ---
    if acf_vals[1] < -ci:
        interpretations.append(
            "Negative ACF at lag 1 (%.3f). The series may be OVER-DIFFERENCED "
            "or have an MA(1) component." % acf_vals[1]
        )
        suggestions.append("If already differenced, try reducing d. Otherwise, include MA term.")

    # --- Summary suggestion ---
    if not suggestions:
        suggestions.append("The ACF/PACF patterns are weak. Consider ETS or ML-based models.")

    results["interpretations"] = interpretations
    results["suggestions"] = suggestions

    if verbose:
        _print_acf_interpretation(results)

    return results


def _print_acf_interpretation(results):
    """Pretty-print ACF/PACF interpretation."""
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        lines = []

        lines.append("[bold]ACF/PACF Analysis[/bold]\n")
        lines.append("[dim]95%% confidence interval: +/-%.4f[/dim]\n" % results["confidence_interval"])

        lines.append("[bold cyan]Significant Lags:[/bold cyan]")
        lines.append("  ACF:  %s" % (results["significant_acf_lags"][:10] or "None"))
        lines.append("  PACF: %s\n" % (results["significant_pacf_lags"][:10] or "None"))

        lines.append("[bold cyan]Interpretation:[/bold cyan]")
        for interp in results["interpretations"]:
            lines.append("  - %s" % interp)

        lines.append("")
        lines.append("[bold green]Suggestions:[/bold green]")
        for sug in results["suggestions"]:
            lines.append("  -> %s" % sug)

        if "seasonal_period" in results:
            lines.append("\n[bold yellow]Seasonal Period: %d[/bold yellow]" % results["seasonal_period"])

        console.print(Panel("\n".join(lines), title="ACF/PACF Interpretation", border_style="cyan"))

    except ImportError:
        print("=== ACF/PACF Interpretation ===")
        print("Significant ACF lags:", results["significant_acf_lags"][:10])
        print("Significant PACF lags:", results["significant_pacf_lags"][:10])
        for interp in results["interpretations"]:
            print(" -", interp)
        print("\nSuggestions:")
        for sug in results["suggestions"]:
            print(" ->", sug)
