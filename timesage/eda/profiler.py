"""Automated EDA profiler for time series data."""

from typing import Any, Dict


def profile(ts, show_plots: bool = True) -> Dict[str, Any]:
    """Run a comprehensive EDA on a TimeSeries object.

    Parameters
    ----------
    ts : TimeSeries
        The time series to profile.
    show_plots : bool
        Whether to display diagnostic plots.

    Returns
    -------
    dict
        Dictionary containing all profiling results.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    results = {}

    # 1. Basic statistics
    desc = ts.describe()
    results["statistics"] = desc

    # 2. Stationarity tests
    stationarity = ts.test_stationarity(verbose=show_plots)
    results["stationarity"] = stationarity

    # 3. Seasonality detection
    seasonality = ts.detect_seasonality(verbose=show_plots)
    results["seasonality"] = seasonality

    # 4. Summary panel
    if show_plots:
        stats_table = Table(title="Summary Statistics", show_header=True, header_style="bold cyan")
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Value", justify="right")
        for key, val in desc.items():
            stats_table.add_row(str(key), "%.4f" % val if isinstance(val, float) else str(val))
        console.print(stats_table)

    # 5. Plots
    if show_plots:
        try:
            ts.plot(show_trend=True, show_outliers=True)
        except Exception:
            pass
        try:
            ts.plot_acf()
        except Exception:
            pass

    return results
