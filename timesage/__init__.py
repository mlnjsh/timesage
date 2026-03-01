"""
╔══════════════════════════════════════════════════════════════╗
║                      ⏳ TimeSage ⏳                          ║
║            The Wise Time Series Library                      ║
║                                                              ║
║  Beautiful EDA · All Models · Plain-English Interpretation   ║
╚══════════════════════════════════════════════════════════════╝

TimeSage makes time series analysis effortless:

    >>> import timesage as ts
    >>> ts.hello()             # Welcome message
    >>> data = ts.load_airline()
    >>> series = ts.TimeSeries(data, target="passengers")
    >>> series.eda()           # Beautiful automated EDA
    >>> series.plot()          # Stunning visualizations
    >>> result = series.forecast(horizon=30)
    >>> result.interpret()     # Plain-English interpretation

"""

__version__ = "0.2.7"
__author__ = "Milan Amrut Joshi"

from timesage.core.timeseries import TimeSeries
from timesage.core.result import ForecastResult
from timesage.eda.profiler import profile
from timesage.plot.theme import set_theme, sage_theme
from timesage.models.auto import AutoForecaster
from timesage.datasets.loader import (
    list_datasets,
    load_airline,
    load_sunspots,
    load_energy,
    load_synthetic_trend,
    load_synthetic_seasonal,
)


def hello():
    """Print a welcome message with library info and quick-start guide."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    message = (
        f"[bold cyan]TimeSage v{__version__}[/bold cyan]\n"
        f"[dim]The Wise Time Series Library[/dim]\n\n"
        f"[bold]Quick Start:[/bold]\n"
        f"  [green]import timesage as ts[/green]\n"
        f"  data = ts.load_airline()\n"
        f"  series = ts.TimeSeries(data, target='passengers')\n"
        f"  series.eda()\n"
        f"  result = series.forecast(horizon=30)\n"
        f"  result.interpret()\n\n"
        f"[bold]Built-in Datasets:[/bold] {', '.join(list_datasets())}\n"
        f"[bold]Models:[/bold] ARIMA, ETS, Theta, RandomForest, XGBoost, LightGBM, Auto\n"
        f"[bold]Author:[/bold] {__author__}"
    )
    console.print(Panel(message, title="Welcome to TimeSage", border_style="cyan"))


__all__ = [
    "TimeSeries",
    "ForecastResult",
    "profile",
    "set_theme",
    "sage_theme",
    "AutoForecaster",
    "hello",
    "list_datasets",
    "load_airline",
    "load_sunspots",
    "load_energy",
    "load_synthetic_trend",
    "load_synthetic_seasonal",
]
