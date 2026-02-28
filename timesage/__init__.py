"""
╔══════════════════════════════════════════════════════════════╗
║                      ⏳ TimeSage ⏳                          ║
║            The Wise Time Series Library                      ║
║                                                              ║
║  Beautiful EDA · All Models · Plain-English Interpretation   ║
╚══════════════════════════════════════════════════════════════╝

TimeSage makes time series analysis effortless:

    >>> import timesage as ts
    >>> series = ts.TimeSeries(df, target="sales", time="date")
    >>> series.eda()           # Beautiful automated EDA
    >>> series.plot()          # Stunning visualizations
    >>> result = series.forecast(horizon=30)
    >>> result.interpret()     # Plain-English interpretation

"""

__version__ = "0.2.0"
__author__ = "Milan Amrut Joshi"

from timesage.core.timeseries import TimeSeries
from timesage.core.result import ForecastResult
from timesage.eda.profiler import profile
from timesage.plot.theme import set_theme, sage_theme
from timesage.models.auto import AutoForecaster

__all__ = [
    "TimeSeries",
    "ForecastResult",
    "profile",
    "set_theme",
    "sage_theme",
    "AutoForecaster",
]
