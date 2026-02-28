"""Built-in datasets for quick experimentation."""

from __future__ import annotations

import pandas as pd
import numpy as np


def list_datasets() -> list[str]:
    """List all available built-in datasets."""
    return ["airline", "sunspots", "energy", "synthetic_trend", "synthetic_seasonal"]


def load_airline() -> pd.DataFrame:
    """Load the classic airline passengers dataset (1949-1960, monthly)."""
    try:
        from statsmodels.datasets import co2
        # Use airline passengers from statsmodels
        import statsmodels.api as sm
        data = sm.datasets.get_rdataset("AirPassengers", "datasets").data
        data.columns = ["time", "passengers"]
        data["time"] = pd.date_range(start="1949-01", periods=len(data), freq="MS")
        data = data.set_index("time")
        return data
    except Exception:
        # Generate synthetic airline-like data
        dates = pd.date_range(start="1949-01", periods=144, freq="MS")
        trend = np.linspace(100, 500, 144)
        seasonal = 50 * np.sin(2 * np.pi * np.arange(144) / 12)
        noise = np.random.normal(0, 15, 144)
        passengers = trend + seasonal + noise
        return pd.DataFrame({"passengers": passengers.astype(int)}, index=dates)


def load_sunspots() -> pd.DataFrame:
    """Load monthly sunspot numbers."""
    try:
        import statsmodels.api as sm
        data = sm.datasets.sunspots.load_pandas().data
        data["YEAR"] = pd.to_datetime(data["YEAR"].astype(int).astype(str), format="%Y")
        data = data.set_index("YEAR")
        data.columns = ["sunspots"]
        return data
    except Exception:
        # Generate synthetic sunspot-like data
        dates = pd.date_range(start="1700-01", periods=300, freq="YS")
        cycle = 80 * np.sin(2 * np.pi * np.arange(300) / 11) + 80
        noise = np.random.normal(0, 20, 300)
        sunspots = np.maximum(0, cycle + noise)
        return pd.DataFrame({"sunspots": sunspots}, index=dates)


def load_energy() -> pd.DataFrame:
    """Load synthetic hourly energy demand dataset."""
    np.random.seed(42)
    hours = 24 * 365
    dates = pd.date_range(start="2023-01-01", periods=hours, freq="h")
    hour_of_day = dates.hour
    day_of_year = dates.dayofyear

    base = 1000
    daily_pattern = 200 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi / 2)
    yearly_pattern = 150 * np.cos(2 * np.pi * day_of_year / 365)
    trend = np.linspace(0, 100, hours)
    noise = np.random.normal(0, 50, hours)

    demand = base + daily_pattern + yearly_pattern + trend + noise
    return pd.DataFrame({"energy_demand": demand}, index=dates)


def load_synthetic_trend(n: int = 500, slope: float = 0.5, noise: float = 10) -> pd.DataFrame:
    """Generate a synthetic time series with trend."""
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    values = slope * np.arange(n) + np.random.normal(0, noise, n) + 100
    return pd.DataFrame({"value": values}, index=dates)


def load_synthetic_seasonal(
    n: int = 730, period: int = 7, amplitude: float = 50, noise: float = 10
) -> pd.DataFrame:
    """Generate a synthetic time series with seasonality."""
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    seasonal = amplitude * np.sin(2 * np.pi * np.arange(n) / period)
    trend = 0.1 * np.arange(n)
    values = 200 + seasonal + trend + np.random.normal(0, noise, n)
    return pd.DataFrame({"value": values}, index=dates)
