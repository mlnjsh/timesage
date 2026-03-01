"""Built-in datasets for quick experimentation."""

from timesage.datasets.loader import (
    list_datasets,
    load_airline,
    load_sunspots,
    load_energy,
    load_synthetic_trend,
    load_synthetic_seasonal,
)

__all__ = [
    "list_datasets",
    "load_airline",
    "load_sunspots",
    "load_energy",
    "load_synthetic_trend",
    "load_synthetic_seasonal",
]
