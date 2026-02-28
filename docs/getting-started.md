# Getting Started with TimeSage

Welcome to TimeSage -- a Python library for time series forecasting that makes
powerful models accessible through a clean, intuitive API. This guide will walk
you through installation, your first forecast, and the core workflow.

---

## Installation

### Basic Installation

Install TimeSage from PyPI:

```bash
pip install timesage-ts
```

This installs the core library with ARIMA, ETS, Theta, and Random Forest models.

### Machine Learning Extras

For XGBoost and LightGBM support, install the ML extras:

```bash
pip install timesage-ts[ml]
```

### Verify Installation

```python
import timesage as ts
print(ts.__version__)
```

### Requirements

- Python 3.10 or higher
- NumPy, pandas, scikit-learn, statsmodels (installed automatically)
- matplotlib for plotting (installed automatically)

---

## Quick Start: Your First Forecast

TimeSage is designed so you can go from raw data to a forecast in three lines:

```python
import timesage as ts

# Load a built-in dataset
series = ts.TimeSeries(ts.load_airline())

# Forecast the next 12 months
result = series.forecast(horizon=12)

# See the results
result.plot()
```

That is it. TimeSage automatically selects the best model, engineers features,
fits to your data, and produces a forecast with confidence intervals.

---

## Core Workflow

Every TimeSage project follows the same four-step pattern:

### Step 1 -- Load Your Data

TimeSage accepts pandas DataFrames, Series, or NumPy arrays:

```python
import pandas as pd
import timesage as ts

# From a CSV file
df = pd.read_csv("sales.csv", parse_dates=["date"])
series = ts.TimeSeries(df, target="revenue", time="date", freq="D")

# From a pandas Series
series = ts.TimeSeries(my_series, name="Daily Sales")

# From a built-in dataset
series = ts.TimeSeries(ts.load_airline(), name="Airline Passengers")
```

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | DataFrame, Series, or array | Your time series data |
| `target` | str or None | Column name for the target variable (if DataFrame) |
| `time` | str or None | Column name for the datetime index (if DataFrame) |
| `freq` | str or None | Frequency string (e.g., "D", "M", "H"). Auto-detected if None |
| `name` | str or None | A human-readable name for the series |

### Step 2 -- Explore Your Data

Before forecasting, understand your data:

```python
# Full exploratory data analysis
series.eda(show_plots=True)

# Descriptive statistics
series.describe()

# Stationarity testing (ADF test)
series.test_stationarity(verbose=True)

# Seasonality detection
series.detect_seasonality(max_lag=90, verbose=True)

# Decomposition into trend, seasonal, and residual components
series.decompose(model="additive", period=12)
```

### Step 3 -- Forecast

Run a forecast with sensible defaults or full control:

```python
# Automatic model selection (recommended for beginners)
result = series.forecast(horizon=30, model="auto")

# Use a specific model
result = series.forecast(horizon=30, model="arima")

# Customize test size and confidence level
result = series.forecast(
    horizon=30,
    model="rf",
    test_size=0.2,
    confidence=0.95
)
```

### Step 4 -- Interpret Results

TimeSage explains forecasts in plain English:

```python
# Plain-English interpretation
result.interpret()

# Quantitative summary with metrics
result.summary()

# Visual plot of forecast vs actuals
result.plot()

# Access raw metrics
print(result.metrics)
# {'MAE': 23.4, 'RMSE': 31.2, 'MAPE': 4.8, 'R2': 0.94, ...}
```

---

## Comparing Models

Not sure which model to use? Compare them all at once:

```python
comparison = series.compare_models(test_size=0.2)
```

This runs every available model on your data, evaluates each on a held-out
test set, and returns a ranked comparison with metrics.

You can also specify which models to compare:

```python
comparison = series.compare_models(
    test_size=0.2,
    models=["arima", "rf", "xgboost"]
)
```

---

## Built-in Datasets

TimeSage ships with several datasets for learning and experimentation:

```python
# Classic airline passengers (monthly, 1949-1960)
airline = ts.load_airline()

# Sunspot observations (monthly)
sunspots = ts.load_sunspots()

# Energy consumption data
energy = ts.load_energy()

# Synthetic data with a clear trend
trend = ts.load_synthetic_trend()

# Synthetic data with seasonal patterns
seasonal = ts.load_synthetic_seasonal()
```

---

## Setting the Visual Theme

TimeSage includes three built-in plot themes:

```python
# Sage green theme (default)
ts.set_theme("sage")

# Dark mode
ts.set_theme("dark")

# Clean minimal theme
ts.set_theme("minimal")
```

---

## What Next?

- **[User Guide](user-guide.md)** -- Detailed tutorials for common workflows
- **[API Reference](api-reference.md)** -- Complete reference for every class and method
- **[Models](models.md)** -- Deep dive into each forecasting model
- **[Examples](examples.md)** -- Real-world use cases with full code

---

## Getting Help

If you run into issues:

1. Check the [FAQ](faq.md) for common problems
2. Ensure your Python version is 3.10+
3. Ensure your data has no missing datetime indices
4. Open an issue on the GitHub repository with a minimal reproducible example
