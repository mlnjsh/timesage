<p align="center">
  <h1 align="center">TimeSage</h1>
  <p align="center"><em>The Wise Time Series Library</em></p>
  <p align="center">
    Beautiful EDA &middot; All Models &middot; Plain-English Interpretation
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/timesage-ts/">
    <img src="https://img.shields.io/pypi/v/timesage-ts?color=2E86AB&style=flat-square" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/timesage-ts/">
    <img src="https://img.shields.io/pypi/pyversions/timesage-ts?color=A23B72&style=flat-square" alt="Python versions">
  </a>
  <a href="https://github.com/mlnjsh/timesage/blob/master/LICENSE">
    <img src="https://img.shields.io/pypi/l/timesage-ts?color=2BA84A&style=flat-square" alt="License">
  </a>
</p>

---

TimeSage makes time series forecasting effortless. Go from raw data to a forecast with plain-English interpretation in just a few lines of Python.

## Features

- **3-Line Forecasting** -- Load data, call `forecast()`, done. TimeSage handles model selection, feature engineering, and evaluation automatically.
- **Plain-English Interpretation** -- Call `result.interpret()` and get a human-readable explanation of accuracy, error patterns, key drivers, and forecast direction.
- **Beautiful Plots** -- Publication-ready visualizations with a custom Sage theme. Decomposition plots, forecast overlays, diagnostics, and more.
- **All Models, One API** -- ARIMA, ETS, Theta, Random Forest, XGBoost, and LightGBM all share the same `series.forecast(model="...")` interface.
- **Smart EDA** -- Automated exploratory data analysis with `series.eda()`: stationarity tests, seasonality detection, decomposition, and descriptive statistics.
- **Automatic Feature Engineering** -- ML models get lag features, rolling statistics, and calendar features created automatically. No manual work required.

## Installation

### Basic (statistical models + Random Forest)

```bash
pip install timesage-ts
```

### Full (adds XGBoost, LightGBM, Plotly)

```bash
pip install timesage-ts[full]
```

### Optional extras

```bash
pip install timesage-ts[ml]           # XGBoost + LightGBM only
pip install timesage-ts[interactive]  # Plotly interactive charts
```

**Requirements:** Python 3.9+

## Quick Start

```python
import timesage as ts

# 1. Create a TimeSeries from your data
series = ts.TimeSeries(df, target="sales", time="date", freq="D")

# 2. Explore your data
series.eda()

# 3. Forecast the next 30 days (auto-selects the best model)
result = series.forecast(horizon=30)

# 4. Get a plain-English interpretation
result.interpret()

# 5. Plot the forecast
result.plot()
```

### What `interpret()` prints

```
Forecast Interpretation
=======================
Overall accuracy : Excellent (MAPE = 3.2%)
Error consistency: Errors are uniform -- no major outlier spikes.
vs. Naive baseline: Model is 2.4x better than a naive repeat-last-value forecast.
Forecast trend  : Upward -- the series is predicted to increase over the horizon.
```

## Available Models

| Model | Key | Type | Install |
|-------|-----|------|---------|
| ARIMA (Auto) | `"arima"` | Statistical | Core |
| ETS (Holt-Winters) | `"ets"` | Statistical | Core |
| Theta | `"theta"` | Statistical | Core |
| Random Forest | `"rf"` | Machine Learning | Core |
| XGBoost | `"xgboost"` | Machine Learning | `pip install timesage-ts[ml]` |
| LightGBM | `"lightgbm"` | Machine Learning | `pip install timesage-ts[ml]` |
| Auto (best of all) | `"auto"` | Automatic | Core |

Use any model with the same interface:

```python
result = series.forecast(horizon=30, model="arima")
result = series.forecast(horizon=30, model="xgboost")
result = series.forecast(horizon=30, model="auto")     # default
```

### Comparing models

```python
comparison = series.compare_models(test_size=0.2)
```

This runs every available model, evaluates each on a held-out test set, and returns a ranked DataFrame with MAE, RMSE, MAPE, and training time.

## EDA in One Line

```python
series.eda()
```

This runs:
- Descriptive statistics (mean, std, min, max, skew, kurtosis)
- Stationarity testing (Augmented Dickey-Fuller)
- Seasonality detection (autocorrelation analysis)
- Trend-seasonal decomposition
- Distribution and time plot visualizations

## Documentation

Full documentation is available in the [`docs/`](https://github.com/mlnjsh/timesage/tree/master/docs) folder:

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, first forecast, core workflow |
| [User Guide](docs/user-guide.md) | Detailed tutorials and common workflows |
| [Models](docs/models.md) | Deep dive into every forecasting model |
| [Features](docs/features.md) | Feature engineering and data handling |
| [Plotting](docs/plotting.md) | Visualization guide and theme customization |
| [Interpretation](docs/interpretation.md) | How plain-English interpretation works |
| [API Reference](docs/api-reference.md) | Complete class and method reference |
| [Examples](docs/examples.md) | Real-world use cases with full code |
| [FAQ](docs/faq.md) | Frequently asked questions |

## Color Palette

TimeSage uses a carefully chosen color palette for all visualizations:

| Color | Hex | Usage |
|-------|-----|-------|
| ![#2E86AB](https://placehold.co/15x15/2E86AB/2E86AB.png) Ocean Blue | `#2E86AB` | Primary -- forecasts, main series |
| ![#A23B72](https://placehold.co/15x15/A23B72/A23B72.png) Berry | `#A23B72` | Secondary -- actuals, comparisons |
| ![#F18F01](https://placehold.co/15x15/F18F01/F18F01.png) Amber | `#F18F01` | Accent -- confidence intervals, highlights |
| ![#2BA84A](https://placehold.co/15x15/2BA84A/2BA84A.png) Sage Green | `#2BA84A` | Success -- good metrics, positive trends |
| ![#E63946](https://placehold.co/15x15/E63946/E63946.png) Coral Red | `#E63946` | Danger -- warnings, poor metrics |

Switch themes with:

```python
import timesage as ts
ts.set_theme("sage")     # Default sage theme
ts.set_theme("dark")     # Dark mode
ts.set_theme("minimal")  # Clean minimal
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Milan Amrut Joshi** ([mlnjsh@gmail.com](mailto:mlnjsh@gmail.com))

GitHub: [github.com/mlnjsh/timesage](https://github.com/mlnjsh/timesage)
