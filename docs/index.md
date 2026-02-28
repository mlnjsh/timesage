# TimeSage Documentation

### The Wise Time Series Library

**Beautiful EDA | All Models | Plain-English Interpretation**

---

## Table of Contents

1. [Getting Started](getting-started.md) -- Installation, first forecast, core concepts
2. [User Guide](user-guide.md) -- Detailed tutorials for every feature
3. [API Reference](api-reference.md) -- Complete reference for all classes and functions
4. [Models Guide](models.md) -- All forecasting models explained
5. [Feature Engineering](features.md) -- Creating powerful features for ML models
6. [Interpretation Engine](interpretation.md) -- Understanding your results in plain English
7. [Plotting & Themes](plotting.md) -- Beautiful visualizations and custom themes
8. [Examples](examples.md) -- Real-world use cases and recipes
9. [FAQ](faq.md) -- Frequently asked questions

---

## Quick Links

```bash
pip install timesage-ts
```

```python
import timesage as ts

# Create a time series
series = ts.TimeSeries(df, target="sales", time="date")

# Full automated EDA
series.eda()

# Forecast with interpretation
result = series.forecast(horizon=30, model="auto")
result.interpret()
```

---

## What Makes TimeSage Different?

1. **3-line forecasting** -- No boilerplate, no configuration hell
2. **Plain-English interpretation** -- Every result explained like a data scientist would
3. **Beautiful by default** -- Custom `sage` theme with 4 variants
4. **All models, one API** -- Statistical (ARIMA, ETS, Theta) + ML (XGBoost, LightGBM, RF) + AutoML
5. **Smart EDA** -- One command gives you statistics, stationarity, seasonality, outliers, and plots
6. **Feature engineering pipeline** -- Lags, rolling windows, temporal features in one call
