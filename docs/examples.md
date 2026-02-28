# Examples & Recipes

This page contains complete, runnable examples for common time series
forecasting workflows using TimeSage. Each example is self-contained and can
be copied into a script or notebook.

---

## Example 1: Retail Sales Forecasting

A complete walkthrough from raw CSV data to an interpreted forecast.

```python
import pandas as pd
import timesage as ts

# -------------------------------------------------------
# Step 1: Load your data
# -------------------------------------------------------
# Assume a CSV with columns: "date", "sales", "store_id"
df = pd.read_csv("retail_sales.csv")
print(df.head())
#         date  sales  store_id
# 0 2022-01-01   1200         1
# 1 2022-01-02   1350         1
# 2 2022-01-03   980          1
# ...

# -------------------------------------------------------
# Step 2: Create a TimeSeries object
# -------------------------------------------------------
series = ts.TimeSeries(df, target="sales", time="date", name="Daily Retail Sales")
print(series)
# TimeSeries(name='Daily Retail Sales', length=730, freq='daily', range=[2022-01-01 -> 2023-12-31])

# -------------------------------------------------------
# Step 3: Automated Exploratory Data Analysis
# -------------------------------------------------------
series.eda()
# Outputs:
#   - Summary statistics table (mean, std, skewness, kurtosis, missing %)
#   - Stationarity tests (ADF + KPSS) with conclusion
#   - Seasonality detection with period identification
#   - Time series plot with trend line and outlier markers
#   - ACF/PACF plots

# -------------------------------------------------------
# Step 4: Deeper stationarity analysis
# -------------------------------------------------------
stationarity = series.test_stationarity()
# Prints a rich table:
#   ADF test: statistic, p-value, stationary?
#   KPSS test: statistic, p-value, stationary?
#   Conclusion: "NON-STATIONARY -- Both tests agree the series has a unit root."

# -------------------------------------------------------
# Step 5: Detect seasonality
# -------------------------------------------------------
seasonality = series.detect_seasonality()
# "Seasonality detected! Primary period: 7"
# (7-day weekly cycle is common in retail)

# -------------------------------------------------------
# Step 6: Compare all available models
# -------------------------------------------------------
comparison = series.compare_models(test_size=0.2)
# Prints a ranked table:
#   Rank | Model         | MAE    | RMSE   | MAPE   | Time (s)
#   1    | RandomForest  | 45.23  | 62.11  | 3.82%  | 1.24
#   2    | XGBoost       | 47.10  | 63.89  | 4.01%  | 0.89
#   3    | ARIMA         | 52.67  | 71.34  | 4.45%  | 3.12
#   ...
#   Winner: RandomForest (MAPE: 3.82%)

# -------------------------------------------------------
# Step 7: Forecast with the best model
# -------------------------------------------------------
result = series.forecast(horizon=30, model="rf")
# Prints plain-English interpretation:
#   Overall Accuracy: [Excellent] ... average error of 3.8%
#   Error Pattern: consistent errors (RMSE/MAE ratio = 1.15)
#   Benchmark: outperforms naive forecast by 62%
#   Key Drivers: 'lag_1' (35.2%), 'roll_mean_7' (22.1%), ...
#   Forecast Direction: UPWARD trend of +5.3% over the horizon

# -------------------------------------------------------
# Step 8: Access results programmatically
# -------------------------------------------------------
print(result.metrics)
# {'MAE': 45.23, 'RMSE': 62.11, 'MSE': 3857.7, 'MedAE': 38.1,
#  'MAPE': 3.82, 'R2': 0.91, 'MASE': 0.38}

print(result.forecast.head())
# 2024-01-01    1312.5
# 2024-01-02    1289.3
# 2024-01-03    1145.7
# ...

# -------------------------------------------------------
# Step 9: Plot the forecast
# -------------------------------------------------------
result.plot()
# Shows: training data, test data, forecast, and confidence interval band

# -------------------------------------------------------
# Step 10: Get a one-line summary DataFrame
# -------------------------------------------------------
print(result.summary())
#               MAE   RMSE    MSE  MedAE  MAPE    R2  MASE
# RandomForest  45.23  62.11  3857.7  38.1  3.82  0.91  0.38
```

---

## Example 2: Stock Price Analysis

Analyzing stock price data for stationarity, patterns, and short-term
forecasting.

```python
import pandas as pd
import timesage as ts

# -------------------------------------------------------
# Load stock data (e.g., from Yahoo Finance CSV export)
# -------------------------------------------------------
df = pd.read_csv("AAPL.csv", parse_dates=["Date"], index_col="Date")
series = ts.TimeSeries(df, target="Close", name="AAPL Closing Price")
print(series)
# TimeSeries(name='AAPL Closing Price', length=252, freq='business_daily', range=[2023-01-03 -> 2023-12-29])

# -------------------------------------------------------
# Test stationarity (stock prices are typically non-stationary)
# -------------------------------------------------------
results = series.test_stationarity()
# Conclusion: "NON-STATIONARY -- Both tests agree the series has a unit root."
# This is expected -- stock prices are a random walk with drift.

# -------------------------------------------------------
# Check enhanced statistics
# -------------------------------------------------------
print(series.describe())
# count     252.000000
# mean      178.432100
# std        12.345678
# min       150.120000
# ...
# skewness    0.234500
# kurtosis   -0.567800
# missing     0.000000
# missing_pct 0.000000
# cv          0.069200

# -------------------------------------------------------
# Detect seasonal patterns
# -------------------------------------------------------
seasonality = series.detect_seasonality()
# "No significant seasonality detected."
# (Stock prices rarely show clean seasonality)

# -------------------------------------------------------
# Decompose the series
# -------------------------------------------------------
series.plot_decomposition(model="additive")
# Shows: observed, trend, seasonal (if any), and residual components

# -------------------------------------------------------
# Forecast with ARIMA (best for random-walk-like series)
# -------------------------------------------------------
result = series.forecast(horizon=10, model="arima")
# ARIMA will find that d=1 is needed (differencing for non-stationarity)

# -------------------------------------------------------
# Examine confidence intervals (important for stock prices!)
# -------------------------------------------------------
print(result.confidence_lower.head())
print(result.confidence_upper.head())
# Wide intervals are expected -- stock prices are hard to predict

# -------------------------------------------------------
# Plot with ACF/PACF for deeper analysis
# -------------------------------------------------------
series.plot_acf(lags=40)
# High ACF values at all lags confirm non-stationarity
```

---

## Example 3: Energy Demand Prediction

Using the built-in energy dataset with an ML model for hourly demand forecasting.

```python
import timesage as ts
from timesage.datasets.loader import load_energy

# -------------------------------------------------------
# Load built-in hourly energy demand dataset
# -------------------------------------------------------
energy_df = load_energy()
print(energy_df.head())
#                      energy_demand
# 2023-01-01 00:00:00       821.34
# 2023-01-01 01:00:00       785.12
# 2023-01-01 02:00:00       762.89
# ...

series = ts.TimeSeries(energy_df, target="energy_demand", name="Hourly Energy Demand")
print(series)
# TimeSeries(name='Hourly Energy Demand', length=8760, freq='hourly', range=[2023-01-01 -> 2023-12-31])

# -------------------------------------------------------
# Run EDA to understand the data
# -------------------------------------------------------
series.eda()
# The profiler will detect:
#   - Strong daily seasonality (period ~24)
#   - Non-stationarity from the trend component
#   - Summary statistics including skewness and kurtosis

# -------------------------------------------------------
# Create custom features for inspection
# -------------------------------------------------------
features = series.create_features(
    lags=[1, 24, 48, 168],       # 1 hour, 1 day, 2 days, 1 week
    windows=[24, 168],            # 1-day and 1-week rolling windows
    temporal=True,                # Include hour, dayofweek, month, etc.
)
print(features.columns.tolist())
# ['energy_demand', 'lag_1', 'lag_24', 'lag_48', 'lag_168',
#  'roll_mean_24', 'roll_std_24', 'roll_min_24', 'roll_max_24',
#  'roll_mean_168', 'roll_std_168', 'roll_min_168', 'roll_max_168',
#  'ewm_7', 'ewm_14', 'ewm_30', 'diff_1', 'diff_7',
#  'pct_change_1', 'pct_change_7',
#  'day_of_week', 'day_of_month', 'month', 'quarter',
#  'week_of_year', 'is_weekend', 'is_month_start', 'is_month_end']

# -------------------------------------------------------
# Forecast with XGBoost (good for complex patterns)
# -------------------------------------------------------
result = series.forecast(horizon=48, model="xgboost")
# Interpretation will highlight:
#   - Accuracy metrics on the test set
#   - Feature importance (likely lag_1 and lag_24 dominate)
#   - Whether the model captures the daily cycle

# -------------------------------------------------------
# Check feature importance
# -------------------------------------------------------
if result.feature_importance:
    for name, score in sorted(
        result.feature_importance.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"  {name}: {score:.1%}")
# Expected output:
#   lag_1: 42.3%
#   lag_7: 18.7%
#   roll_mean_7: 12.1%
#   dayofweek: 8.4%
#   month: 5.2%

# -------------------------------------------------------
# Plot the forecast
# -------------------------------------------------------
result.plot()
```

---

## Example 4: Seasonal Time Series (Airline Passengers)

The classic airline passengers dataset -- monthly data with trend and
multiplicative seasonality.

```python
import timesage as ts
from timesage.datasets.loader import load_airline

# -------------------------------------------------------
# Load the built-in airline passengers dataset
# -------------------------------------------------------
airline_df = load_airline()
print(airline_df.head())
#             passengers
# 1949-01-01         112
# 1949-02-01         118
# 1949-03-01         132
# ...

series = ts.TimeSeries(airline_df, target="passengers", name="Airline Passengers")
print(series)
# TimeSeries(name='Airline Passengers', length=144, freq='monthly', range=[1949-01-01 -> 1960-12-01])

# -------------------------------------------------------
# Seasonal decomposition
# -------------------------------------------------------
series.plot_decomposition(model="additive", period=12)
# Decomposes into:
#   - Trend: upward growth over the 12-year period
#   - Seasonal: 12-month repeating pattern (peak in summer)
#   - Residual: what is left after removing trend and season

# -------------------------------------------------------
# Detect seasonality automatically
# -------------------------------------------------------
seasonality = series.detect_seasonality()
# "Seasonality detected! Primary period: 12"

# -------------------------------------------------------
# Forecast with ETS (designed for trend + seasonality)
# -------------------------------------------------------
result = series.forecast(horizon=24, model="ets")

# -------------------------------------------------------
# Interpret the results
# -------------------------------------------------------
result.interpret()
# Prints:
#   Overall Accuracy: [Good] ... average error of 7.2%
#   Error Pattern: consistent errors
#   Benchmark: outperforms naive by 45%
#   Forecast Direction: UPWARD trend of +18.3%

# -------------------------------------------------------
# Compare against other approaches
# -------------------------------------------------------
comparison = series.compare_models(
    test_size=0.2,
    models=["arima", "ets", "theta", "rf"]
)
# See which model handles the seasonal pattern best

# -------------------------------------------------------
# Visualize
# -------------------------------------------------------
result.plot()
# Shows the historical data, test split, forecast, and confidence bands
```

---

## Example 5: Quick Model Comparison

When you want to quickly find the best model without manual experimentation.

```python
import pandas as pd
import timesage as ts

# Load your data
df = pd.read_csv("my_data.csv", parse_dates=["date"], index_col="date")
series = ts.TimeSeries(df, target="value")

# -------------------------------------------------------
# Compare all available models in one call
# -------------------------------------------------------
comparison = series.compare_models(test_size=0.2)

# The comparison table is printed automatically:
#   Rank | Model         | MAE    | RMSE   | MAPE   | Time (s)
#   1    | RandomForest  | 12.34  | 16.78  | 5.21%  | 0.98
#   2    | XGBoost       | 13.01  | 17.22  | 5.49%  | 0.67
#   3    | ETS           | 14.56  | 19.01  | 6.13%  | 0.12
#   4    | Theta         | 15.23  | 20.11  | 6.42%  | 0.08
#   5    | ARIMA         | 16.89  | 22.34  | 7.11%  | 2.45
#
#   Winner: RandomForest (MAPE: 5.21%)

# -------------------------------------------------------
# The result is a DataFrame -- use it programmatically
# -------------------------------------------------------
print(comparison.columns.tolist())
# ['Model', 'MAE', 'RMSE', 'MAPE', 'Time (s)']

best_model = comparison.iloc[0]["Model"]
print(f"Best model: {best_model}")

# -------------------------------------------------------
# Forecast with the winning model
# -------------------------------------------------------
# Map the display name back to the model key
model_key_map = {
    "ARIMA": "arima", "ETS": "ets", "Theta": "theta",
    "RandomForest": "rf", "XGBoost": "xgboost", "LightGBM": "lightgbm",
}
winner_key = model_key_map.get(best_model, "auto")
result = series.forecast(horizon=30, model=winner_key)

# -------------------------------------------------------
# Compare only specific models
# -------------------------------------------------------
subset = series.compare_models(
    test_size=0.2,
    models=["arima", "rf", "xgboost"]
)
```

---

## Example 6: Custom Feature Engineering

Using TimeSage's feature pipeline with your own models or with scikit-learn
directly.

```python
import pandas as pd
import timesage as ts
from timesage.features.lag import create_lag_features
from timesage.features.window import create_window_features
from timesage.features.temporal import create_temporal_features
from timesage.features.pipeline import FeaturePipeline
from sklearn.linear_model import Ridge

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
df = pd.read_csv("sales.csv", parse_dates=["date"], index_col="date")
series = ts.TimeSeries(df, target="sales")

# -------------------------------------------------------
# Option A: Use the high-level create_features method
# -------------------------------------------------------
features = series.create_features(
    lags=[1, 2, 3, 7, 14, 28],
    windows=[7, 14, 30],
    temporal=True,
    drop_na=True,
)
print(f"Shape: {features.shape}")
print(f"Columns: {features.columns.tolist()}")

# -------------------------------------------------------
# Option B: Use individual feature modules for fine control
# -------------------------------------------------------
# Start with the raw DataFrame
featured = df.copy()

# Add lag features with custom lags
featured = create_lag_features(
    featured, target="sales",
    lags=[1, 3, 7, 14, 30, 60, 90],  # Include longer lookbacks
    drop_na=False,
)

# Add rolling window features with custom functions
featured = create_window_features(
    featured, target="sales",
    windows=[7, 14, 30, 60],
    functions=["mean", "std", "min", "max", "median"],
    ewm_spans=[7, 30],
    drop_na=False,
)

# Add temporal features with cyclical encoding
featured = create_temporal_features(
    featured,
    cyclical=True,    # sin/cos encoding for periodic features
    holidays=True,    # US federal holiday indicators
)

# Drop NaN rows from lagging
featured = featured.dropna()

# -------------------------------------------------------
# Option C: Use the FeaturePipeline directly
# -------------------------------------------------------
pipeline = FeaturePipeline(
    target="sales",
    lags=[1, 2, 3, 7, 14, 28],
    windows=[7, 14, 30],
    temporal=True,
)
features = pipeline.transform(df, drop_na=True)

# -------------------------------------------------------
# Use features with any scikit-learn model
# -------------------------------------------------------
X = features.drop(columns=["sales"])
y = features["sales"]

# Train/test split (time-aware -- no shuffling!)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
print(f"RMSE: {mean_squared_error(y_test, predictions, squared=False):.2f}")
```

---

## Example 7: Residual Diagnostics

After generating a forecast, use residual diagnostics to assess model quality
and identify potential improvements.

```python
import timesage as ts
from timesage.interpret.diagnostics import diagnose_residuals
from timesage.interpret.explainer import explain_forecast
from timesage.interpret.report import generate_report

# -------------------------------------------------------
# Generate a forecast
# -------------------------------------------------------
df = ts.TimeSeries(your_dataframe, target="value")
result = df.forecast(horizon=30, model="arima")

# -------------------------------------------------------
# Option A: Quick residual diagnostics
# -------------------------------------------------------
if result.residuals is not None:
    diagnostics = diagnose_residuals(result.residuals)
    # Prints a detailed report with four tests:
    #
    #   PASS BIAS: Residuals have a mean of 0.0023 (p=0.8721).
    #     No significant bias detected.
    #
    #   ISSUE NORMALITY: Residuals are NOT normally distributed
    #     (Shapiro p=0.0012, JB p=0.0034).
    #     Prediction intervals may be unreliable.
    #
    #   PASS AUTOCORRELATION: Ljung-Box test p=0.3456.
    #     No significant autocorrelation in residuals.
    #
    #   PASS HETEROSCEDASTICITY: Variance ratio: 1.34.
    #     Residual variance is relatively stable over time.
    #
    #   Summary: Issues detected: non-normal distribution.
    #     See individual diagnostics above for recommendations.

# -------------------------------------------------------
# Option B: Access diagnostic results programmatically
# -------------------------------------------------------
diag = diagnose_residuals(result.residuals, verbose=False)

# Check each test
print(f"Biased? {diag['bias']['is_biased']}")
print(f"Normal? {diag['normality']['is_normal']}")
print(f"Autocorrelated? {diag['autocorrelation']['has_autocorrelation']}")
print(f"Heteroscedastic? {diag['heteroscedasticity']['is_heteroscedastic']}")
print(f"Summary: {diag['summary']}")

# -------------------------------------------------------
# Option C: Detailed forecast explanation
# -------------------------------------------------------
explanations = explain_forecast(result)
# Prints detailed sections:
#   OVERALL ACCURACY
#   ERROR PATTERN
#   BENCHMARK
#   PREDICTABILITY
#   FORECAST DIRECTION
#   KEY DRIVERS (for ML models)
#   PATTERN TYPE (for ML models)

# -------------------------------------------------------
# Option D: Full report (combines everything)
# -------------------------------------------------------
report = generate_report(result)
# Combines:
#   - Forecast analysis (accuracy, errors, benchmark)
#   - Residual diagnostics (bias, normality, autocorrelation, heteroscedasticity)
#   - Actionable recommendations

# -------------------------------------------------------
# Interpreting the diagnostics
# -------------------------------------------------------
# BIAS: If the model is biased (p < 0.05), it systematically over- or
# under-predicts. Fix: review feature engineering, try a different model,
# or add an intercept correction.
#
# NORMALITY: If residuals are not normal, confidence intervals are
# unreliable. Fix: apply a log or Box-Cox transform to the target,
# or use bootstrap-based intervals.
#
# AUTOCORRELATION: If residuals are autocorrelated, the model is missing
# temporal patterns. Fix: add more lag features, increase ARIMA order,
# or switch to a model that captures more complex dependencies.
#
# HETEROSCEDASTICITY: If variance changes over time, the model is more
# accurate in some periods than others. Fix: log-transform the target,
# use weighted regression, or consider GARCH for variance modeling.
```

---

## Example 8: Working with Built-in Datasets

TimeSage ships with several datasets for quick experimentation.

```python
from timesage.datasets.loader import (
    list_datasets,
    load_airline,
    load_sunspots,
    load_energy,
    load_synthetic_trend,
    load_synthetic_seasonal,
)
import timesage as ts

# -------------------------------------------------------
# See all available datasets
# -------------------------------------------------------
print(list_datasets())
# ['airline', 'sunspots', 'energy', 'synthetic_trend', 'synthetic_seasonal']

# -------------------------------------------------------
# Airline passengers (monthly, 1949-1960, 144 obs)
# -------------------------------------------------------
airline = load_airline()
series = ts.TimeSeries(airline, target="passengers")
series.eda()

# -------------------------------------------------------
# Sunspots (yearly, long history)
# -------------------------------------------------------
sunspots = load_sunspots()
series = ts.TimeSeries(sunspots, target="sunspots")
result = series.forecast(horizon=20, model="arima")

# -------------------------------------------------------
# Energy demand (hourly, 1 year, 8760 obs)
# -------------------------------------------------------
energy = load_energy()
series = ts.TimeSeries(energy, target="energy_demand")
# Good for testing ML models on large data
result = series.forecast(horizon=48, model="lightgbm")

# -------------------------------------------------------
# Synthetic trend (customizable)
# -------------------------------------------------------
trend_data = load_synthetic_trend(n=500, slope=0.5, noise=10)
series = ts.TimeSeries(trend_data, target="value")
# Useful for testing trend detection and ARIMA

# -------------------------------------------------------
# Synthetic seasonal (customizable)
# -------------------------------------------------------
seasonal_data = load_synthetic_seasonal(n=730, period=7, amplitude=50, noise=10)
series = ts.TimeSeries(seasonal_data, target="value")
# 2 years of daily data with a 7-day cycle
# Good for testing seasonality detection
seasonality = series.detect_seasonality()
# "Seasonality detected! Primary period: 7"
```

---

## Example 9: Plotting and Visualization

TimeSage provides themed plotting capabilities for exploratory analysis and
forecast visualization.

```python
import timesage as ts

# -------------------------------------------------------
# Set the visual theme
# -------------------------------------------------------
ts.set_theme("sage")     # Default warm theme
ts.set_theme("dark")     # Dark background
ts.set_theme("minimal")  # Clean, minimal style

# -------------------------------------------------------
# Basic time series plot
# -------------------------------------------------------
series.plot()

# -------------------------------------------------------
# Plot with trend line and outlier detection
# -------------------------------------------------------
series.plot(show_trend=True, show_outliers=True, figsize=(16, 6))

# -------------------------------------------------------
# ACF and PACF plots (side by side)
# -------------------------------------------------------
series.plot_acf(lags=40)

# -------------------------------------------------------
# Seasonal decomposition plot
# -------------------------------------------------------
series.plot_decomposition(model="additive", period=12)

# -------------------------------------------------------
# Forecast plot (after generating a forecast)
# -------------------------------------------------------
result = series.forecast(horizon=30, model="rf")
result.plot(figsize=(16, 6))
# Shows: historical data, test predictions, future forecast, confidence bands
```
