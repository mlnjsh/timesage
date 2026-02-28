# TimeSage User Guide

A practical, tutorial-style guide to every feature in the TimeSage time series library.

Throughout this guide, we use the convention:

```python
import timesage as ts
import pandas as pd
import numpy as np
```

---

## 1. Loading Data

TimeSage works with time series data through the `TimeSeries` class. There are several ways to load your data.

### From a DataFrame with a date column

If your DataFrame has a column containing dates (strings or datetime objects), pass it along with the name of that column:

```python
df = pd.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    "sales": [100, 120, 95, 130, 110]
})

series = ts.TimeSeries(df, date_column="date", value_column="sales")
print(series)
# TimeSeries: 5 observations from 2024-01-01 to 2024-01-05 (freq=D)
```

TimeSage automatically parses the date strings, sets them as the index, and infers the frequency.

### From a DataFrame with a DatetimeIndex

If your DataFrame already uses a `DatetimeIndex`, you only need to specify which column holds the values:

```python
dates = pd.date_range("2024-01-01", periods=100, freq="D")
df = pd.DataFrame({"revenue": np.random.randn(100).cumsum() + 500}, index=dates)

series = ts.TimeSeries(df, value_column="revenue")
print(series)
# TimeSeries: 100 observations from 2024-01-01 to 2024-04-09 (freq=D)
```

### From a pandas Series

You can pass a pandas Series directly. If it has a DatetimeIndex, TimeSage uses it. If not, TimeSage will create a default daily index starting from today:

```python
# With DatetimeIndex
values = pd.Series(
    [10, 20, 15, 25, 30],
    index=pd.date_range("2024-06-01", periods=5, freq="M"),
    name="monthly_totals"
)
series = ts.TimeSeries(values)
print(series)
# TimeSeries: 5 observations from 2024-06-30 to 2024-10-31 (freq=M)

# Without DatetimeIndex (daily index is created automatically)
raw_values = pd.Series([10, 20, 15, 25, 30], name="readings")
series = ts.TimeSeries(raw_values)
print(series)
# TimeSeries: 5 observations from 2026-02-28 to 2026-03-04 (freq=D)
```

### Handling different date formats

TimeSage relies on pandas for date parsing, which handles most standard formats automatically. For unusual formats, parse the dates yourself before passing them in:

```python
# European format (DD/MM/YYYY) -- parse explicitly
df = pd.DataFrame({
    "date": ["28/01/2024", "29/01/2024", "30/01/2024"],
    "value": [100, 105, 98]
})
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

series = ts.TimeSeries(df, date_column="date", value_column="value")
print(series)
# TimeSeries: 3 observations from 2024-01-28 to 2024-01-30 (freq=D)
```

```python
# Unix timestamps
df = pd.DataFrame({
    "timestamp": [1704067200, 1704153600, 1704240000],
    "value": [42, 45, 41]
})
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

series = ts.TimeSeries(df, date_column="timestamp", value_column="value")
```

### Specifying frequency manually

If TimeSage cannot infer the frequency (for example, with irregular data or gaps), you can specify it explicitly:

```python
# Data with a missing day -- frequency inference might fail
df = pd.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-04"],  # Jan 3 is missing
    "value": [10, 20, 40]
})

series = ts.TimeSeries(df, date_column="date", value_column="value", freq="D")
# TimeSage will fill the gap and set freq=D
print(series)
# TimeSeries: 4 observations from 2024-01-01 to 2024-01-04 (freq=D)
# Note: missing dates are forward-filled or interpolated depending on config
```

Common frequency codes:

| Code | Meaning     |
|------|-------------|
| `D`  | Daily       |
| `W`  | Weekly      |
| `M`  | Monthly     |
| `Q`  | Quarterly   |
| `H`  | Hourly      |
| `B`  | Business day|

---

## 2. Exploratory Data Analysis (EDA)

TimeSage provides a one-command EDA workflow that gives you a complete overview of your time series.

### Running full automated EDA with `series.eda()`

```python
# Load a built-in dataset
series = ts.datasets.load("airline")
print(series)
# TimeSeries: 144 observations from 1949-01-01 to 1960-12-01 (freq=MS)

# Run full EDA
report = series.eda()
```

Expected output:

```
================================================================================
                    TIME SERIES EXPLORATORY DATA ANALYSIS
================================================================================

BASIC STATISTICS
----------------
  Observations : 144
  Date Range   : 1949-01-01 to 1960-12-01
  Frequency    : MS (Month Start)
  Mean         : 280.30
  Std Dev      : 119.97
  Min          : 104.00
  Max          : 622.00
  Skewness     : 0.58
  Kurtosis     : -0.71
  Missing      : 0 (0.00%)

STATIONARITY ANALYSIS
---------------------
  ADF Test     : p=0.9919 (NON-STATIONARY)
  KPSS Test    : p=0.0100 (NON-STATIONARY)
  Conclusion   : NON-STATIONARY
  Recommendation: Consider differencing or detrending before modeling.

SEASONALITY DETECTION
---------------------
  Seasonal     : Yes
  Period       : 12
  ACF Peaks    : [12, 24, 36]

DECOMPOSITION
-------------
  Model        : additive
  Trend        : extracted (see decomposition plot)
  Seasonal     : period=12
  Residual     : mean=-0.02, std=17.14

OUTLIERS
--------
  Detected     : 2 potential outliers
  Indices      : [1960-07-01, 1960-08-01]
  Method       : IQR (1.5x)

================================================================================
```

The `eda()` method returns an `EDAReport` object that you can inspect programmatically:

```python
report.statistics       # dict of descriptive stats
report.stationarity     # StationarityResult object
report.seasonality      # SeasonalityResult object
report.outliers         # list of outlier dates
report.decomposition    # DecompositionResult object
```

### Custom analysis: individual EDA functions

You do not have to run the full EDA. Each analysis step is available on its own.

#### `describe()` -- Summary statistics

```python
stats = series.describe()
print(stats)
# Returns a dict:
# {
#     "count": 144,
#     "mean": 280.30,
#     "std": 119.97,
#     "min": 104.0,
#     "25%": 180.0,
#     "50%": 265.5,
#     "75%": 360.5,
#     "max": 622.0,
#     "skewness": 0.58,
#     "kurtosis": -0.71,
#     "missing": 0,
#     "missing_pct": 0.0
# }
```

#### `test_stationarity()` -- ADF and KPSS tests

```python
result = series.test_stationarity()
print(result)
# StationarityResult(
#     adf_statistic=-0.8787,
#     adf_pvalue=0.9919,
#     kpss_statistic=1.0527,
#     kpss_pvalue=0.01,
#     conclusion="NON-STATIONARY"
# )
```

#### `detect_seasonality()` -- ACF-based seasonality detection

```python
result = series.detect_seasonality()
print(result)
# SeasonalityResult(
#     is_seasonal=True,
#     period=12,
#     acf_peaks=[12, 24, 36]
# )
```

#### `decompose()` -- Trend-seasonal decomposition

```python
result = series.decompose(model="additive", period=12)
print(result.trend.head())
print(result.seasonal.head())
print(result.residual.head())
```

---

## 3. Visualization

TimeSage provides several plotting methods, all available directly from the `TimeSeries` object.

### Basic plotting: `series.plot()`

```python
series = ts.datasets.load("airline")

# Simple time series plot
series.plot()
```

This creates a clean line chart with the date on the x-axis and values on the y-axis. The plot title defaults to the series name.

```python
# Customize the plot
series.plot(
    title="Monthly Airline Passengers",
    ylabel="Passengers (thousands)",
    figsize=(12, 5)
)
```

### With trend overlay: `series.plot(show_trend=True)`

```python
series.plot(show_trend=True)
```

This overlays a smoothed trend line (computed via STL or moving average) on top of the raw data. The trend appears as a thicker, semi-transparent line so you can see the underlying trajectory.

```python
# Control the trend smoothing window
series.plot(show_trend=True, trend_window=24)
```

### With outlier highlighting: `series.plot(show_outliers=True)`

```python
series.plot(show_outliers=True)
```

Outliers are detected using the IQR method and marked with red circles on the plot. This is helpful for quickly spotting anomalies.

```python
# Combine trend and outliers
series.plot(show_trend=True, show_outliers=True, title="Airline Data with Trend and Outliers")
```

### ACF/PACF: `series.plot_acf(lags=40)`

The autocorrelation function (ACF) and partial autocorrelation function (PACF) plots are essential for understanding the correlation structure of your data and for selecting ARIMA parameters.

```python
# ACF plot
series.plot_acf(lags=40)

# PACF plot
series.plot_pacf(lags=40)

# Both side by side
series.plot_acf_pacf(lags=40)
```

The blue shaded region shows the 95% confidence interval. Spikes outside this region indicate statistically significant correlations at those lags.

### Decomposition: `series.plot_decomposition()`

```python
series.plot_decomposition(model="additive", period=12)
```

This produces a four-panel plot showing:
1. **Observed** -- the original data
2. **Trend** -- the long-term movement
3. **Seasonal** -- the repeating pattern
4. **Residual** -- what remains after removing trend and seasonal components

```python
# Multiplicative decomposition (better for data with increasing variance)
series.plot_decomposition(model="multiplicative", period=12)
```

### Themes: sage (default), dark, minimal

TimeSage ships with three visual themes you can apply globally or per-plot:

```python
# Set theme globally
ts.set_theme("dark")
series.plot()

# Or per-plot
series.plot(theme="minimal")
series.plot(theme="sage")       # the default green-tinted theme
series.plot(theme="dark")       # dark background, light lines
```

Theme descriptions:

| Theme     | Background  | Style                                  |
|-----------|-------------|----------------------------------------|
| `sage`    | White       | Soft green accents, clean gridlines    |
| `dark`    | Dark gray   | Light lines on dark background         |
| `minimal` | White       | No gridlines, minimal chrome           |

---

## 4. Stationarity Testing

### What is stationarity and why it matters

A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) do not change over time. Many forecasting models (ARIMA, ETS) assume stationarity or need it as a preprocessing step.

A non-stationary series has trends, changing variance, or structural breaks that must be addressed before modeling.

### ADF test explanation

The **Augmented Dickey-Fuller (ADF)** test checks whether a unit root is present in the series.

- **Null hypothesis (H0):** The series has a unit root (non-stationary).
- **Alternative (H1):** The series is stationary.
- **Interpretation:** If `p-value < 0.05`, reject H0 and conclude the series is stationary.

```python
result = series.test_stationarity()
print(f"ADF statistic: {result.adf_statistic:.4f}")
print(f"ADF p-value:   {result.adf_pvalue:.4f}")
# ADF statistic: -0.8787
# ADF p-value:   0.9919
# p > 0.05, so we cannot reject H0 -- the series is non-stationary
```

### KPSS test explanation

The **Kwiatkowski-Phillips-Schmidt-Shin (KPSS)** test is the reverse of ADF.

- **Null hypothesis (H0):** The series is stationary (or trend-stationary).
- **Alternative (H1):** The series has a unit root.
- **Interpretation:** If `p-value < 0.05`, reject H0 and conclude the series is non-stationary.

```python
print(f"KPSS statistic: {result.kpss_statistic:.4f}")
print(f"KPSS p-value:   {result.kpss_pvalue:.4f}")
# KPSS statistic: 1.0527
# KPSS p-value:   0.0100
# p < 0.05, so we reject H0 -- the series is non-stationary
```

### The 4 possible conclusions

TimeSage combines the ADF and KPSS tests to give one of four conclusions:

| ADF Result     | KPSS Result    | Conclusion               | Meaning                                                  |
|----------------|----------------|--------------------------|----------------------------------------------------------|
| Stationary     | Stationary     | `STATIONARY`             | Both tests agree the series is stationary.                |
| Non-stationary | Non-stationary | `NON-STATIONARY`         | Both tests agree the series is not stationary.            |
| Stationary     | Non-stationary | `TREND-STATIONARY`       | Stationary around a deterministic trend. Detrend it.      |
| Non-stationary | Stationary     | `DIFFERENCE-STATIONARY`  | A unit root is present. Differencing will make it stationary. |

```python
result = series.test_stationarity()
print(result.conclusion)
# "NON-STATIONARY"
```

### What to do when a series is non-stationary

```python
# Option 1: First-order differencing
diff_series = series.diff(1)
diff_result = diff_series.test_stationarity()
print(diff_result.conclusion)
# "STATIONARY" -- differencing removed the trend

# Option 2: Seasonal differencing (for seasonal non-stationarity)
seasonal_diff = series.diff(12)  # remove yearly seasonality
seasonal_result = seasonal_diff.test_stationarity()

# Option 3: Log transform + differencing (for multiplicative trends)
log_series = series.transform(np.log)
log_diff = log_series.diff(1)
```

Most forecasting methods in TimeSage handle differencing internally, so you usually do not need to difference manually before calling `forecast()`. But understanding stationarity helps you interpret model behavior and diagnostics.

---

## 5. Seasonality Detection

### How TimeSage detects seasonality (ACF peaks)

TimeSage detects seasonality by computing the autocorrelation function (ACF) and looking for periodic peaks. A peak at lag `k` means the series is correlated with itself `k` time steps ago. Regularly spaced peaks indicate a seasonal pattern.

```python
series = ts.datasets.load("airline")
result = series.detect_seasonality()
print(result)
# SeasonalityResult(
#     is_seasonal=True,
#     period=12,
#     acf_peaks=[12, 24, 36]
# )
```

### Understanding the output: period, peaks

- **`is_seasonal`**: `True` if significant periodic peaks were found in the ACF.
- **`period`**: The dominant seasonal period (the lag of the first significant peak). For monthly data with yearly seasonality, this is 12.
- **`acf_peaks`**: All detected peak lags. Peaks at [12, 24, 36] confirm a period-12 pattern (the correlation repeats every 12, 24, and 36 months).

```python
if result.is_seasonal:
    print(f"Seasonal period detected: {result.period}")
    print(f"The series repeats every {result.period} time steps.")
else:
    print("No significant seasonality detected.")
```

### Using detected seasonality in decomposition

The detected period flows naturally into decomposition and modeling:

```python
# Use the detected period for decomposition
seasonality = series.detect_seasonality()
series.plot_decomposition(period=seasonality.period)

# The forecast method also auto-detects seasonality
# but you can override it
forecast_result = series.forecast(horizon=24, seasonal_period=seasonality.period)
```

---

## 6. Feature Engineering

TimeSage can automatically generate a rich set of features from your time series. This is especially useful for machine learning models (Random Forest, XGBoost, LightGBM) that need tabular input.

### `create_features()` overview

```python
series = ts.datasets.load("airline")
features_df = series.create_features()
print(features_df.head())
```

Expected output:

```
            value  lag_1  lag_2  lag_3  ...  rolling_mean_7  rolling_std_7  month  day_of_week  is_weekend
1949-02-01  118    112    NaN    NaN   ...  NaN             NaN            2      2             0
1949-03-01  132    118    112    NaN   ...  NaN             NaN            3      2             0
1949-04-01  129    132    118    112   ...  NaN             NaN            4      5             1
...
```

By default, `create_features()` generates lag features, rolling window statistics, and calendar features. You can customize what gets created.

### Lag features: what they are, how to choose

Lag features are previous values of the series. `lag_1` is the value from one time step ago, `lag_7` is from seven steps ago, and so on.

```python
# Create only lag features
features_df = series.create_features(
    lags=[1, 2, 3, 6, 12],  # specific lags
)
print(features_df[["value", "lag_1", "lag_6", "lag_12"]].dropna().head())
```

```
            value  lag_1  lag_6  lag_12
1950-01-01  115    140    104    112
1950-02-01  126    115    118    118
1950-03-01  141    126    135    132
```

**How to choose lags:**
- For daily data: lags 1-7 (last week), 14, 21, 28, 365
- For monthly data: lags 1-3, 6, 12 (last year same month)
- For hourly data: lags 1, 24 (same hour yesterday), 168 (same hour last week)
- Rule of thumb: include lags matching the seasonal period

### Rolling window features: mean, std, min, max

Rolling window features capture recent trends and volatility:

```python
features_df = series.create_features(
    rolling_windows=[3, 6, 12],
    rolling_stats=["mean", "std", "min", "max"]
)
print(features_df.columns.tolist())
# [..., 'rolling_mean_3', 'rolling_std_3', 'rolling_min_3', 'rolling_max_3',
#  'rolling_mean_6', 'rolling_std_6', 'rolling_min_6', 'rolling_max_6',
#  'rolling_mean_12', 'rolling_std_12', 'rolling_min_12', 'rolling_max_12']
```

- **`rolling_mean_k`**: Average of the last `k` values. Smooths out noise, shows recent trend.
- **`rolling_std_k`**: Standard deviation of the last `k` values. Measures recent volatility.
- **`rolling_min_k`** / **`rolling_max_k`**: Range of recent values.

### Exponential weighted features

Exponential weighted statistics give more weight to recent observations. The `span` parameter controls how fast the weights decay:

```python
features_df = series.create_features(
    ewm_spans=[3, 12],
    ewm_stats=["mean", "std"]
)
print(features_df[["value", "ewm_mean_3", "ewm_mean_12"]].dropna().head())
```

```
            value  ewm_mean_3  ewm_mean_12
1949-04-01  129    126.43      121.87
1949-05-01  121    124.71      121.74
1949-06-01  135    129.36      123.78
```

A smaller span (e.g., 3) reacts faster to recent changes. A larger span (e.g., 12) is smoother and more stable.

### Difference and percent change

```python
features_df = series.create_features(
    diff_orders=[1, 12],       # first difference and seasonal difference
    pct_change_periods=[1, 12] # period-over-period percent change
)
print(features_df[["value", "diff_1", "diff_12", "pct_change_1", "pct_change_12"]].dropna().head())
```

```
            value  diff_1  diff_12  pct_change_1  pct_change_12
1950-02-01  126    11.0    8.0      0.0957        0.0678
1950-03-01  141    15.0    9.0      0.1190        0.0682
1950-04-01  135    -6.0    6.0     -0.0426        0.0465
```

- **`diff_1`**: Change from the previous period (first difference).
- **`diff_12`**: Change from 12 periods ago (seasonal difference for monthly data).
- **`pct_change_1`**: Percentage change from the previous period.
- **`pct_change_12`**: Year-over-year percentage change.

### Temporal/calendar features

Calendar features capture time-based patterns like weekday effects and seasonal cycles:

```python
features_df = series.create_features(
    calendar_features=True
)
print(features_df[["value", "month", "quarter", "day_of_week", "is_weekend", "day_of_year"]].head())
```

```
            value  month  quarter  day_of_week  is_weekend  day_of_year
1949-01-01  112    1      1        5            1           1
1949-02-01  118    2      1        1            0           32
1949-03-01  132    3      1        1            0           60
1949-04-01  129    4      2        4            0           91
1949-05-01  121    5      2        6            1           121
```

Available calendar features:
- `year`, `month`, `quarter`, `day_of_month`, `day_of_week`, `day_of_year`
- `week_of_year`, `is_weekend`, `is_month_start`, `is_month_end`
- `is_quarter_start`, `is_quarter_end`, `hour` (for sub-daily data)

### Custom feature configuration

Combine all feature types in a single call:

```python
features_df = series.create_features(
    lags=[1, 2, 3, 12],
    rolling_windows=[3, 6],
    rolling_stats=["mean", "std"],
    ewm_spans=[6],
    ewm_stats=["mean"],
    diff_orders=[1],
    pct_change_periods=[1, 12],
    calendar_features=True
)

print(f"Generated {features_df.shape[1]} features from {series.n_obs} observations")
# Generated 21 features from 144 observations

# Drop rows with NaN from lagging/rolling operations
features_clean = features_df.dropna()
print(f"Usable rows after dropping NaN: {features_clean.shape[0]}")
# Usable rows after dropping NaN: 131
```

---

## 7. Forecasting

### Basic forecast: `series.forecast(horizon=30)`

The simplest way to generate a forecast:

```python
series = ts.datasets.load("airline")

result = series.forecast(horizon=24)
print(result)
# ForecastResult(
#     model="auto",
#     horizon=24,
#     mae=15.32,
#     rmse=19.87,
#     mape=4.21,
#     test_size=36
# )
```

This automatically selects the best model, trains it, evaluates on a held-out test set, and generates 24 future predictions.

```python
# Access the forecast values
print(result.predictions.head())
#             forecast  lower_bound  upper_bound
# 1961-01-01   445.2     410.3       480.1
# 1961-02-01   421.8     385.7       457.9
# 1961-03-01   462.5     425.2       499.8

# Plot the forecast
result.plot()
```

### Choosing a model

TimeSage supports several forecasting models. Specify one with the `model` parameter:

```python
# Statistical models
result_arima = series.forecast(horizon=24, model="arima")
result_ets   = series.forecast(horizon=24, model="ets")
result_theta = series.forecast(horizon=24, model="theta")

# Machine learning models
result_rf    = series.forecast(horizon=24, model="rf")
result_xgb   = series.forecast(horizon=24, model="xgboost")
result_lgbm  = series.forecast(horizon=24, model="lightgbm")

# Automatic selection (default)
result_auto  = series.forecast(horizon=24, model="auto")
```

| Model       | Code         | Best For                                         |
|-------------|--------------|--------------------------------------------------|
| ARIMA       | `"arima"`    | Stationary data, strong autocorrelation           |
| ETS         | `"ets"`      | Data with clear trend and/or seasonality          |
| Theta       | `"theta"`    | Simple, robust baseline; good for competitions    |
| Random Forest | `"rf"`     | Complex nonlinear patterns, many features         |
| XGBoost     | `"xgboost"`  | High accuracy, handles missing values             |
| LightGBM    | `"lightgbm"` | Fast training, large datasets, categorical features|
| Auto        | `"auto"`     | Let TimeSage pick the best model automatically    |

### Understanding test_size parameter

The `test_size` parameter controls how many observations are held out for evaluation. The model is trained on everything except the last `test_size` observations, then evaluated on those held-out points.

```python
# Default: test_size is set automatically (roughly 20% of data)
result = series.forecast(horizon=12, test_size=36)
# The last 36 months are used for testing
# The model trains on months 1-108, tests on 109-144, then forecasts 12 ahead

# Smaller test size (more training data, less reliable evaluation)
result = series.forecast(horizon=12, test_size=12)

# Explicitly specify
result = series.forecast(horizon=12, test_size=24)
print(f"Train size: {144 - 24} observations")
print(f"Test size:  24 observations")
print(f"Forecast:   12 steps ahead")
```

### Confidence intervals

Every forecast includes prediction intervals:

```python
result = series.forecast(horizon=24, model="arima")

# Default: 95% confidence interval
print(result.predictions[["forecast", "lower_bound", "upper_bound"]].head())
#             forecast  lower_bound  upper_bound
# 1961-01-01   445.2     410.3       480.1
# 1961-02-01   421.8     385.7       457.9

# The interval widens as you forecast further into the future
print(result.predictions[["forecast", "lower_bound", "upper_bound"]].tail())
#             forecast  lower_bound  upper_bound
# 1962-11-01   510.3     420.1       600.5
# 1962-12-01   480.7     388.2       573.2
```

For machine learning models, confidence intervals are estimated using quantile regression or bootstrapping.

```python
# Plot with confidence intervals visible
result.plot(show_ci=True)
```

### When to use which model (guidelines)

- **Short series (< 50 observations):** Use `"ets"` or `"theta"`. ML models need more data.
- **Strong seasonality:** Use `"ets"` or `"arima"`. They handle seasonal patterns natively.
- **Nonlinear patterns:** Use `"xgboost"` or `"lightgbm"`. They can capture complex interactions.
- **Multiple external features:** Use ML models (`"rf"`, `"xgboost"`, `"lightgbm"`).
- **Quick baseline:** Use `"theta"`. It is simple but surprisingly competitive.
- **Not sure:** Use `"auto"` and let TimeSage decide.

---

## 8. Model Comparison

### `series.compare_models()` walkthrough

Instead of trying each model one at a time, compare them all at once:

```python
series = ts.datasets.load("airline")

comparison = series.compare_models()
print(comparison)
```

Expected output:

```
================================================================================
                         MODEL COMPARISON RESULTS
================================================================================

  Rank  Model      MAE     RMSE    MAPE(%)   Training Time
  ----  --------   -----   ------  --------  -------------
  1     ETS        13.41   17.25   3.87      0.42s
  2     ARIMA      15.32   19.87   4.21      1.83s
  3     LightGBM   16.05   20.11   4.55      0.31s
  4     XGBoost    16.89   21.34   4.78      0.52s
  5     Theta      18.21   22.56   5.12      0.08s
  6     RF         20.45   25.67   5.89      0.67s

  Best Model: ETS (MAPE: 3.87%)

================================================================================
```

### Reading the comparison table

The models are ranked by MAPE (Mean Absolute Percentage Error) by default. Each row shows:
- **Rank**: Position based on the ranking metric
- **Model**: The model name
- **MAE**: Mean Absolute Error (in the units of your data)
- **RMSE**: Root Mean Squared Error (in the units of your data, penalizes large errors)
- **MAPE(%)**: Mean Absolute Percentage Error (scale-independent, easy to interpret)
- **Training Time**: How long the model took to fit

### Interpreting MAPE, MAE, RMSE

- **MAE (Mean Absolute Error):** The average size of the errors, in the same units as your data. If MAE = 15.32 for airline passengers (thousands), the model is off by about 15,320 passengers on average.

- **RMSE (Root Mean Squared Error):** Similar to MAE but penalizes large errors more heavily. RMSE is always >= MAE. A big gap between RMSE and MAE means the model makes a few very large errors.

- **MAPE (Mean Absolute Percentage Error):** The average error as a percentage of the actual value. MAPE = 4.21% means the model is off by about 4.21% on average. Scale-independent, so it works for comparing across different datasets.

### Specifying which models to compare

```python
# Compare only specific models
comparison = series.compare_models(
    models=["arima", "ets", "xgboost"]
)

# Compare with specific test size
comparison = series.compare_models(
    models=["arima", "ets", "theta", "lightgbm"],
    test_size=24
)

# Rank by a different metric
comparison = series.compare_models(
    rank_by="rmse"  # or "mae", "mape"
)
```

You can also get the best model's result directly:

```python
best_result = comparison.best_result
print(f"Best model: {best_result.model_name}")
print(f"Best MAPE: {best_result.mape:.2f}%")

# Use the best model to forecast
forecast = best_result.forecast(horizon=24)
forecast.plot()
```

---

## 9. Understanding Results

### `result.interpret()` walkthrough

After generating a forecast, call `interpret()` to get a human-readable analysis of the results:

```python
series = ts.datasets.load("airline")
result = series.forecast(horizon=24, model="ets")

interpretation = result.interpret()
print(interpretation)
```

Expected output:

```
================================================================================
                        FORECAST INTERPRETATION
================================================================================

MODEL: ETS (Error-Trend-Seasonal)

ACCURACY ASSESSMENT
-------------------
  MAE  : 13.41       (average error: 13,410 passengers)
  RMSE : 17.25       (large-error-penalized: 17,250 passengers)
  MAPE : 3.87%       --> EXCELLENT accuracy (< 5%)
  MASE : 0.42        --> Beats naive baseline by 58%
  R2   : 0.96        --> Model explains 96% of variance

ERROR PATTERN ANALYSIS
----------------------
  RMSE/MAE Ratio: 1.29
  Interpretation: Errors are relatively consistent. No extreme outlier
                  predictions detected (ratio < 1.5 is good).

NAIVE BENCHMARK (MASE)
----------------------
  MASE = 0.42 means this model's errors are 42% the size of a naive
  "repeat last value" forecast. Values < 1.0 mean the model adds value
  over the simplest possible approach.

FORECAST DIRECTION
------------------
  The forecast predicts an UPWARD trend over the next 24 periods.
  Average forecasted value: 476.3 (vs. last observed: 432.0)
  Expected change: +10.3%

================================================================================
```

### Metrics explained

#### MAE (Mean Absolute Error)

The average of the absolute differences between predictions and actual values.

```
MAE = mean(|actual - predicted|)
```

Intuition: "On average, the forecast is off by this much."

#### RMSE (Root Mean Squared Error)

The square root of the average squared errors.

```
RMSE = sqrt(mean((actual - predicted)^2))
```

Intuition: Like MAE, but large errors count disproportionately more. If you care about avoiding big misses, pay attention to RMSE.

#### MAPE (Mean Absolute Percentage Error)

The average percentage error.

```
MAPE = mean(|actual - predicted| / |actual|) * 100
```

Intuition: "On average, the forecast is off by this percentage." Scale-independent, so 5% means the same thing regardless of whether you are forecasting 100 or 100,000.

#### MASE (Mean Absolute Scaled Error)

Compares your model's errors to a naive "repeat the last value" forecast.

```
MASE = MAE_model / MAE_naive
```

- MASE < 1.0: Your model beats the naive baseline.
- MASE = 1.0: Your model is as good as just repeating the last observation.
- MASE > 1.0: The naive method would have been better.

#### R-squared (R2)

The proportion of variance in the actual values explained by the model.

- R2 = 1.0: Perfect predictions.
- R2 = 0.0: The model is no better than predicting the mean.
- R2 < 0.0: The model is worse than predicting the mean.

### Accuracy levels

TimeSage classifies MAPE into intuitive accuracy levels:

| MAPE Range | Level      | Meaning                                      |
|------------|------------|----------------------------------------------|
| < 5%       | Excellent  | Very accurate, suitable for critical decisions|
| 5% - 10%   | Good       | Reliable for most planning purposes           |
| 10% - 20%  | Moderate   | Useful but treat with caution                 |
| > 20%      | Poor       | Consider alternative models or more data      |

### Error pattern analysis (RMSE/MAE ratio)

The ratio of RMSE to MAE reveals how consistent the errors are:

- **Ratio close to 1.0** (1.0 - 1.2): Errors are very consistent in size. The model does not make occasional large mistakes.
- **Ratio around 1.3 - 1.5**: Some variance in error size, but acceptable.
- **Ratio > 1.5**: The model makes a few very large errors. Investigate which time periods have the worst predictions.

```python
ratio = result.rmse / result.mae
print(f"RMSE/MAE ratio: {ratio:.2f}")
if ratio < 1.3:
    print("Errors are consistent -- good.")
elif ratio < 1.5:
    print("Some large errors, but acceptable.")
else:
    print("Warning: a few predictions are very far off. Investigate outliers.")
```

### Naive benchmark (MASE)

```python
print(f"MASE: {result.mase:.2f}")
if result.mase < 0.5:
    print("Model is more than 2x better than the naive approach.")
elif result.mase < 1.0:
    print("Model beats the naive approach.")
else:
    print("Model does not beat the naive approach. Consider a different model.")
```

### Feature importance (ML models)

For machine learning models (Random Forest, XGBoost, LightGBM), you can inspect which features matter most:

```python
result = series.forecast(horizon=24, model="xgboost")
interpretation = result.interpret()

# Feature importance is included in the interpretation for ML models
print(interpretation.feature_importance)
# {
#     "lag_12": 0.342,
#     "lag_1": 0.198,
#     "rolling_mean_6": 0.145,
#     "month": 0.089,
#     "lag_2": 0.076,
#     ...
# }
```

For statistical models (ARIMA, ETS, Theta), feature importance is not applicable.

### Forecast direction

The interpretation includes a summary of the forecast direction:

```python
print(interpretation.direction)
# "UPWARD"

print(interpretation.direction_detail)
# "The forecast predicts an UPWARD trend over the next 24 periods.
#  Average forecasted value: 476.3 (vs. last observed: 432.0)
#  Expected change: +10.3%"
```

---

## 10. Residual Diagnostics

After fitting a model, you should check the residuals (the differences between actual and predicted values on the test set). Good residuals should look like white noise -- no patterns, no bias, normally distributed.

### `diagnose_residuals()` function

```python
from timesage.interpret.diagnostics import diagnose_residuals

series = ts.datasets.load("airline")
result = series.forecast(horizon=24, model="ets")

diagnostics = diagnose_residuals(result)
print(diagnostics)
```

Expected output:

```
================================================================================
                        RESIDUAL DIAGNOSTICS
================================================================================

BIAS TEST (t-test)
------------------
  Mean residual : -0.87
  t-statistic   : -0.34
  p-value       : 0.7352
  Result        : PASS -- No significant bias detected.
  Interpretation: The model does not systematically over- or under-predict.

NORMALITY
---------
  Shapiro-Wilk  : W=0.976, p=0.6012 --> PASS (residuals are normal)
  Jarque-Bera   : JB=1.23, p=0.5412 --> PASS (residuals are normal)
  Interpretation: Residuals follow a normal distribution, supporting the
                  validity of confidence intervals.

AUTOCORRELATION (Ljung-Box)
---------------------------
  Ljung-Box Q   : 8.45
  p-value       : 0.3912
  Lags tested   : 10
  Result        : PASS -- No significant autocorrelation.
  Interpretation: Residuals are independent. The model has captured all
                  temporal patterns in the data.

HETEROSCEDASTICITY (Variance Ratio)
-----------------------------------
  First-half variance  : 142.3
  Second-half variance : 178.9
  Variance ratio       : 1.26
  Result               : PASS -- No significant heteroscedasticity.
  Interpretation: The spread of errors is consistent over time.

OVERALL ASSESSMENT
------------------
  4/4 tests passed.
  The residuals behave like white noise. The model is well-specified.

================================================================================
```

### Bias test (t-test)

Tests whether the mean of the residuals is significantly different from zero.

- **PASS** (p > 0.05): No systematic bias. The model does not consistently over- or under-predict.
- **FAIL** (p < 0.05): The model has a bias. It tends to predict too high or too low.

### Normality (Shapiro-Wilk, Jarque-Bera)

Tests whether residuals follow a normal distribution. This matters because confidence intervals assume normality.

- **Shapiro-Wilk**: Best for small samples (n < 50). Tests the overall shape of the distribution.
- **Jarque-Bera**: Tests specifically for skewness and kurtosis.
- **PASS** (p > 0.05): Residuals are approximately normal.
- **FAIL** (p < 0.05): Residuals are non-normal. Confidence intervals may be unreliable.

### Autocorrelation (Ljung-Box)

Tests whether residuals are correlated with each other at various lags. If residuals are autocorrelated, the model has missed some temporal pattern.

- **PASS** (p > 0.05): No significant autocorrelation. The model captured all temporal dependencies.
- **FAIL** (p < 0.05): Residuals are autocorrelated. Consider adding more lags or a different model.

### Heteroscedasticity (variance ratio)

Tests whether the variance of residuals changes over time. If it does, the model's errors are inconsistent, and confidence intervals are not uniformly reliable.

- **PASS** (ratio < 2.0): Variance is roughly constant.
- **FAIL** (ratio >= 2.0): Variance changes over time. Consider a log transform or a model that handles heteroscedasticity.

### What to do when diagnostics fail

| Diagnostic Failed    | What to Try                                                    |
|----------------------|----------------------------------------------------------------|
| Bias test            | Try a different model, add trend terms, or use differencing    |
| Normality            | Apply a Box-Cox or log transform to the series before modeling |
| Autocorrelation      | Increase ARIMA order (p,q), add more lag features for ML models|
| Heteroscedasticity   | Log-transform the data, or use a multiplicative model          |

```python
# Example: fixing non-normality with a log transform
log_series = series.transform(np.log)
result = log_series.forecast(horizon=24, model="ets")
diagnostics = diagnose_residuals(result)
# Re-check: normality test should now pass
```

---

## 11. Built-in Datasets

TimeSage ships with several classic time series datasets for learning and testing.

### airline

Monthly totals of international airline passengers (1949-1960). 144 observations with clear trend and multiplicative seasonality.

```python
series = ts.datasets.load("airline")
print(series)
# TimeSeries: 144 observations from 1949-01-01 to 1960-12-01 (freq=MS)
series.plot(title="Airline Passengers")
```

Best for: Learning decomposition, seasonal ARIMA, ETS.

### sunspots

Monthly mean sunspot numbers (1749-1983). A long series with a roughly 11-year cycle.

```python
series = ts.datasets.load("sunspots")
print(series)
# TimeSeries: 2820 observations from 1749-01-01 to 1983-12-01 (freq=MS)
series.plot(title="Monthly Sunspots")
```

Best for: Long-range periodicity, testing with large datasets.

### energy

Daily energy consumption data. Shows weekly seasonality and holiday effects.

```python
series = ts.datasets.load("energy")
print(series)
# TimeSeries: 1461 observations from 2018-01-01 to 2021-12-31 (freq=D)
series.plot(title="Daily Energy Consumption")
```

Best for: Daily forecasting, weekly patterns, ML models.

### synthetic_trend

A synthetically generated series with a linear upward trend and Gaussian noise. No seasonality.

```python
series = ts.datasets.load("synthetic_trend")
print(series)
# TimeSeries: 365 observations from 2024-01-01 to 2024-12-31 (freq=D)
series.plot(title="Synthetic Trend Data")
```

Best for: Testing trend detection and differencing.

### synthetic_seasonal

A synthetically generated series with a sine-wave seasonal pattern (period=30) and noise. No trend.

```python
series = ts.datasets.load("synthetic_seasonal")
print(series)
# TimeSeries: 365 observations from 2024-01-01 to 2024-12-31 (freq=D)
series.plot(title="Synthetic Seasonal Data")
```

Best for: Testing seasonality detection, seasonal decomposition.

### Loading and using each

All datasets follow the same interface:

```python
# Load by name
series = ts.datasets.load("airline")

# List all available datasets
available = ts.datasets.list_datasets()
print(available)
# ["airline", "sunspots", "energy", "synthetic_trend", "synthetic_seasonal"]

# Get metadata about a dataset
info = ts.datasets.info("airline")
print(info)
# {
#     "name": "airline",
#     "description": "Monthly international airline passengers, 1949-1960",
#     "observations": 144,
#     "frequency": "MS",
#     "source": "Box & Jenkins (1976)",
#     "features": ["trend", "seasonality", "multiplicative"]
# }
```

---

## 12. Advanced Usage

### Using AutoForecaster directly

The `series.forecast()` method is a convenience wrapper around the `AutoForecaster` class. For more control, use the class directly:

```python
from timesage.forecast import AutoForecaster

series = ts.datasets.load("airline")

forecaster = AutoForecaster(
    models=["arima", "ets", "xgboost"],
    test_size=36,
    seasonal_period=12,
    n_lags=24,
    feature_config={
        "lags": [1, 2, 3, 6, 12, 24],
        "rolling_windows": [3, 6, 12],
        "rolling_stats": ["mean", "std"],
        "calendar_features": True
    }
)

# Fit on the series
forecaster.fit(series)

# Access individual model results
for name, model_result in forecaster.results_.items():
    print(f"{name}: MAPE={model_result.mape:.2f}%, MAE={model_result.mae:.2f}")
# arima: MAPE=4.21%, MAE=15.32
# ets: MAPE=3.87%, MAE=13.41
# xgboost: MAPE=4.55%, MAE=16.05

# Get the best model
best = forecaster.best_model_
print(f"Best model: {best.name}")
# Best model: ets

# Generate a forecast with the best model
predictions = forecaster.predict(horizon=24)
print(predictions.head())
```

### Accessing model internals

Each fitted model exposes its internals for inspection:

```python
result = series.forecast(horizon=24, model="arima")

# Access the underlying fitted model
model = result.model_
print(model.summary())
# Prints the full ARIMA model summary (order, coefficients, AIC, BIC, etc.)

# For ARIMA: get the order
print(f"ARIMA order: {model.order}")
print(f"Seasonal order: {model.seasonal_order}")
# ARIMA order: (1, 1, 1)
# Seasonal order: (1, 1, 1, 12)

# Access fitted values on training data
print(result.fitted_values.head())

# Access residuals
print(result.residuals.head())
```

For ML models:

```python
result = series.forecast(horizon=24, model="xgboost")

# Access the XGBoost model
xgb_model = result.model_
print(type(xgb_model))
# <class 'xgboost.XGBRegressor'>

# Get feature importance
importance = pd.Series(
    xgb_model.feature_importances_,
    index=result.feature_names_
).sort_values(ascending=False)
print(importance.head(10))

# Access the training features DataFrame
print(result.X_train_.head())
print(result.X_test_.head())
```

### Custom feature pipelines

You can create a custom feature pipeline and pass it to ML models:

```python
from timesage.features import FeatureEngine

# Create a custom feature engine
engine = FeatureEngine(
    lags=[1, 7, 14, 28],
    rolling_windows=[7, 14, 28],
    rolling_stats=["mean", "std", "min", "max"],
    ewm_spans=[7, 28],
    ewm_stats=["mean"],
    diff_orders=[1, 7],
    pct_change_periods=[1, 7],
    calendar_features=True,
    custom_transforms={
        "log_value": lambda s: np.log(s),
        "squared_value": lambda s: s ** 2
    }
)

# Generate features
features_df = engine.transform(series)
print(f"Generated {features_df.shape[1]} features")

# Use the custom engine with AutoForecaster
forecaster = AutoForecaster(
    models=["xgboost", "lightgbm"],
    feature_engine=engine,
    test_size=30
)
forecaster.fit(series)
predictions = forecaster.predict(horizon=14)
```

### Combining with scikit-learn

TimeSage's feature engineering integrates with the scikit-learn ecosystem:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from timesage.features import FeatureEngine

series = ts.datasets.load("energy")

# Step 1: Generate features
engine = FeatureEngine(
    lags=[1, 7, 14, 28],
    rolling_windows=[7, 14],
    rolling_stats=["mean", "std"],
    calendar_features=True
)
features_df = engine.transform(series).dropna()

# Step 2: Split into X and y
y = features_df["value"]
X = features_df.drop(columns=["value"])

train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Step 3: Build a scikit-learn pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    ))
])

# Step 4: Train and evaluate
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
```

You can also wrap your scikit-learn model back into a TimeSage result for interpretation:

```python
from timesage.interpret import ResultInterpreter

interpreter = ResultInterpreter(
    actual=y_test,
    predicted=pd.Series(predictions, index=y_test.index),
    model_name="GradientBoosting (custom)",
    feature_names=X.columns.tolist(),
    feature_importance=dict(zip(
        X.columns,
        pipeline.named_steps["model"].feature_importances_
    ))
)

print(interpreter.interpret())
# Prints the same rich interpretation as result.interpret()
```

---

## Quick Reference

### Common workflows

```python
import timesage as ts

# Load data
series = ts.TimeSeries(df, date_column="date", value_column="value")

# Full EDA in one command
series.eda()

# Quick forecast
result = series.forecast(horizon=30)
result.plot()
result.interpret()

# Compare all models
comparison = series.compare_models()

# Use the best model
best = comparison.best_result
best.plot()
```

### Import paths

```python
import timesage as ts                              # Main entry point
from timesage.forecast import AutoForecaster       # Direct forecaster access
from timesage.features import FeatureEngine        # Custom feature engineering
from timesage.interpret import ResultInterpreter   # Wrap custom models
from timesage.interpret.diagnostics import diagnose_residuals  # Residual checks
```

### Cheat sheet: method signatures

```python
# TimeSeries methods
series.eda()                                         # Full EDA report
series.describe()                                    # Summary statistics
series.test_stationarity()                           # ADF + KPSS tests
series.detect_seasonality()                          # ACF peak detection
series.decompose(model="additive", period=12)        # Decomposition
series.create_features(lags=..., rolling_windows=...) # Feature engineering
series.forecast(horizon=30, model="auto")            # Forecast
series.compare_models(models=["arima", "ets", ...])  # Compare models

# Plotting methods
series.plot(show_trend=True, show_outliers=True)     # Time series plot
series.plot_acf(lags=40)                             # ACF plot
series.plot_pacf(lags=40)                            # PACF plot
series.plot_acf_pacf(lags=40)                        # Both ACF and PACF
series.plot_decomposition(model="additive")          # Decomposition plot

# Result methods
result.plot(show_ci=True)                            # Forecast plot
result.interpret()                                   # Human-readable analysis

# Datasets
ts.datasets.load("airline")                          # Load dataset
ts.datasets.list_datasets()                          # List all datasets
ts.datasets.info("airline")                          # Dataset metadata
```
