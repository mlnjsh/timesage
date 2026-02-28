# Feature Engineering Guide

TimeSage automatically creates features for machine learning models (Random Forest, XGBoost, LightGBM). This guide explains every feature type, how they are computed, and when they matter.

---

## Overview

When you call a forecast with an ML model, TimeSage generates features from the raw time series before training. You can also create features manually:

    import timesage as ts
    series = ts.TimeSeries(ts.load_airline())
    features_df = series.create_features(lags=[1, 2, 3, 7, 14, 28], windows=[7, 14, 30], temporal=True)

Features fall into five categories:

| Category | Count | Description |
|----------|-------|-------------|
| Lag features | 6 (default) | Past values at specific offsets |
| Rolling statistics | 12 (default) | Aggregations over sliding windows |
| EWM features | 3 | Exponentially weighted means |
| Diff features | 4 | Changes and percent changes |
| Temporal features | 9 | Calendar-based indicators |

Total: approximately 34 features by default.

---

## Lag Features

Lag features capture the direct relationship between the current value and past values at specific time offsets. The default lag set is [1, 2, 3, 7, 14, 28].

| Feature | Formula | Meaning |
|---------|---------|---------|
| lag_1 | y(t-1) | Previous period value |
| lag_2 | y(t-2) | Two periods ago |
| lag_3 | y(t-3) | Three periods ago |
| lag_7 | y(t-7) | One week ago (daily data) |
| lag_14 | y(t-14) | Two weeks ago |
| lag_28 | y(t-28) | Four weeks ago |

Lag features are the most important features for most time series. They encode the autocorrelation structure -- how much a value depends on its own recent history.

### Custom Lags

For monthly data:

    features_df = series.create_features(lags=[1, 2, 3, 6, 12, 24], windows=[3, 6, 12], temporal=True)

For hourly data:

    features_df = series.create_features(lags=[1, 2, 6, 12, 24, 168], windows=[6, 12, 24, 168], temporal=True)

Tips: match lags to your data frequency. Too many lags can cause overfitting on small datasets.

---

## Rolling Statistics

Rolling features compute aggregations over a sliding window of recent values. Default windows: [7, 14, 30].

| Feature | Formula | Description |
|---------|---------|-------------|
| rolling_mean_W | mean(y[t-W:t]) | Average over the window |
| rolling_std_W | std(y[t-W:t]) | Volatility over the window |
| rolling_min_W | min(y[t-W:t]) | Minimum in the window |
| rolling_max_W | max(y[t-W:t]) | Maximum in the window |

With three windows, this produces 12 features total. Rolling mean captures local trend level. Rolling std captures recent volatility. Min/max capture the recent range.

---

## EWM Features

Exponentially weighted mean features give more weight to recent observations. Default spans: [7, 14, 30].

| Feature | Span | Description |
|---------|------|-------------|
| ewm_7 | 7 | Short-term smoothed average |
| ewm_14 | 14 | Medium-term smoothed average |
| ewm_30 | 30 | Long-term smoothed average |

Unlike rolling mean which weights all values equally, EWM applies exponentially decaying weights so the most recent value has the highest influence.

---

## Diff Features

Diff features measure how the series is changing over time.

| Feature | Formula | Description |
|---------|---------|-------------|
| diff_1 | y(t) - y(t-1) | One-period absolute change |
| diff_7 | y(t) - y(t-7) | Week-over-week absolute change |
| pct_change_1 | (y(t)-y(t-1))/y(t-1) | One-period percent change |
| pct_change_7 | (y(t)-y(t-7))/y(t-7) | Week-over-week percent change |

These help models learn momentum patterns -- for example, that a 5% weekly drop tends to bounce back.

---

## Temporal Features

Calendar-based features from the datetime index:

| Feature | Range | Description |
|---------|-------|-------------|
| day_of_week | 0-6 | Monday=0, Sunday=6 |
| day_of_month | 1-31 | Day within the month |
| month | 1-12 | Month of the year |
| quarter | 1-4 | Quarter of the year |
| week_of_year | 1-53 | ISO week number |
| is_weekend | 0 or 1 | 1 if Saturday or Sunday |
| is_month_start | 0 or 1 | 1 if first day of month |
| is_month_end | 0 or 1 | 1 if last day of month |

Disable with temporal=False if calendar patterns are irrelevant.

---

## Using FeaturePipeline Directly

    from timesage.features import FeaturePipeline
    pipeline = FeaturePipeline(lags=[1, 2, 3, 7, 14, 28], windows=[7, 14, 30], temporal=True)
    features_df = pipeline.transform(series.dataframe)
    print(f"Features: {features_df.shape[1]} columns")

---

## Feature Importance

    result = series.forecast(horizon=30, model="rf")
    if result.feature_importance:
        for name, score in sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {name}: {score:.4f}")

Typical patterns: lag_1 is almost always most important, rolling_mean_7 captures level, month/day_of_week captures seasonality.

---

## Handling NaN Values

Lag and rolling features create NaN values at the beginning. TimeSage drops incomplete rows internally. For manual use:

    features_df = series.create_features(lags=[1, 7, 28], windows=[7, 14, 30])
    print(f"Raw rows: {len(series)}")
    print(f"After NaN removal: {features_df.dropna().shape[0]}")

---

## Best Practices

1. Start with defaults -- they work well for most daily and monthly data.
2. Match lags to your data frequency.
3. Avoid too many features if your dataset is small (fewer than 200 rows).
4. Check feature importance to prune irrelevant features.
5. Consider domain knowledge when selecting features.

---

## Additional Details on Each Feature Type

### Lag Feature Deep Dive

The mathematical formulation of a lag feature at lag k is simply:

    X_lag_k(t) = y(t - k)

This creates a new column where each row contains the value from k periods ago. Missing values at the start of the series (the first k rows) are set to NaN and later dropped.

When choosing lags, consider the autocorrelation function (ACF) of your series. Peaks in the ACF at specific lags suggest those lags will be informative features. You can visualize this with:

    series.plot_acf(lags=60)

### Rolling Feature Deep Dive

Rolling features use a fixed-size sliding window. For a window of size W, the rolling mean at time t is:

    rolling_mean_W(t) = (1/W) * sum(y[t-W+1], y[t-W+2], ..., y[t])

The rolling standard deviation measures how spread out values are within the window. High rolling_std suggests the series is volatile; low rolling_std suggests stability.

### When to Increase Window Sizes

If your data has long-term patterns (annual cycles in monthly data), use larger windows:

    features_df = series.create_features(lags=[1, 3, 6, 12], windows=[3, 6, 12, 24], temporal=True)

### EWM Mathematical Detail

The exponentially weighted mean with span s is computed as:

    ewm_s(t) = alpha * y(t) + (1 - alpha) * ewm_s(t-1)
    where alpha = 2 / (s + 1)

A larger span produces smoother output (less responsive to recent changes). A smaller span is noisier but more responsive.

### Combining Features Effectively

The best models use a combination of all feature types:
- Lags provide direct historical values
- Rolling stats provide aggregated context
- EWM provides smoothed signals
- Diffs provide momentum information
- Temporal features provide calendar context

Together, these give the ML model a rich representation of the time series that captures both recent dynamics and longer-term patterns.
