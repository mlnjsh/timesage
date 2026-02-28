# Models Guide

TimeSage provides a unified API across all forecasting models. Every model is accessed
through the same interface:

```python
result = series.forecast(horizon=30, model="arima")
```

This guide covers every model in depth: how it works, when to use it, its
limitations, and the implementation details inside TimeSage.

---

## Statistical Models

Statistical models are available with the core installation (`pip install timesage-ts`).
They require no extra dependencies beyond `statsmodels`.

### ARIMA (Auto-ARIMA)

**What it is:**
ARIMA stands for AutoRegressive Integrated Moving Average. It models a time series
using three components:

- **AR (AutoRegressive):** Uses past values of the series to predict the future.
  The parameter `p` controls how many past values are used.
- **I (Integrated):** Differences the series to remove trends and make it
  stationary. The parameter `d` controls the number of differencing steps.
- **MA (Moving Average):** Uses past forecast errors (residuals) to refine
  predictions. The parameter `q` controls how many past errors are used.

**How TimeSage implements it:**
TimeSage performs an automatic grid search over the order parameters:

- `p` in {0, 1, 2}
- `d` in {0, 1}
- `q` in {0, 1, 2}

This produces 18 candidate models. Each is fitted and scored by AIC (Akaike
Information Criterion). The model with the lowest AIC is selected. The search
runs silently with warnings suppressed to avoid noisy output from poor-fitting
candidates.

Under the hood, TimeSage uses `statsmodels.tsa.arima.model.ARIMA`.

**Best for:**

- Univariate time series
- Linear patterns (trends that grow or shrink at a constant rate)
- Short to medium term forecasts
- Series where recent past values are strong predictors of future values

**Limitations:**

- Cannot capture non-linear patterns
- Struggles with multiple seasonalities (e.g., daily + weekly + yearly)
- No built-in handling of exogenous variables (regressors)
- The grid search over 18 combinations can be slow on very long series

**Confidence intervals:**
ARIMA provides native prediction intervals from `statsmodels`. These are
analytically derived from the model structure and are generally reliable when
the residuals are approximately normally distributed. The default confidence
level is 95%.

**Usage:**

```python
result = series.forecast(horizon=30, model="arima")

# Access the forecast
result.forecast            # pd.Series of predictions
result.confidence_lower    # Lower bound of 95% CI
result.confidence_upper    # Upper bound of 95% CI
result.metrics             # {"MAE": ..., "RMSE": ..., "MAPE": ..., ...}
```

**Series length requirement:** At least 30 observations recommended. Fewer
observations reduce the reliability of the AIC-based order selection.

---

### ETS (Exponential Smoothing)

**What it is:**
ETS stands for Error-Trend-Seasonality, also known as the Holt-Winters method.
It forecasts by giving exponentially decreasing weights to older observations.
The core idea: recent data matters more than distant data.

The ETS framework decomposes a forecast into:

- **Error:** How the model handles randomness (additive or multiplicative)
- **Trend:** The long-term direction of the series
- **Seasonality:** Repeating patterns at fixed intervals

**How TimeSage implements it:**
TimeSage uses `statsmodels.tsa.holtwinters.ExponentialSmoothing` with:

- **Trend:** Additive with damping enabled. Damping prevents the trend from
  projecting unrealistically far into the future.
- **Seasonality:** Set to `None` by default (no explicit seasonal component).
  If you need seasonal ETS, consider using `decompose()` first.
- **Optimization:** Parameters are optimized automatically by `statsmodels`.
- **Fallback:** If the damped-trend model fails to converge, TimeSage falls back
  to a simple additive trend without damping.

**Best for:**

- Series with clear trend and/or seasonality
- Simple, interpretable patterns
- Short to medium term forecasts
- When you want a fast, reliable baseline

**Limitations:**

- Cannot model complex non-linear relationships
- The current implementation does not fit a seasonal component automatically.
  For seasonal data, ARIMA or ML models may be better choices.
- Point forecasts extrapolate the trend indefinitely (damping helps but does not
  eliminate this issue)

**Confidence intervals:**
ETS confidence intervals are estimated from the standard deviation of in-sample
residuals. They assume constant error variance and normal residuals. This is an
approximation -- the intervals are symmetric around the forecast and do not widen
over time the way ARIMA intervals do.

**Usage:**

```python
result = series.forecast(horizon=30, model="ets")
```

**Series length requirement:** At least 20 observations.

---

### Theta Method

**What it is:**
The Theta method decomposes the time series into two "theta lines" -- one
capturing the long-term trend (with curvature amplified) and another capturing
short-term dynamics (with curvature dampened). The final forecast is a
combination of these two lines.

Despite its simplicity, the Theta method won the M3 Forecasting Competition in
2000, beating many more complex models. It remains one of the strongest simple
baselines in forecasting.

**How TimeSage implements it:**
TimeSage uses `statsmodels.tsa.forecasting.theta.ThetaModel`. The implementation
is straightforward -- fit the Theta model and generate forecasts.

**Best for:**

- When you need a simple, competition-winning baseline
- Series without strong seasonality
- Quick benchmarking against more complex models
- When interpretability of the method matters less than performance

**Limitations:**

- Does not explicitly model seasonality
- No built-in handling of exogenous variables
- Less well-known and less tuneable than ARIMA or ETS

**Confidence intervals:**
Estimated from residual standard deviation, similar to ETS. The intervals are
symmetric and assume normality.

**Usage:**

```python
result = series.forecast(horizon=30, model="theta")
```

**Series length requirement:** At least 20 observations.

---

## Machine Learning Models

ML models convert the forecasting problem into a supervised learning task by
engineering features from the time series. All ML models in TimeSage share the
same feature engineering pipeline and the same recursive multi-step prediction
strategy.

### Feature Engineering (All ML Models)

When you use an ML model, TimeSage automatically creates the following features:

| Feature | Description |
|---------|-------------|
| `lag_1`, `lag_2`, `lag_3` | Value at t-1, t-2, t-3 |
| `lag_7`, `lag_14` | Value at t-7, t-14 (weekly/biweekly lookback) |
| `roll_mean_7`, `roll_mean_14` | 7-day and 14-day rolling average |
| `roll_std_7`, `roll_std_14` | 7-day and 14-day rolling standard deviation |
| `dayofweek` | Day of the week (0=Monday, 6=Sunday) |
| `month` | Month of the year (1-12) |

Temporal features (`dayofweek`, `month`) are only created when the data has a
`DatetimeIndex`.

Rows with NaN values (from lagging and rolling) are dropped before training.

### Recursive Multi-Step Prediction

All ML models in TimeSage use recursive (iterative) multi-step forecasting:

1. Predict the next value using the current feature set.
2. Append the predicted value to the series.
3. Recompute features using the extended series.
4. Repeat for each step in the forecast horizon.

This approach uses the model's own predictions as inputs for subsequent steps.
It is flexible (works with any model) but errors can compound over long horizons.

---

### Random Forest

**What it is:**
Random Forest builds an ensemble of 100 independent decision trees, each trained
on a random subset of the data and features. The final prediction is the average
of all trees. This "wisdom of crowds" approach is robust to noise and outliers.

**How TimeSage implements it:**

- Uses `sklearn.ensemble.RandomForestRegressor`
- 100 estimators (trees)
- `random_state=42` for reproducibility
- `n_jobs=-1` to use all CPU cores
- Feature engineering as described above
- Recursive multi-step prediction

**Best for:**

- Robust baseline with minimal tuning
- Series with outliers or noisy data
- When you want feature importance rankings
- When you need a model that "just works"

**Limitations:**

- Cannot extrapolate beyond the range of training data (tree-based limitation)
- Recursive prediction can degrade over long horizons
- Slower than gradient boosting for equivalent accuracy

**Feature importance:**
Random Forest provides feature importance scores (normalized to sum to 1.0).
Access them via `result.feature_importance`, which returns a dictionary mapping
feature names to their importance scores.

**Confidence intervals:**
Estimated from the standard deviation of in-sample residuals. Not native to
the model -- this is an approximation.

**Usage:**

```python
result = series.forecast(horizon=30, model="rf")

# Feature importance
result.feature_importance
# {'lag_1': 0.35, 'roll_mean_7': 0.22, 'dayofweek': 0.15, ...}
```

**Series length requirement:** At least 50 observations recommended (need enough
data after dropping NaN rows from feature engineering).

---

### XGBoost

**Requires:** `pip install timesage-ts[ml]`

**What it is:**
XGBoost (eXtreme Gradient Boosting) is a gradient boosting algorithm that builds
trees sequentially, where each new tree corrects the errors of the previous ones.
It excels at capturing complex non-linear patterns and feature interactions.

**How TimeSage implements it:**

- Uses `xgboost.XGBRegressor`
- 100 estimators (boosting rounds)
- `max_depth=6` (tree depth)
- `learning_rate=0.1`
- `random_state=42` for reproducibility
- `verbosity=0` to suppress output
- Same feature engineering and recursive prediction as other ML models

**Best for:**

- Complex non-linear patterns
- Series with many interacting features
- When Random Forest is too slow or not accurate enough
- Kaggle-style competitions and high-accuracy needs

**Limitations:**

- Requires the `xgboost` package (not in the core installation)
- More prone to overfitting than Random Forest without careful tuning
- Cannot extrapolate beyond training data range
- Recursive prediction error accumulation over long horizons

**Usage:**

```python
result = series.forecast(horizon=30, model="xgboost")
```

**Series length requirement:** At least 50 observations.

---

### LightGBM

**Requires:** `pip install timesage-ts[ml]`

**What it is:**
LightGBM is Microsoft's gradient boosting framework that uses histogram-based
learning. It bins continuous features into discrete histograms, making training
dramatically faster than traditional gradient boosting.

**How TimeSage implements it:**

- Uses `lightgbm.LGBMRegressor`
- 100 estimators
- `max_depth=6`
- `learning_rate=0.1`
- `random_state=42` for reproducibility
- `verbose=-1` to suppress output
- Same feature engineering and recursive prediction as other ML models

**Best for:**

- Large datasets where training speed matters
- Complex non-linear patterns
- When XGBoost is too slow
- High-frequency data (hourly, minutely)

**Limitations:**

- Requires the `lightgbm` package (not in the core installation)
- Same extrapolation limitation as all tree-based models
- May require tuning for optimal performance on small datasets

**Usage:**

```python
result = series.forecast(horizon=30, model="lightgbm")
```

**Series length requirement:** At least 50 observations.

---

## AutoForecaster

The AutoForecaster automates model selection by trying multiple models and
picking the best one.

**How it works:**

1. **Candidate models:** ARIMA, ETS, Theta, and Random Forest (RF is included
   only when the series has more than 50 observations).
2. **Validation split:** Holds out the last 15% of the series (minimum 5
   observations) as a validation set.
3. **Scoring:** Each candidate is trained on the training portion, and its
   predictions on the validation set are scored by MAPE (Mean Absolute
   Percentage Error).
4. **Selection:** The model with the lowest MAPE wins.
5. **Refit:** The winning model is refitted on the full series before generating
   the final forecast.
6. **Fallback:** If all candidates fail, ETS is used as the default fallback.

**Note:** XGBoost and LightGBM are NOT included in the AutoForecaster candidates.
If you want to evaluate them, use `series.compare_models()` instead.

**Usage:**

```python
# AutoForecaster is the default
result = series.forecast(horizon=30)

# Explicitly:
result = series.forecast(horizon=30, model="auto")
```

---

## Model Selection Guide

### Quick Reference Table

| Scenario | Recommended Model | Why |
|----------|-------------------|-----|
| First time with a new dataset | `"auto"` | Let TimeSage pick for you |
| Univariate, linear trend | `"arima"` | Strong at linear patterns |
| Clear trend + seasonality | `"ets"` | Built for trend/season decomposition |
| Quick reliable baseline | `"theta"` | Competition-winning simplicity |
| Non-linear patterns, outliers | `"rf"` | Robust ensemble, no tuning needed |
| Complex patterns, need accuracy | `"xgboost"` | Best at non-linear relationships |
| Very large dataset | `"lightgbm"` | Fastest gradient boosting |
| No idea what to use | `"auto"` then `compare_models()` | Try everything, pick the winner |

### When to Use Statistical vs. ML Models

**Use statistical models (ARIMA, ETS, Theta) when:**

- Your series is relatively short (under 200 observations)
- The patterns are linear and well-behaved
- You need analytically derived confidence intervals
- Interpretability of the model parameters matters
- You want fast fitting times

**Use ML models (RF, XGBoost, LightGBM) when:**

- Your series is long (200+ observations)
- Patterns are non-linear or have complex interactions
- Feature importance insights are valuable
- You have domain knowledge to encode as features
- Accuracy is more important than interpretability

### Series Length Requirements

| Model | Minimum | Recommended |
|-------|---------|-------------|
| ARIMA | 20 | 50+ |
| ETS | 15 | 30+ |
| Theta | 15 | 30+ |
| RF | 30 | 100+ |
| XGBoost | 30 | 100+ |
| LightGBM | 30 | 100+ |
| Auto | 30 | 50+ |

The minimums account for the feature engineering overhead in ML models (14
observations are consumed by the longest lag, plus additional rows by rolling
windows). Shorter series may work but will have reduced accuracy.

### Comparing Models Programmatically

Use `compare_models()` to benchmark all available models on your data:

```python
comparison = series.compare_models(test_size=0.2)
```

This returns a DataFrame ranked by MAPE with columns for MAE, RMSE, MAPE, and
training time. It automatically detects which ML extras are installed and
includes XGBoost/LightGBM only if available.

To compare a specific subset:

```python
comparison = series.compare_models(
    test_size=0.2,
    models=["arima", "ets", "rf", "xgboost"]
)
```

---

## Understanding the Output

### Metrics

Every `ForecastResult` provides these metrics (computed on the test set):

| Metric | What It Means | Good Values |
|--------|---------------|-------------|
| **MAE** | Average absolute error in original units | Depends on scale |
| **RMSE** | Root mean squared error (penalizes large errors more) | Depends on scale |
| **MSE** | Mean squared error | Depends on scale |
| **MedAE** | Median absolute error (robust to outliers) | Depends on scale |
| **MAPE** | Average percentage error | < 10% is good |
| **R2** | Proportion of variance explained | > 0.8 is good |
| **MASE** | Error relative to naive forecast | < 1.0 means model beats naive |

### Interpretation

Call `result.interpret()` to get a plain-English explanation of:

- **Overall accuracy** (Excellent / Good / Moderate / Poor based on MAPE)
- **Error consistency** (RMSE/MAE ratio indicates whether errors are uniform or have outlier spikes)
- **Benchmark comparison** (MASE tells you if the model adds value beyond a naive approach)
- **Key drivers** (which features matter most, for ML models)
- **Forecast direction** (upward, downward, or flat trend in the predictions)
