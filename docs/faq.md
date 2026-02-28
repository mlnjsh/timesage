# Frequently Asked Questions

---

## Installation

### Q: How do I install TimeSage?

The core package installs with:

```bash
pip install timesage-ts
```

This includes everything needed for statistical models (ARIMA, ETS, Theta),
Random Forest, EDA, plotting, and interpretation. The core dependencies are
numpy, pandas, matplotlib, scipy, scikit-learn, statsmodels, and rich.

### Q: How do I install XGBoost and LightGBM support?

Install the `ml` extras:

```bash
pip install timesage-ts[ml]
```

Or install everything at once:

```bash
pip install timesage-ts[full]
```

The `[full]` extra includes XGBoost, LightGBM, Plotly (interactive charts), and
Kaleido (chart export).

### Q: I get `ModuleNotFoundError: No module named 'xgboost'` when using `model="xgboost"`.

XGBoost is not included in the core installation. Install it with:

```bash
pip install timesage-ts[ml]
```

or directly:

```bash
pip install xgboost
```

The same applies to LightGBM. TimeSage detects available packages at runtime and
only offers models whose dependencies are installed.

### Q: What Python versions are supported?

TimeSage requires Python 3.9 or higher. It is tested on Python 3.9, 3.10, 3.11,
and 3.12.

### Q: How do I install the latest development version?

Install directly from GitHub:

```bash
pip install git+https://github.com/mlnjsh/timesage.git
```

---

## Data Format

### Q: What data formats does TimeSage accept?

TimeSage accepts:

- **pandas DataFrame** with a date column and a numeric target column
- **pandas DataFrame** with a `DatetimeIndex` and a numeric target column
- **pandas Series** with a `DatetimeIndex`

Examples:

```python
# DataFrame with date column
series = ts.TimeSeries(df, target="sales", time="date")

# DataFrame with DatetimeIndex
series = ts.TimeSeries(df, target="sales")

# pandas Series
series = ts.TimeSeries(df["sales"])
```

### Q: Does my data need to be sorted by date?

No. TimeSage automatically sorts the data by the datetime index during
initialization. However, providing pre-sorted data avoids unnecessary work.

### Q: What datetime formats are supported?

Any format that `pd.to_datetime()` can parse. If your index is not already a
`DatetimeIndex`, TimeSage will attempt to convert it. If conversion fails, a
`ValueError` is raised.

### Q: What if my data has non-uniform time intervals?

TimeSage infers the frequency using `pd.infer_freq()`. If the frequency cannot
be inferred (e.g., missing dates, irregular intervals), it defaults to `"D"`
(daily). You can override this by specifying the `freq` parameter:

```python
series = ts.TimeSeries(df, target="value", freq="W")  # Weekly
series = ts.TimeSeries(df, target="value", freq="M")  # Monthly
series = ts.TimeSeries(df, target="value", freq="H")  # Hourly
```

---

## Missing Data

### Q: How does TimeSage handle missing values?

TimeSage drops NaN values before fitting models (via `.dropna()`). It does not
perform imputation automatically. If your series has missing values, the
`describe()` method and EDA profiler will report the count and percentage of
missing observations.

For best results, handle missing data before creating a `TimeSeries`:

```python
# Forward fill
df["value"] = df["value"].ffill()

# Linear interpolation
df["value"] = df["value"].interpolate(method="linear")

# Drop missing rows
df = df.dropna(subset=["value"])
```

### Q: What if there are gaps in my date index (e.g., weekends missing)?

If you have business-day data (no weekends), set `freq="B"`:

```python
series = ts.TimeSeries(df, target="close", freq="B")
```

For other gaps, consider reindexing your DataFrame to a complete date range and
filling missing values before creating a TimeSeries.

---

## Model Selection

### Q: Which model should I use?

If you are unsure, start with `model="auto"`. The AutoForecaster tries ARIMA,
ETS, Theta, and Random Forest, then picks the one with the lowest MAPE on a
validation set.

For a more thorough comparison, use `compare_models()`:

```python
comparison = series.compare_models(test_size=0.2)
```

As a rough guide:

| Your situation | Try this |
|----------------|----------|
| Short series (< 100 obs), simple patterns | `"arima"` or `"ets"` |
| Medium series, clear trend/seasonality | `"ets"` or `"theta"` |
| Long series (> 200 obs), complex patterns | `"rf"` or `"xgboost"` |
| Very large dataset, need speed | `"lightgbm"` |
| No idea | `"auto"` |

### Q: When should I use statistical vs. ML models?

**Statistical models** (ARIMA, ETS, Theta) are better when:

- The series is short (fewer than 100-200 observations)
- Patterns are linear and well-behaved
- You need interpretable model parameters
- You want analytically derived confidence intervals

**ML models** (RF, XGBoost, LightGBM) are better when:

- The series is long (200+ observations)
- Patterns are non-linear or have complex interactions
- You want feature importance insights
- Accuracy matters more than model interpretability

### Q: Does AutoForecaster try XGBoost and LightGBM?

No. The AutoForecaster only tries ARIMA, ETS, Theta, and Random Forest. This is
a deliberate design choice to keep the automatic selection fast and to avoid
requiring optional dependencies.

To include XGBoost and LightGBM in your comparison, use `compare_models()`:

```python
comparison = series.compare_models(
    models=["arima", "ets", "theta", "rf", "xgboost", "lightgbm"]
)
```

---

## Metrics & Interpretation

### Q: What is a good MAPE?

MAPE (Mean Absolute Percentage Error) measures the average percentage deviation
of predictions from actuals. TimeSage uses these thresholds:

| MAPE Range | Rating |
|------------|--------|
| < 5% | Excellent |
| 5% - 10% | Good |
| 10% - 20% | Moderate |
| 20% - 40% | Poor |
| > 40% | Very poor |

However, "good" depends on the domain. A 10% MAPE might be excellent for stock
prices but poor for electricity demand. Compare against a naive baseline (MASE)
for a domain-independent assessment.

### Q: What is MASE and why does it matter?

MASE (Mean Absolute Scaled Error) compares your model's error against a naive
forecast (predicting tomorrow's value as today's value).

- **MASE < 1.0:** Your model is better than the naive approach. The further
  below 1.0, the more value the model adds.
- **MASE = 1.0:** Your model is equivalent to the naive approach.
- **MASE > 1.0:** Your model is worse than simply repeating the last value.
  Consider a simpler approach.

MASE is particularly useful because it is scale-independent and works across
different types of time series.

### Q: How are confidence intervals computed?

It depends on the model:

- **ARIMA:** Native analytical intervals from `statsmodels`. These are the most
  reliable -- they account for the model structure and widen over the forecast
  horizon.
- **ETS and Theta:** Estimated from the standard deviation of in-sample
  residuals. The intervals are symmetric and assume normal residuals. They do
  not widen over time.
- **ML models (RF, XGBoost, LightGBM):** Estimated from residual standard
  deviation, similar to ETS. These are approximations -- tree-based models do
  not have a native probabilistic framework.

The default confidence level is 95%. You can change it:

```python
result = series.forecast(horizon=30, model="arima", confidence=0.90)
```

### Q: What does the RMSE/MAE ratio tell me?

The ratio of RMSE to MAE indicates error consistency:

- **Ratio close to 1.0:** Errors are very uniform -- the model is equally
  accurate across all predictions.
- **Ratio around 1.2-1.3:** Typical, healthy range.
- **Ratio above 1.5:** The model has outlier errors -- it is accurate most of
  the time but makes large mistakes occasionally. This could indicate regime
  changes, outliers, or patterns the model is missing.

### Q: How do I interpret feature importance?

For ML models (RF, XGBoost, LightGBM), feature importance tells you which
features contribute most to predictions. Values are normalized to sum to 1.0.

Common patterns:

- **`lag_1` dominates:** The series has strong autocorrelation (today's value
  strongly predicts tomorrow's).
- **`dayofweek` or `month` are high:** The series has strong calendar-driven
  seasonality.
- **Rolling features are high:** Trends and momentum matter more than point
  values.
- **Low importance features:** Consider removing them to reduce noise and
  potential overfitting.

Access feature importance:

```python
result = series.forecast(horizon=30, model="rf")
print(result.feature_importance)
# {'lag_1': 0.35, 'roll_mean_7': 0.22, 'dayofweek': 0.15, ...}
```

---

## Stationarity

### Q: What does "stationary" mean?

A stationary time series has statistical properties (mean, variance,
autocorrelation) that do not change over time. Most forecasting models
(especially ARIMA) work best on stationary data.

TimeSage runs two tests:

- **ADF (Augmented Dickey-Fuller):** Tests the null hypothesis that the series
  has a unit root (is non-stationary). A p-value < 0.05 means the series IS
  stationary.
- **KPSS (Kwiatkowski-Phillips-Schmidt-Shin):** Tests the null hypothesis that
  the series IS stationary. A p-value >= 0.05 means the series IS stationary.

TimeSage combines both tests to give one of four conclusions:

| ADF says | KPSS says | Conclusion |
|----------|-----------|------------|
| Stationary | Stationary | **STATIONARY** -- Both agree |
| Non-stationary | Non-stationary | **NON-STATIONARY** -- Both agree, has unit root |
| Stationary | Non-stationary | **TREND-STATIONARY** -- Stationary around a trend |
| Non-stationary | Stationary | **DIFFERENCE-STATIONARY** -- Unit root but no deterministic trend |

### Q: My series is non-stationary. What should I do?

You have several options:

1. **Use ARIMA** -- It handles non-stationarity automatically through the
   differencing parameter `d`. TimeSage's auto-ARIMA will select `d=1` if
   needed.
2. **Difference the series manually** before feeding it to other models:
   ```python
   df["value_diff"] = df["value"].diff()
   ```
3. **Use ML models** -- Tree-based models (RF, XGBoost, LightGBM) can handle
   non-stationary data because they learn from features (lags, rolling stats)
   rather than the raw series.
4. **Apply a log transform** for series with multiplicative trends:
   ```python
   import numpy as np
   df["log_value"] = np.log(df["value"])
   ```

---

## Seasonality & Decomposition

### Q: How does TimeSage detect seasonality?

TimeSage uses autocorrelation function (ACF) analysis. It computes the ACF up to
a maximum lag (default: 90 or one-third of the series length, whichever is
smaller), then looks for significant peaks -- lags where the autocorrelation is
higher than its neighbors and exceeds the significance threshold
(1.96 / sqrt(n)).

The first significant peak determines the primary seasonal period.

```python
results = series.detect_seasonality(max_lag=90)
print(results["period"])       # e.g., 7 for weekly, 12 for monthly
print(results["peaks"][:5])    # Top 5 peaks: [(7, 0.82), (14, 0.65), ...]
```

### Q: Can TimeSage handle multiple seasonalities?

The `detect_seasonality()` method identifies the primary seasonal period. The
`peaks` field in the result contains additional significant lags, which may
correspond to secondary seasonalities.

However, the built-in models do not explicitly model multiple seasonalities. For
series with complex seasonal patterns (e.g., hourly data with daily + weekly +
yearly cycles), consider:

- Using ML models (RF, XGBoost) which capture multiple seasonalities through
  temporal features (hour, dayofweek, month)
- Decomposing the series and modeling residuals
- Using external libraries like `prophet` or `statsmodels` SARIMAX for multi-seasonal ARIMA

---

## Multivariate & Exogenous Variables

### Q: Does TimeSage support multivariate time series?

Not yet. TimeSage currently focuses on univariate forecasting -- predicting a
single target variable from its own history. Multivariate support (predicting
multiple related series simultaneously) is planned for a future release.

### Q: Can I use external variables (regressors) with TimeSage models?

Not directly through the `forecast()` API. The statistical models (ARIMA, ETS,
Theta) and ML models all operate on the target series and its automatically
engineered features.

However, you can work around this by using the feature engineering pipeline with
your own models:

```python
# Create features from the time series
features = series.create_features(lags=[1, 7, 14], windows=[7, 14])

# Add your own external features
features["temperature"] = temperature_data
features["is_holiday"] = holiday_flags

# Use with scikit-learn or any ML library
from sklearn.ensemble import GradientBoostingRegressor
X = features.drop(columns=["sales"])
y = features["sales"]
model = GradientBoostingRegressor()
model.fit(X, y)
```

---

## AutoForecaster

### Q: How does AutoForecaster decide which model is best?

The AutoForecaster follows these steps:

1. **Candidate pool:** ARIMA, ETS, Theta, and Random Forest (RF is only included
   if the series has more than 50 observations).
2. **Validation split:** The last 15% of the series is held out (minimum 5
   observations).
3. **Training:** Each candidate model is trained on the training portion.
4. **Scoring:** Predictions are generated for the validation period and scored
   by MAPE.
5. **Selection:** The model with the lowest MAPE wins.
6. **Refit:** The winning model is refitted on the full series.
7. **Fallback:** If all candidates fail (e.g., due to convergence issues), ETS
   is used as the default.

### Q: Why doesn't AutoForecaster include XGBoost or LightGBM?

Two reasons:

1. **Dependency management:** XGBoost and LightGBM are optional dependencies.
   The AutoForecaster should work with the core installation alone.
2. **Speed:** Adding more candidates increases the selection time. The current
   four candidates balance coverage with speed.

Use `compare_models()` for a comprehensive comparison that includes all
installed models.

---

## Comparison with Other Libraries

### Q: How does TimeSage compare to statsmodels?

`statsmodels` is a lower-level statistical library. TimeSage wraps statsmodels
models (ARIMA, ETS, Theta) and adds:

- Automatic parameter selection (auto-ARIMA grid search)
- A unified API across all model types
- Plain-English interpretation of results
- Automated EDA (stationarity tests, seasonality detection)
- Feature engineering for ML models
- Model comparison in one line
- Beautiful terminal output with `rich`

If you need fine-grained control over model parameters, use statsmodels directly.
If you want quick, interpreted results, use TimeSage.

### Q: How does TimeSage compare to Facebook Prophet?

Prophet is a Bayesian structural time series model designed for business
forecasting. Key differences:

- **Prophet** excels at series with strong seasonality, holidays, and trend
  changepoints. It has a dedicated API for adding custom holidays and regressors.
- **TimeSage** provides a broader set of models (statistical + ML) with automatic
  selection and comparison. It focuses on interpretation and education.

Use Prophet when you have complex seasonal patterns with known holidays. Use
TimeSage when you want to quickly compare multiple model families and get
plain-English insights.

### Q: How does TimeSage compare to sktime?

`sktime` is a comprehensive time series framework with many algorithms,
transformers, and pipelines. It follows scikit-learn's API design. Key
differences:

- **sktime** is a framework for building complex time series pipelines with
  many algorithms. It has a steeper learning curve.
- **TimeSage** is a higher-level library focused on simplicity and
  interpretation. It prioritizes getting results quickly with minimal code.

Use sktime for advanced pipelines, cross-validation strategies, and algorithm
research. Use TimeSage for quick analysis, model comparison, and interpretable
forecasts.

---

## Residual Diagnostics

### Q: What does `diagnose_residuals` check?

It runs four tests on the model residuals:

1. **Bias test** (t-test for mean = 0): Checks if the model systematically
   over- or under-predicts.
2. **Normality test** (Shapiro-Wilk + Jarque-Bera): Checks if residuals follow
   a normal distribution. Non-normal residuals make confidence intervals
   unreliable.
3. **Autocorrelation test** (Ljung-Box at lag 10): Checks if residuals are
   correlated with each other. Autocorrelated residuals mean the model is
   missing temporal patterns.
4. **Heteroscedasticity test** (variance ratio, first half vs. second half):
   Checks if the error variance changes over time. Changing variance means the
   model is more accurate in some periods than others.

Each test returns a pass/fail status with a plain-English interpretation and
actionable recommendations.

### Q: All my residual diagnostics show "ISSUE". Is my model useless?

Not necessarily. Residual diagnostics are idealized statistical tests. In
practice:

- **Non-normal residuals** are common in real-world data and do not mean the
  point forecasts are bad -- only that confidence intervals may be inaccurate.
- **Mild autocorrelation** may not significantly affect forecast accuracy.
- **Heteroscedasticity** is common in financial and economic data.

Focus on the practical metrics (MAPE, MASE) first. Use residual diagnostics as
guidance for improvement, not as a binary pass/fail gate.

---

## General

### Q: What license is TimeSage released under?

TimeSage is released under the MIT License. You are free to use, modify, and
distribute it in personal and commercial projects.

### Q: How can I contribute to TimeSage?

Contributions are welcome. Visit the GitHub repository at
[github.com/mlnjsh/timesage](https://github.com/mlnjsh/timesage):

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install timesage-ts[dev]`
4. Make your changes
5. Run tests: `pytest`
6. Submit a pull request

Areas where contributions are particularly welcome:

- Additional forecasting models (e.g., Prophet wrapper, neural networks)
- Multivariate support
- Additional built-in datasets
- Improved documentation and tutorials
- Bug reports and fixes

### Q: The `rich` output looks garbled in my terminal. What do I do?

Make sure your terminal supports Unicode and 256 colors. Most modern terminals
(iTerm2, Windows Terminal, GNOME Terminal) work fine. If you are using an older
terminal, try:

- Upgrading to a modern terminal emulator
- Setting the `TERM` environment variable: `export TERM=xterm-256color`
- If all else fails, TimeSage falls back to plain text output when `rich`
  rendering fails

### Q: Can I use TimeSage in a Jupyter notebook?

Yes. TimeSage works in Jupyter notebooks. The `rich` library renders tables and
panels in notebook cells. Matplotlib plots display inline as usual. For
interactive plots, install the `interactive` extras:

```bash
pip install timesage-ts[interactive]
```

### Q: How long does fitting take?

Typical fitting times (on a modern laptop):

| Model | 100 obs | 1,000 obs | 10,000 obs |
|-------|---------|-----------|------------|
| ETS | < 0.1s | < 0.2s | < 0.5s |
| Theta | < 0.1s | < 0.2s | < 0.5s |
| ARIMA (auto) | 1-5s | 3-10s | 10-30s |
| Random Forest | 0.5-2s | 2-5s | 5-15s |
| XGBoost | 0.3-1s | 1-3s | 3-10s |
| LightGBM | 0.2-0.5s | 0.5-2s | 2-5s |
| AutoForecaster | 3-10s | 5-20s | 20-60s |

ARIMA is the slowest because it fits 18 candidate models during the grid search.
AutoForecaster is slowest overall because it fits multiple model types. LightGBM
is the fastest ML model due to its histogram-based algorithm.
