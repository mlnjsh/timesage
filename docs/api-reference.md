# TimeSage API Reference

Complete API reference for the TimeSage Python library (v0.1.0).

---

## Table of Contents

- [Top-Level Exports](#top-level-exports)
- [timesage.core.timeseries.TimeSeries](#timesagecoretimeseriestimeseries)
- [timesage.core.result.ForecastResult](#timesagecoreresultforecastresult)
- [timesage.eda.profiler](#timesageedaprofiler)
- [timesage.plot.theme](#timesageplottheme)
- [timesage.plot.timeplots](#timesageplottimeplots)
- [timesage.plot.diagnostic](#timesageplotdiagnostic)
- [timesage.features.pipeline.FeaturePipeline](#timesagefeaturespipelinefeaturepipeline)
- [timesage.models.statistical](#timesagemodelsstatistical)
- [timesage.models.ml](#timesagemodelsml)
- [timesage.models.auto.AutoForecaster](#timesagemodelsautoautoforecaster)
- [timesage.interpret.explainer](#timesageinterpretexplainer)
- [timesage.interpret.diagnostics](#timesageinterpretdiagnostics)
- [timesage.datasets.loader](#timesagedatasetsloader)
- [timesage.utils.helpers](#timesageutilshelpers)

---

## Top-Level Exports

```python
import timesage
```

| Name | Type | Description |
|------|------|-------------|
| `__version__` | `str` | Library version. Currently `"0.1.0"`. |
| `TimeSeries` | class | Main time series container and analysis class. |
| `ForecastResult` | dataclass | Container for forecast outputs, metrics, and visualizations. |
| `AutoForecaster` | class | Automatic model selection and forecasting. |
| `profile(ts, show_plots=True)` | function | Run full exploratory data analysis on a `TimeSeries` object. |
| `set_theme(theme="sage")` | function | Set the global plot theme. Accepts `"sage"`, `"dark"`, or `"minimal"`. |
| `hello()` | function | Print the TimeSage welcome banner. |

---

## timesage.core.timeseries.TimeSeries

The primary class for loading, analyzing, and forecasting time series data.

### Constructor

```python
TimeSeries(data, target=None, time=None, freq=None, name=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` or `pd.Series` | *(required)* | The input data. If a DataFrame, `target` specifies which column to use. If a Series, it is used directly. |
| `target` | `str` or `None` | `None` | Column name of the target variable in a DataFrame. |
| `time` | `str` or `None` | `None` | Column name of the datetime column. If `None`, the DataFrame index is used. |
| `freq` | `str` or `None` | `None` | Frequency override string (e.g., `"D"`, `"M"`, `"H"`). If `None`, frequency is auto-detected. |
| `name` | `str` or `None` | `None` | A display name for the time series. |

### Properties

#### `values`

```python
@property
values -> pd.Series
```

Returns the target column as a pandas Series.

#### `target`

```python
@property
target -> str
```

Returns the name of the target column.

#### `frequency`

```python
@property
frequency -> str
```

Returns a human-readable frequency string such as `"daily"`, `"monthly"`, `"hourly"`, etc.

#### `dataframe`

```python
@property
dataframe -> pd.DataFrame
```

Returns the underlying DataFrame.

#### `__len__`

```python
__len__() -> int
```

Returns the number of observations in the time series.

### Methods

#### `describe`

```python
describe() -> pd.Series
```

Returns enhanced descriptive statistics. Includes all standard `pd.Series.describe()` fields plus the following additional metrics:

| Extra Field | Description |
|-------------|-------------|
| `skewness` | Skewness of the distribution. |
| `kurtosis` | Excess kurtosis of the distribution. |
| `missing` | Count of missing (NaN) values. |
| `missing_pct` | Percentage of missing values. |
| `cv` | Coefficient of variation (std / mean). |

**Returns:** `pd.Series` with the combined statistics.

---

#### `test_stationarity`

```python
test_stationarity(verbose=True) -> dict
```

Runs Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity tests.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `True` | If `True`, prints a formatted rich table with results. |

**Returns:** `dict` with the following structure:

```python
{
    "adf": {
        "statistic": float,   # ADF test statistic
        "p_value": float,     # p-value
        "stationary": bool    # True if p_value < 0.05
    },
    "kpss": {
        "statistic": float,   # KPSS test statistic
        "p_value": float,     # p-value
        "stationary": bool    # True if p_value >= 0.05
    },
    "conclusion": str         # Human-readable summary
}
```

---

#### `detect_seasonality`

```python
detect_seasonality(max_lag=90, verbose=True) -> dict
```

Detects seasonal patterns using autocorrelation function (ACF) analysis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_lag` | `int` | `90` | Maximum lag to examine in the ACF. |
| `verbose` | `bool` | `True` | If `True`, prints a formatted summary. |

**Returns:** `dict` with the following structure:

```python
{
    "seasonal": bool,           # Whether seasonality was detected
    "period": int or None,      # Dominant seasonal period (None if not seasonal)
    "peaks": list[tuple],       # List of (lag, acf_value) tuples at significant peaks
    "acf_values": np.ndarray    # Full ACF values array
}
```

---

#### `decompose`

```python
decompose(model="additive", period=None)
```

Decomposes the time series into trend, seasonal, and residual components.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"additive"` | Decomposition model: `"additive"` or `"multiplicative"`. |
| `period` | `int` or `None` | `None` | Seasonal period. If `None`, auto-detected from the data frequency. |

**Returns:** `statsmodels.tsa.seasonal.DecompositionResult` with attributes:

- `trend` -- Trend component (pd.Series)
- `seasonal` -- Seasonal component (pd.Series)
- `resid` -- Residual component (pd.Series)
- `observed` -- Original observed values (pd.Series)

---

#### `eda`

```python
eda(show_plots=True) -> dict
```

Runs a full automated exploratory data analysis pipeline, including descriptive statistics, stationarity tests, and seasonality detection.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_plots` | `bool` | `True` | If `True`, displays summary plots. |

**Returns:** `dict` with keys:

- `"statistics"` -- Output of `describe()`
- `"stationarity"` -- Output of `test_stationarity()`
- `"seasonality"` -- Output of `detect_seasonality()`

---

#### `plot`

```python
plot(show_trend=False, show_outliers=False, figsize=(14, 5))
```

Plots the time series with the active theme.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_trend` | `bool` | `False` | Overlay a trend line on the plot. |
| `show_outliers` | `bool` | `False` | Highlight detected outliers. |
| `figsize` | `tuple[int, int]` | `(14, 5)` | Figure size in inches `(width, height)`. |

**Returns:** `matplotlib.figure.Figure`

---

#### `plot_acf`

```python
plot_acf(lags=40, figsize=(14, 4))
```

Plots the ACF and PACF side by side.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lags` | `int` | `40` | Number of lags to display. |
| `figsize` | `tuple[int, int]` | `(14, 4)` | Figure size in inches. |

**Returns:** `matplotlib.figure.Figure`

---

#### `plot_decomposition`

```python
plot_decomposition(model="additive", period=None, figsize=(14, 10))
```

Plots a 4-panel decomposition (observed, trend, seasonal, residual).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"additive"` | `"additive"` or `"multiplicative"`. |
| `period` | `int` or `None` | `None` | Seasonal period. Auto-detected if `None`. |
| `figsize` | `tuple[int, int]` | `(14, 10)` | Figure size in inches. |

**Returns:** `matplotlib.figure.Figure`

---

#### `create_features`

```python
create_features(lags=None, windows=None, temporal=True, drop_na=True) -> pd.DataFrame
```

Engineers machine learning features from the time series.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lags` | `list[int]` or `None` | `None` | Lag values to create. Defaults to `[1, 2, 3, 7, 14, 28]`. |
| `windows` | `list[int]` or `None` | `None` | Rolling window sizes. Defaults to `[7, 14, 30]`. |
| `temporal` | `bool` | `True` | Whether to create calendar/temporal features. |
| `drop_na` | `bool` | `True` | Whether to drop rows with NaN values introduced by lagging/rolling. |

**Returns:** `pd.DataFrame` containing the following generated columns:

| Feature Group | Columns Created |
|---------------|-----------------|
| Lag features | `lag_1`, `lag_2`, `lag_3`, `lag_7`, `lag_14`, `lag_28` |
| Rolling mean | `roll_mean_7`, `roll_mean_14`, `roll_mean_30` |
| Rolling std | `roll_std_7`, `roll_std_14`, `roll_std_30` |
| Rolling min | `roll_min_7`, `roll_min_14`, `roll_min_30` |
| Rolling max | `roll_max_7`, `roll_max_14`, `roll_max_30` |
| EWM | `ewm_7`, `ewm_14`, `ewm_30` |
| Differencing | `diff_1`, `diff_7` |
| Percent change | `pct_change_1`, `pct_change_7` |
| Temporal (calendar) | `day_of_week`, `day_of_month`, `month`, `quarter`, `week_of_year`, `is_weekend`, `is_month_start`, `is_month_end` |

---

#### `forecast`

```python
forecast(horizon=30, model="auto", test_size=0.2, confidence=0.95, verbose=True) -> ForecastResult
```

Forecasts future values of the time series.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `horizon` | `int` | `30` | Number of future periods to forecast. |
| `model` | `str` | `"auto"` | Forecasting model to use. See table below. |
| `test_size` | `float` | `0.2` | Fraction of data reserved for evaluation (time-respecting split). |
| `confidence` | `float` | `0.95` | Confidence level for prediction intervals (0 to 1). |
| `verbose` | `bool` | `True` | If `True`, prints progress and evaluation metrics. |

**Available models:**

| Model string | Description | Extra dependencies |
|--------------|-------------|-------------------|
| `"auto"` | Automatic model selection via `AutoForecaster`. | None |
| `"arima"` | Auto ARIMA with AIC-based grid search. | None |
| `"ets"` | Exponential Smoothing (Holt-Winters) with damped trend. | None |
| `"theta"` | Theta method. | None |
| `"rf"` | Random Forest regressor (100 trees). | None |
| `"xgboost"` | XGBoost gradient boosting. | `pip install timesage-ts[ml]` |
| `"lightgbm"` | LightGBM gradient boosting. | `pip install timesage-ts[ml]` |

**Returns:** [`ForecastResult`](#timesagecoreresultforecastresult)

---

#### `compare_models`

```python
compare_models(test_size=0.2, models=None, verbose=True) -> pd.DataFrame
```

Compares multiple forecasting models on the same train/test split.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_size` | `float` | `0.2` | Fraction of data for the test set. |
| `models` | `list[str]` or `None` | `None` | List of model names to compare. If `None`, auto-detects all available models. |
| `verbose` | `bool` | `True` | If `True`, prints a comparison table. |

**Returns:** `pd.DataFrame` with one row per model, ranked by MAPE (ascending). Columns include model name and all computed error metrics.

---

## timesage.core.result.ForecastResult

A dataclass that encapsulates all outputs from a forecasting operation.

```python
from timesage import ForecastResult
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `forecast` | `pd.Series` | Forecasted values for the future horizon. |
| `actual` | `pd.Series` or `None` | Actual values from the test set (if available). |
| `train` | `pd.Series` or `None` | Training data used for fitting. |
| `test_predictions` | `pd.Series` or `None` | Model predictions on the test set. |
| `confidence_lower` | `pd.Series` or `None` | Lower bound of the confidence interval. |
| `confidence_upper` | `pd.Series` or `None` | Upper bound of the confidence interval. |
| `model_name` | `str` | Name of the model that produced the forecast. |
| `feature_importance` | `dict[str, float]` or `None` | Feature importance scores (ML models only). |
| `residuals` | `pd.Series` or `None` | Residuals from the test set evaluation. |

### Properties

#### `metrics`

```python
@property
metrics -> dict[str, float]
```

Lazily computed dictionary of evaluation metrics comparing `actual` to `test_predictions`.

| Key | Description |
|-----|-------------|
| `"MAE"` | Mean Absolute Error |
| `"RMSE"` | Root Mean Squared Error |
| `"MSE"` | Mean Squared Error |
| `"MedAE"` | Median Absolute Error |
| `"MAPE"` | Mean Absolute Percentage Error |
| `"R2"` | R-squared (coefficient of determination) |
| `"MASE"` | Mean Absolute Scaled Error (vs. naive forecast) |

### Methods

#### `interpret`

```python
interpret() -> None
```

Prints a rich-formatted panel with a comprehensive interpretation of the forecast results. Sections include:

- **Overall Accuracy** -- Rating derived from MAPE.
- **Error Pattern** -- Analysis of the RMSE/MAE ratio.
- **Benchmark** -- MASE comparison against naive forecasting.
- **Key Drivers** -- Feature importance breakdown (if available).
- **Forecast Direction** -- Detected trend in the forecast.

---

#### `summary`

```python
summary() -> pd.DataFrame
```

Returns all evaluation metrics as a single-row DataFrame, useful for programmatic comparison.

**Returns:** `pd.DataFrame` with one row and columns for each metric key.

---

#### `plot`

```python
plot(figsize=(14, 5))
```

Plots the full forecast visualization including training data, actual test values, test predictions, future forecast, and confidence intervals.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `figsize` | `tuple[int, int]` | `(14, 5)` | Figure size in inches. |

**Returns:** `matplotlib.figure.Figure`

---

## timesage.eda.profiler

Standalone EDA profiling function.

### `profile`

```python
profile(ts, show_plots=True) -> dict
```

Runs a full profiling pipeline on a `TimeSeries` object: descriptive statistics, stationarity testing, seasonality detection, summary table output, and optional plot display.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ts` | `TimeSeries` | *(required)* | A `TimeSeries` instance to profile. |
| `show_plots` | `bool` | `True` | If `True`, renders summary plots. |

**Returns:** `dict` containing `"statistics"`, `"stationarity"`, and `"seasonality"` results.

---

## timesage.plot.theme

Global theming system for all TimeSage plots.

### Constants

#### `COLORS`

```python
COLORS: dict[str, str]
```

Default color palette dictionary used across all plots:

| Key | Hex Value | Usage |
|-----|-----------|-------|
| `"primary"` | `#2E86AB` | Main data series, primary elements |
| `"secondary"` | `#A23B72` | Secondary data series |
| `"accent"` | `#F18F01` | Highlights, call-outs |
| `"success"` | `#2BA84A` | Positive indicators |
| `"danger"` | `#E63946` | Negative indicators, outliers |
| `"neutral"` | `#6C757D` | Muted elements |
| `"background"` | `#FAFBFC` | Figure background |
| `"grid"` | `#E8ECEF` | Grid lines |
| `"text"` | `#2D3436` | Text and labels |

#### `PALETTE`

```python
PALETTE: list[str]
```

A list of 8 colors used for multi-series plots. Cycles through these colors when plotting multiple series on the same axes.

### Functions

#### `sage_theme`

```python
sage_theme() -> None
```

Applies the default "sage" matplotlib rcParams globally. Called internally when the sage theme is active.

---

#### `set_theme`

```python
set_theme(theme="sage") -> None
```

Sets the global plot theme for all subsequent TimeSage plots.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `theme` | `str` | `"sage"` | Theme name. One of `"sage"`, `"dark"`, or `"minimal"`. |

**Available themes:**

| Theme | Description |
|-------|-------------|
| `"sage"` | Default TimeSage theme with custom colors and styling. |
| `"dark"` | Dark background theme for presentations. |
| `"minimal"` | Clean, minimal theme with reduced visual elements. |

---

## timesage.plot.timeplots

Core plotting functions for time series visualization.

### `plot_series`

```python
plot_series(series, show_trend=False, show_outliers=False, title="Time Series", figsize=(14, 5))
```

Renders the main time series plot.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `series` | `pd.Series` | *(required)* | The time series data to plot. |
| `show_trend` | `bool` | `False` | Overlay a smoothed trend line. |
| `show_outliers` | `bool` | `False` | Highlight IQR-detected outliers. |
| `title` | `str` | `"Time Series"` | Plot title. |
| `figsize` | `tuple[int, int]` | `(14, 5)` | Figure size in inches. |

**Returns:** `matplotlib.figure.Figure`

---

### `plot_components`

```python
plot_components(decomposition, title="Decomposition", figsize=(14, 10))
```

Renders a 4-panel decomposition plot (observed, trend, seasonal, residual).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decomposition` | `DecompositionResult` | *(required)* | Output from `TimeSeries.decompose()` or `statsmodels` seasonal decomposition. |
| `title` | `str` | `"Decomposition"` | Plot title. |
| `figsize` | `tuple[int, int]` | `(14, 10)` | Figure size in inches. |

**Returns:** `matplotlib.figure.Figure`

---

### `plot_forecast`

```python
plot_forecast(result, figsize=(14, 5))
```

Renders the forecast visualization with training data, actuals, predictions, and confidence intervals.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `ForecastResult` | *(required)* | A `ForecastResult` instance. |
| `figsize` | `tuple[int, int]` | `(14, 5)` | Figure size in inches. |

**Returns:** `matplotlib.figure.Figure`

---

## timesage.plot.diagnostic

Diagnostic plotting functions.

### `plot_acf_pacf`

```python
plot_acf_pacf(series, lags=40, figsize=(14, 4))
```

Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) side by side.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `series` | `pd.Series` | *(required)* | The time series data. |
| `lags` | `int` | `40` | Number of lags to display. |
| `figsize` | `tuple[int, int]` | `(14, 4)` | Figure size in inches. |

**Returns:** `matplotlib.figure.Figure`

---

## timesage.features.pipeline.FeaturePipeline

Automated feature engineering pipeline for ML-based forecasting.

### Constructor

```python
FeaturePipeline(target, lags=None, windows=None, temporal=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | `str` | *(required)* | Name of the target column in the DataFrame. |
| `lags` | `list[int]` or `None` | `None` | Lag values to generate. Defaults to `[1, 2, 3, 7, 14, 28]`. |
| `windows` | `list[int]` or `None` | `None` | Rolling window sizes. Defaults to `[7, 14, 30]`. |
| `temporal` | `bool` | `True` | Whether to generate calendar/temporal features. |

### Methods

#### `transform`

```python
transform(df, drop_na=True) -> pd.DataFrame
```

Applies all configured feature transformations to the input DataFrame.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | *(required)* | Input DataFrame with a datetime index and the target column. |
| `drop_na` | `bool` | `True` | Drop rows with NaN values introduced by lag/rolling operations. |

**Returns:** `pd.DataFrame` with all generated features appended as new columns. See [`TimeSeries.create_features()`](#create_features) for the full list of generated columns.

---

## timesage.models.statistical

Classical statistical forecasting models. All models share a common interface.

### Common Interface

All statistical forecasters implement the following methods:

```python
fit(series, full_df) -> self
predict(horizon) -> pd.Series
confidence_intervals(horizon, confidence) -> tuple[pd.Series, pd.Series]
residuals() -> pd.Series
```

| Method | Description |
|--------|-------------|
| `fit(series, full_df)` | Fit the model to the training series. `full_df` is the complete DataFrame for context. |
| `predict(horizon)` | Generate point forecasts for `horizon` future periods. Returns `pd.Series`. |
| `confidence_intervals(horizon, confidence)` | Returns a tuple `(lower, upper)` of `pd.Series` for the prediction interval at the given confidence level. |
| `residuals()` | Returns in-sample residuals as `pd.Series`. |

---

### `ARIMAForecaster`

```python
ARIMAForecaster()
```

Auto ARIMA model that performs a grid search over the following parameter space, selecting the combination with the lowest AIC:

- `p`: 0, 1, 2
- `d`: 0, 1
- `q`: 0, 1, 2

No constructor parameters.

---

### `ETSForecaster`

```python
ETSForecaster()
```

Exponential Smoothing (Holt-Winters) model with damped trend. Automatically configures additive or multiplicative seasonality based on the data.

No constructor parameters.

---

### `ThetaForecaster`

```python
ThetaForecaster()
```

Theta method forecaster. A simple, robust method that decomposes the series using a theta parameter and combines linear extrapolation with exponential smoothing.

No constructor parameters.

---

## timesage.models.ml

Machine learning forecasting models. These models automatically create lag, rolling, and temporal features internally and use recursive multi-step prediction for the forecast horizon.

### Common Interface

All ML forecasters implement the following methods:

```python
fit(series, full_df) -> self
predict(horizon) -> pd.Series
confidence_intervals(horizon, confidence) -> tuple[pd.Series, pd.Series]
feature_importance() -> dict[str, float]
residuals() -> pd.Series
```

| Method | Description |
|--------|-------------|
| `fit(series, full_df)` | Fit the model using auto-generated features. |
| `predict(horizon)` | Recursive multi-step forecast for `horizon` periods. Returns `pd.Series`. |
| `confidence_intervals(horizon, confidence)` | Prediction intervals. Returns `(lower, upper)` tuple of `pd.Series`. |
| `feature_importance()` | Returns a dictionary mapping feature names to their importance scores. |
| `residuals()` | Returns in-sample residuals as `pd.Series`. |

---

### `RandomForestForecaster`

```python
RandomForestForecaster()
```

Random Forest regressor with 100 trees. Automatically engineers lag, rolling window, and temporal features from the input series. Uses recursive multi-step prediction for forecasting beyond a single step.

No constructor parameters. No extra dependencies.

---

### `XGBoostForecaster`

```python
XGBoostForecaster()
```

XGBoost gradient boosting forecaster. Same feature engineering and recursive prediction strategy as `RandomForestForecaster`.

**Requires:** `pip install timesage-ts[ml]`

---

### `LightGBMForecaster`

```python
LightGBMForecaster()
```

LightGBM gradient boosting forecaster. Same feature engineering and recursive prediction strategy as `RandomForestForecaster`.

**Requires:** `pip install timesage-ts[ml]`

---

## timesage.models.auto.AutoForecaster

Automatic model selection that benchmarks multiple models and selects the best performer.

```python
AutoForecaster()
```

### Selection Strategy

1. Fits ARIMA, ETS, Theta, and RandomForest models on the training data.
2. Evaluates each model on a 15% validation split.
3. Selects the model with the lowest MAPE.
4. Refits the winning model on the full training data.

### Methods

```python
fit(series, full_df) -> self
predict(horizon) -> pd.Series
confidence_intervals(horizon, confidence) -> tuple[pd.Series, pd.Series]
feature_importance() -> dict[str, float] or None
residuals() -> pd.Series
```

| Method | Description |
|--------|-------------|
| `fit(series, full_df)` | Run the model selection tournament and fit the best model. |
| `predict(horizon)` | Forecast using the selected best model. Returns `pd.Series`. |
| `confidence_intervals(horizon, confidence)` | Prediction intervals from the best model. Returns `(lower, upper)` tuple. |
| `feature_importance()` | Feature importance from the best model. Returns `dict` or `None` if not applicable. |
| `residuals()` | Residuals from the best model. Returns `pd.Series`. |

---

## timesage.interpret.explainer

Functions for generating human-readable interpretations of forecast results and models.

### `explain_forecast`

```python
explain_forecast(result, verbose=True) -> dict
```

Produces a detailed interpretation of a `ForecastResult`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `ForecastResult` | *(required)* | The forecast result to interpret. |
| `verbose` | `bool` | `True` | If `True`, prints a formatted rich panel. |

**Returns:** `dict` with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `"overall_accuracy"` | `str` | Rating based on MAPE (e.g., "Excellent", "Good", "Fair"). |
| `"error_pattern"` | `str` | Analysis of the RMSE-to-MAE ratio. |
| `"benchmark"` | `str` | MASE-based comparison against a naive forecast. |
| `"predictability"` | `str` | Assessment of how predictable the series is. |
| `"key_drivers"` | `dict` or `None` | Feature importance breakdown (ML models only). |
| `"pattern_type"` | `str` | Detected pattern type in the data. |
| `"forecast_direction"` | `str` | Trend direction of the forecast (e.g., "upward", "downward", "flat"). |

---

### `explain_model`

```python
explain_model(model_name, params, verbose=True) -> dict
```

Generates a plain-English description of a forecasting model and its parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | *(required)* | Name of the model (e.g., `"ARIMA"`, `"RandomForest"`). |
| `params` | `dict` | *(required)* | Model parameters to describe. |
| `verbose` | `bool` | `True` | If `True`, prints the explanation. |

**Returns:** `dict` with model description details.

---

## timesage.interpret.diagnostics

Residual diagnostic tools for evaluating forecast model assumptions.

### `diagnose_residuals`

```python
diagnose_residuals(residuals, verbose=True) -> dict
```

Runs a suite of statistical tests on model residuals to assess model adequacy.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `residuals` | `pd.Series` | *(required)* | Residuals from a fitted model. |
| `verbose` | `bool` | `True` | If `True`, prints formatted diagnostic results. |

**Returns:** `dict` with the following diagnostic tests:

| Key | Test(s) | Fields |
|-----|---------|--------|
| `"bias"` | One-sample t-test for zero mean | `statistic`, `p_value`, `significant`, `interpretation` |
| `"normality"` | Shapiro-Wilk + Jarque-Bera | `shapiro` (dict), `jarque_bera` (dict), `interpretation` |
| `"autocorrelation"` | Ljung-Box test | `statistic`, `p_value`, `significant`, `interpretation` |
| `"heteroscedasticity"` | Variance ratio (first half vs. second half) | `ratio`, `significant`, `interpretation` |

Each sub-dictionary includes an `"interpretation"` string with a plain-English assessment.

---

## timesage.datasets.loader

Built-in datasets for experimentation and testing.

### `list_datasets`

```python
list_datasets() -> list[str]
```

Returns a list of all available built-in dataset names.

**Returns:** `["airline", "sunspots", "energy", "synthetic_trend", "synthetic_seasonal"]`

---

### `load_airline`

```python
load_airline() -> pd.DataFrame
```

Loads the classic airline passengers dataset (monthly, 1949--1960).

**Returns:** `pd.DataFrame` with a datetime index and a single column `"passengers"`.

---

### `load_sunspots`

```python
load_sunspots() -> pd.DataFrame
```

Loads the historical yearly sunspot numbers dataset.

**Returns:** `pd.DataFrame` with a datetime index and a single column `"sunspots"`.

---

### `load_energy`

```python
load_energy() -> pd.DataFrame
```

Loads a synthetic hourly energy demand dataset.

**Returns:** `pd.DataFrame` with a datetime index and a single column `"energy_demand"`.

---

### `load_synthetic_trend`

```python
load_synthetic_trend(n=500, slope=0.5, noise=10) -> pd.DataFrame
```

Generates a synthetic time series with a linear trend and Gaussian noise.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | `500` | Number of data points to generate. |
| `slope` | `float` | `0.5` | Slope of the linear trend. |
| `noise` | `float` | `10` | Standard deviation of Gaussian noise. |

**Returns:** `pd.DataFrame` with a datetime index and a target column.

---

### `load_synthetic_seasonal`

```python
load_synthetic_seasonal(n=730, period=7, amplitude=50, noise=10) -> pd.DataFrame
```

Generates a synthetic time series with a repeating seasonal pattern and noise.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | `730` | Number of data points to generate. |
| `period` | `int` | `7` | Seasonal period length (e.g., 7 for weekly). |
| `amplitude` | `float` | `50` | Amplitude of the seasonal component. |
| `noise` | `float` | `10` | Standard deviation of Gaussian noise. |

**Returns:** `pd.DataFrame` with a datetime index and a target column.

---

## timesage.utils.helpers

Utility functions used internally and available for general use.

### `ensure_datetime_index`

```python
ensure_datetime_index(df) -> pd.DataFrame
```

Ensures the DataFrame has a proper `DatetimeIndex`. Converts string or numeric index to datetime if possible.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Input DataFrame. |

**Returns:** `pd.DataFrame` with a `DatetimeIndex`.

---

### `infer_frequency`

```python
infer_frequency(index) -> str
```

Infers the frequency of a `DatetimeIndex` and returns a human-readable string.

| Parameter | Type | Description |
|-----------|------|-------------|
| `index` | `pd.DatetimeIndex` | The datetime index to analyze. |

**Returns:** `str` -- Frequency string (e.g., `"daily"`, `"monthly"`, `"hourly"`).

---

### `detect_outliers_iqr`

```python
detect_outliers_iqr(series, factor=1.5) -> pd.Series
```

Detects outliers using the Interquartile Range (IQR) method.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `series` | `pd.Series` | *(required)* | The data series to check. |
| `factor` | `float` | `1.5` | IQR multiplier. Values beyond `Q1 - factor*IQR` or `Q3 + factor*IQR` are flagged. |

**Returns:** `pd.Series` -- Boolean mask where `True` indicates an outlier.

---

### `safe_import`

```python
safe_import(module_name) -> module or None
```

Attempts to import a module by name. Returns the module if successful, or `None` if the import fails. Used internally for optional dependency handling (e.g., XGBoost, LightGBM).

| Parameter | Type | Description |
|-----------|------|-------------|
| `module_name` | `str` | Fully qualified module name (e.g., `"xgboost"`). |

**Returns:** The imported module, or `None` if unavailable.

---

### `format_number`

```python
format_number(n, decimals=2) -> str
```

Formats a number with human-readable suffixes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `float` or `int` | *(required)* | The number to format. |
| `decimals` | `int` | `2` | Number of decimal places. |

**Returns:** `str` -- Formatted string. Examples: `"1.23M"`, `"4.56K"`, `"7.89"`.

---

### `train_test_split_ts`

```python
train_test_split_ts(series, test_size=0.2) -> tuple[pd.Series, pd.Series]
```

Performs a time-respecting train/test split. Unlike random splitting, this preserves temporal ordering by taking the last `test_size` fraction of the data as the test set.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `series` | `pd.Series` | *(required)* | The time series to split. |
| `test_size` | `float` | `0.2` | Fraction of data to use as the test set (taken from the end). |

**Returns:** `tuple[pd.Series, pd.Series]` -- `(train, test)` where `train` contains the first `1 - test_size` fraction and `test` contains the remainder.
