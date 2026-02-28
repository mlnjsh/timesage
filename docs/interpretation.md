# Interpretation Engine

TimeSage includes a plain-English interpretation engine that translates forecast results into understandable insights. No statistics background required.

---

## Overview

After running a forecast, you can get a human-readable explanation:

    import timesage as ts

    series = ts.TimeSeries(ts.load_airline(), name="Airline Passengers")
    result = series.forecast(horizon=12, model="auto")

    # Plain-English interpretation
    result.interpret()

This prints an analysis covering accuracy, error patterns, benchmark comparison, key drivers, and forecast direction.

---

## What interpret() Reports

The interpretation covers five areas:

### 1. Accuracy Rating

A qualitative rating based on MAPE (Mean Absolute Percentage Error):

| MAPE Range | Rating |
|------------|--------|
| Below 5% | EXCELLENT |
| 5% to 10% | GOOD |
| 10% to 20% | FAIR |
| 20% to 50% | POOR |
| Above 50% | UNRELIABLE |

Example output: "The model achieved a MAPE of 4.2%, meaning predictions are typically within 4.2% of actual values."

### 2. Error Pattern

Analyzes the residuals (actual minus predicted) to detect:

- **Bias:** Are predictions systematically too high or too low?
- **Balance:** Are errors roughly symmetric?

Example output: "Errors are roughly balanced between over-predictions and under-predictions, with no systematic bias detected."

### 3. Benchmark Comparison

Compares the forecast against a naive baseline using MASE (Mean Absolute Scaled Error):

| MASE Value | Interpretation |
|------------|---------------|
| MASE below 1 | Model outperforms the naive baseline |
| MASE equals 1 | Model matches the naive baseline |
| MASE above 1 | Model is worse than the naive baseline |

The naive baseline simply repeats the last observed value.

### 4. Key Drivers (ML Models Only)

For tree-based models (RF, XGBoost, LightGBM), lists the top features by importance. For statistical models, this section explains model components (trend, seasonality, AR/MA terms).

### 5. Direction

Reports whether the forecast predicts an upward, downward, or flat trend over the forecast horizon.

---

## Detailed Summary

For a more quantitative view, use summary():

    result.summary()

This prints all metrics with model details:

    === Forecast Summary ===
    Model: Random Forest (rf)
    Training observations: 120
    Test observations: 24
    Forecast horizon: 12
    Metrics:
      MAE:   23.4
      RMSE:  31.2
      MSE:   973.4
      MedAE: 18.7
      MAPE:  4.8%
      R2:    0.94
      MASE:  0.72

---

## Advanced Interpretation Tools

### explain_forecast(result)

For a structured explanation you can use programmatically:

    from timesage.interpret.explainer import explain_forecast
    explanation = explain_forecast(result)

This returns a dictionary with keys like overall_accuracy, error_pattern, benchmark, key_drivers, forecast_direction, and more.

### diagnose_residuals(residuals)

Performs four statistical tests on the residuals to check if the model has captured all patterns:

    from timesage.interpret.diagnostics import diagnose_residuals
    diagnosis = diagnose_residuals(result.residuals)

---

## Residual Diagnostics in Detail

### Bias Detection

Checks if the mean of residuals is significantly different from zero.

- **No bias:** "Residuals are centered around zero. The model is not systematically over- or under-predicting."
- **Positive bias:** "Residuals have a positive mean. The model tends to under-predict."
- **Negative bias:** "Residuals have a negative mean. The model tends to over-predict."

### Normality Test

Runs the Shapiro-Wilk test to check if residuals follow a normal distribution.

- **Normal (p > 0.05):** "Residuals are approximately normally distributed. Confidence intervals are reliable."
- **Non-normal (p < 0.05):** "Residuals are NOT normally distributed. Confidence intervals may be wider than reported."

### Autocorrelation Test

Runs the Ljung-Box test to check if residuals contain leftover patterns.

- **No autocorrelation:** "No significant autocorrelation detected. The model has captured the temporal patterns."
- **Autocorrelation detected:** "Significant autocorrelation found. The model may be missing a seasonal pattern."

### Heteroscedasticity Test

Checks if the variance of residuals changes over time.

- **Constant variance:** "Residual variance is stable over time. The model performs consistently."
- **Changing variance:** "Residual variance changes over time. The model is less accurate during volatile periods."

---

## Interpreting Specific Metrics

### MAE (Mean Absolute Error)

"On average, the forecast is off by X units." This is in the same units as your data, making it easy to understand. Good for answering "how far off are the predictions?"

### RMSE (Root Mean Squared Error)

Similar to MAE but penalizes large errors more heavily. If RMSE is much larger than MAE, there are occasional large errors (outlier predictions).

### MAPE (Mean Absolute Percentage Error)

"The forecast is typically off by X%." Scale-independent, making it useful for comparing across different datasets. Unreliable when values are close to zero.

### R2 (R-Squared)

"The model explains X% of the variance in the data." An R2 of 0.95 means the model captures 95% of the variation. Useful for understanding overall fit quality.

### MASE (Mean Absolute Scaled Error)

Compares the model against a naive forecast. A MASE of 0.5 means the model is twice as accurate as simply repeating the last value. This is one of the most robust error metrics.

### MedAE (Median Absolute Error)

The median of absolute errors. Less sensitive to outliers than MAE. If MedAE is much lower than MAE, a few large errors are pulling the MAE up.

---

## Tips for Better Interpretations

1. **Check residual diagnostics** before trusting the forecast. If the diagnostics show autocorrelation, the model is missing patterns.

2. **Compare RMSE and MAE.** If RMSE is much larger than MAE, look for outliers or sudden shifts in your data.

3. **Use feature importance** to understand what drives the forecast. If lag_1 dominates everything else, the model is mostly using the recent value.

4. **Look at the plot.** Sometimes a visual inspection reveals issues that metrics miss, like the model consistently missing peaks or troughs.

5. **Run diagnostics on multiple models.** If one model shows autocorrelation in residuals but another does not, the second model captures more structure.

6. **Consider the use case.** A MAPE of 10% might be excellent for long-range weather forecasting but poor for next-day stock prices.

---

## Example: Full Interpretation Workflow

    import timesage as ts
    from timesage.interpret.diagnostics import diagnose_residuals

    # Load and forecast
    series = ts.TimeSeries(ts.load_airline())
    result = series.forecast(horizon=12, model="rf")

    # Step 1: Quick interpretation
    result.interpret()

    # Step 2: Detailed metrics
    result.summary()

    # Step 3: Residual diagnostics
    diagnosis = diagnose_residuals(result.residuals)

    # Step 4: Visual inspection
    result.plot()
