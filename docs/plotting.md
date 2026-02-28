# Plotting and Themes Guide

TimeSage includes built-in plotting for time series visualization, forecast results, and diagnostic charts. All plots use a consistent, publication-ready style.

---

## Quick Start

    import timesage as ts

    series = ts.TimeSeries(ts.load_airline(), name="Airline Passengers")

    # Basic time series plot
    series.plot()

    # Plot with trend overlay and outlier markers
    series.plot(show_trend=True, show_outliers=True)

---

## Available Plot Types

### 1. Time Series Plot

    series.plot(show_trend=False, show_outliers=False)

Displays the raw time series. Options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| show_trend | False | Overlay a smoothed trend line |
| show_outliers | False | Highlight detected outlier points |

### 2. Autocorrelation Function (ACF) Plot

    series.plot_acf(lags=40)

Shows how correlated the series is with its own lagged values. Useful for identifying seasonal periods (peaks at regular intervals), checking stationarity (slow decay suggests non-stationarity), and selecting lag features for ML models.

### 3. Decomposition Plot

    series.decompose(model="additive", period=12)
    series.plot_decomposition()

Displays four panels: the original series, trend component (long-term direction), seasonal component (repeating patterns), and residual component (what remains after removing trend and seasonality).

### 4. Forecast Plot

    result = series.forecast(horizon=12, model="auto")
    result.plot()

Shows training data in the primary color, test period actuals as a dashed line, test predictions overlaid for comparison, future forecast extending beyond the data, and the confidence interval as a shaded band.

---

## Themes

TimeSage ships with three visual themes. Set the theme before plotting.

### Sage (Default)

    ts.set_theme("sage")

A calm, professional theme with muted greens and blues. Good for reports and presentations.

### Dark

    ts.set_theme("dark")

A dark background theme with bright accent colors. Good for dashboards and screen presentations.

### Minimal

    ts.set_theme("minimal")

A clean, minimal theme with thin lines and no grid. Good for publications and academic papers.

---

## Color Palette

TimeSage uses a consistent five-color palette across all themes:

| Name | Hex Code | Usage |
|------|----------|-------|
| Primary | #2E86AB | Main series line, primary data |
| Secondary | #A23B72 | Secondary lines, comparisons |
| Accent | #F18F01 | Highlights, forecast lines |
| Success | #2BA84A | Positive indicators, good metrics |
| Danger | #E63946 | Negative indicators, outliers, errors |

These colors are designed to be distinguishable in colorblind simulations, readable on both light and dark backgrounds, and visually harmonious when used together.

---

## Customizing Plots with Matplotlib

TimeSage plots are built on matplotlib, so you can customize them further:

    import matplotlib.pyplot as plt

    # Create the TimeSage plot
    series.plot(show_trend=True)

    # Customize with matplotlib
    plt.title("Custom Title", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Passengers (thousands)")
    plt.tight_layout()
    plt.savefig("my_plot.png", dpi=300)
    plt.show()

### Adjusting Figure Size

    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    series.plot()
    plt.tight_layout()
    plt.show()

### Saving Plots

    import matplotlib.pyplot as plt

    result.plot()
    plt.savefig("forecast.png", dpi=300, bbox_inches="tight")
    plt.savefig("forecast.pdf", bbox_inches="tight")  # Vector format

---

## EDA Plots

The eda() method produces a multi-panel diagnostic view:

    series.eda(show_plots=True)

This generates: time series plot with trend, ACF plot, seasonal decomposition, and distribution histogram of values.

To run EDA without plots (text output only):

    series.eda(show_plots=False)

---

## Comparing Forecasts Visually

To visually compare multiple models, run them separately and plot:

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = ["arima", "ets", "rf", "xgboost"]
    for ax, model_name in zip(axes.flat, models):
        result = series.forecast(horizon=12, model=model_name)
        plt.sca(ax)
        result.plot()
        ax.set_title(f"{model_name.upper()}")

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300)
    plt.show()

---

## Theme Details

### Sage Theme Colors

The default "sage" theme uses earthy, professional tones:
- Background: light warm gray
- Grid: subtle dotted lines
- Text: dark gray
- Primary line: #2E86AB (teal blue)

### Dark Theme Colors

The "dark" theme uses a dark background with vibrant lines:
- Background: dark charcoal (#1a1a2e)
- Grid: dim gray lines
- Text: light gray
- Primary line: bright variants of the standard palette

### Minimal Theme

The "minimal" theme strips away visual clutter:
- Background: pure white
- Grid: none
- Text: black
- Borders: only left and bottom axes (no top/right spines)
- Lines: thinner, cleaner

---

## Plot Troubleshooting

### Plots not showing

If plots do not appear, ensure you are using an interactive matplotlib backend:

    import matplotlib
    matplotlib.use("TkAgg")  # or "Qt5Agg", "WebAgg"

Or in Jupyter notebooks, use the inline magic at the top of your notebook:

    %matplotlib inline

### Overlapping labels

Use plt.tight_layout() after creating the plot to automatically adjust spacing.

### Large date ranges

For very long series, x-axis date labels can overlap. Matplotlib handles this automatically in most cases, but you can set the locator manually:

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    series.plot()
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()
    plt.show()

---

## Summary

| What You Want | Method |
|---------------|--------|
| Plot the raw series | series.plot() |
| Show trend and outliers | series.plot(show_trend=True, show_outliers=True) |
| Autocorrelation chart | series.plot_acf(lags=40) |
| Decomposition panels | series.plot_decomposition() |
| Full EDA with plots | series.eda(show_plots=True) |
| Forecast visualization | result.plot() |
| Change the theme | ts.set_theme("dark") |
