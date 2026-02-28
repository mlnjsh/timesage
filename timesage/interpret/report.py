"""Generate comprehensive forecast reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from timesage.core.result import ForecastResult


def generate_report(result: "ForecastResult", verbose: bool = True) -> str:
    """Generate a full interpretation report combining all insights.

    This is the ultimate one-stop interpretation tool:
    - Forecast accuracy breakdown
    - Error pattern analysis
    - Residual diagnostics
    - Feature importance insights
    - Actionable recommendations
    """
    from timesage.interpret.explainer import explain_forecast
    from timesage.interpret.diagnostics import diagnose_residuals

    sections = []
    sections.append(f"{'='*60}")
    sections.append(f"  TIMESAGE FORECAST REPORT — {result.model_name}")
    sections.append(f"{'='*60}\n")

    # Section 1: Forecast interpretation
    explanations = explain_forecast(result, verbose=False)
    sections.append("  FORECAST ANALYSIS")
    sections.append("  " + "-" * 40)
    for key, text in explanations.items():
        label = key.replace("_", " ").title()
        sections.append(f"\n  {label}:")
        sections.append(f"  {text}")

    # Section 2: Residual diagnostics
    if result.residuals is not None:
        sections.append(f"\n\n  RESIDUAL DIAGNOSTICS")
        sections.append("  " + "-" * 40)
        diag = diagnose_residuals(result.residuals, verbose=False)
        for key in ["bias", "normality", "autocorrelation", "heteroscedasticity"]:
            sections.append(f"\n  {key.title()}: {diag[key]['interpretation']}")
        sections.append(f"\n  Summary: {diag['summary']}")

    # Section 3: Recommendations
    sections.append(f"\n\n  RECOMMENDATIONS")
    sections.append("  " + "-" * 40)
    recommendations = _generate_recommendations(result, explanations)
    for i, rec in enumerate(recommendations, 1):
        sections.append(f"  {i}. {rec}")

    sections.append(f"\n{'='*60}")

    report = "\n".join(sections)

    if verbose:
        try:
            from rich.console import Console
            from rich.panel import Panel
            Console().print(Panel(report, title="TimeSage Report", style="cyan"))
        except ImportError:
            print(report)

    return report


def _generate_recommendations(result: "ForecastResult", explanations: dict) -> list[str]:
    """Generate actionable recommendations based on results."""
    recs = []

    if "MAPE" in result.metrics:
        mape = result.metrics["MAPE"]
        if mape > 20:
            recs.append(
                "High error rate detected. Try: (1) Adding more features, "
                "(2) Using a different model, (3) Checking for outliers or data quality issues."
            )

    if "MASE" in result.metrics and result.metrics["MASE"] > 1:
        recs.append(
            "Model underperforms naive baseline. Consider simplifying or "
            "checking if the series has enough signal for prediction."
        )

    if result.feature_importance is not None:
        imp = result.feature_importance
        if len(imp) > 0:
            bottom = imp.nsmallest(max(1, len(imp) // 3))
            if len(bottom) > 0:
                recs.append(
                    f"Consider removing low-importance features ({len(bottom)} identified) "
                    f"to reduce model complexity and potential overfitting."
                )

    if not recs:
        recs.append(
            "Model performance looks good! Consider: "
            "(1) Testing on more recent data, "
            "(2) Monitoring for concept drift, "
            "(3) Updating the model periodically."
        )

    return recs
