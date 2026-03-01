"""Plain-English interpretation of forecasts and diagnostics."""

from timesage.interpret.explainer import explain_forecast, explain_model
from timesage.interpret.diagnostics import diagnose_residuals
from timesage.interpret.report import generate_report

__all__ = [
    "explain_forecast",
    "explain_model",
    "diagnose_residuals",
    "generate_report",
]
