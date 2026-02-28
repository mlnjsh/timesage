"""Residual diagnostics with plain-English interpretation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def diagnose_residuals(residuals: pd.Series, verbose: bool = True) -> dict[str, any]:
    """Comprehensive residual diagnostics with interpretation.

    Tests for: normality, autocorrelation, heteroscedasticity, bias.
    Returns plain-English explanation of what each test means.
    """
    r = residuals.dropna()
    results = {}

    # 1. Mean (bias test)
    mean_val = r.mean()
    t_stat, t_p = stats.ttest_1samp(r, 0)
    biased = t_p < 0.05
    results["bias"] = {
        "mean": float(mean_val),
        "p_value": float(t_p),
        "is_biased": biased,
        "interpretation": (
            f"Residuals have a mean of {mean_val:.4f} (p={t_p:.4f}). "
            + ("The model has a SYSTEMATIC BIAS — it consistently over- or under-predicts. "
               "Consider adding an intercept correction or reviewing feature engineering."
               if biased else
               "No significant bias detected — the model doesn't systematically over- or under-predict.")
        ),
    }

    # 2. Normality
    shapiro_stat, shapiro_p = stats.shapiro(r[:min(5000, len(r))])
    jb_stat, jb_p = stats.jarque_bera(r)
    is_normal = shapiro_p > 0.05 and jb_p > 0.05
    results["normality"] = {
        "shapiro_p": float(shapiro_p),
        "jarque_bera_p": float(jb_p),
        "is_normal": is_normal,
        "interpretation": (
            f"Residuals are {'normally distributed' if is_normal else 'NOT normally distributed'} "
            f"(Shapiro p={shapiro_p:.4f}, JB p={jb_p:.4f}). "
            + ("This validates the prediction intervals and statistical inference."
               if is_normal else
               "Prediction intervals may be unreliable. Consider: "
               "transforming the target (log, Box-Cox), or using bootstrap CIs.")
        ),
    }

    # 3. Autocorrelation (Ljung-Box)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_result = acorr_ljungbox(r, lags=[10], return_df=True)
    lb_p = float(lb_result["lb_pvalue"].iloc[0])
    has_autocorr = lb_p < 0.05
    results["autocorrelation"] = {
        "ljung_box_p": lb_p,
        "has_autocorrelation": has_autocorr,
        "interpretation": (
            f"Ljung-Box test p={lb_p:.4f}. "
            + ("Residuals show SIGNIFICANT autocorrelation — the model is missing "
               "temporal patterns. Consider: adding more lag features, increasing model "
               "order (p in ARIMA), or trying a different model."
               if has_autocorr else
               "No significant autocorrelation in residuals — the model has captured "
               "the temporal dependencies well.")
        ),
    }

    # 4. Heteroscedasticity
    mid = len(r) // 2
    first_half_var = r.iloc[:mid].var()
    second_half_var = r.iloc[mid:].var()
    ratio = max(first_half_var, second_half_var) / min(first_half_var, second_half_var) if min(first_half_var, second_half_var) > 0 else 1
    is_hetero = ratio > 2.0
    results["heteroscedasticity"] = {
        "variance_ratio": float(ratio),
        "is_heteroscedastic": is_hetero,
        "interpretation": (
            f"Variance ratio (first half vs second half): {ratio:.2f}. "
            + ("Residuals show CHANGING VARIANCE over time — the model's accuracy "
               "varies across the series. Consider: log transformation, weighted "
               "regression, or GARCH-type modeling for the variance."
               if is_hetero else
               "Residual variance is relatively stable over time — good sign.")
        ),
    }

    # 5. Summary
    issues = []
    if biased:
        issues.append("systematic bias")
    if not is_normal:
        issues.append("non-normal distribution")
    if has_autocorr:
        issues.append("residual autocorrelation")
    if is_hetero:
        issues.append("changing variance")

    if not issues:
        results["summary"] = "Residuals look healthy! The model assumptions are well-satisfied."
    else:
        results["summary"] = (
            f"Issues detected: {', '.join(issues)}. "
            f"See individual diagnostics above for specific recommendations."
        )

    if verbose:
        _print_diagnostics(results)

    return results


def _print_diagnostics(results: dict):
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        console.print()
        console.print(Panel("[bold]Residual Diagnostics[/bold]", style="blue"))

        for key in ["bias", "normality", "autocorrelation", "heteroscedasticity"]:
            info = results[key]
            label = key.upper()
            passed = not info.get("is_biased", not info.get("is_normal", not info.get("has_autocorrelation", info.get("is_heteroscedastic"))))
            status = "[green]PASS[/green]" if not any([
                info.get("is_biased"), not info.get("is_normal"),
                info.get("has_autocorrelation"), info.get("is_heteroscedastic"),
            ]) else "[red]ISSUE[/red]"

            console.print(f"  {status} [bold]{label}:[/bold] {info['interpretation']}")
            console.print()

        summary_style = "green" if "healthy" in results["summary"] else "yellow"
        console.print(f"  [{summary_style}]Summary: {results['summary']}[/{summary_style}]\n")

    except ImportError:
        for key in ["bias", "normality", "autocorrelation", "heteroscedasticity"]:
            print(f"  {key.upper()}: {results[key]['interpretation']}")
        print(f"\n  Summary: {results['summary']}\n")
