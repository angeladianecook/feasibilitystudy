#!/usr/bin/env python
"""Model patients_enrolled over a site's first 3 months of participation.

Target: for each site, sum of patients_enrolled across all study-site pairs
where enrollment.month <= 3 (a site's ramp-up window), aggregated from
data/enrollment.csv. Only sites that actually participated in a study (i.e.
appear in enrollment.csv) are included.

Features (all from data/sites.csv, known at/before site activation):
eligible_population, prior_enrollment_rate, site_experience,
investigator_experience, startup_days, quality_score, competing_trials.

Fits OLS (statsmodels), Poisson regression, and negative binomial
regression on the same target/features, prints coefficients, AIC, and a
predicted-vs-actual variance comparison to show overdispersion, then writes
a short comparison to outputs/enrollment_model_summary.md.

Usage:
    python scripts/model_enrollment.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

FEATURES = [
    "eligible_population",
    "prior_enrollment_rate",
    "site_experience",
    "investigator_experience",
    "startup_days",
    "quality_score",
    "competing_trials",
]
TARGET = "patients_enrolled_3mo"


def build_dataset() -> pd.DataFrame:
    sites = pd.read_csv(DATA_DIR / "sites.csv")
    enrollment = pd.read_csv(DATA_DIR / "enrollment.csv")

    target = (
        enrollment[enrollment["month"] <= 3]
        .groupby("site_id")["patients_enrolled"]
        .sum()
        .rename(TARGET)
    )
    df = sites.set_index("site_id").join(target, how="inner").reset_index()
    return df[["site_id", TARGET] + FEATURES]


def standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    # Feature scales vary hugely (eligible_population ~10^5 vs quality_score
    # ~10^2), which makes the Poisson/NegBin Hessian ill-conditioned. Z-scoring
    # puts all features on a common scale; coefficients are then "per std dev".
    df = df.copy()
    for col in FEATURES:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


def fit_models(df: pd.DataFrame):
    df_std = standardize_features(df)
    formula = f"{TARGET} ~ " + " + ".join(FEATURES)
    y = df_std[TARGET]
    X = sm.add_constant(df_std[FEATURES])

    ols = smf.ols(formula, data=df_std).fit()
    poisson = smf.glm(formula, data=df_std, family=sm.families.Poisson()).fit()
    negbin = sm.NegativeBinomial(y, X).fit(method="bfgs", maxiter=200, disp=0)

    return ols, poisson, negbin, X, y


def coefficient_table(ols, poisson, negbin) -> pd.DataFrame:
    negbin_params = negbin.params.rename({"const": "Intercept"})
    table = pd.DataFrame({
        "OLS": ols.params,
        "Poisson": poisson.params,
        "NegBinomial": negbin_params,
    })
    return table.round(4)


def variance_comparison(y: pd.Series, ols, poisson, negbin, X) -> pd.DataFrame:
    pred_ols = ols.fittedvalues
    pred_poisson = poisson.fittedvalues
    pred_negbin = negbin.predict(X)
    alpha = negbin.params["alpha"]
    negbin_pred_var = (pred_negbin + alpha * pred_negbin ** 2).mean()
    pearson_dispersion = poisson.pearson_chi2 / poisson.df_resid

    rows = [
        ("Raw target", y.mean(), y.var(), y.var() / y.mean(), "empirical var/mean ratio"),
        ("OLS", pred_ols.mean(), (y - pred_ols).var(),
         (y - pred_ols).var() / max(pred_ols.mean(), 1e-9), "residual variance, assumed constant"),
        ("Poisson", pred_poisson.mean(), pred_poisson.mean(), 1.0,
         f"assumes Var=Mean; Pearson chi2/df={pearson_dispersion:.2f}"),
        ("NegBinomial", pred_negbin.mean(), negbin_pred_var, negbin_pred_var / pred_negbin.mean(),
         f"Var=Mean+alpha*Mean^2, alpha={alpha:.3f}"),
    ]
    return pd.DataFrame(rows, columns=["model", "pred_mean", "variance", "var_mean_ratio", "note"])


def write_summary(df, ols, poisson, negbin, var_table: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    actual_ratio = var_table.loc[var_table["model"] == "Raw target", "var_mean_ratio"].iloc[0]
    pearson_dispersion = poisson.pearson_chi2 / poisson.df_resid
    alpha = negbin.params["alpha"]
    alpha_pvalue = negbin.pvalues["alpha"]

    lines = [
        "# Enrollment Model Summary",
        "",
        f"Target: `patients_enrolled_3mo` (sum of `patients_enrolled` over a site's "
        f"first 3 months of participation), aggregated from `data/enrollment.csv` for "
        f"n = {len(df)} sites that participated in at least one study.",
        "",
        "## Model fit (AIC, lower is better)",
        "",
        "| Model | AIC | Log-likelihood |",
        "|---|---|---|",
        f"| OLS | {ols.aic:.1f} | {ols.llf:.1f} |",
        f"| Poisson | {poisson.aic:.1f} | {poisson.llf:.1f} |",
        f"| Negative Binomial | {negbin.aic:.1f} | {negbin.llf:.1f} |",
        "",
        "OLS's AIC is on a Gaussian likelihood and isn't directly comparable in scale "
        "to the two count-model AICs, but all three are reported for completeness. "
        "Ranked by AIC: negative binomial (best) < OLS < Poisson (worst). A naive "
        "Poisson fit is actually the *worst* of the three here, which is itself the "
        "overdispersion story below.",
        "",
        "## Why negative binomial fits better than OLS (and plain Poisson doesn't)",
        "",
        f"- The target is a non-negative integer count with a right-skewed "
        f"distribution (empirical variance/mean ratio = {actual_ratio:.2f}, i.e. "
        f"variance is ~{actual_ratio:.0f}x the mean). OLS assumes a continuous, "
        f"homoscedastic, potentially-negative outcome, none of which hold for "
        f"enrollment counts: it can and does predict negative enrollment for "
        f"some sites, and its constant-variance assumption understates spread for "
        f"high-enrollment sites and overstates it for low ones.",
        f"- Poisson regression models the count correctly (non-negative, integer) "
        f"but assumes Var(Y) = Mean(Y). The Pearson chi2/df statistic from the "
        f"fitted Poisson model is {pearson_dispersion:.2f}, far above 1.0, confirming "
        f"severe overdispersion. Because Poisson's likelihood is tightly coupled to "
        f"that (wrong) Var=Mean assumption, the mismatch is punished directly in the "
        f"log-likelihood, which is why Poisson's AIC ({poisson.aic:.1f}) ends up "
        f"*worse* than OLS's ({ols.aic:.1f}) despite modeling the right kind of "
        f"outcome. A good mean-structure fit is not enough if the variance "
        f"assumption is this wrong.",
        f"- Negative binomial adds a dispersion parameter (alpha = {alpha:.3f}, "
        f"p = {alpha_pvalue:.2e}) that lets predicted variance grow faster than the "
        f"mean (Var = Mean + alpha*Mean^2), matching the extra-Poisson variance "
        f"actually present in site-level enrollment (site heterogeneity in "
        f"execution, referral surges, seasonal effects, etc. that Poisson's "
        f"single-parameter mean-variance link can't absorb). That's why it has the "
        f"lowest AIC of the three ({negbin.aic:.1f}) and is the appropriate model "
        f"for this target: it keeps Poisson's correct count structure while "
        f"actually fitting the observed variance.",
        "",
        "## Coefficients",
        "",
        coefficient_table(ols, poisson, negbin).to_markdown(),
        "",
        "## Predicted vs. actual variance",
        "",
        "Poisson's canonical log link forces its average prediction to exactly "
        "match the actual mean; negative binomial's MLE has no such constraint, "
        "so a small gap between its predicted mean and the actual mean (below) is "
        "expected, not a bug.",
        "",
        var_table.round(3).to_markdown(index=False),
        "",
    ]

    out_path = OUT_DIR / "enrollment_model_summary.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main():
    df = build_dataset()
    print(f"Dataset: {len(df)} sites with enrollment history\n")

    ols, poisson, negbin, X, y = fit_models(df)

    print("=== Coefficients ===")
    print(coefficient_table(ols, poisson, negbin).to_string())

    print("\n=== AIC ===")
    print(f"OLS:          {ols.aic:.1f}")
    print(f"Poisson:      {poisson.aic:.1f}")
    print(f"NegBinomial:  {negbin.aic:.1f}")

    var_table = variance_comparison(y, ols, poisson, negbin, X)
    print("\n=== Predicted vs. actual variance (overdispersion check) ===")
    print(var_table.round(3).to_string(index=False))

    out_path = write_summary(df, ols, poisson, negbin, var_table)
    print(f"\nSummary written to: {out_path}")


if __name__ == "__main__":
    main()
