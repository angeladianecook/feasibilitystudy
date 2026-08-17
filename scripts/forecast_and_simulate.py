#!/usr/bin/env python
"""Forecast one study's enrollment and Monte Carlo-simulate time-to-target.

Study: STU-0051 (Cardiology, Phase 3, target_enrollment=1,326, 27
participating sites, 24 months of observed data). Picked because its
historical cumulative enrollment crosses 500 patients around month 10 --
close enough to the 12-month checkpoint that "will it hit 500 by month 12"
is a genuinely uncertain question, not a foregone conclusion either way.

Part 1 -- forecasting: aggregates the study's monthly enrollment across all
27 sites (sum of patients_enrolled per month) into a single time series,
then fits an exponential smoothing model (Holt, damped additive trend) and
an ARIMA(1,1,1) model (both statsmodels) to forecast the next 6 months.

Part 2 -- Monte Carlo simulation: independent of Part 1's fitted models.
For each of the study's 27 sites, its historical mean monthly enrollment
rate (from enrollment.csv) seeds a Poisson process. Each of 5,000
iterations independently samples, per site: (a) an activation delay (0 or 1
month, common startup jitter) plus a fixed +2 month delay applied every
iteration to 3 designated "at-risk" sites (the 3 lowest quality_score sites
in the cohort) -- a stress-tested scenario, not a random event -- and (b) a
per-iteration rate multiplier (lognormal, sigma=0.25) representing
uncertainty in the site's true enrollment capability, on top of Poisson
sampling noise for the monthly counts themselves. Trajectories are simulated
for 24 months (this study's actual observed duration). Note the raw
enrollment.csv month index is relative to each site's own participation
start; this simulation treats month 1 as a common study-level activation
point and layers sampled delays on top of it, to test how staggered site
activation affects overall time-to-target.

Reports P(cumulative enrollment >= 500 patients by month 12) and the
50th/80th/90th percentile time-to-target (a percentile that falls in the
un-reached tail is reported as "> horizon", not silently dropped). Saves a
fan chart of simulated trajectories to outputs/monte_carlo_fanchart.png.

Usage:
    python scripts/forecast_and_simulate.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

STUDY_ID = "STU-0051"
FORECAST_HORIZON = 6

TARGET_PATIENTS = 500
TARGET_CHECKPOINT_MONTH = 12
N_ITER = 5_000
SIM_HORIZON = 24
N_LATE_SITES = 3
LATE_SITE_DELAY = 2
RANDOM_SEED = 42

PALETTE = {
    "band_outer": "#b7d3f6",
    "band_mid": "#86b6ef",
    "band_inner": "#5598e7",
    "median": "#256abf",
    "actual": "#eb6834",
    "target_line": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "surface": "#fcfcfb",
}


def load_study_monthly(study_id: str):
    enrollment = pd.read_csv(DATA_DIR / "enrollment.csv")
    sub = enrollment[enrollment["study_id"] == study_id].copy()
    monthly = sub.groupby("month")["patients_enrolled"].sum().sort_index()
    monthly.index = monthly.index.astype(int)
    return monthly, sub


def fit_forecasts(monthly: pd.Series):
    # statsmodels wants a proper time index to forecast against; the actual
    # calendar start is arbitrary since "month" here is study-relative.
    ts = monthly.copy()
    ts.index = pd.period_range(start="2023-01", periods=len(ts), freq="M")

    ets_fit = ExponentialSmoothing(ts, trend="add", damped_trend=True, seasonal=None).fit()
    ets_forecast = ets_fit.forecast(FORECAST_HORIZON)

    arima_fit = ARIMA(ts, order=(1, 1, 1)).fit()
    arima_result = arima_fit.get_forecast(FORECAST_HORIZON)
    arima_forecast = arima_result.predicted_mean
    arima_ci = arima_result.conf_int(alpha=0.20)

    return ets_fit, ets_forecast, arima_fit, arima_forecast, arima_ci


def run_monte_carlo(sub: pd.DataFrame, sites: pd.DataFrame, rng: np.random.Generator):
    site_rate = sub.groupby("site_id")["patients_enrolled"].mean().rename("mean_rate")
    site_ids = site_rate.index.to_numpy()
    base_rates = site_rate.to_numpy()
    n_sites = len(site_ids)

    quality = sites.set_index("site_id").loc[site_ids, "quality_score"]
    late_sites = quality.sort_values().index[:N_LATE_SITES].to_numpy()
    late_mask = np.isin(site_ids, late_sites)

    jitter = rng.choice([0, 1], size=(N_ITER, n_sites), p=[0.85, 0.15])
    activation_delay = jitter + np.where(late_mask, LATE_SITE_DELAY, 0)[None, :]
    activation_month = 1 + activation_delay  # shape (N_ITER, n_sites)

    rate_multiplier = rng.lognormal(mean=0.0, sigma=0.25, size=(N_ITER, n_sites))
    # A per-iteration shock shared across all sites (referral pipeline strength,
    # seasonal effects, competing-trial launches) -- unlike per-site noise, this
    # doesn't diversify away as more sites are pooled, so it's what keeps
    # P(reach target) from collapsing to ~100% just because there are 27 sites.
    systemic_shock = rng.lognormal(mean=0.0, sigma=0.15, size=(N_ITER, 1))
    effective_rate = base_rates[None, :] * rate_multiplier * systemic_shock  # (N_ITER, n_sites)

    months = np.arange(1, SIM_HORIZON + 1)
    active = months[None, None, :] >= activation_month[:, :, None]  # (N_ITER, n_sites, SIM_HORIZON)

    monthly_counts = rng.poisson(lam=effective_rate[:, :, None], size=(N_ITER, n_sites, SIM_HORIZON))
    monthly_counts = monthly_counts * active
    monthly_total = monthly_counts.sum(axis=1)  # (N_ITER, SIM_HORIZON)
    cumulative = monthly_total.cumsum(axis=1)  # (N_ITER, SIM_HORIZON)

    reached_mask = (cumulative >= TARGET_PATIENTS).any(axis=1)
    first_month_idx = np.argmax(cumulative >= TARGET_PATIENTS, axis=1)
    reach_month = np.where(reached_mask, first_month_idx + 1, np.inf)

    return cumulative, reach_month, reached_mask, late_sites


def summarize_monte_carlo(reach_month: np.ndarray, reached_mask: np.ndarray) -> dict:
    prob_by_12 = float(np.mean(reached_mask & (reach_month <= TARGET_CHECKPOINT_MONTH)))
    fraction_reached = float(reached_mask.mean())

    percentiles = {}
    for p in (50, 80, 90):
        val = np.percentile(reach_month, p)
        percentiles[p] = f"> {SIM_HORIZON} mo (not reached within horizon)" if np.isinf(val) else f"{val:.1f} mo"

    return {"prob_by_12": prob_by_12, "fraction_reached": fraction_reached, "percentiles": percentiles}


def plot_fanchart(cumulative: np.ndarray, monthly_actual: pd.Series, out_path: Path):
    months = np.arange(1, cumulative.shape[1] + 1)
    bands = {p: np.percentile(cumulative, p, axis=0) for p in (5, 10, 25, 50, 75, 90, 95)}
    actual_cum = monthly_actual.reindex(months, fill_value=0).cumsum()

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])

    ax.fill_between(months, bands[5], bands[95], color=PALETTE["band_outer"], label="5th-95th pct", zorder=1)
    ax.fill_between(months, bands[10], bands[90], color=PALETTE["band_mid"], label="10th-90th pct", zorder=2)
    ax.fill_between(months, bands[25], bands[75], color=PALETTE["band_inner"], label="25th-75th pct", zorder=3)
    ax.plot(months, bands[50], color=PALETTE["median"], linewidth=2.5, label="Median (simulated)", zorder=4)
    ax.plot(months, actual_cum.values, color=PALETTE["actual"], linewidth=2, linestyle="-",
             marker="o", markersize=4, label="Actual (historical)", zorder=5)

    ax.axhline(TARGET_PATIENTS, color=PALETTE["target_line"], linestyle="--", linewidth=1.5, zorder=1)
    ax.text(0.5, TARGET_PATIENTS + 15, f"Target: {TARGET_PATIENTS} patients",
            color=PALETTE["ink_secondary"], fontsize=10)
    ax.axvline(TARGET_CHECKPOINT_MONTH, color=PALETTE["target_line"], linestyle=":", linewidth=1.2, zorder=1)

    ax.set_title(f"Monte Carlo Enrollment Simulation: {STUDY_ID} ({N_ITER:,} iterations)",
                 color=PALETTE["ink"], fontsize=13, fontweight="bold")
    ax.set_xlabel("Month", color=PALETTE["ink"])
    ax.set_ylabel("Cumulative patients enrolled", color=PALETTE["ink"])
    ax.set_xlim(1, cumulative.shape[1])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(PALETTE["axis"])
    ax.tick_params(colors=PALETTE["ink_secondary"])
    ax.legend(frameon=False, loc="upper left", labelcolor=PALETTE["ink_secondary"])

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    sites = pd.read_csv(DATA_DIR / "sites.csv")

    monthly, sub = load_study_monthly(STUDY_ID)
    print(f"Study {STUDY_ID}: {len(monthly)} months observed, "
          f"{sub['site_id'].nunique()} participating sites, "
          f"final cumulative enrollment {monthly.sum()}\n")

    print("=== Forecasting (next 6 months) ===")
    ets_fit, ets_forecast, arima_fit, arima_forecast, arima_ci = fit_forecasts(monthly)
    forecast_table = pd.DataFrame({
        "month": range(monthly.index.max() + 1, monthly.index.max() + 1 + FORECAST_HORIZON),
        "exp_smoothing": ets_forecast.values.round(1),
        "arima": arima_forecast.values.round(1),
        "arima_lo80": arima_ci.iloc[:, 0].values.round(1),
        "arima_hi80": arima_ci.iloc[:, 1].values.round(1),
    })
    print(forecast_table.to_string(index=False))
    print(f"\nExponential smoothing AIC: {ets_fit.aic:.1f}   ARIMA(1,1,1) AIC: {arima_fit.aic:.1f}\n")

    print("=== Monte Carlo simulation ===")
    cumulative, reach_month, reached_mask, late_sites = run_monte_carlo(sub, sites, rng)
    print(f"Designated late-activating sites (2-month delay, lowest quality_score): {list(late_sites)}\n")

    summary = summarize_monte_carlo(reach_month, reached_mask)
    print(f"P(reach {TARGET_PATIENTS} patients by month {TARGET_CHECKPOINT_MONTH}): {summary['prob_by_12']:.1%}")
    print(f"Reached {TARGET_PATIENTS} patients within {SIM_HORIZON}-month horizon: {summary['fraction_reached']:.1%} of iterations")
    for p, val in summary["percentiles"].items():
        print(f"{p}th percentile time-to-target: {val}")

    fanchart_path = OUT_DIR / "monte_carlo_fanchart.png"
    plot_fanchart(cumulative, monthly, fanchart_path)
    print(f"\nFan chart written to: {fanchart_path}")


if __name__ == "__main__":
    main()
