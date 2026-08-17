#!/usr/bin/env python
"""Binary classification of site success: logistic regression vs. gradient
boosting, with calibration diagnostics.

Label (successful_site): a site is "successful" if it averaged at least
1.5 enrolled patients per site-month across all of its study
participations in data/enrollment.csv (sites that never participated in
any study count as 0/month). 1.5 patients/month is close to the median
site's performance (median ~1.31) and gives a roughly balanced label
(~45% positive over all 1,000 sites), an operationally meaningful bar
(most feasibility teams would call anything enrolling less than 1-2
patients a month a stalled site) rather than an arbitrary quantile split.

Features (all known at/before site activation, from data/sites.csv):
eligible_population, prior_enrollment_rate, site_experience,
investigator_experience, startup_days, quality_score, competing_trials.

Fits logistic regression (scaled) and gradient boosting on an identical
train/test split, reports AUC, precision/recall, and a confusion matrix
for both, plots a calibration curve (reliability diagram) with a count
histogram to outputs/calibration_plot.png, and writes a short summary,
including whether sites predicted at ~90% actually succeed ~90% of the
time, to outputs/site_success_model_summary.md.

Usage:
    python scripts/model_site_success.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
TARGET = "successful_site"
SUCCESS_THRESHOLD = 1.5  # avg. patients enrolled per site-month
N_BINS = 10

PALETTE = {
    "logreg": "#2a78d6",
    "gbm": "#eb6834",
    "diagonal": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "surface": "#fcfcfb",
}


def build_dataset() -> pd.DataFrame:
    sites = pd.read_csv(DATA_DIR / "sites.csv")
    enrollment = pd.read_csv(DATA_DIR / "enrollment.csv")

    rate = enrollment.groupby("site_id")["patients_enrolled"].mean().rename("enrolled_per_site_month")
    df = sites.set_index("site_id").join(rate, how="left").reset_index()
    df["enrolled_per_site_month"] = df["enrolled_per_site_month"].fillna(0.0)
    df[TARGET] = (df["enrolled_per_site_month"] >= SUCCESS_THRESHOLD).astype(int)
    return df[["site_id", TARGET, "enrolled_per_site_month"] + FEATURES]


def fit_models(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.25, random_state=42, stratify=df[TARGET]
    )

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    logreg.fit(X_train, y_train)

    gbm = GradientBoostingClassifier(random_state=42)
    gbm.fit(X_train, y_train)

    return logreg, gbm, X_test, y_test


def evaluate(name: str, model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, pred, target_names=["not_successful", "successful"])
    cm = confusion_matrix(y_test, pred)
    brier = brier_score_loss(y_test, proba)

    print(f"\n=== {name} ===")
    print(f"AUC: {auc:.3f}   Brier score: {brier:.3f}")
    print(report)
    print("Confusion matrix ([[TN, FP], [FN, TP]]):")
    print(cm)

    return {"name": name, "proba": proba, "auc": auc, "report": report, "cm": cm, "brier": brier}


def near_90_calibration(prob_true, prob_pred, counts):
    idx = np.argmin(np.abs(prob_pred - 0.9))
    return {
        "bin_mean_predicted": prob_pred[idx],
        "bin_observed_rate": prob_true[idx],
        "bin_n": counts[idx],
    }


def plot_calibration(results, y_test, out_path: Path):
    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 7.5), height_ratios=[3, 1], sharex=True,
        gridspec_kw={"hspace": 0.08}, constrained_layout=True,
    )
    fig.patch.set_facecolor(PALETTE["surface"])

    ax_cal.plot([0, 1], [0, 1], linestyle="--", linewidth=2, color=PALETTE["diagonal"],
                label="Perfectly calibrated", zorder=1)

    near_90 = {}
    for res, color in zip(results, [PALETTE["logreg"], PALETTE["gbm"]]):
        prob_true, prob_pred = calibration_curve(y_test, res["proba"], n_bins=N_BINS, strategy="uniform")
        bin_edges = np.linspace(0, 1, N_BINS + 1)
        bin_idx = np.digitize(res["proba"], bin_edges[1:-1])
        counts = np.array([np.sum(bin_idx == i) for i in range(len(bin_edges) - 1)])
        counts_nonempty = counts[counts > 0]

        ax_cal.plot(prob_pred, prob_true, marker="o", markersize=8, linewidth=2,
                    color=color, label=res["name"], zorder=2)

        ax_hist.hist(res["proba"], bins=bin_edges, color=color, alpha=0.55,
                     edgecolor=PALETTE["surface"], linewidth=0.5, label=res["name"])

        near_90[res["name"]] = near_90_calibration(prob_true, prob_pred, counts_nonempty)

    ax_cal.set_ylabel("Observed fraction of successful sites", color=PALETTE["ink"])
    ax_cal.set_title("Calibration Curve (Reliability Diagram): Site Success Prediction",
                      color=PALETTE["ink"], fontsize=13, fontweight="bold")
    ax_cal.set_xlim(-0.02, 1.02)
    ax_cal.set_ylim(-0.02, 1.02)
    ax_cal.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax_cal.spines[["top", "right"]].set_visible(False)
    ax_cal.spines[["left", "bottom"]].set_color(PALETTE["axis"])
    ax_cal.tick_params(colors=PALETTE["ink_secondary"])
    ax_cal.legend(frameon=False, loc="upper left", labelcolor=PALETTE["ink_secondary"])

    ax_hist.set_xlabel("Mean predicted probability", color=PALETTE["ink"])
    ax_hist.set_ylabel("Count", color=PALETTE["ink"])
    ax_hist.grid(True, color=PALETTE["grid"], linewidth=0.8, axis="y")
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.spines[["left", "bottom"]].set_color(PALETTE["axis"])
    ax_hist.tick_params(colors=PALETTE["ink_secondary"])
    ax_hist.legend(frameon=False, loc="upper right", labelcolor=PALETTE["ink_secondary"])

    for ax in (ax_cal, ax_hist):
        ax.set_facecolor(PALETTE["surface"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    return near_90


def write_summary(df, results, near_90, out_path: Path):
    positive_rate = df[TARGET].mean()

    lines = [
        "# Site Success Model Summary",
        "",
        "## Label definition",
        "",
        f"`successful_site` = 1 if a site's average `patients_enrolled` per "
        f"site-month in `data/enrollment.csv` is >= {SUCCESS_THRESHOLD} "
        f"(sites that never participated in a study count as 0/month), else 0. "
        f"Positive rate across all {len(df)} sites: {positive_rate:.1%}. "
        f"{SUCCESS_THRESHOLD} patients/month sits close to the median site's "
        f"performance and represents a stalled-vs-functioning-site cutoff, "
        f"chosen for a roughly balanced label rather than an arbitrary quantile split.",
        "",
        "## Model comparison",
        "",
        "| Model | AUC | Brier score |",
        "|---|---|---|",
    ]
    for res in results:
        lines.append(f"| {res['name']} | {res['auc']:.3f} | {res['brier']:.3f} |")

    lines += ["", "### Classification reports (threshold = 0.5)", ""]
    for res in results:
        lines += [f"**{res['name']}**", "", "```", res["report"].rstrip(), "```", ""]
        cm = res["cm"]
        lines += [
            f"Confusion matrix: TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}",
            "",
        ]

    lines += ["## Calibration at ~90% predicted probability", ""]
    for name, info in near_90.items():
        gap = info["bin_observed_rate"] - info["bin_mean_predicted"]
        verdict = "well-calibrated" if abs(gap) < 0.10 else "miscalibrated"
        lines.append(
            f"- **{name}**: nearest bin to 90% predicted probability has mean "
            f"predicted probability {info['bin_mean_predicted']:.1%} "
            f"(n={info['bin_n']} sites), and those sites actually succeeded "
            f"{info['bin_observed_rate']:.1%} of the time "
            f"(gap = {gap:+.1%}), **{verdict}** at that bin."
        )
    lines += [
        "",
        "See `outputs/calibration_plot.png` for the full reliability diagram "
        "(top) and the predicted-probability histogram per model (bottom), "
        "which shows how many test sites actually fall near each bin. A "
        "single bin's calibration is only as trustworthy as its sample size.",
        "",
    ]

    out_path.write_text("\n".join(lines) + "\n")


def main():
    df = build_dataset()
    print(f"Dataset: {len(df)} sites, {df[TARGET].sum()} successful ({df[TARGET].mean():.1%})")

    logreg, gbm, X_test, y_test = fit_models(df)

    results = [
        evaluate("Logistic Regression", logreg, X_test, y_test),
        evaluate("Gradient Boosting", gbm, X_test, y_test),
    ]

    plot_path = OUT_DIR / "calibration_plot.png"
    near_90 = plot_calibration(results, y_test, plot_path)
    print(f"\nCalibration plot written to: {plot_path}")

    print("\n=== Calibration near 90% predicted probability ===")
    for name, info in near_90.items():
        print(f"{name}: predicted {info['bin_mean_predicted']:.1%}, "
              f"observed {info['bin_observed_rate']:.1%} (n={info['bin_n']})")

    summary_path = OUT_DIR / "site_success_model_summary.md"
    write_summary(df, results, near_90, summary_path)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
