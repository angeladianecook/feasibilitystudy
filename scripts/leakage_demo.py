#!/usr/bin/env python
"""Demonstrate target leakage in the site-success classifier.

Takes model_site_success.py's 7 legitimate features (all known at/before
site activation) and adds one engineered from the eventual/future study
outcome: final_enrollment_total, a site's total patients_enrolled summed
across its ENTIRE observed history in data/enrollment.csv (every month of
every study it participated in, not just early data). That number can only
exist once a site has finished enrolling. A deployed feasibility model
would never have it at decision time, and it is mechanically close to the
target itself: successful_site is defined as (mean enrolled/month >=
threshold), and final_enrollment_total is that same history's numerator
before dividing by months observed. Including it should make the classifier
look unrealistically good.

Fits logistic regression and gradient boosting twice each, with vs. without
the leaked feature (identical train/test split both times), and prints AUC
side by side, plus where the leaked feature ranks in each model's
importances/coefficients when it's included.

Usage:
    python scripts/leakage_demo.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import model_site_success  # noqa: E402

ROOT = SCRIPTS_DIR.parent
DATA_DIR = ROOT / "data"

FEATURES = model_site_success.FEATURES
TARGET = model_site_success.TARGET
LEAKED_FEATURE = "final_enrollment_total"
RANDOM_STATE = 42


def build_dataset_with_leak() -> pd.DataFrame:
    df = model_site_success.build_dataset()
    enrollment = pd.read_csv(DATA_DIR / "enrollment.csv")
    total = enrollment.groupby("site_id")["patients_enrolled"].sum().rename(LEAKED_FEATURE)
    df = df.merge(total, on="site_id", how="left")
    df[LEAKED_FEATURE] = df[LEAKED_FEATURE].fillna(0.0)
    return df


def logreg_ctor():
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])


def gbm_ctor():
    return GradientBoostingClassifier(random_state=RANDOM_STATE)


def fit_and_eval(ctor, df: pd.DataFrame, feature_cols: list, train_idx, test_idx):
    model = ctor()
    model.fit(df.loc[train_idx, feature_cols], df.loc[train_idx, TARGET])
    proba = model.predict_proba(df.loc[test_idx, feature_cols])[:, 1]
    auc = roc_auc_score(df.loc[test_idx, TARGET], proba)
    return model, auc


def leaked_feature_rank(model, feature_cols: list) -> str:
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        rank = list(importances.index).index(LEAKED_FEATURE) + 1
        return f"#{rank}/{len(feature_cols)} by importance ({importances[LEAKED_FEATURE]:.3f})"

    coefs = pd.Series(model.named_steps["clf"].coef_[0], index=feature_cols)
    ranked = coefs.abs().sort_values(ascending=False)
    rank = list(ranked.index).index(LEAKED_FEATURE) + 1
    return f"#{rank}/{len(feature_cols)} by |coef| ({coefs[LEAKED_FEATURE]:+.3f} standardized)"


def main():
    df = build_dataset_with_leak()
    leaked_features = FEATURES + [LEAKED_FEATURE]

    train_idx, test_idx = train_test_split(
        df.index, test_size=0.25, random_state=RANDOM_STATE, stratify=df[TARGET]
    )

    print(f"Dataset: {len(df)} sites, {df[TARGET].mean():.1%} successful")
    print(f"Leaked feature: {LEAKED_FEATURE} = total patients_enrolled summed across a site's "
          f"ENTIRE enrollment.csv history (all months, all studies), only knowable after a "
          f"site has finished enrolling, not at decision time.\n")

    rows = []
    for name, ctor in [("LogisticRegression", logreg_ctor), ("GradientBoosting", gbm_ctor)]:
        model_leak, auc_leak = fit_and_eval(ctor, df, leaked_features, train_idx, test_idx)
        _, auc_clean = fit_and_eval(ctor, df, FEATURES, train_idx, test_idx)
        rows.append({
            "model": name,
            "AUC_with_leak": round(auc_leak, 3),
            "AUC_without_leak": round(auc_clean, 3),
            "inflation": round(auc_leak - auc_clean, 3),
            "leaked_feature_rank": leaked_feature_rank(model_leak, leaked_features),
        })

    table = pd.DataFrame(rows)
    print("=== AUC: with leaked feature vs. without (identical train/test split) ===")
    print(table.to_string(index=False))

    print(
        "\nThe leaked feature inflates AUC because it is mechanically close to the target: "
        f"successful_site is defined as (mean patients_enrolled/month >= "
        f"{model_site_success.SUCCESS_THRESHOLD}), and {LEAKED_FEATURE} is that same enrollment "
        "history's numerator, just not divided by months observed. A real feasibility model can "
        "never see this at decision time; it doesn't exist until the site has already finished "
        "enrolling. So the 'with leak' numbers above are not a model anyone could deploy; they're "
        "the warning sign to look for whenever a model's performance looks too good to be true."
    )


if __name__ == "__main__":
    main()
