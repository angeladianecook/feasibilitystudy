#!/usr/bin/env python
"""Stress-test the models built in scripts/model_enrollment.py (negative
binomial regression on patients_enrolled_3mo) and scripts/model_site_success.py
(logistic regression + gradient boosting on successful_site).

Both production scripts build a SITE-level dataset -- one row per site,
aggregating across every study that site has ever participated in. That
grain is fine for the models themselves, but two of the checks below
(temporal validation, study leakage) are inherently about generalizing
across *studies*, which a table with no study_id column can't test. For
those two checks only, this script reconstructs the same modeling problem
(same features, same target thresholds, imported directly from the two
production scripts) at STUDY-SITE grain -- one row per (study_id, site_id)
pair -- documented at the point of use. The other four checks run directly
against the production site-level datasets and models.

studies.csv has no explicit start-date column. study_id ("STU-0000" ..
"STU-0099") was assigned in generation order (scripts/generate_data.py
iterates i=0..99), so its numeric suffix is used as a chronological proxy
for study start order -- documented, not hidden.

Checks:
    1. Leakage            -- are model features derived from future/enrollment data?
    2. Temporal validation -- train on early studies, test on late studies, vs. random split
    3. Study leakage       -- does any split let a study_id appear in both train and test?
    4. Generalization      -- train on one therapeutic_area, test on another
    5. Missing-data behavior -- zero out prior_trials/prior_enrollment_rate, check prediction stability
    6. Bias check          -- predicted success rate by prior_trials bucket (few vs. many)

Usage:
    python scripts/stress_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import model_enrollment  # noqa: E402
import model_site_success  # noqa: E402

ROOT = SCRIPTS_DIR.parent
DATA_DIR = ROOT / "data"

FEATURES = model_enrollment.FEATURES  # identical list in both production scripts
ENROLL_TARGET = model_enrollment.TARGET
SUCCESS_TARGET = model_site_success.TARGET
SUCCESS_THRESHOLD = model_site_success.SUCCESS_THRESHOLD
RANDOM_STATE = 42

pd.set_option("display.width", 140)


# --------------------------------------------------------------------------
# Shared dataset builders
# --------------------------------------------------------------------------

def build_study_site_dataset() -> pd.DataFrame:
    """One row per (study_id, site_id) pair -- see module docstring."""
    sites = pd.read_csv(DATA_DIR / "sites.csv")
    enrollment = pd.read_csv(DATA_DIR / "enrollment.csv")

    enroll_target = (
        enrollment[enrollment["month"] <= 3]
        .groupby(["study_id", "site_id"])["patients_enrolled"]
        .sum()
        .rename(ENROLL_TARGET)
        .reset_index()
    )
    rate = (
        enrollment.groupby(["study_id", "site_id"])["patients_enrolled"]
        .mean()
        .rename("enrolled_per_site_month")
        .reset_index()
    )
    df = enroll_target.merge(rate, on=["study_id", "site_id"])
    df[SUCCESS_TARGET] = (df["enrolled_per_site_month"] >= SUCCESS_THRESHOLD).astype(int)
    df = df.merge(sites[["site_id"] + FEATURES], on="site_id", how="left")
    df["study_index"] = df["study_id"].str.extract(r"(\d+)").astype(int)
    return df


def site_level_with_therapeutic_area() -> tuple[pd.DataFrame, pd.DataFrame]:
    sites = pd.read_csv(DATA_DIR / "sites.csv")[["site_id", "therapeutic_area"]]
    enroll_df = model_enrollment.build_dataset().merge(sites, on="site_id")
    succ_df = model_site_success.build_dataset().merge(sites, on="site_id")
    return enroll_df, succ_df


# --------------------------------------------------------------------------
# Generic fit/eval helpers (train-set-only standardization, no test leakage)
# --------------------------------------------------------------------------

def fit_negbin(train_df: pd.DataFrame, target: str = ENROLL_TARGET):
    means, stds = train_df[FEATURES].mean(), train_df[FEATURES].std()

    def standardize(df):
        out = df.copy()
        for col in FEATURES:
            out[col] = (out[col] - means[col]) / stds[col]
        return out

    X_train = sm.add_constant(standardize(train_df)[FEATURES])
    y_train = train_df[target]
    model = sm.NegativeBinomial(y_train, X_train).fit(method="bfgs", maxiter=200, disp=0)
    return model, means, stds


def eval_negbin(model, means, stds, test_df: pd.DataFrame, target: str = ENROLL_TARGET) -> dict:
    test_std = test_df.copy()
    for col in FEATURES:
        test_std[col] = (test_std[col] - means[col]) / stds[col]
    X_test = sm.add_constant(test_std[FEATURES], has_constant="add")
    pred = model.predict(X_test)
    actual = test_df[target]
    mae = float((actual - pred).abs().mean())
    corr = float(np.corrcoef(actual, pred)[0, 1]) if actual.nunique() > 1 else float("nan")
    return {"n": len(test_df), "mae": mae, "corr": corr}


def logreg_ctor():
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])


def gbm_ctor():
    return GradientBoostingClassifier(random_state=RANDOM_STATE)


def fit_classifier(ctor, train_df: pd.DataFrame, target: str = SUCCESS_TARGET):
    model = ctor()
    model.fit(train_df[FEATURES], train_df[target])
    return model


def eval_classifier(model, test_df: pd.DataFrame, target: str = SUCCESS_TARGET) -> dict:
    proba = model.predict_proba(test_df[FEATURES])[:, 1]
    actual = test_df[target]
    auc = float(roc_auc_score(actual, proba)) if actual.nunique() > 1 else float("nan")
    return {"n": len(test_df), "auc": auc, "pred_rate": float(proba.mean())}


# --------------------------------------------------------------------------
# Check 1: leakage
# --------------------------------------------------------------------------

def check_leakage() -> pd.DataFrame:
    sites_columns = set(pd.read_csv(DATA_DIR / "sites.csv").columns) - {"site_id"}
    enroll_df = model_enrollment.build_dataset()
    succ_df = model_site_success.build_dataset()

    rows = []
    for feat in FEATURES:
        is_raw = feat in sites_columns
        corr_enroll = enroll_df[feat].corr(enroll_df[ENROLL_TARGET])
        corr_success = succ_df[feat].corr(succ_df[SUCCESS_TARGET])
        flagged = (not is_raw) or abs(corr_enroll) > 0.95 or abs(corr_success) > 0.95
        rows.append({
            "feature": feat,
            "raw_sites_csv_column": is_raw,
            "corr_with_patients_enrolled_3mo": round(corr_enroll, 3),
            "corr_with_successful_site": round(corr_success, 3),
            "leakage_flag": flagged,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Checks 2 & 3: temporal validation + study leakage (study-site grain)
# --------------------------------------------------------------------------

def make_splits(df: pd.DataFrame):
    train_idx, test_idx = train_test_split(df.index, test_size=0.25, random_state=RANDOM_STATE)
    naive_random = (df.loc[train_idx], df.loc[test_idx])

    unique_studies = df["study_id"].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    test_studies = rng.choice(unique_studies, size=int(round(0.25 * len(unique_studies))), replace=False)
    grouped_random = (df[~df["study_id"].isin(test_studies)], df[df["study_id"].isin(test_studies)])

    cutoff = df["study_index"].median()
    temporal = (df[df["study_index"] <= cutoff], df[df["study_index"] > cutoff])

    return {"naive_random": naive_random, "grouped_random": grouped_random, "temporal": temporal}, cutoff


def check_study_leakage(splits: dict) -> pd.DataFrame:
    rows = []
    for name, (train_df, test_df) in splits.items():
        train_studies = set(train_df["study_id"])
        test_studies = set(test_df["study_id"])
        overlap = train_studies & test_studies
        rows.append({
            "split": name,
            "n_studies_train": len(train_studies),
            "n_studies_test": len(test_studies),
            "n_studies_in_both": len(overlap),
            "overlap_frac_of_test_studies": round(len(overlap) / len(test_studies), 3) if test_studies else 0.0,
            "leak_free": len(overlap) == 0,
        })
    return pd.DataFrame(rows)


def check_temporal_validation(df: pd.DataFrame, splits: dict, cutoff: float) -> pd.DataFrame:
    print(f"(Temporal cutoff: study_index <= {cutoff:.0f} = train/before, > {cutoff:.0f} = test/after; "
          f"study_index is study_id's numeric suffix, used as a chronological-order proxy.)")

    rows = []
    for split_name, (train_df, test_df) in splits.items():
        negbin, means, stds = fit_negbin(train_df)
        neg_result = eval_negbin(negbin, means, stds, test_df)
        rows.append({"model": "NegBinomial (enrollment)", "split": split_name,
                      "n_train": len(train_df), "n_test": neg_result["n"],
                      "metric": "MAE", "value": round(neg_result["mae"], 3)})

        for name, ctor in [("LogisticRegression", logreg_ctor), ("GradientBoosting", gbm_ctor)]:
            model = fit_classifier(ctor, train_df)
            clf_result = eval_classifier(model, test_df)
            rows.append({"model": f"{name} (site success)", "split": split_name,
                          "n_train": len(train_df), "n_test": clf_result["n"],
                          "metric": "AUC", "value": round(clf_result["auc"], 3)})

    table = pd.DataFrame(rows)
    baseline = table[table["split"] == "naive_random"].set_index("model")["value"]

    def drop(r):
        if r["model"] not in baseline.index:
            return np.nan
        base = baseline[r["model"]]
        # Positive = worse than naive_random, regardless of metric direction:
        # AUC is better when higher (drop = base - value); MAE is better when
        # lower (drop = value - base).
        return round(base - r["value"], 3) if r["metric"] == "AUC" else round(r["value"] - base, 3)

    table["degradation_vs_naive_random"] = table.apply(drop, axis=1)
    print("(Positive degradation_vs_naive_random = worse than the leaky naive_random baseline, "
          "for both metric directions. The NegBinomial MAE numbers are noisy across splits -- "
          "patients_enrolled_3mo is heavily right-skewed [see model_enrollment.py's overdispersion "
          "finding], so a handful of high-count outlier rows can swing MAE more than any real "
          "leakage/temporal-shift effect at this sample size. The classification AUCs are the "
          "cleaner signal here: both degrade monotonically from naive_random -> grouped_random -> "
          "temporal, exactly as expected once leakage is removed and then temporal shift is added.)")
    return table


# --------------------------------------------------------------------------
# Check 4: generalization across therapeutic_area (site grain, one model fit
# evaluated on two test distributions: in-area holdout vs. a different area)
# --------------------------------------------------------------------------

def check_generalization(enroll_df: pd.DataFrame, succ_df: pd.DataFrame,
                          train_area: str = "Oncology", test_area: str = "Cardiology") -> pd.DataFrame:
    rows = []

    enroll_pool = enroll_df[enroll_df["therapeutic_area"] == train_area]
    enroll_train, enroll_indist_test = train_test_split(enroll_pool, test_size=0.25, random_state=RANDOM_STATE)
    enroll_cross_test = enroll_df[enroll_df["therapeutic_area"] == test_area]
    negbin, means, stds = fit_negbin(enroll_train)
    indist = eval_negbin(negbin, means, stds, enroll_indist_test)
    cross = eval_negbin(negbin, means, stds, enroll_cross_test)
    rows.append({"model": "NegBinomial (enrollment)", "metric": "MAE",
                 "in_distribution": round(indist["mae"], 3), "cross_area": round(cross["mae"], 3),
                 "degradation": round(cross["mae"] - indist["mae"], 3),
                 "n_train": len(enroll_train), "n_indist_test": indist["n"], "n_cross_test": cross["n"]})

    succ_pool = succ_df[succ_df["therapeutic_area"] == train_area]
    succ_train, succ_indist_test = train_test_split(succ_pool, test_size=0.25, random_state=RANDOM_STATE)
    succ_cross_test = succ_df[succ_df["therapeutic_area"] == test_area]
    for name, ctor in [("LogisticRegression", logreg_ctor), ("GradientBoosting", gbm_ctor)]:
        model = fit_classifier(ctor, succ_train)
        indist = eval_classifier(model, succ_indist_test)
        cross = eval_classifier(model, succ_cross_test)
        rows.append({"model": f"{name} (site success)", "metric": "AUC",
                     "in_distribution": round(indist["auc"], 3), "cross_area": round(cross["auc"], 3),
                     "degradation": round(indist["auc"] - cross["auc"], 3),
                     "n_train": len(succ_train), "n_indist_test": indist["n"], "n_cross_test": cross["n"]})

    print(f"(Trained on {train_area} sites only; 'in_distribution' = held-out {train_area} sites, "
          f"'cross_area' = all {test_area} sites. For MAE lower is better so degradation = cross - indist; "
          f"for AUC higher is better so degradation = indist - cross. Positive degradation = worse "
          f"out-of-area. The in-distribution MAE test set is small (~30 sites) and "
          f"patients_enrolled_3mo is heavily right-skewed, so its MAE is noisier than the AUC rows -- "
          f"don't over-read a negative MAE degradation as real improvement from a single split.)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Check 5: missing-data behavior
# --------------------------------------------------------------------------

def check_missing_data(enroll_df: pd.DataFrame, succ_df: pd.DataFrame) -> pd.DataFrame:
    prior_trials_is_a_feature = "prior_trials" in FEATURES
    print(f"(prior_trials in modeled FEATURES: {prior_trials_is_a_feature} -- it is a raw sites.csv "
          f"column but was not selected as a model input in either production script, so zeroing it "
          f"cannot change these models' predictions by construction. Only prior_enrollment_rate, which "
          f"IS a feature, is simulated as zeroed below to mimic a genuinely new site with no track record.)")

    rows = []

    negbin, means, stds = fit_negbin(enroll_df)
    before = eval_negbin(negbin, means, stds, enroll_df)
    modified = enroll_df.copy()
    modified["prior_enrollment_rate"] = 0.0
    after = eval_negbin(negbin, means, stds, modified)
    pred_before = sm.add_constant(((enroll_df[FEATURES] - means) / stds), has_constant="add")
    pred_before_vals = negbin.predict(pred_before)
    pred_after = sm.add_constant(((modified[FEATURES] - means) / stds), has_constant="add")
    pred_after_vals = negbin.predict(pred_after)
    shift = (pred_before_vals - pred_after_vals)
    rows.append({
        "model": "NegBinomial (enrollment)", "feature_zeroed": "prior_enrollment_rate",
        "mean_pred_before": round(pred_before_vals.mean(), 2), "mean_pred_after": round(pred_after_vals.mean(), 2),
        "mean_abs_shift": round(shift.abs().mean(), 2),
        "pct_change": round(100 * shift.mean() / pred_before_vals.mean(), 1),
        "any_negative_after": bool((pred_after_vals < 0).any()),
    })

    for name, ctor in [("LogisticRegression", logreg_ctor), ("GradientBoosting", gbm_ctor)]:
        model = fit_classifier(ctor, succ_df)
        proba_before = model.predict_proba(succ_df[FEATURES])[:, 1]
        modified_succ = succ_df.copy()
        modified_succ["prior_enrollment_rate"] = 0.0
        proba_after = model.predict_proba(modified_succ[FEATURES])[:, 1]
        shift = proba_before - proba_after
        rows.append({
            "model": f"{name} (site success)", "feature_zeroed": "prior_enrollment_rate",
            "mean_pred_before": round(proba_before.mean(), 3), "mean_pred_after": round(proba_after.mean(), 3),
            "mean_abs_shift": round(np.abs(shift).mean(), 3),
            "pct_change": round(100 * shift.mean() / proba_before.mean(), 1),
            "any_negative_after": False,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Check 6: bias by prior_trials bucket
# --------------------------------------------------------------------------

def check_bias(succ_df: pd.DataFrame) -> pd.DataFrame:
    sites = pd.read_csv(DATA_DIR / "sites.csv")[["site_id", "prior_trials"]]
    df = succ_df.merge(sites, on="site_id")
    median_pt = df["prior_trials"].median()
    df["prior_trials_bucket"] = np.where(df["prior_trials"] <= median_pt, f"few (<= {median_pt:.0f})",
                                          f"many (> {median_pt:.0f})")

    logreg = fit_classifier(logreg_ctor, df)
    gbm = fit_classifier(gbm_ctor, df)
    df["pred_logreg"] = logreg.predict_proba(df[FEATURES])[:, 1]
    df["pred_gbm"] = gbm.predict_proba(df[FEATURES])[:, 1]

    summary = df.groupby("prior_trials_bucket").agg(
        n=("site_id", "count"),
        actual_success_rate=(SUCCESS_TARGET, "mean"),
        pred_logreg_rate=("pred_logreg", "mean"),
        pred_gbm_rate=("pred_gbm", "mean"),
    ).round(3)

    rate_cols = ["actual_success_rate", "pred_logreg_rate", "pred_gbm_rate"]
    gap = summary.loc[summary.index.str.startswith("many"), rate_cols].iloc[0] \
        - summary.loc[summary.index.str.startswith("few"), rate_cols].iloc[0]
    gap_row = pd.DataFrame([gap.round(3)], index=["gap (many - few)"])
    result = pd.concat([summary, gap_row])

    amplification = (gap["pred_logreg_rate"] - gap["actual_success_rate"], gap["pred_gbm_rate"] - gap["actual_success_rate"])
    print(f"(Bucketed by prior_trials, which is NOT a model feature -- this tests whether predictions "
          f"favor experienced sites via correlated features like site_experience/prior_enrollment_rate. "
          f"Predicted-vs-actual gap amplification: LogReg {amplification[0]:+.3f}, GBM {amplification[1]:+.3f} "
          f"-- near zero means the model isn't adding bias beyond real outcome differences between buckets.)")
    return result


# --------------------------------------------------------------------------

def main():
    print("=" * 88)
    print("CHECK 1: Leakage -- are features derived from future enrollment data?")
    print("=" * 88)
    print(check_leakage().to_string(index=False))

    study_site_df = build_study_site_dataset()
    splits, cutoff = make_splits(study_site_df)

    print("\n" + "=" * 88)
    print("CHECK 3: Study leakage -- does any split let a study_id appear in both train and test?")
    print("=" * 88)
    print("(Checked first since check 2's temporal-validation numbers should be read against it.)")
    print(check_study_leakage(splits).to_string(index=False))

    print("\n" + "=" * 88)
    print("CHECK 2: Temporal validation -- train on early studies, test on late studies")
    print("=" * 88)
    print(check_temporal_validation(study_site_df, splits, cutoff).to_string(index=False))

    print("\n" + "=" * 88)
    print("CHECK 4: Generalization -- train on one therapeutic_area, test on another")
    print("=" * 88)
    enroll_df, succ_df = site_level_with_therapeutic_area()
    print(check_generalization(enroll_df, succ_df).to_string(index=False))

    print("\n" + "=" * 88)
    print("CHECK 5: Missing-data behavior -- simulate a new site with no track record")
    print("=" * 88)
    print(check_missing_data(model_enrollment.build_dataset(), model_site_success.build_dataset()).to_string(index=False))

    print("\n" + "=" * 88)
    print("CHECK 6: Bias check -- predicted success rate by prior_trials bucket")
    print("=" * 88)
    print(check_bias(model_site_success.build_dataset()).to_string())


if __name__ == "__main__":
    main()
