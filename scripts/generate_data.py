#!/usr/bin/env python
"""Generate synthetic clinical-trial feasibility data.

Writes four CSVs to data/: studies.csv, sites.csv, enrollment.csv,
patients.csv. Uses a fixed random seed throughout, so re-running the
script reproduces byte-identical output.

Site "capability" (a hidden combination of experience, quality, and
competing-trial load) drives both `sites.prior_enrollment_rate` and the
actual monthly enrollment in enrollment.csv, but each is drawn with its
own noise term, so the two are correlated without being identical.
`eligible_population` is generated mostly independently of capability, so
some large-population sites perform poorly and some small ones overperform,
deliberately, so downstream models have to learn that population alone
is not predictive.

Usage:
    python scripts/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

N_STUDIES = 100
N_SITES = 1_000
N_PATIENTS = 100_000

THERAPEUTIC_AREAS = [
    "Oncology", "Cardiology", "Neurology", "Immunology",
    "Infectious Disease", "Endocrinology", "Respiratory", "Rare Disease",
]
DIAGNOSES_BY_AREA = {
    "Oncology": ["Breast Cancer", "Lung Cancer", "Colorectal Cancer", "Prostate Cancer", "Lymphoma"],
    "Cardiology": ["Heart Failure", "Atrial Fibrillation", "Hypertension", "Coronary Artery Disease"],
    "Neurology": ["Alzheimer's Disease", "Parkinson's Disease", "Multiple Sclerosis", "Epilepsy"],
    "Immunology": ["Rheumatoid Arthritis", "Psoriasis", "Lupus", "Crohn's Disease"],
    "Infectious Disease": ["HIV", "Hepatitis C", "Tuberculosis", "COVID-19"],
    "Endocrinology": ["Type 2 Diabetes", "Type 1 Diabetes", "Thyroid Disorder", "Obesity"],
    "Respiratory": ["Asthma", "COPD", "Pulmonary Fibrosis"],
    "Rare Disease": ["Cystic Fibrosis", "Duchenne Muscular Dystrophy", "Huntington's Disease"],
}
ALL_DIAGNOSES = [d for ds in DIAGNOSES_BY_AREA.values() for d in ds]
PHASES = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
PHASE_DURATION_MONTHS = {"Phase 1": (6, 12), "Phase 2": (9, 18), "Phase 3": (12, 24), "Phase 4": (6, 12)}
PHASE_ENROLLMENT_FACTOR = {"Phase 1": 0.6, "Phase 2": 1.0, "Phase 3": 1.8, "Phase 4": 1.3}
COUNTRIES = {
    "USA": "North America", "Canada": "North America", "Mexico": "North America",
    "Germany": "Europe", "France": "Europe", "UK": "Europe", "Poland": "Europe",
    "Spain": "Europe", "Italy": "Europe", "Netherlands": "Europe",
    "Brazil": "Latin America", "Argentina": "Latin America", "Chile": "Latin America",
    "India": "Asia Pacific", "South Korea": "Asia Pacific", "Japan": "Asia Pacific",
    "Australia": "Asia Pacific", "China": "Asia Pacific",
    "South Africa": "Africa/Middle East", "Israel": "Africa/Middle East",
}


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def gen_studies(rng: np.random.Generator) -> pd.DataFrame:
    therapeutic_area = rng.choice(THERAPEUTIC_AREAS, N_STUDIES)
    phase = rng.choice(PHASES, N_STUDIES, p=[0.15, 0.30, 0.40, 0.15])
    protocol_complexity = rng.integers(1, 11, N_STUDIES)
    number_of_countries = rng.integers(1, 31, N_STUDIES)

    phase_factor = np.array([PHASE_ENROLLMENT_FACTOR[p] for p in phase])
    base = 20 + 15 * number_of_countries
    target_enrollment = (base * phase_factor * rng.lognormal(0, 0.35, N_STUDIES)).round().astype(int)
    target_enrollment = np.clip(target_enrollment, 20, None)

    return pd.DataFrame({
        "study_id": [f"STU-{i:04d}" for i in range(N_STUDIES)],
        "therapeutic_area": therapeutic_area,
        "phase": phase,
        "target_enrollment": target_enrollment,
        "protocol_complexity": protocol_complexity,
        "number_of_countries": number_of_countries,
    })


def gen_sites(rng: np.random.Generator) -> pd.DataFrame:
    country_names = list(COUNTRIES.keys())
    country = rng.choice(country_names, N_SITES)
    region = np.array([COUNTRIES[c] for c in country])
    therapeutic_area = rng.choice(THERAPEUTIC_AREAS, N_SITES)

    site_experience = rng.integers(0, 26, N_SITES)
    investigator_experience = rng.integers(0, 31, N_SITES)
    prior_trials = rng.poisson(lam=np.clip(1 + site_experience * 0.6, 0.2, None))
    competing_trials = rng.poisson(lam=3, size=N_SITES)

    quality_score = np.clip(
        rng.normal(60 + site_experience * 0.9 + investigator_experience * 0.4, 12, N_SITES), 0, 100
    )
    startup_days = np.clip(
        rng.normal(200 - site_experience * 3.5 - investigator_experience * 1.0, 45, N_SITES), 20, 450
    ).round().astype(int)

    # Hidden capability latent: drives both prior_enrollment_rate and actual
    # monthly enrollment in enrollment.csv, each with independent noise.
    capability = (
        0.35 * zscore(site_experience.astype(float))
        + 0.25 * zscore(investigator_experience.astype(float))
        + 0.25 * zscore(quality_score)
        - 0.15 * zscore(competing_trials.astype(float))
        + rng.normal(0, 0.5, N_SITES)
    )

    # eligible_population: mostly independent of capability (weak 12% weight)
    # so some high-population sites underperform and some low-population
    # sites overperform, on purpose.
    population_base = rng.lognormal(mean=9.2, sigma=1.0, size=N_SITES)
    eligible_population = np.clip(
        (population_base * (1 + 0.12 * capability)).round().astype(int), 100, None
    )

    prior_enrollment_rate = np.clip(
        2.0 + 2.8 * capability + rng.normal(0, 1.6, N_SITES), 0.1, None
    ).round(2)

    df = pd.DataFrame({
        "site_id": [f"SITE-{i:05d}" for i in range(N_SITES)],
        "country": country,
        "region": region,
        "therapeutic_area": therapeutic_area,
        "site_experience": site_experience,
        "investigator_experience": investigator_experience,
        "prior_trials": prior_trials,
        "prior_enrollment_rate": prior_enrollment_rate,
        "startup_days": startup_days,
        "quality_score": quality_score.round(1),
        "eligible_population": eligible_population,
        "competing_trials": competing_trials,
    })
    df["_capability"] = capability  # kept in-memory only, not written to CSV
    return df


def gen_enrollment(rng: np.random.Generator, studies: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    rows = []
    site_ta = sites["therapeutic_area"].to_numpy()
    site_ids = sites["site_id"].to_numpy()
    capability = sites["_capability"].to_numpy()
    quality = sites["quality_score"].to_numpy()
    competing = sites["competing_trials"].to_numpy()

    for _, study in studies.iterrows():
        match_mask = site_ta == study["therapeutic_area"]
        n_match, n_other = match_mask.sum(), (~match_mask).sum()
        weights = np.where(match_mask, 0.8 / max(n_match, 1), 0.2 / max(n_other, 1))
        weights = weights / weights.sum()

        n_participating = int(np.clip(rng.integers(5, 51), 1, N_SITES))
        chosen_idx = rng.choice(N_SITES, size=n_participating, replace=False, p=weights)

        lo, hi = PHASE_DURATION_MONTHS[study["phase"]]
        duration = int(rng.integers(lo, hi + 1))
        complexity_drag = 1 + 0.05 * study["protocol_complexity"]

        for idx in chosen_idx:
            base_lambda = np.clip(
                (3 + 4 * capability[idx]) / complexity_drag - 0.15 * competing[idx], 0.05, None
            )
            months = np.arange(1, duration + 1)
            screened = rng.poisson(lam=base_lambda, size=duration)
            conv_prob = np.clip(sigmoid(0.04 * (quality[idx] - 50) + 0.5 * capability[idx]), 0.15, 0.85)
            enrolled = rng.binomial(screened, conv_prob)

            block = pd.DataFrame({
                "study_id": study["study_id"],
                "site_id": site_ids[idx],
                "month": months,
                "patients_screened": screened,
                "patients_enrolled": enrolled,
            })
            rows.append(block)

    return pd.concat(rows, ignore_index=True)


def gen_patients(rng: np.random.Generator, sites: pd.DataFrame) -> pd.DataFrame:
    site_ids = sites["site_id"].to_numpy()
    weights = sites["eligible_population"].to_numpy().astype(float)
    weights = weights / weights.sum()

    site_idx = rng.choice(N_SITES, size=N_PATIENTS, p=weights)
    patient_site_id = site_ids[site_idx]
    patient_country = sites["country"].to_numpy()[site_idx]
    patient_ta = sites["therapeutic_area"].to_numpy()[site_idx]
    patient_quality = sites["quality_score"].to_numpy()[site_idx]

    age = np.clip(rng.normal(58, 16, N_PATIENTS), 18, 95).round().astype(int)
    sex = rng.choice(["F", "M"], N_PATIENTS, p=[0.52, 0.48])
    comorbidity = rng.poisson(lam=1.3, size=N_PATIENTS)
    prior_treatment = rng.random(N_PATIENTS) < 0.3

    diagnosis = np.empty(N_PATIENTS, dtype=object)
    on_area = rng.random(N_PATIENTS) < 0.9
    for area, options in DIAGNOSES_BY_AREA.items():
        mask = on_area & (patient_ta == area)
        diagnosis[mask] = rng.choice(options, mask.sum())
    off_mask = ~on_area
    diagnosis[off_mask] = rng.choice(ALL_DIAGNOSES, off_mask.sum())

    start_day = pd.Timestamp("2021-01-01").value // 10**9
    end_day = pd.Timestamp("2025-12-31").value // 10**9
    diagnosis_date = pd.to_datetime(
        rng.integers(start_day, end_day, N_PATIENTS), unit="s"
    ).normalize()

    age_score = np.where((age >= 18) & (age <= 75), 1.0, -1.0)
    logit = (
        0.5 * age_score
        - 0.3 * comorbidity
        + 0.02 * (patient_quality - 50)
        - 0.4 * prior_treatment.astype(float)
        + rng.normal(0, 1, N_PATIENTS)
    )
    eligible = rng.random(N_PATIENTS) < sigmoid(logit)

    return pd.DataFrame({
        "patient_id": [f"PT-{i:06d}" for i in range(N_PATIENTS)],
        "site_id": patient_site_id,
        "age": age,
        "sex": sex,
        "diagnosis": diagnosis,
        "diagnosis_date": diagnosis_date,
        "prior_treatment": prior_treatment,
        "comorbidity": comorbidity,
        "geography": patient_country,
        "eligible": eligible,
    })


def print_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name} ({len(df):,} rows, {len(df.columns)} cols) ===")
    print(df.dtypes.to_string())
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        print(numeric.describe().T[["mean", "std", "min", "max"]].round(2).to_string())
    categorical = df.select_dtypes(include=["object", "bool"])
    for col in categorical.columns:
        if df[col].nunique() <= 15:
            print(f"-- {col} --")
            print(df[col].value_counts().to_string())


def main():
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    studies = gen_studies(rng)
    sites = gen_sites(rng)
    enrollment = gen_enrollment(rng, studies, sites)
    patients = gen_patients(rng, sites)

    sites_out = sites.drop(columns=["_capability"])

    studies.to_csv(DATA_DIR / "studies.csv", index=False)
    sites_out.to_csv(DATA_DIR / "sites.csv", index=False)
    enrollment.to_csv(DATA_DIR / "enrollment.csv", index=False)
    patients.to_csv(DATA_DIR / "patients.csv", index=False)

    print_summary("studies.csv", studies)
    print_summary("sites.csv", sites_out)
    print_summary("enrollment.csv", enrollment)
    print_summary("patients.csv", patients)

    # Sanity-check the correlation structure requested: prior_enrollment_rate
    # should track actual enrollment loosely, and eligible_population should
    # NOT be a strong predictor on its own.
    site_actual = enrollment.groupby("site_id")["patients_enrolled"].mean().rename("actual_mean_enrolled")
    check = sites_out.set_index("site_id").join(site_actual, how="inner")
    corr_prior_actual = check["prior_enrollment_rate"].corr(check["actual_mean_enrolled"])
    corr_pop_prior = check["eligible_population"].corr(check["prior_enrollment_rate"])
    corr_pop_actual = check["eligible_population"].corr(check["actual_mean_enrolled"])

    print("\n=== Correlation sanity checks ===")
    print(f"prior_enrollment_rate vs actual mean enrolled/month : {corr_prior_actual:.3f}  (want moderate, not ~1.0)")
    print(f"eligible_population vs prior_enrollment_rate         : {corr_pop_prior:.3f}  (want weak)")
    print(f"eligible_population vs actual mean enrolled/month    : {corr_pop_actual:.3f}  (want weak)")

    print(f"\nCSVs written to: {DATA_DIR}")


if __name__ == "__main__":
    main()
