#!/usr/bin/env python
"""Generate synthetic clinical-trial feasibility data.

Writes sites.csv, patients.csv, and screening_events.csv to data/raw/, and
loads the same tables into a SQLite database at data/feasibility.db using
the schema in sql/schema.sql.

Usage:
    python scripts/generate_synthetic_data.py [--seed 42] [--n-sites 60]
"""
import argparse
import sqlite3
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "feasibility.db"
SCHEMA_PATH = ROOT / "sql" / "schema.sql"

COUNTRIES = ["USA", "Germany", "Poland", "Brazil", "India", "South Korea", "Canada", "Spain"]
REGION_BY_COUNTRY = {
    "USA": "North America", "Canada": "North America",
    "Germany": "Europe", "Poland": "Europe", "Spain": "Europe",
    "Brazil": "Latin America",
    "India": "Asia Pacific", "South Korea": "Asia Pacific",
}
SITE_TYPES = ["academic", "community", "hospital_network"]
REFERRAL_SOURCES = ["physician", "advertisement", "registry", "walk_in"]
SCREEN_FAIL_REASONS = [
    "inclusion_criteria_not_met", "exclusion_criteria_met",
    "withdrew_consent", "lost_to_followup", "lab_values_out_of_range",
]


def gen_sites(rng: np.random.Generator, fake: Faker, n_sites: int, study_start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for i in range(n_sites):
        country = rng.choice(COUNTRIES)
        site_type = rng.choice(SITE_TYPES, p=[0.3, 0.45, 0.25])
        experience = int(rng.gamma(shape=3.0, scale=3.0))
        prior_trials = int(rng.poisson(lam=max(experience / 3, 0.5)))
        catchment = int(rng.lognormal(mean=12.5, sigma=0.6))
        competing = int(rng.poisson(lam=2.5))
        activation_date = study_start + timedelta(days=int(rng.integers(0, 180)))
        target_enrollment = int(max(5, rng.normal(loc=25, scale=10)))

        # Underlying "true" propensity to succeed, driven by realistic
        # feasibility factors. This is deliberately not saved to disk -
        # only its downstream effects (enrollment counts, site_success)
        # are, so the exercises have to recover the signal from noisy data.
        propensity = (
            0.35 * (experience / 15)
            + 0.25 * (prior_trials / 10)
            + 0.20 * (catchment / 5_000_000)
            - 0.15 * (competing / 8)
            + rng.normal(0, 0.15)
        )
        rows.append(dict(
            site_id=f"SITE-{i:04d}",
            country=country,
            region=REGION_BY_COUNTRY[country],
            site_type=site_type,
            investigator_experience_years=experience,
            prior_trials_conducted=prior_trials,
            catchment_population=catchment,
            competing_trials_active=competing,
            activation_date=activation_date.date().isoformat(),
            target_enrollment=target_enrollment,
            _propensity=propensity,
        ))
    return pd.DataFrame(rows)


def gen_patients_and_events(rng: np.random.Generator, sites: pd.DataFrame, study_end: pd.Timestamp):
    patients, events = [], []

    for _, site in sites.iterrows():
        activation = pd.Timestamp(site["activation_date"])
        enrollment_window_days = max((study_end - activation).days, 1)
        # Higher propensity sites screen more candidates and convert more of them.
        base_rate = 0.5 + 2.5 * max(site["_propensity"], -0.4)
        n_referred = max(0, int(rng.poisson(lam=max(base_rate, 0.1) * site["target_enrollment"] * 1.8)))

        n_randomized_site = 0
        for _ in range(n_referred):
            referral_day = int(rng.integers(0, enrollment_window_days + 1))
            screen_date = activation + timedelta(days=referral_day)
            if screen_date > study_end:
                continue

            patient_id = f"PT-{uuid.uuid4().hex[:10]}"
            age = int(np.clip(rng.normal(55, 15), 18, 90))
            sex = rng.choice(["F", "M"])
            comorbidity_count = int(rng.poisson(lam=1.2))
            referral_source = rng.choice(REFERRAL_SOURCES, p=[0.5, 0.2, 0.2, 0.1])

            events.append(dict(
                event_id=f"EV-{uuid.uuid4().hex[:10]}", patient_id=patient_id,
                site_id=site["site_id"], funnel_stage="referred",
                stage_date=screen_date.date().isoformat(), screen_fail_reason=None,
            ))

            screen_pass_prob = np.clip(0.55 + 0.3 * site["_propensity"] - 0.01 * comorbidity_count, 0.05, 0.95)
            passed = rng.random() < screen_pass_prob

            randomized = 0
            randomization_date = arm = event_observed = time_to_event_days = dropout = None

            if passed:
                events.append(dict(
                    event_id=f"EV-{uuid.uuid4().hex[:10]}", patient_id=patient_id,
                    site_id=site["site_id"], funnel_stage="randomized",
                    stage_date=screen_date.date().isoformat(), screen_fail_reason=None,
                ))
                randomized = 1
                n_randomized_site += 1
                randomization_date = screen_date
                arm = rng.choice(["treatment", "control"])

                max_followup = max((study_end - randomization_date).days, 1)
                true_event_time = rng.weibull(a=1.3) * 400 * (1.15 if arm == "control" else 1.0)
                censor_time = min(max_followup, int(rng.exponential(scale=500)))
                time_to_event_days = int(min(true_event_time, censor_time))
                event_observed = int(true_event_time <= censor_time)
                dropout = int((not event_observed) and rng.random() < 0.12)
            else:
                reason = rng.choice(SCREEN_FAIL_REASONS)
                events.append(dict(
                    event_id=f"EV-{uuid.uuid4().hex[:10]}", patient_id=patient_id,
                    site_id=site["site_id"], funnel_stage="screen_failed",
                    stage_date=screen_date.date().isoformat(), screen_fail_reason=reason,
                ))

            patients.append(dict(
                patient_id=patient_id, site_id=site["site_id"], age=age, sex=sex,
                comorbidity_count=comorbidity_count, referral_source=referral_source,
                screen_date=screen_date.date().isoformat(), randomized=randomized,
                randomization_date=randomization_date.date().isoformat() if randomization_date is not None else None,
                arm=arm, event_observed=event_observed, time_to_event_days=time_to_event_days,
                dropout=dropout,
            ))

    return pd.DataFrame(patients), pd.DataFrame(events), sites


def finalize_sites(sites: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    randomized_counts = (
        patients[patients["randomized"] == 1].groupby("site_id").size().rename("n_randomized")
    )
    sites = sites.merge(randomized_counts, on="site_id", how="left")
    sites["n_randomized"] = sites["n_randomized"].fillna(0).astype(int)
    sites["site_success"] = (sites["n_randomized"] >= 0.8 * sites["target_enrollment"]).astype(int)
    return sites.drop(columns=["_propensity", "n_randomized"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sites", type=int, default=60)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    fake = Faker()
    Faker.seed(args.seed)

    study_start = pd.Timestamp("2023-01-01")
    study_end = pd.Timestamp("2025-06-30")

    sites_raw = gen_sites(rng, fake, args.n_sites, study_start)
    patients, events, sites_raw = gen_patients_and_events(rng, sites_raw, study_end)
    sites = finalize_sites(sites_raw, patients)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sites.to_csv(RAW_DIR / "sites.csv", index=False)
    patients.to_csv(RAW_DIR / "patients.csv", index=False)
    events.to_csv(RAW_DIR / "screening_events.csv", index=False)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA_PATH.read_text())
        sites.to_sql("sites", conn, if_exists="replace", index=False)
        patients.to_sql("patients", conn, if_exists="replace", index=False)
        events.to_sql("screening_events", conn, if_exists="replace", index=False)

    print(f"sites: {len(sites)} rows -> {RAW_DIR / 'sites.csv'}")
    print(f"patients: {len(patients)} rows -> {RAW_DIR / 'patients.csv'}")
    print(f"screening_events: {len(events)} rows -> {RAW_DIR / 'screening_events.csv'}")
    print(f"SQLite database -> {DB_PATH}")


if __name__ == "__main__":
    main()
