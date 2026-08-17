# feasibilitystudy

Synthetic clinical-trial feasibility data, plus a series of modeling
exercises built on top of it: enrollment prediction, site success
classification, survival analysis, forecasting/simulation, and leakage
stress-testing.

Nothing here is real patient data — `scripts/generate_synthetic_data.py`
generates the entire dataset from a seeded random process.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Generate the data

```bash
python scripts/generate_synthetic_data.py --seed 42 --n-sites 60
```

This writes `sites.csv`, `patients.csv`, and `screening_events.csv` to
`data/raw/`, and loads the same three tables into a SQLite database at
`data/feasibility.db` using the schema in `sql/schema.sql`. Regenerate any
time — the generated CSVs and database are gitignored.

## Data model

- **sites** — one row per trial site: country/region, site type, investigator
  experience, prior trials conducted, catchment population, competing
  trials, activation date, enrollment target, and the `site_success` label
  (met >=80% of target enrollment).
- **patients** — one row per screened candidate: demographics, comorbidity
  count, referral source, screening/randomization dates, arm assignment,
  and (for randomized patients) survival fields `time_to_event_days` /
  `event_observed` / `dropout`.
- **screening_events** — one row per funnel-stage transition per patient
  (`referred` -> `screen_failed` or `randomized`), used to reconstruct the
  screening funnel over time.

See `sql/schema.sql` for full column definitions and a few example
feasibility queries (monthly enrollment, screen-fail reasons, site success
balance).

## Project layout

```
data/
  raw/          generated CSVs (gitignored, recreate via the script below)
  processed/    feature tables / model-ready datasets produced by notebooks
  feasibility.db  SQLite copy of the same tables (gitignored)
scripts/
  generate_synthetic_data.py   synthetic data generator
sql/
  schema.sql    table definitions + example queries
notebooks/
  01_enrollment_prediction.ipynb
  02_site_success_classification.ipynb
  03_survival_analysis.ipynb
  04_forecasting_simulation.ipynb
  05_leakage_stress_testing.ipynb
```

## Modeling exercises

1. **Enrollment prediction** — predict site- and study-level enrollment
   counts/rates from site features and early screening-funnel activity.
2. **Site success classification** — classify whether a site will hit its
   enrollment target using only features known at/before activation.
3. **Survival analysis** — Kaplan-Meier and Cox proportional-hazards
   modeling of time-to-event outcomes, treatment vs. control.
4. **Forecasting/simulation** — forecast monthly enrollment trajectories and
   simulate trial completion timelines under different site-activation
   scenarios.
5. **Leakage stress-testing** — deliberately probe the earlier models for
   target and temporal leakage, and demonstrate how performance changes
   once leakage is removed.

Run `jupyter lab` from the project root (with `venv` activated) to work
through the notebooks.
