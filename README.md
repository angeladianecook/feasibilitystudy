# feasibilitystudy

Synthetic clinical-trial feasibility data, plus a series of modeling
exercises built on top of it: enrollment prediction, site success
classification, survival analysis, forecasting/simulation, and leakage
stress-testing.

Nothing here is real patient data: `scripts/generate_data.py` generates
the entire dataset from a fixed random seed.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Generate the data

```bash
python scripts/generate_data.py
python scripts/load_duckdb.py
```

The first command writes `studies.csv`, `sites.csv`, `enrollment.csv`, and
`patients.csv` to `data/`. The second loads those four CSVs into a local
DuckDB database at `data/icon.duckdb`. Both are safe to rerun any time; the
generated CSVs and database are gitignored.

## Data model

- **studies**: one row per study: therapeutic area, phase, target
  enrollment, protocol complexity, number of countries.
- **sites**: one row per trial site: country/region, therapeutic area,
  investigator/site experience, prior trials, prior enrollment rate,
  startup days, quality score, eligible population, competing trials.
- **enrollment**: one row per (study, site, month): patients screened and
  enrolled that month. Site-level enrollment as a function of site
  covariates plus noise, so downstream models have real signal to recover.
- **patients**: one row per patient: age, sex, diagnosis, diagnosis date,
  prior treatment, comorbidity count, geography, eligibility.

See `sql/feasibility_queries.sql` for example feasibility queries (eligible
patient count per site, historical enrollment rate, sites with strong
population/performance/startup-speed, and the "population exists but the
site can't recruit it" gap), runnable via `scripts/run_queries.py`.

## Project layout

```
data/
  studies.csv, sites.csv, enrollment.csv, patients.csv   generated CSVs (gitignored)
  icon.duckdb   DuckDB copy of the same four tables (gitignored)
  processed/    feature tables / model-ready datasets, if any exercise needs them
scripts/
  generate_data.py        synthetic data generator
  load_duckdb.py           loads the CSVs into data/icon.duckdb
  run_queries.py           runs sql/feasibility_queries.sql, prints formatted tables
  model_enrollment.py      OLS / Poisson / negative binomial enrollment models
  model_site_success.py    logistic regression / gradient boosting classifier + calibration
  survival_analysis.py     Kaplan-Meier + Cox PH on time to an enrollment milestone
  forecast_and_simulate.py exponential smoothing / ARIMA + Monte Carlo simulation
  stress_test.py           six leakage/generalization/bias audits of the models above
  leakage_demo.py          before/after demo of what real target leakage looks like
sql/
  feasibility_queries.sql  the four feasibility queries above
notebooks/
  01_enrollment_prediction.ipynb
  02_site_success_classification.ipynb
  03_survival_analysis.ipynb
  04_forecasting_simulation.ipynb
  05_leakage_stress_testing.ipynb
outputs/
  *.md   model summaries and interview talking points
  *.png  calibration curve, KM curve, Monte Carlo fan chart
```

The scripts in `scripts/` are the canonical, runnable implementation of
each exercise; the notebooks are lightweight stubs over the same data for
interactive exploration.

## Modeling exercises

1. **Enrollment prediction**: predict a site's enrolled-patient count over
   its first 3 months of participation from site features known at/before
   activation. `scripts/model_enrollment.py`.
2. **Site success classification**: classify whether a site will sustain
   enough enrollment to be worth activating. `scripts/model_site_success.py`.
3. **Survival analysis**: Kaplan-Meier and Cox proportional-hazards
   modeling of time to an enrollment milestone, with censoring.
   `scripts/survival_analysis.py`.
4. **Forecasting/simulation**: forecast monthly enrollment for an active
   study and Monte Carlo-simulate time to a target enrollment count under
   site-activation-delay scenarios. `scripts/forecast_and_simulate.py`.
5. **Leakage stress-testing**: audit the enrollment and site-success models
   for target/temporal/study leakage, generalization gaps, missing-data
   behavior, and bias, then demonstrate what real leakage does to
   performance. `scripts/stress_test.py` and `scripts/leakage_demo.py`.

Run `jupyter lab` from the project root (with `venv` activated) to work
through the notebooks, or run any script in `scripts/` directly for the
full analysis and saved outputs.

## Run everything

From a clean checkout, this is the full pipeline in order (each modeling
script depends on `data/*.csv` from `generate_data.py`; `run_queries.py`
additionally depends on `data/icon.duckdb` from `load_duckdb.py`):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/load_duckdb.py
python scripts/run_queries.py
python scripts/model_enrollment.py
python scripts/model_site_success.py
python scripts/survival_analysis.py
python scripts/forecast_and_simulate.py
python scripts/stress_test.py
python scripts/leakage_demo.py
```
