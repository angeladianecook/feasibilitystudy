# Interview Talking Points

Outline only — built from what's actually in this repo (`outputs/*.md`, the three
plots, and live reruns of `stress_test.py` / `leakage_demo.py`), not from memory.
One entry per exercise: the question it answers, what I actually found, and one
sentence tying it to a site-selection or feasibility call a Director would make.

## 1. Enrollment prediction (`scripts/model_enrollment.py`)

- **Question:** given a site's known-at-activation characteristics (eligible
  population, historical rate, experience, quality, competing trials), how many
  patients will it enroll in its first 3 months?
- **Found:** OLS, Poisson, and negative binomial fit on the same 962 sites. The
  target is badly overdispersed (variance ≈26x the mean) — that breaks Poisson's
  Var=Mean assumption so hard that naive Poisson's AIC (14,417) is *worse* than
  plain OLS (8,004). Negative binomial (dispersion parameter alpha=1.04, highly
  significant) wins clearly at AIC 6,595. `prior_enrollment_rate` is the
  strongest predictor, ahead of `quality_score` and `site_experience`;
  `competing_trials` pulls enrollment down.
- **Director angle:** modeling the right *kind* of outcome (a count) isn't
  enough if the variance assumption is wrong — a Director should distrust any
  enrollment forecast built on plain Poisson without an overdispersion check,
  and should weight a site's own track record over its raw catchment population.

## 2. Site success classification (`scripts/model_site_success.py`)

- **Question:** will a site sustain enough enrollment (≥1.5 patients/month) to
  be worth activating?
- **Found:** logistic regression (AUC 0.920) edges out gradient boosting (AUC
  0.902) — but the real finding is calibration, not AUC: near 90% predicted
  probability, logistic regression is accurate (85.3% predicted → 82.6%
  observed) while gradient boosting overstates confidence by 12 points (85.0%
  predicted → 72.7% observed). See `calibration_plot.png`.
- **Director angle:** if a model says "90% chance this site succeeds," that
  number is only actionable if it's calibrated — AUC alone would have hidden
  the fact that the fancier model (gradient boosting) is the less trustworthy
  one exactly where it matters most, at the high-confidence end.

## 3. Survival analysis (`scripts/survival_analysis.py`)

- **Question:** not just *whether* a site reaches a meaningful cohort size (20
  patients), but *how long* it takes — properly accounting for sites that
  haven't gotten there yet.
- **Found:** median time to 20 patients is 6 months, but 37% of sites (358/962)
  are censored, never reaching it within the observed window — dropping them or
  counting their censor time as failure would bias the estimate. Cox PH:
  `prior_enrollment_rate` (HR=2.15) and `site_experience` (HR=1.53) speed up
  time-to-milestone, `competing_trials` (HR=0.79) slows it, and
  `eligible_population` is *not* significant (p=0.47) once the others are in
  the model — the same "population isn't destiny" pattern the SQL feasibility
  queries surfaced independently.
- **Director angle:** given two sites with identical catchment population, the
  one with the stronger track record and more experience will hit a usable
  cohort size roughly twice as fast — time-to-milestone planning should be
  built from demonstrated performance, not addressable population.

## 4. Forecasting / simulation (`scripts/forecast_and_simulate.py`)

- **Question:** for an active study (STU-0051), what does the next 6 months of
  enrollment look like, and what's the real probability and date range for
  hitting a 500-patient target?
- **Found:** exponential smoothing and ARIMA(1,1,1) both settle around
  54-56 patients/month steady-state. Monte Carlo (5,000 iterations, seeded from
  each site's actual historical rate, with a built-in stress scenario — a
  2-month activation delay on the 3 lowest-quality-score sites) puts P(reach
  500 patients by month 12) at 94.7%, with 50th/80th/90th percentile
  time-to-target at 10/11/12 months.
- **Director angle:** "will we hit 500 patients by the month-12 review" gets
  answered with a probability and a range (94.7%, 10-12 months) instead of a
  single optimistic point estimate, and the cost of a specific known risk
  (3 slow-starting sites) is quantified rather than hand-waved.

## 5. Leakage stress-testing (`scripts/stress_test.py` + `scripts/leakage_demo.py`)

- **Question:** do the enrollment and site-success models actually generalize,
  or do their good numbers depend on how the data happened to get split or what
  got fed into them?
- **Found:** six checks — no feature is derived from future enrollment data
  (all are raw `sites.csv` columns, correlations ≤0.62); a naive random split
  put 99/100 studies in both train and test (real leakage), while grouped and
  temporal splits were confirmed clean; classification AUC degraded cleanly and
  monotonically as leakage was removed and then temporal shift added
  (0.968 → 0.962 → 0.954, gradient boosting); a genuinely new site with no
  track record (`prior_enrollment_rate` zeroed) sees predictions drop 48-62%;
  and predicted success rates by site-experience bucket track the real observed
  gap almost exactly, i.e. no model-added bias beyond what real outcomes
  support. Separately, `leakage_demo.py` showed the failure mode directly:
  adding one feature engineered from the eventual outcome
  (`final_enrollment_total`) inflates AUC from a real 0.90-0.92 to an
  unrealistic 0.98, dominating the model as the #1 feature.
- **Director angle:** before trusting any vendor or internal model for
  site-selection decisions, ask how it was validated — a random train/test
  split on multi-study, multi-site data can look great while silently leaking
  information across a study's own sites, which is exactly the failure mode
  that makes an undeployable model look deployable.
