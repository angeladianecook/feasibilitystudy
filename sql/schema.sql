-- Schema for synthetic clinical-trial feasibility data.
-- Mirrors the tables written by scripts/generate_synthetic_data.py.
-- Grain: one row per site in `sites`, one row per randomized patient in
-- `patients`, one row per screened-patient funnel event in `screening_events`.

CREATE TABLE IF NOT EXISTS sites (
    site_id             TEXT PRIMARY KEY,
    country             TEXT NOT NULL,
    region              TEXT NOT NULL,
    site_type           TEXT NOT NULL,      -- academic | community | hospital_network
    investigator_experience_years INTEGER NOT NULL,
    prior_trials_conducted INTEGER NOT NULL,
    catchment_population INTEGER NOT NULL,
    competing_trials_active INTEGER NOT NULL,
    activation_date     DATE NOT NULL,
    target_enrollment   INTEGER NOT NULL,
    site_success        INTEGER NOT NULL    -- 1 = met >=80% of target_enrollment on time, else 0
);

CREATE TABLE IF NOT EXISTS patients (
    patient_id          TEXT PRIMARY KEY,
    site_id             TEXT NOT NULL REFERENCES sites(site_id),
    age                 INTEGER NOT NULL,
    sex                 TEXT NOT NULL,
    comorbidity_count   INTEGER NOT NULL,
    referral_source     TEXT NOT NULL,      -- physician | advertisement | registry | walk_in
    screen_date         DATE NOT NULL,
    randomized          INTEGER NOT NULL,   -- 1 if passed screening and randomized
    randomization_date  DATE,
    arm                 TEXT,               -- treatment | control (null if not randomized)
    event_observed      INTEGER,            -- 1 = event (e.g. discontinuation/relapse) occurred, 0 = censored
    time_to_event_days  INTEGER,            -- follow-up duration in days from randomization
    dropout             INTEGER             -- 1 if patient discontinued before trial end
);

CREATE TABLE IF NOT EXISTS screening_events (
    event_id            TEXT PRIMARY KEY,
    patient_id          TEXT NOT NULL REFERENCES patients(patient_id),
    site_id             TEXT NOT NULL REFERENCES sites(site_id),
    funnel_stage        TEXT NOT NULL,      -- referred | pre_screened | consented | screen_failed | randomized
    stage_date          DATE NOT NULL,
    screen_fail_reason  TEXT                -- populated only when funnel_stage = 'screen_failed'
);

-- Example feasibility queries -------------------------------------------

-- Monthly enrollment counts by site (input to enrollment prediction / forecasting).
-- SELECT site_id, strftime('%Y-%m', randomization_date) AS month, COUNT(*) AS n_randomized
-- FROM patients
-- WHERE randomized = 1
-- GROUP BY site_id, month
-- ORDER BY site_id, month;

-- Screen-failure rate by reason (feeds leakage / data-quality checks).
-- SELECT screen_fail_reason, COUNT(*) AS n
-- FROM screening_events
-- WHERE funnel_stage = 'screen_failed'
-- GROUP BY screen_fail_reason
-- ORDER BY n DESC;

-- Site-level success label balance (input to site success classification).
-- SELECT site_success, COUNT(*) FROM sites GROUP BY site_success;
