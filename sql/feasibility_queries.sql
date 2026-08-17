-- Feasibility queries against data/icon.duckdb.
-- Tables are loaded from data/{studies,sites,enrollment,patients}.csv via
-- scripts/load_duckdb.py. Each query below is parsed and run in order by
-- scripts/run_queries.py, which looks for the "-- N. Title" header lines.

-- 1. Eligible patient count per site
SELECT
    site_id,
    COUNT(*)                              AS total_patients,
    COUNT(*) FILTER (WHERE eligible)      AS eligible_patients,
    ROUND(COUNT(*) FILTER (WHERE eligible) * 1.0 / COUNT(*), 3) AS eligible_rate
FROM patients
GROUP BY site_id
ORDER BY eligible_patients DESC
;

-- 2. Historical enrollment rate per site (enrolled/screened, and enrolled per site-month)
SELECT
    site_id,
    COUNT(*)                                     AS site_months,
    SUM(patients_screened)                       AS total_screened,
    SUM(patients_enrolled)                       AS total_enrolled,
    ROUND(SUM(patients_enrolled) * 1.0 / NULLIF(SUM(patients_screened), 0), 3) AS enrolled_per_screened,
    ROUND(SUM(patients_enrolled) * 1.0 / NULLIF(COUNT(*), 0), 3)               AS enrolled_per_site_month
FROM enrollment
GROUP BY site_id
ORDER BY enrolled_per_site_month DESC
;

-- 3. Sites with high eligible_population, good historical performance, and low startup_days
WITH ranked AS (
    SELECT
        site_id,
        country,
        region,
        eligible_population,
        prior_enrollment_rate,
        startup_days,
        NTILE(3) OVER (ORDER BY eligible_population ASC)   AS population_tercile,   -- 3 = highest population
        NTILE(3) OVER (ORDER BY prior_enrollment_rate ASC) AS performance_tercile,  -- 3 = highest historical rate
        NTILE(3) OVER (ORDER BY startup_days DESC)         AS speed_tercile         -- 3 = lowest (fastest) startup_days
    FROM sites
)
SELECT
    ROW_NUMBER() OVER (ORDER BY eligible_population DESC, prior_enrollment_rate DESC, startup_days ASC) AS site_rank,
    site_id,
    country,
    region,
    eligible_population,
    prior_enrollment_rate,
    startup_days
FROM ranked
WHERE population_tercile = 3
  AND performance_tercile = 3
  AND speed_tercile = 3
ORDER BY site_rank
;

-- 4. Sites with high eligible_population but poor historical enrollment
--    (the "population exists but site can't recruit it" gap)
WITH ranked AS (
    SELECT
        site_id,
        country,
        region,
        eligible_population,
        prior_enrollment_rate,
        startup_days,
        NTILE(3) OVER (ORDER BY eligible_population ASC)   AS population_tercile,   -- 3 = highest population
        NTILE(3) OVER (ORDER BY prior_enrollment_rate ASC) AS performance_tercile   -- 1 = lowest historical rate
    FROM sites
)
SELECT
    site_id,
    country,
    region,
    eligible_population,
    prior_enrollment_rate,
    startup_days,
    ROUND(eligible_population / NULLIF(prior_enrollment_rate, 0), 1) AS population_per_unit_rate
FROM ranked
WHERE population_tercile = 3
  AND performance_tercile = 1
ORDER BY population_per_unit_rate DESC
;
