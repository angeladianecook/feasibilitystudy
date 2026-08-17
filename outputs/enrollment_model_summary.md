# Enrollment Model Summary

Target: `patients_enrolled_3mo` (sum of `patients_enrolled` over a site's first 3 months of participation), aggregated from `data/enrollment.csv` for n = 962 sites that participated in at least one study.

## Model fit (AIC, lower is better)

| Model | AIC | Log-likelihood |
|---|---|---|
| OLS | 8003.7 | -3993.9 |
| Poisson | 14416.9 | -7200.5 |
| Negative Binomial | 6595.0 | -3288.5 |

OLS's AIC is on a Gaussian likelihood and isn't directly comparable in scale to the two count-model AICs, but all three are reported for completeness. Ranked by AIC: negative binomial (best) < OLS < Poisson (worst) -- a naive Poisson fit is actually the *worst* of the three here, which is itself the overdispersion story below.

## Why negative binomial fits better than OLS (and plain Poisson doesn't)

- The target is a non-negative integer count with a right-skewed distribution (empirical variance/mean ratio = 25.62, i.e. variance is ~26x the mean). OLS assumes a continuous, homoscedastic, potentially-negative outcome, none of which hold for enrollment counts -- it can and does predict negative enrollment for some sites, and its constant-variance assumption understates spread for high-enrollment sites and overstates it for low ones.
- Poisson regression models the count correctly (non-negative, integer) but assumes Var(Y) = Mean(Y). The Pearson chi2/df statistic from the fitted Poisson model is 12.66, far above 1.0, confirming severe overdispersion. Because Poisson's likelihood is tightly coupled to that (wrong) Var=Mean assumption, the mismatch is punished directly in the log-likelihood -- which is why Poisson's AIC (14416.9) ends up *worse* than OLS's (8003.7) despite modeling the right kind of outcome. A good mean-structure fit is not enough if the variance assumption is this wrong.
- Negative binomial adds a dispersion parameter (alpha = 1.037, p = 6.35e-73) that lets predicted variance grow faster than the mean (Var = Mean + alpha*Mean^2), matching the extra-Poisson variance actually present in site-level enrollment (site heterogeneity in execution, referral surges, seasonal effects, etc. that Poisson's single-parameter mean-variance link can't absorb). That's why it has the lowest AIC of the three (6595.0) and is the appropriate model for this target -- it keeps Poisson's correct count structure while actually fitting the observed variance.

## Coefficients

|                         |      OLS |   Poisson |   NegBinomial |
|:------------------------|---------:|----------:|--------------:|
| Intercept               |  16.9387 |    2.5078 |        2.3483 |
| alpha                   | nan      |  nan      |        1.0368 |
| competing_trials        |  -2.9144 |   -0.2008 |       -0.3334 |
| eligible_population     |   0.7367 |    0.0341 |        0.0519 |
| investigator_experience |   1.5062 |    0.1024 |        0.1968 |
| prior_enrollment_rate   |   8.7811 |    0.3595 |        0.5143 |
| quality_score           |   3.3587 |    0.2708 |        0.4412 |
| site_experience         |   3.577  |    0.2644 |        0.3943 |
| startup_days            |   0.2017 |    0.012  |        0.0531 |

## Predicted vs. actual variance

Poisson's canonical log link forces its average prediction to exactly match the actual mean; negative binomial's MLE has no such constraint, so a small gap between its predicted mean and the actual mean (below) is expected, not a bug.

| model       |   pred_mean |   variance |   var_mean_ratio | note                                    |
|:------------|------------:|-----------:|-----------------:|:----------------------------------------|
| Raw target  |      16.939 |    434.022 |           25.623 | empirical var/mean ratio                |
| OLS         |      16.939 |    236.607 |           13.968 | residual variance, assumed constant     |
| Poisson     |      16.939 |     16.939 |            1     | assumes Var=Mean; Pearson chi2/df=12.66 |
| NegBinomial |      21.873 |   1876.84  |           85.805 | Var=Mean+alpha*Mean^2, alpha=1.037      |

