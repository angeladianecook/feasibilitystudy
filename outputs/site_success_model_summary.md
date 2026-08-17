# Site Success Model Summary

## Label definition

`successful_site` = 1 if a site's average `patients_enrolled` per site-month in `data/enrollment.csv` is >= 1.5 (sites that never participated in a study count as 0/month), else 0. Positive rate across all 1000 sites: 45.4%. 1.5 patients/month sits close to the median site's performance and represents a stalled-vs-functioning-site cutoff, chosen for a roughly balanced label rather than an arbitrary quantile split.

## Model comparison

| Model | AUC | Brier score |
|---|---|---|
| Logistic Regression | 0.920 | 0.114 |
| Gradient Boosting | 0.902 | 0.128 |

### Classification reports (threshold = 0.5)

**Logistic Regression**

```
                precision    recall  f1-score   support

not_successful       0.84      0.84      0.84       136
    successful       0.81      0.81      0.81       114

      accuracy                           0.82       250
     macro avg       0.82      0.82      0.82       250
  weighted avg       0.82      0.82      0.82       250
```

Confusion matrix: TN=114, FP=22, FN=22, TP=92

**Gradient Boosting**

```
                precision    recall  f1-score   support

not_successful       0.83      0.84      0.84       136
    successful       0.81      0.80      0.80       114

      accuracy                           0.82       250
     macro avg       0.82      0.82      0.82       250
  weighted avg       0.82      0.82      0.82       250
```

Confusion matrix: TN=114, FP=22, FN=23, TP=91

## Calibration at ~90% predicted probability

- **Logistic Regression**: nearest bin to 90% predicted probability has mean predicted probability 85.3% (n=23 sites), and those sites actually succeeded 82.6% of the time (gap = -2.6%), **well-calibrated** at that bin.
- **Gradient Boosting**: nearest bin to 90% predicted probability has mean predicted probability 85.0% (n=22 sites), and those sites actually succeeded 72.7% of the time (gap = -12.3%), **miscalibrated** at that bin.

See `outputs/calibration_plot.png` for the full reliability diagram (top) and the predicted-probability histogram per model (bottom), which shows how many test sites actually fall near each bin. A single bin's calibration is only as trustworthy as its sample size.

