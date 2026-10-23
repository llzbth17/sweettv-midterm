# Lead-in residual reranker — forward validation (rejected)

Candidate file: `submission_stable2_h.csv`. The returned Kaggle score was worse than the
confirmed `0.44224` public score, so the candidate is rejected and must not be uploaded
again.

## Features

- **Lead-in / channel flow.** For each user, source programmes are those watched in the
  last 7 days before the cut. Global transitions keep `0 < next_start - air_start <= 180`
  minutes, weight each transition by the product of the two screen-time ratios, and keep
  the top 40 destinations per source. The user score is
  `sum_source engagement(source) * P(destination | source)`, normalised by the user's
  maximum.
- **Residual reranker.** Logistic regression for the hidden Top-5 indicator with
  `log(hand)` as an offset. Residual features: `lead`, `vt`, repeat share,
  channel/category/slot affinity, popularity, user-slot exposure, user activity, trend,
  full-view rate, and full-view count. The correction is shrunk by `lambda = 0.5`.
- **Standardisation.** The model is fitted on the training-window feature scale; the test
  window is standardised with its own mean/std. This removes the observed level shift in
  full-view counts and recent age between the May-July windows and the July-October test
  window. The choice was made by forward validation, not by the test labels.

## Forward validation

Training uses only strictly earlier windows, so no target overlaps the holdout. `w4` is
the latest 12-week window and is the closest analogue of the final test window.

| Holdout | Base `hand` MAP@5 | Reranked MAP@5 | Gain | Paired bootstrap 95% CI |
|---|---:|---:|---:|---|
| `w4` (2020-05-04 to 2020-07-27) | 0.4089 | 0.4124 | +0.0035 | [+0.0005, +0.0065] |
| `w3` (2020-04-27 to 2020-07-20) | 0.4107 | 0.4119 | +0.0013 | [-0.0019, +0.0044] |
| `w2` (2020-04-20 to 2020-07-13) | 0.4198 | 0.4217 | +0.0018 | [-0.0015, +0.0052] |

`w4` win/loss rates are 17.1% / 13.5% of users (the rest tie), and the gain is positive in
both random user halves (+0.0028 and +0.0043). The candidate preserves 740/1,260 Top-5
sets exactly versus `submission.csv`; average set intersection is 4.55/5.

## Why this is not the rejected GBM

The rejected CatBoost candidate had a local `+0.0018` but scored `0.43022` private /
`0.44072` public. The new reranker uses a **linear residual with a fixed offset**, a
**single new temporal feature**, and **feature-scale matching**; there is no tree model
over 74 features. The validation gain is consistent in sign on all three forward windows,
but the `w3`/`w2` intervals include zero. The public test confirmed that the gain did not
transfer: the candidate scored worse than `0.44224`, so it is rejected. Keep the confirmed
fallbacks.
