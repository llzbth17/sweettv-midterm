# Final feature / ensemble search — 2026-10-23

Goal: beat the confirmed best (0.44224 public / 0.43453 private).
Protocol: five 12-week temporal mirrors ending on the test cut (w0 04-06/29 … w4 05-04/27),
base = `hand` (recent-persistence personal model). A candidate is only promotable if its
mean delta over the five mirrors is clearly positive AND it does not lose badly in any one
mirror (to protect the 50/50 user split).

## What was tested (all flat or negative vs base)

| Direction | dmean (5 mirrors) | dmin | verdict |
|---|---:|---:|---|
| `full_rate` factor (b 0.2/0.3/0.5) | -0.0005…-0.0009 | -0.0021 | reject (short mirrors had misled earlier) |
| 14/28-day recent persistence re-weight (h14/h28) | -0.0002 / -0.0005 | -0.0013 | reject |
| slot / category affinity re-weight | -0.0005 / +0.0003 | -0.0012 | reject |
| decay 0.85 / 0.95 exposure variants | -0.0005 / -0.0007 | -0.0017 | reject |
| channel/recent rank or log blend | -0.0004…+0.0002 | -0.0010 | reject |
| rank ensemble hand+ref+pop (best: wh1 wp2) | +0.0005 | -0.0001 | noise (w4 CI [-0.0012,+0.0020]) |
| channel-exposure blend | +0.0004 | -0.0014 | noise |
| CF (LightFM-style user-kNN) quota 1–3 for unseen | -0.034 (k1) … -0.14 (k3) | — | strongly reject |
| film slot (popularity / channel / genre / slot) | -0.031 (force 1 film) | — | reject (99.6% of film targets unseen) |
| adaptive per-user pick between channel & recent models | 0.0000 (identical) | — | no separable signal |

## Conclusion

The personal ranker is at its local optimum. Every independently-constructed signal that is
available at inference time has been added, re-weighted, ensembled, or used as a quota, and
none survives the 12-week horizon. The two confirmed Kaggle submissions are within 0.00025
private of each other and no leakage-free feature separates them, so there is no safe
per-user ensemble either.

The remaining score is the unpredictable unseen tail (~30% of targets unseen, plus 14% film
targets that are almost all one-off cold items). The public winner is 0.44297; our 0.44224
is 0.00073 behind. 0.5 is not reachable on this label.

## Final recommendation

- Best public: `submission_personal_043453.csv` (0.44224 public / 0.43453 private).
- Best private: `submission_personal_043478.csv` (0.43478 private / 0.44048 public) — current `submission.csv`.
- Do not upload `submission_gbm_ch04.csv` or `submission_stable2_h.csv` again (both regressed).
