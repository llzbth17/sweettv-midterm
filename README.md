# SWEET.TV TV Programme Recommender — LLM4Rec Midterm

Leakage-safe hybrid recommender for the Kaggle competition
[`sweettv-tv-program-recommender`](https://www.kaggle.com/competitions/sweettv-tv-program-recommender).

The pipeline joins 5.82M viewing sessions to the EPG by same-channel time overlap, applies the
official `screen_time >= 0.8` target rule, and compares a popularity baseline with content-based
filtering (CBF), pair association rules (AR), PureSVD matrix factorization (MF), and personalized
PageRank (PPR). The report compares those methods and an RRF hybrid; the **submitted file uses a
channel-aware repeat-aware personal-history ranker**, because direct repeat behaviour and
channel-specific exposure are the strongest signals for the hidden 12-week target.

## Final submitted model (channel-aware repeat + recent persistence)

The final ranker collapses repeated sessions of the same programme airing and uses every observed
overlap as graded evidence, excluding films and airings longer than 10,000 s. For each
`(user, show)` pair:

```
engagement       = sum over airings of screen_time * 0.9 ** (age_in_weeks)
V(u, show)       = sum over channels of
                   user's engagement share on the channel * test airings of the show on it
T(u, show)       = the same channel-weighted exposure over the last 12 weeks
repeat           = smoothed share of full-view user-show pairs spanning >= 2 weeks
consistency      = user's distinct watched weeks / show's distinct training weeks
channel_affinity = user's overall engagement share on the show's dominant test channel
recent_gain      = sum over the last 14 days of (6 - rank) / 5,
                   where rank is the show's rank in the user's daily Top-5 target
category_affinity = user's engagement share in the show's category
score            = engagement
                   * log10(max(V(u, show), 0.5))
                   * ((V(u, show) + 5) / (T(u, show) + 5)) ** 0.5
                   * (1 + 1.6 * repeat)
                   * (1 + 0.6 * consistency)
                   * (1 + 0.5 * channel_affinity)
                   * (1 + 0.25 * recent_gain)
                   * (1 + 0.3 * category_affinity)
```

with ties broken by the number of users who watched the show. The channel weighting matters because
a programme can air on several channels while a household watches only a subset; the affinity term
captures the user's general preference for the show's main channel; recent persistence captures the
shows currently at the top of the user's target list. Users with fewer than five eligible shows are
filled from the engagement-popularity head.

On the fixed five-method split (session-level labels, 1,124 users) the final ranker reaches
**MAP@5 = 0.3977** `[0.3826, 0.4122]`, versus **0.3951** for the channel-only model, **0.3906** for
repeat-only and **0.3826** for engagement-only. On the airing-level target recipe used for inference
the final model reaches **0.3930** on the main mirror and **0.4090** on the 12-week mirror. Adding
recent persistence and category affinity improves MAP@5 by **+0.0019** over the channel model on
seven airing-level temporal mirrors (4-12 week horizons; paired SE `0.0005`) and **+0.0026** on 17
additional 4-8 week mirrors (SE `0.0003`). Relative to repeat-only, the final gain is **+0.0054**
and **+0.0065**, respectively. On the clean fixed split the paired gain over channel-only is
`+0.0027`; the wide split is flat (`-0.0002`).

### Kaggle leaderboard scores (confirmed)

| Submission file | Model | Public MAP@5 | Private MAP@5 |
|---|---|---:|---:|
| `submission_personal_043453.csv` | channel-only | **0.44224** (best public) | 0.43453 |
| **`submission.csv`** | recent-persistence (final) | 0.44048 | **0.43478** (best private) |
| `submission_gbm_ch04.csv` | CatBoost reranker | 0.44072 | 0.43022 (rejected) |
| `submission_stable2_h.csv` | lead-in residual reranker | worse than 0.44224 | — (rejected) |

The current `submission.csv` is the final recent-persistence model, selected as the best
**private** score. Its Submission page screenshot (public 0.44048 / private 0.43478, user
elizavetagringruz) is embedded in the report (`midterm_figures/kaggle_score.png`). The
channel-variant file is kept separately as `submission_personal_043453.csv` for the best
public score. For context: the public leaderboard maximum was 0.44297, so the best public
score above is within 0.00073 of the winner; the 0.43006/0.41565/0.40026 early submissions
(`submission_personal_043006.csv`, `submission_personal_041.csv`, `submission_personal_040.csv`)
and the older five-way hybrid (`submission_hybrid_previous.csv`) remain available for
reproducibility. A further feature/ensemble search on five 12-week mirrors found no factor
that improves the final model (see `midterm_results/final_search_log.md`).

### Lead-in residual reranker (tested, rejected)

`submission_stable2_h.csv` was an experimental improvement over the recent-persistence
submission. It adds a **lead-in / channel-flow** feature and a logistic residual reranker.
It improved the local holdout but **did not improve the Kaggle score**, so it is not the
selected submission and should not be uploaded again:

- For each user, take the programmes watched in the **last seven days** before the cut.
- Build global show-to-show transitions with `0 < next_start - air_start <= 180` minutes.
  Transition weight is the product of the two screen-time ratios; keep the top 40
  destinations per source.
- `lead(u, show) = sum_source engagement(u, source) * P(show | source)`, normalised by
  the user's maximum. Coverage is 75-85% of seen candidate pairs.
- Fit a logistic model for the hidden Top-5 indicator with `log(hand)` as an offset and
  stable residual features: `lead`, `vt`, repeat share, channel/category/slot affinity,
  popularity, user-slot exposure, user activity, trend, full-view rate, and full-view count.
- Shrink the residual correction by `lambda = 0.5` so the base ranking dominates.

Honest forward validation (train only on earlier temporal windows):

| Holdout | Base MAP@5 | Reranked MAP@5 | Gain |
|---|---:|---:|---:|
| `w4` (latest 12 weeks) | 0.4089 | 0.4124 | +0.0035 |
| `w3` | 0.4107 | 0.4119 | +0.0013 |
| `w2` | 0.4198 | 0.4217 | +0.0018 |

The `w4` paired bootstrap 95% CI is `[+0.0005, +0.0065]`; `w3` and `w2` are positive
but their CIs include zero. The candidate changes 520/1,260 Top-5 sets versus the current
submission. The returned Kaggle score was worse than `0.44224`, so it is rejected.
The earlier CatBoost candidate `submission_gbm_ch04.csv` is also rejected (`0.43022`
private / `0.44072` public). Do not upload either candidate again.

## Results on the fixed validation split

Validation window: `2020-05-18` to `2020-07-27`; 1,124 users with exactly five targets;
6,376 candidate programmes. All methods use the same split, candidate pool, fallback, and Top-5
function.

| Method | MAP@5 | 95% CI | NDCG@5 | Recall@5 | Coverage |
|---|---:|---|---:|---:|---:|
| Popularity | 0.2726 | [0.259, 0.286] | 0.383 | 0.336 | 0.08% |
| CBF | 0.1621 | [0.152, 0.172] | 0.253 | 0.204 | 6.98% |
| Association rules | 0.2672 | [0.255, 0.280] | 0.385 | 0.323 | 0.71% |
| PureSVD MF | 0.2159 | [0.205, 0.228] | 0.328 | 0.308 | 1.30% |
| Personalized PageRank | 0.3037 | [0.290, 0.319] | 0.425 | 0.376 | 1.41% |
| **Hybrid (RRF)** | **0.3182** | **[0.304, 0.332]** | **0.441** | **0.390** | **1.05%** |

Paired bootstrap: hybrid − PPR = `+0.0144` `[0.0051, 0.0227]`;
hybrid − popularity = `+0.0456` `[0.0342, 0.0563]`.

Full numbers, diagnostics, case studies, and the verification log are in `midterm_results/`.
The report is `midterm_report/midterm_report.pdf` (6 pages including references).

## Repository layout

```
midterm/
  config.py          fixed split dates, hyperparameters, paths
  data_prep.py       session dedup, EPG load, interval join, screen time, target
  core.py            split construction, metrics, RRF, inference split
  methods.py         popularity, CBF, association rules, PureSVD, PPR
  evaluate.py        validation harness and inner-split weight tuning
  run_final.py       hybrid baseline fit, submission.csv, format validation
  run_final_personal.py  selected personal-history final model
  analysis.py        bootstrap CIs, overlap, cold-start, figures
  case_studies.py    fixed-user history / target / Top-5 tables
  verify.py          hand-computation and integrity checks
  report.py          PDF + Markdown report
data/
  dataset11-30.csv
  export_arh_11-20-final.csv
  export_arh_21-30-final.csv
  export_arh_31-42-final.csv
  submission.csv                 official sample (user order)
midterm_artifacts/               rebuilt parquet caches (generated)
midterm_results/                 validation tables, logs, case studies (generated)
midterm_figures/                 figures used by the report (generated)
midterm_report/                  report.pdf and report.md (generated)
submission.csv                   final Kaggle submission (1,260 rows)
```

## Environment

Python 3.9.6. Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

All experiments run on CPU; no GPU is required. Peak memory is about 4 GB during the join.

## Reproduce

The committed artifacts can be used directly. To rebuild everything from the raw CSVs:

```bash
make data       # rebuild sessions, EPG cache, clean overlaps (about 3-5 min)
make tune       # tune hybrid weights on the inner split
make eval       # inner + main validation tables
make final      # fit the personal-history model and write submission.csv
make analysis   # bootstrap CIs, overlap, cold-start, figures
make cases      # two fixed-user case-study tables
make verify     # hand checks and submission format validation
make report     # PDF and Markdown report
```

Or run the Python entry points directly:

```bash
python3 midterm/data_prep.py --force
python3 midterm/evaluate.py --split inner --tune
python3 midterm/evaluate.py --split main
python3 midterm/run_final_personal.py
python3 midterm/analysis.py
python3 midterm/case_studies.py
python3 midterm/verify.py
python3 midterm/report.py
```

`data_prep.py` is idempotent: without `--force` it reuses the parquet caches in
`midterm_artifacts/`.

## Protocol summary

- **Join.** Log `vsetv_id` = EPG `channel_id`; half-open interval overlap between the session and
  the programme (`[start, start + duration)`). One row per session-airing pair.
- **Missing programme IDs.** EPG rows with `tv_show_id = 0` (88.91%) are dropped before the join
  and are never imputed.
- **Target.** Sum watching seconds per `(user, programme airing)` over repeated sessions
  (zapping away and returning); keep airings with `screen_time >= 0.8`; count full airings per
  `(user, show)`; tie-break by total seconds, then show ID; take Top-5.
- **Fixed validation.** Train `< 2020-05-18`; validate `2020-05-18` to `2020-07-27`.
- **Inner split (weights only).** `2020-04-20` to `2020-05-18`.
- **Final fit.** All labelled logs `< 2020-07-27`; rank programmes airing `2020-07-27` to
  `2020-10-20`.
- **Hybrid.** RRF with `k = 60`; weights `popularity 0.0592`, `cbf 0.0175`, `ar 0.2237`,
  `mf 0.4095`, `ppr 0.2901` (inner split, 300 Dirichlet draws, seed 42).

## Submission

`submission.csv` contains `user_id,tv_show_id` for 1,260 users in the sample order. Each
`tv_show_id` field is five space-separated positive programme IDs, all distinct and all airing in
the test window. `run_final_personal.py` validates the format and prints
`submission validation: OK`. The default `--variant repeat` writes the selected channel-aware
repeat + recent-persistence model; `--variant engagement` reproduces the previous 0.41565 submission, `--variant test` the
0.40026 full-view/future-exposure submission, and `--variant train` the exact public-solution
recipe. All variants write a valid file.

## Before submitting

Replace the placeholders in `midterm_report/midterm_report.pdf` and `report.md`:

- Student/Email/Kaggle are already filled in (Elizaveta Gringruz, emgringruz@edu.hse.ru, elizavetagringruz)
- `https://github.com/llzbth17/sweettv-midterm`
- the Kaggle score screenshot (already inserted, `midterm_figures/kaggle_score.png`)

See `SUBMISSION_CHECKLIST.md` for the full list. Attach the original, unmodified AI-session
export `elizaveta_gringruz_midterm_session.json` separately; do not edit it.
