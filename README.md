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

| Submission | Model | Public MAP@5 | Private MAP@5 |
|---|---|---:|---:|
| **`submission_personal_repeat.csv`** | recent-persistence (final) | 0.44048 | **0.43478** (best private) |
| channel-only variant | channel-aware repeat, no recency | **0.44224** (best public) | 0.43453 |
| CatBoost reranker | (local `submission_gbm_ch04.csv`) | 0.44072 | 0.43022 (rejected) |
| lead-in residual reranker | (local `submission_stable2_h.csv`) | worse than 0.44224 | — (rejected) |

The file uploaded to Kaggle and selected as the final one (best **private** score) is
`submission_personal_repeat.csv` — the recent-persistence model. In this repository
`submission.csv` is the **official Kaggle sample** (zeroed template that fixes the user
order); the best-public channel-only variant and the two rejected candidates were scored on
Kaggle but are kept locally, not committed here. The Submission-page screenshot (public
0.44048 / private 0.43478, user elizavetagringruz) is committed as `kaggle_score.png` and
embedded in the report. For context, the public leaderboard maximum was 0.44297, so the best
public score is within 0.00073 of the winner. Earlier submissions (0.43006 / 0.41565 /
0.40026) and the older five-way hybrid (`submission_final.csv`) stay reproducible via
`run_final_personal.py --variant ...`. A further feature/ensemble search on five 12-week
mirrors found no factor that improves the final model (see `final_search_log.md`).

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

Full numbers, diagnostics, case studies, and the verification log are committed at the repo
root (`*.log`, `*.json`, `validation_*.csv`, `case_studies.*`, `final_*.csv`, `score_ceiling.md`,
`final_search_log.md`). The report is `midterm_report.pdf` (6 pages including references), with
the Markdown twin `report.md`.

## Repository layout

All files are at the repository root (flat structure). The raw Kaggle CSVs are not committed
(Kaggle license).

| File | What it is |
|---|---|
| `submission_personal_repeat.csv` | **final Kaggle submission** (1,260 rows; 0.44048 public / 0.43478 private) |
| `submission.csv` | official Kaggle **sample** (zeroed template; user order) |
| `submission_final.csv` | RRF hybrid candidate (previous best, kept for reproducibility) |
| `submission_personal.csv` | first personal-history submission (0.40026) |
| `submission_personal_{engagement,train,blend}.csv` | `run_final_personal.py --variant ...` outputs |
| `midterm_report.pdf` / `report.md` | 6-page report and its Markdown twin |
| `kaggle_score.png`, `kaggle_submission_screenshot.png` | Kaggle Submission-page screenshot embedded in the report |
| `config.py`, `data_prep.py`, `core.py`, `methods.py`, `evaluate.py`, `run_final.py`, `run_final_personal.py`, `analysis.py`, `case_studies.py`, `verify.py`, `report.py` | the full pipeline code |
| `validation_main.csv`, `validation_inner.csv`, `final_info.json`, `hybrid_weights_inner.json`, `personal_validation*.json` | validation and tuning results |
| `final_top5_long*.csv`, `case_studies.*`, `analysis_*.csv`/`.json`, `method_overlap_main.csv` | generated tables behind the report numbers |
| `*.log` (`prep`, `final`, `eval_*`, `analysis`, `case_studies`, `verification`, `report_build`, `robustness_all_users`) | run logs |
| `final_search_log.md`, `score_ceiling.md`, `lead_reranker_validation.md`, `case_studies.md` | analysis notes |
| `fig_pipeline.png`, `fig_map_ci.png`, `fig_overlap.png` | report figures |
| `Makefile`, `requirements.txt` | environment / entry points |

## Environment

Python 3.9.6. Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

All experiments run on CPU; no GPU is required. Peak memory is about 4 GB during the join.

## Reproduce

The files in this repository are committed flat (root level), while the code is written for
the canonical two-level layout — `config.py` resolves `data/`, `midterm_artifacts/`,
`midterm_results/`, `midterm_figures/` and `midterm_report/` relative to the parent of the
`midterm/` package. To run the pipeline, first restore that layout (the ten `.py` files into
`midterm/`, the raw Kaggle CSVs into `data/`), then:

```bash
make data       # rebuild sessions, EPG cache, clean overlaps (about 3-5 min)
make tune       # tune hybrid weights on the inner split
make eval       # inner + main validation tables
make final      # fit the personal-history model, write submission.csv
make analysis   # bootstrap CIs, overlap, cold-start, figures
make cases      # two fixed-user case-study tables
make verify     # hand checks and submission format validation
make report     # PDF and Markdown report
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

`submission_personal_repeat.csv` (1,260 rows) is the file uploaded to Kaggle (0.44048 public /
0.43478 private); `submission.csv` in this repository is the official zeroed **sample** that
fixes the user order. Each field is five space-separated positive programme IDs, all distinct
and all airing in the test window. `run_final_personal.py` validates the format and prints
`submission validation: OK`. The default `--variant repeat` writes the selected channel-aware
repeat + recent-persistence model; `--variant engagement` reproduces the previous 0.41565
submission, `--variant test` the 0.40026 full-view/future-exposure submission, and
`--variant train` the exact public-solution recipe. All variants write a valid file.

## Before submitting

- Student/Email/Kaggle are already filled in (Elizaveta Gringruz, emgringruz@edu.hse.ru, elizavetagringruz)
- GitHub URL: `https://github.com/llzbth17/sweettv-midterm` (already inserted in the report)
- the Kaggle score screenshot is already inserted (`kaggle_score.png`)

Attach the original, unmodified AI-session export `elizaveta_gringruz_midterm_session.json`
separately; do not edit it.
