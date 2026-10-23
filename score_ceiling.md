# Why 0.5 is not reachable on this competition

## Confirmed leaderboard

From the archived public leaderboard (36 teams, 578 submissions):

| Rank | Team | Public MAP@5 |
|---:|---|---:|
| 1 | Magic City | 0.44297 |
| 2 | Nikita Melentev | 0.44252 |
| 3 | x0x0w1 | 0.43639 |
| 4 | Too much ... | 0.43566 |
| 5 | AfterParty | 0.43521 |

The second-place solution reports `0.44203` public / `0.42905` private in its write-up
("Two-layered recommender system methodology", Nikita Melentev, 2021).

Our confirmed submissions:

| File | Private | Public |
|---|---:|---:|
| `submission_personal_043453.csv` | 0.43453 | **0.44224** |
| `submission_personal_043478.csv` | **0.43478** | 0.44048 |
| `submission_stable2_h.csv` | worse | worse |

The best public score is therefore already within `0.00073` of the public winner and
above the published second-place public score. A target of `0.5` would require `+0.05776`
over the winner, not a feature-level gain.

## Why the missing score cannot be recovered from films

On the latest 12-week holdout (`w4`):

- 5,995 target rows; **839 (14.0%) are category `Фильмы`**.
- **836 / 839 film targets (99.6%) are unseen** in the user's history.
- 1,821 / 5,156 non-film targets (35.3%) are also unseen.
- The best simple film-only ranker (test airing count) reaches only **1.8% hit@1** and
  **3.6% hit@5** over film candidates.
- Forcing one film into the base Top-5 loses **0.0314 MAP@5**; channel/genre/slot film
  profiles are weaker (median target rank around 918 of 5,929).
- The film candidate pool has 5,929 shows for 538 target films; 313 target films have no
  prior full view by any user, so there is no collaborative or popularity signal for them.

Ignoring films caps MAP@5 at 0.86 even with perfect non-film ranking. Recovering the
remaining gap would require predicting unseen one-off films, which the data do not
support. The original contest's top solutions also plateau at about 0.44 public / 0.43
private.

## Further feature search (12-week test-length mirrors)

Every candidate was re-tested on five 12-week mirrors ending on the test cut
(`w0` 2020-04-06/29 through `w4` 2020-05-04/27, base = `hand` recent-persistence model).
On this honest horizon all of them are flat or negative, so none was promoted:

| Candidate | dmean vs base | dmin | dmax |
|---|---:|---:|---:|
| `full_rate` (beta 0.2/0.3/0.5) | -0.0005..-0.0009 | -0.0021 | +0.0045 |
| `h14` persistence (0.2/0.3) | -0.0002 / -0.0001 | -0.0012 | +0.0007 |
| `h28` (0.3) | -0.0005 | -0.0013 | 0.0000 |
| `slot_affinity` (0.2) | -0.0005 | -0.0012 | +0.0004 |
| `mean_ratio` (0.2) | -0.0002 | -0.0014 | +0.0006 |
| channel mix 0.5·ch + 0.5·recent (rank/log) | -0.0001..+0.0001 | -0.0010 | +0.0008 |
| decay 0.85 / 0.95 exposure variants | -0.0005 / -0.0007 | -0.0017 | +0.0011 |

Note: the short-horizon (4-10 week) mirrors that were used earlier overstate
`full_rate` and recency-type factors; on the 12-week horizon that matches the real
test they do not help. The personal model is at its local optimum; the remaining
score gap is the unpredictable unseen tail (see above), not a missing feature.

## Decision

Do not chase `0.5` with more feature variants. It is outside the measurable ceiling and
the last two local-gain candidates (`submission_gbm_ch04.csv`, `submission_stable2_h.csv`)
both regressed on Kaggle. Use a confirmed fallback:

- best public: `submission_personal_043453.csv` (`0.44224`);
- best private: `submission_personal_043478.csv` (`0.43478`, currently `submission.csv`).
