# Leakage-Safe Hybrid TV Programme Recommendation on SWEET.TV

**Student:** Elizaveta Gringruz  |  **Team:** Individual  |  **Email:** emgringruz@edu.hse.ru  
**Date:** 2026-10-22  |  **Assignment:** LLM4Rec Midterm Exam  
**Kaggle:** elizavetagringruz  |  **GitHub:** https://github.com/llzbth17/sweettv-midterm

## Abstract

SWEET.TV must rank five programmes for 1,260 users, yet only 11.1% of EPG airings carry a usable programme identifier. We join 5.82M viewing sessions to identified airings by same-channel time overlap, apply the official screen-time >= 0.8 rule, and compare a popularity baseline with content-based filtering, pair association rules, PureSVD matrix factorization, and personalized PageRank on one fixed 10-week temporal split (6,376 candidates, 1,124 users). A reciprocal-rank hybrid reaches MAP@5 = 0.318 [95% CI 0.304-0.332], versus 0.273 for popularity and 0.304 for PageRank; the +0.014 gain over PageRank is small but its paired bootstrap interval excludes zero. The binding constraint is coverage rather than model class: 71.3% of validation candidates have no training interaction, and a mixed family account still fails when one profile must serve children, news, and serials. The submitted ranker is a channel-aware repeat model with recent Top-5 persistence; on the same fixed split it reaches MAP@5 = 0.398 [0.383, 0.412], so the hybrid remains the interpretable benchmark rather than the deployed model.

**Index Terms** — recommender systems, implicit feedback, hybrid ranking, personalized PageRank, TV EPG.

## 1. Introduction

SWEET.TV has 5.82M viewing sessions for 4,914 users and a programme schedule (EPG) covering weeks 11-42 of 2020. The task is to produce a ranked Top-5 list of programmes for each of 1,260 test users. The central difficulty is not the ranking model: 88.91% of EPG rows have `tv_show_id = 0` ("programme missing"), so only 381,892 of 3.44M airings can be linked to an item. Any recommender must first decide how to join a session to a schedule without inventing items.

A good recommender increases tune-in and retention by putting the right news, kids, film, or serial programme in front of the right viewer. The business value is highest where the schedule is fragmented: a household that watches children's animation in the morning and serials at night is poorly served by a single global popularity list.

Concrete example: user `6519574208115912303` completed `ТСН.` nine times and `Утро с Интером.` four times in the validation window. A popularity list gives four of the five targets, but it always puts the same national news first. The content model recovers `Утро с Интером.` from the morning-show profile; the hybrid keeps the safe news items and adds the morning show, reaching AP@5 = 1.0.

**Contributions.** (i) A leakage-safe, reproducible session-to-EPG pipeline with an explicit matching rule and a documented missing-ID policy. (ii) A fixed temporal split and candidate pool used unchanged by five methods and one hybrid. (iii) A validation comparison, two fixed-user Top-5 case studies, and a reproducible submission.

## 2. Related Work

Content-based filtering represents items by metadata and users by the items they consumed [5]. Item-based collaborative filtering replaces item content with co-consumption similarity [4], while implicit-feedback matrix factorization models confidence rather than ratings [3]. Association rules provide an interpretable item-to-item mapping through support, confidence, and lift [1]. Personalized PageRank propagates relevance over a user-item graph with a restart to the target user [2]; BPR is the corresponding pairwise-ranking alternative for implicit data [6]. The competition page defines the task and the official target recipe [7].

**Alternatives considered.** Item-kNN instead of PureSVD: rejected because its Top-5 was almost a popularity list on this sparse matrix; association rules kept as the interpretable co-occurrence model. BPR instead of PureSVD: rejected because no GPU was available and the deterministic SVD was easier to reproduce. Higher-order association rules: pair rules were exact, fast, and sufficient at 0.29% density. A sequence or LLM recommender: promising but out of scope for a one-week midterm and impossible to validate without more temporal folds.

## 3. Method

### 3.1 Session-to-schedule join and target

The log gives `(user_id, vsetv_id, start_time, stop_time)`. `vsetv_id` is the EPG `channel_id`; 197 of the 202 log channels appear in the EPG (five channels are absent). A session is joined to every identified programme on the **same channel** whose half-open interval `[start, start + duration)` intersects the session. The interaction strength is `overlap_seconds`; the official screen time is `overlap_seconds / programme_duration`.

Programmes with `tv_show_id = 0` are dropped before the join and are **not imputed**. They are 88.91% of EPG rows. A synthetic ID would create false interactions and contaminate both training and the target, so the pipeline keeps only the 381,892 identified airings. The join yields 3,847,592 session-airing rows, 2,963,956 matched sessions (50.9% of 5.82M), 4,895 users, and 4,363 programmes. 122 duplicate session starts were collapsed to the longest stop before joining, so each session-airing pair is emitted once.

The target follows the organisers' recipe exactly: drop `tv_show_id = 0`, keep airings with screen time >= 0.8, count full airings per `(user, programme)`, break ties by total seconds and then programme ID, and take the five most frequent programmes. A candidate pool is every identified programme airing in the evaluation window; it contains 100% of validation targets (815/815).

### 3.2 Fixed protocol and leakage control

The fixed validation cut is **2020-05-18**: training uses all log rows before the cut (10 weeks, 2.38M overlap rows) and validation uses the next 10 weeks (1.46M rows, 6,376 candidates, 1,124 users with exactly five targets). The inner split for hybrid weights is 2020-04-20 to 2020-05-18. The final model trains on all labelled rows through 2020-07-26 and ranks the 6,646 programmes that air in weeks 31-42.

No post-cutoff viewing behaviour enters any model. Programme metadata (title, category, genres, year, channel) and the evaluation-window schedule are allowed because they are published in advance. Popularity, user profiles, baskets, the SVD, and the PageRank graph are all built from `air_start < cut`. The hybrid weights are selected on the inner split, not on the reported validation split. All methods receive the same `Split` object and the same candidate array; cold items are filled with the same popularity ordering.

### 3.3 Methods

**Popularity (reference).** `p_i = sum_u X_ui`, where `X` counts full-view airings. Every user receives the same Top-5. It is not one of the two required methods.

**Content-based filtering (CBF).** Each programme is a sparse vector: character 3-5 gram TF-IDF of the title plus one-hot category, genre 1-3, decade, and channel. A user profile is the L2-normalised, full-view-weighted sum of the vectors of programmes they completed. The score is cosine similarity between profile and candidate vector. CBF is the only method that can score the 71.3% of candidates with no training interaction.

**Association rules.** Baskets are the sets of programmes each user completed in training. We mine exact 2-item rules `i -> j` with support >= 3 users, confidence >= 0.02, and lift > 1. The score for user `u` and candidate `j` is `sum_i X_ui * conf(i->j) * log1p(lift(i->j))`, keeping the 50 strongest consequents per antecedent. Pair rules are the right granularity at this density; triples are too sparse.

**Matrix factorization (PureSVD).** `A = L2_normalise(log1p(X) * IDF)`. A 32-factor truncated SVD gives `U S V^T`; the score is the reconstruction `U_u . V_j`. Log1p damps marathon re-watches, IDF prevents wall-to-wall channels from dominating, and row normalisation puts users on a comparable scale. The fit explains 29.1% of the variance on the validation split.

**Personalized PageRank (PPR).** On the bipartite user-programme graph, `R` is user->item and `C` is item->user. With restart `alpha = 0.15` and 20 iterations, `p_(t+1) = alpha * seed_u + (1-alpha) * R^T C p_t`; the seed is the user's training history. The item-item transition is never materialised: each iteration is two sparse products, which keeps the 6,646-item test pool tractable.

**Hybrid.** Scores are fused by reciprocal-rank fusion (RRF, `k = 60`) with weights tuned by 300 Dirichlet draws on the inner split: popularity 0.059, CBF 0.018, AR 0.224, MF 0.410, PPR 0.290. RRF is used instead of raw-score blending because the five score scales are not comparable. A method with no positive score for a user is excluded and the remaining weights are renormalised for that user.

![Leakage-safe pipeline](midterm_figures/fig_pipeline.png)

**Figure 1.** Leakage-safe pipeline. Every stage before the five rankers is shared, so differences in the results table come from the model, not from data preparation.

## 4. Experiments

### 4.1 Setup

All experiments run on an Apple M4 Pro laptop (CPU only, Python 3.9.6). The validation protocol is fixed: 1,124 users with exactly five relevant programmes, 6,376 candidates, MAP@5 as the primary metric, and NDCG@5, Recall@5, catalog coverage, and novelty as secondary diagnostics. Coverage is the share of the candidate pool that appears in any user's Top-5; novelty is the mean `-log2(p_i)` of recommended items. Fit times are wall-clock seconds for the 1,124 x 6,376 score matrix; the hybrid adds about 1.2 s of rank fusion.


| Method | MAP@5 | 95% CI | NDCG@5 | Recall@5 | Coverage | Items | Novelty | Fit (s) |
|---|---|---|---|---|---|---|---|---|
| POPULARITY | 0.273 | [0.259, 0.286] | 0.383 | 0.336 | 0.08% | 5 | 4.47 | 0.00 |
| CBF | 0.162 | [0.152, 0.172] | 0.253 | 0.204 | 6.98% | 445 | 12.62 | 0.76 |
| AR | 0.267 | [0.255, 0.280] | 0.385 | 0.323 | 0.71% | 45 | 4.59 | 0.23 |
| MF | 0.216 | [0.205, 0.228] | 0.328 | 0.308 | 1.30% | 83 | 6.06 | 0.05 |
| PPR | 0.304 | [0.290, 0.319] | 0.425 | 0.376 | 1.41% | 90 | 4.64 | 1.07 |
| HYBRID | 0.318 | [0.304, 0.332] | 0.441 | 0.390 | 1.05% | 67 | 5.09 | 2.11 |

**Table 1.** Validation comparison. All rows use the same split, candidate pool, users, fallback, and Top-5 function. Confidence intervals are 1,000-user bootstraps. The full table is in `midterm_results/validation_main.csv`.

![Validation MAP@5](midterm_figures/fig_map_ci.png)

**Figure 2.** MAP@5 with 95% bootstrap intervals on the fixed validation split.

### 4.2 Comparison with the baseline

The hybrid reaches MAP@5 = 0.318 [0.304, 0.332], a **+16.7% relative lift** over popularity (0.273) and **+4.7%** over PPR (0.304). The paired bootstrap difference hybrid - PPR is +0.0144 [0.0051, 0.0227]; 37.3% of users improve, 28.2% get worse, and the rest tie. PPR alone is +11.4% over popularity. Association rules are statistically close to popularity (0.267), MF is 20.8% below popularity (0.216), and CBF is 40.5% below (0.162). The ranking is stable when users with fewer than five targets are included: hybrid 0.305, PPR 0.295, popularity 0.265 on 1,236 users.

### 4.3 Verification

Every claim below is reproduced by `python3 midterm/verify.py` and stored in `midterm_results/verification.log`.
- One interval row was recomputed from raw timestamps: session 2020-03-11 22:00:04 -> 2020-03-12 00:04:12 on channel 4, programme 5200303 (22:00:00 + 6300 s). Overlap = 6296 s, screen time = 6296/6300 = 0.999365; the table stores 6296 s and 0.999365.
- The target recipe was recomputed for user 6519574208115912303: full-view frequencies 9, 4, 3, 2, 2, 1, ... produce `[2400480, 200337, 20088, 240081, 2400508]`, exactly the stored target.
- AP@5 for user 14874594108965844718 was hand-computed from the hybrid list: hits at ranks 1, 2, 4 -> `(1/1 + 2/2 + 3/4) / 5 = 0.55`, identical to the implementation.
- Join integrity: 0 duplicate session-airing keys and 0 rows with screen time > 1. Target containment is 815/815 (100%). The submission has 1,260 rows, exactly five unique positive IDs per row, and every ID airs in the test window.

### 4.4 Top-5 case studies

Two users were fixed before inspecting model outputs. User A (`6519574208115912303`) is a news/film generalist with 23 distinct training programmes and 41 full views. User B (`14874594108965844718`) is a mixed family/serial viewer with 30 distinct programmes and 44 full views. Both have exactly five validation targets.

**Table 2.** Training history and validation target for user A.

| Type | # | ID | Title | Category |
|---|---|---|---|---|
| History | 1 | 2400480 | ТСН. | Инфо |
| History | 2 | 240081 | Факты. | Инфо |
| History | 3 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения |
| History | 4 | 2400508 | Сегодня. | Инфо |
| History | 5 | 5900256 | Утро с Украиной. | Инфо |
| Target | 1 | 2400480 | ТСН. | Инфо |
| Target | 2 | 200337 | Утро с Интером. | Развлечения |
| Target | 3 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения |
| Target | 4 | 240081 | Факты. | Инфо |
| Target | 5 | 2400508 | Сегодня. | Инфо |

**Table 3.** Training history and validation target for user B.

| Type | # | ID | Title | Category |
|---|---|---|---|---|
| History | 1 | 12001734 | М/с 'Волшебные покровители'. | Детям |
| History | 2 | 12001682 | М/с 'Губка Боб Квадратные Штаны'. | Детям |
| History | 3 | 2400480 | ТСН. | Инфо |
| History | 4 | 12002355 | М/с 'Кунг-фу панда: Удивительные легенды'. | Детям |
| History | 5 | 10003012 | Т/с 'Однажды в Одессе'. | Сериалы |
| Target | 1 | 2400480 | ТСН. | Инфо |
| Target | 2 | 12001682 | М/с 'Губка Боб Квадратные Штаны'. | Детям |
| Target | 3 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения |
| Target | 4 | 1000589 | Т/с 'Отчаянные домохозяйки', 1 сезон, 10 с. 'Возвращайся'. | Сериалы |
| Target | 5 | 10003012 | Т/с 'Однажды в Одессе'. | Сериалы |

**Table 4.** Method Top-5 for the same two users. `Y` marks a hit against the official target. IDs are programme IDs; titles and categories are from the EPG.

| Method | # | A: ID | A: Title | A: Cat. | Hit | B: ID | B: Title | B: Cat. | Hit |
|---|---|---|---|---|---|---|---|---|---|
| popularity | 1 | 2400480 | ТСН. | Инфо | Y | 2400480 | ТСН. | Инфо | Y |
|  | 2 | 240081 | Факты. | Инфо | Y | 240081 | Факты. | Инфо |  |
|  | 3 | 2400467 | 'Викна-Новости'. | Инфо |  | 2400467 | 'Викна-Новости'. | Инфо |  |
|  | 4 | 2400508 | Сегодня. | Инфо | Y | 2400508 | Сегодня. | Инфо |  |
|  | 5 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | Y | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | Y |
| cbf | 1 | 2400508 | Сегодня. | Инфо | Y | 12001734 | М/с 'Волшебные покровители'. | Детям |  |
|  | 2 | 5900256 | Утро с Украиной. | Инфо |  | 12001732 | М/с 'Пингвины из 'Мадагаскара'. | Детям |  |
|  | 3 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | Y | 12002540 | Kids' Time. М/с 'Шоу Луни Тюнз'. | Детям |  |
|  | 4 | 240081 | Факты. | Инфо | Y | 12001682 | М/с 'Губка Боб Квадратные Штаны'. | Детям | Y |
|  | 5 | 5900275 | 'Утро России'. | Инфо |  | 120070654 | М/ф 'Облачно, возможны осадки в виде фрикаделек'. | Детям |  |
| ar | 1 | 2400480 | ТСН. | Инфо | Y | 240081 | Факты. | Инфо |  |
|  | 2 | 240081 | Факты. | Инфо | Y | 2400480 | ТСН. | Инфо | Y |
|  | 3 | 2400508 | Сегодня. | Инфо | Y | 2400467 | 'Викна-Новости'. | Инфо |  |
|  | 4 | 2400467 | 'Викна-Новости'. | Инфо |  | 200344 | 'Чистоnews'. | Развлечения |  |
|  | 5 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | Y | 2400508 | Сегодня. | Инфо |  |
| mf | 1 | 200337 | Утро с Интером. | Развлечения | Y | 12001682 | М/с 'Губка Боб Квадратные Штаны'. | Детям | Y |
|  | 2 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | Y | 12001734 | М/с 'Волшебные покровители'. | Детям |  |
|  | 3 | 2400508 | Сегодня. | Инфо | Y | 1000589 | Т/с 'Отчаянные домохозяйки', 1 сезон, 10 с. 'Возвращайся'. | Сериалы | Y |
|  | 4 | 240081 | Факты. | Инфо | Y | 6200371 | ДжеДАИ. | Познавательное |  |
|  | 5 | 2400480 | ТСН. | Инфо | Y | 10003012 | Т/с 'Однажды в Одессе'. | Сериалы | Y |
| ppr | 1 | 2400480 | ТСН. | Инфо | Y | 2400480 | ТСН. | Инфо | Y |
|  | 2 | 240081 | Факты. | Инфо | Y | 240081 | Факты. | Инфо |  |
|  | 3 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | Y | 12001682 | М/с 'Губка Боб Квадратные Штаны'. | Детям | Y |
|  | 4 | 2400508 | Сегодня. | Инфо | Y | 2400467 | 'Викна-Новости'. | Инфо |  |
|  | 5 | 2400467 | 'Викна-Новости'. | Инфо |  | 12001734 | М/с 'Волшебные покровители'. | Детям |  |
| hybrid | 1 | 2400480 | ТСН. | Инфо | Y | 12001682 | М/с 'Губка Боб Квадратные Штаны'. | Детям | Y |
|  | 2 | 240081 | Факты. | Инфо | Y | 2400480 | ТСН. | Инфо | Y |
|  | 3 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | Y | 12001734 | М/с 'Волшебные покровители'. | Детям |  |
|  | 4 | 2400508 | Сегодня. | Инфо | Y | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | Y |
|  | 5 | 200337 | Утро с Интером. | Развлечения | Y | 240081 | Факты. | Инфо |  |

**User A.** All methods except CBF recover at least four of the five news/morning targets. Popularity and PPR both pick `Викна-Новости` (2400467), a national news programme the user did not complete; CBF replaces it with `Утро с Украиной` (5900256), another morning show. MF recovers `Утро с Интером.` (200337), the user's second most frequent target. The hybrid keeps the four safe hits and adds `Утро с Интером.`, so AP@5 = 1.0. The example shows why content and collaborative signals are complementary: CBF knows the morning-show genre, MF knows the co-view pattern.

**User B.** This is the informative failure. The history mixes films (10), children's animation (5), news (5), and serials (5); the target includes two serials (`Отчаянные домохозяйки`, `Однажды в Одессе`), one children's show, one news, and one entertainment show. CBF recommends five children's shows and hits only `Губка Боб`; MF recovers both serials and hits 3/5; AR and popularity recommend almost only news and hit 1/5 and 2/5. The hybrid reaches 3/5 (AP@5 = 0.55) but still misses both serials, because AR, PPR, and popularity outvote MF. This is a household-profile problem, not a scoring bug: one latent vector cannot represent a viewer who switches between children's animation and evening serials.

## 5. Discussion

### 5.1 Business and algorithmic analysis

**Selected strategy.** The submitted ranker is a channel-aware repeat model with recent Top-5 persistence and category affinity, not the five-way hybrid. On the fixed method split (1,124 users) it reaches MAP@5 = 0.398 [0.383, 0.412], +0.0027 over the channel-only model (0.395), +0.0071 over repeat-only (0.391), and far above the hybrid (0.318). On the 12-week airing-level mirror it reaches 0.409. The cost is a few groupbys over 3.0M airing-level overlap rows plus a Top-5 sort; there is no iterative training. The same 6,376-candidate pool and temporal cut are used as in the comparison. A post-hoc lead-in residual reranker (`submission_stable2_h.csv`) raised the latest 12-week holdout from 0.4089 to 0.4124 (+0.0035, 95% CI [+0.0005, +0.0065]), but the gain did not transfer to Kaggle: the public score was worse than the confirmed 0.44224, so the candidate is rejected.

**Why this strategy.** The target counts complete airings in the next 12 weeks, so the user's own recent viewing is the strongest signal. Engagement weights every observed overlap by screen-time fraction and `0.9^(age in weeks)`; a schedule ratio rewards shows that become more frequent in the test window; repeat and consistency factors reward programmes the user returns to across weeks. Exposure is channel-weighted: airings on channels the user watches are counted fully, airings on channels they never use are discounted, and a small affinity term rewards the show's dominant channel. Recent persistence adds the user's daily Top-5 target ranks over the last 14 days, and category affinity adds a light prior from the user's category mix. The hybrid's collaborative models remain useful as cold-start and explanation components, but they dilute the personal signal: on the same split the best hybrid is 0.318 and PPR 0.304, while the personal ranker is 0.398. The personal model uses only identified programmes in the six allowed categories and fills 8 users with fewer than five eligible shows from the popularity head.

**User experience.** The final submission recommends 154 distinct programmes across 1,260 users, versus 5 for pure popularity; 49.6% of final slots are news, 20.3% children, 14.8% entertainment, 9.4% serials, 4.7% educational, and 1.2% other. That is closer to the validation target mix than the old hybrid but still news-heavy. A production system should add a diversity or exploration quota (for example, at most three news items per user and at least one item from a second category) and monitor household-level satisfaction rather than only MAP@5.

**Algorithmic differences.** Mean Top-5 Jaccard overlap is 0.708 between popularity and PPR, 0.652 between AR and PPR, 0.482 between MF and hybrid, and 0.202 between CBF and hybrid. PPR and popularity agree on the news head; MF and CBF contribute the long tail. The hybrid's 0.448 overlap with popularity is therefore not a collapse to the baseline: it keeps the safe head while admitting 20% of slots from other models.

### 5.2 Failure case: duplicated join rows

The first implementation of the interval join emitted the best-overlapping programme twice: once in the forward expansion over programmes starting inside the session, and again in the `best` array. On the old artifact this produced 5,948,563 overlap rows with 1,343,480 duplicate `(user, programme, airing)` groups and 2,467,243 extra rows. The duplicate seconds inflated screen time above 1.0 and over-counted the frequency of exactly the programmes a user watched most, which is the worst possible bias for a Top-5 target.

**Root cause.** The forward loop appended every overlapping programme, then a second pass appended `best` for every session. There was no deduplication key and no invariant check. A second, smaller cause was 122 duplicate session starts in the raw log, which created duplicate overlaps even after the first bug was fixed.

**Fix and verification.** The join is now a single offset expansion per session; the `best` append was removed; sessions are deduplicated by `(user, channel, start)` keeping the longest stop; and `data_prep.py` asserts 0 duplicate session-airing keys and `screen_time <= 1`. The corrected table has 3,847,592 rows, 0 duplicates, and 0 ratios above 1.0. `verify.py` also recomputes one row from raw timestamps (0.999365) and the target frequencies for a fixed user. This changed the local MAP scale but not the method ranking, which is why the report uses only the corrected artifacts.

### 5.3 What worked, what surprised me

**Worked.** (i) Aligning the training matrix with the official full-view target instead of using all seconds. (ii) PPR: a simple graph walk was the strongest single personalised model and the cheapest strong model. (iii) Association rules: pair rules are interpretable and nearly match popularity without any content. (iv) A single popularity fallback for every method made cold-start behaviour comparable instead of hiding it inside each model.

**Surprised.** (i) CBF has the highest novelty (12.62) and the worst MAP (0.162); novelty is not value. (ii) PureSVD is below the popularity baseline (0.216 vs 0.273) at 0.29% density, even though it is the best model for the serial-heavy User B. (iii) The hybrid still leaves 71.3% of validation candidates unscored by four of five methods; coverage, not model class, is the binding constraint, and the personal ranker avoids this by scoring every show the user has already watched and falling back to popularity. (iv) The local MAP scale (0.16-0.40) is not directly comparable to the Kaggle leaderboard (top around 0.43) because the windows, users, and hidden target differ; the report only claims the measured local gains.

### 5.4 Limitations

- **Schedule coverage.** 88.91% of EPG rows are unidentified; 49.1% of sessions never match an identified programme. A programme that only airs with `tv_show_id = 0` can never be recommended or evaluated.
- **Cold candidates.** 71.3% of validation candidates and 57.5% of test candidates have no training interaction; only CBF can score them, and its ranking quality is low.
- **Target definition.** The 0.8 threshold discards partial viewers; users with fewer than five full views are excluded from the main table (112 of 1,236 users with any target). The official test labels are hidden and may use a different tie-break.
- **Distribution shift.** Validation is May-July; the test is August-October, with a new school season and schedule changes. One 10-week split cannot measure seasonal drift.
- **Model limits.** PureSVD is untuned (32 factors, no negative sampling); association rules are pair-only; the hybrid weights come from a four-week inner split with 1,004 users and may not transfer to every segment.
- **No online evidence.** There is no A/B test, impression log, or revenue measurement; all business claims are offline and associational.

### 5.5 Next improvement

The concrete next step is **household/session decomposition**: cluster a user's sessions by time-of-day and content, learn 2-3 profiles per heavy account, and route each request to the matching profile. User B is the motivating case: a single vector averages children's animation with evening serials. A second step is to add partial-view and time-decay signals as auxiliary confidence, not as target labels, and to evaluate on three rolling temporal folds. A third is a content embedding (or an LLM-generated genre/style vector) for the 71% cold candidates, blended into PPR rather than used as a standalone ranker.

### 5.6 Submission and reproducibility

The final `submission.csv` has 1,260 rows, five unique positive IDs per row, and only programmes airing in weeks 31-42. The final model is the channel-aware repeat ranker with recent Top-5 persistence and category affinity; it trains on all labelled logs through 2020-07-26 and ranks candidates airing in weeks 31-42. The exact commands are listed below; the repository layout is described in `README.md`.

```
python3 midterm/data_prep.py --force        # rebuild sessions, EPG, overlaps
python3 midterm/evaluate.py --split inner --tune
python3 midterm/evaluate.py --split main
python3 midterm/run_final_personal.py --variant repeat  # writes submission.csv
python3 midterm/analysis.py                # tables, CIs, figures
python3 midterm/case_studies.py
python3 midterm/verify.py
python3 midterm/report.py                  # this PDF
```

**GitHub:** `https://github.com/llzbth17/sweettv-midterm`. The repository contains `midterm/` (code), `midterm_results/` (tables, logs, case studies), `midterm_figures/` (figures), and the final `submission.csv`.

![Kaggle submission score](midterm_figures/kaggle_score.png)

**Kaggle score.** Submission page for the designated final `submission.csv`: public score **0.44048**, private score **0.43478**, user elizavetagringruz. A separate channel-variant submission scored 0.44224 public / 0.43453 private and is kept as `submission_personal_043453.csv`.

## 6. AI Usage Disclosure

**Tools.** OpenCode 1.18.30 with DeepSeek V4.1 Flash Free; Python 3.9.6; pandas 2.3.3; NumPy 2.0.2; SciPy 1.13.1; scikit-learn 1.6.1; ReportLab 5.0.1; matplotlib 3.9.4.

**How AI was used.** Code generation and refactoring for the join, the five models, the evaluation harness, and the report generator; debugging the duplicate-join failure; figure specifications; and a first draft of the report text. The AI did not have access to any post-cutoff viewing behaviour and did not tune on the reported validation labels.

**What I personally verified.** I re-ran the full pipeline from raw CSVs and checked: (i) one overlap row by hand (6296 s / 6300 s = 0.999365); (ii) the full-view target frequencies for user 6519574208115912303; (iii) AP@5 for user 14874594108965844718 by hand (0.55); (iv) the duplicate-key and screen-time invariants; (v) the 815/815 target containment; and (vi) the 1,260 x 5 submission format. The verification log is generated by `midterm/verify.py`.

**What I trusted without independent verification.** The official Kaggle MAP@5 implementation and the hidden private-test distribution; the exact organiser tie-break for users with fewer than five full views; and the assumption that EPG metadata for the evaluation window is available at recommendation time. These are stated as assumptions, not verified facts.

**Session log.** The original, unmodified AI-session export is attached separately (`<student>_midterm_session.json`). It was not edited. The report claims and the session should be read together.

## References

[1] R. Agrawal, T. Imieliński, and A. Swami, "Mining association rules between sets of items in large databases," in *Proc. 1993 ACM SIGMOD Int. Conf. Management of Data*, 1993, pp. 207-216. [Online]. Available: https://doi.org/10.1145/170035.170072. [Accessed: 2026-10-22].

[2] S. Brin and L. Page, "The anatomy of a large-scale hypertextual Web search engine," *Computer Networks and ISDN Systems*, vol. 30, no. 1-7, pp. 107-117, 1998. [Online]. Available: https://doi.org/10.1016/S0169-7552(98)00110-X. [Accessed: 2026-10-22].

[3] Y. Hu, Y. Koren, and C. Volinsky, "Collaborative filtering for implicit feedback datasets," in *Proc. 8th IEEE Int. Conf. Data Mining (ICDM)*, 2008, pp. 263-272. [Online]. Available: https://doi.org/10.1109/ICDM.2008.22. [Accessed: 2026-10-22].

[4] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, "Item-based collaborative filtering recommendation algorithms," in *Proc. 10th Int. Conf. World Wide Web (WWW)*, 2001, pp. 285-295. [Online]. Available: https://doi.org/10.1145/371920.372071. [Accessed: 2026-10-22].

[5] M. J. Pazzani and D. Billsus, "Content-based recommendation systems," in *The Adaptive Web*, LNCS 4321, Springer, 2007, pp. 325-341. [Online]. Available: https://doi.org/10.1007/978-3-540-72079-9_10. [Accessed: 2026-10-22].

[6] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, "BPR: Bayesian personalized ranking from implicit feedback," in *Proc. 25th Conf. Uncertainty in Artificial Intelligence (UAI)*, 2009. [Online]. Available: https://arxiv.org/abs/1205.2618. [Accessed: 2026-10-22].

[7] Kaggle, "sweet.tv - TV program recommender," competition page, 2020. [Online]. Available: https://www.kaggle.com/competitions/sweettv-tv-program-recommender. [Accessed: 2026-10-22].
