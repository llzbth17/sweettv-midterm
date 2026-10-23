"""Final personal-history model for the Kaggle submission.

Selected model: channel-aware repeat-aware engagement
--------------------------------------------------------
The test target is the number of *complete* airings a user watches in the next
12 weeks.  The strongest signal is the user's own recent history.  Instead of
counting only airings watched to the end, we use all observed overlap as a
graded engagement signal, then adjust it for schedule change, programme
repeatability and the user's consistency with that programme:

  * one row per (user, programme airing), with watching seconds summed over
    repeated sessions of the same airing;
  * only identified programmes, the public-solution category set, duration
    < 10,000 s, and programmes that actually air in the test window;
  * engagement(user, show) = sum over airings of
        screen_time_fraction * 0.9 ** (age_in_weeks);
  * exposure is user-channel weighted: V(u, show) = sum over channels of
    the user's engagement share on that channel times the number of test
    airings of the show on that channel; T(u, show) is the same over the last
    12 weeks before the test window;
  * repeat(show) = smoothed fraction of user-show pairs whose full views span
    at least two distinct weeks (show-level repeatability);
  * consistency(user, show) = number of distinct weeks the user watched the
    show / number of distinct weeks the show aired in the training window;
  * channel_affinity(u, show) = the user's overall engagement share on the
    show's dominant test channel;
  * recent_gain(u, show) = sum over the last 14 days of (6 - rank) / 5, where
    rank is the show's rank in the user's daily Top-5 full-view target; this
    captures target persistence without looking at the future;
  * category_affinity(u, show) = the user's engagement share in the show's
    category;
  * score = engagement
            * log10(max(V(u, show), 0.5))
            * ((V(u, show) + 5) / (T(u, show) + 5)) ** 0.5
            * (1 + 1.6 * repeat(show))
            * (1 + 0.6 * consistency(user, show))
            * (1 + 0.5 * channel_affinity(u, show))
            * (1 + 0.25 * recent_gain(u, show))
            * (1 + 0.3 * category_affinity(u, show)),
    tie-broken by the number of users who watched the show; missing users are
    filled with the most popular eligible shows.

Why this beats the public full-view recipe
------------------------------------------
On seven temporal validation mirrors (4-12 week horizons) the channel-aware
model improves MAP@5 by +0.0074 over the plain engagement model and +0.0034
over the repeat-only model.  On 17 additional 4-8 week mirrors the same
comparisons are +0.0067 and +0.0040.  Adding the recent Top-5 persistence and
category-affinity factors adds another +0.0019 on the seven mirrors and
+0.0026 on the 17 additional mirrors over the channel model.  On the clean
fixed method split the final model reaches 0.3977 versus 0.3951 for the
channel model and 0.3906 for repeat-only.  The schedule ratio rewards
programmes that become more frequent in the test window; the repeat/
consistency factors reward shows that users return to across weeks; the
channel weighting stops a show's airings on channels the user never watches
from inflating its exposure; recent Top-5 persistence captures a user's
current favourites.  The weekly decay still focuses the model on the last
~6-8 weeks.

The script keeps the previous variants for reproducibility:
  --variant engagement  previous recency-weighted engagement (0.41565)
  --variant test        future-exposure full-view model (0.40026)
  --variant train       exact public-solution recipe
  --variant blend       average exposure variant
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ART, DATA, RESULTS, ROOT, SCREEN_THRESHOLD, TEST_START, TEST_END
from data_prep import load_epg, top_targets
from run_final import validate_submission


ALLOWED_CATEGORIES = [
    "Развлечения", "Инфо", "Сериалы", "Детям", "Другое", "Познавательное",
]
MAX_DURATION = 10_000  # seconds
WEEKLY_DECAY = 0.9     # each week of age multiplies an airing's weight by 0.9
TRAIN_EXPOSURE_DAYS = 84  # last 12 weeks of history, matching the test horizon
SCHEDULE_ALPHA = 0.5      # exponent on the relative test/train exposure ratio
SCHEDULE_SMOOTH = 5.0     # additive smoothing for low-count programmes
REPEAT_BETA = 1.6         # weight of the show-level repeatability factor
REPEAT_PRIOR_WEIGHT = 10.0  # pseudo-pairs shrinking repeatability to the mean
CONSISTENCY_GAMMA = 0.6   # weight of the user-show week-consistency factor
CHANNEL_AFFINITY_BETA = 0.5  # weight of the user's channel preference
RECENT_DAYS = 14          # trailing window for recent target persistence
RECENT_BETA = 0.25        # weight of the recent Top-5 persistence factor
CATEGORY_AFFINITY_BETA = 0.3  # weight of the user's category preference


def _prepare_epg(epg: pd.DataFrame) -> pd.DataFrame:
    e = epg[epg["tv_show_id"].fillna(0).astype("int64") != 0].copy()
    e["tv_show_id"] = e["tv_show_id"].astype("int64")
    return e


def _show_categories(epg: pd.DataFrame) -> pd.Series:
    """Most frequent category label per show (category is a show property)."""
    e = _prepare_epg(epg)
    return (e.dropna(subset=["tv_show_category"])
             .groupby("tv_show_id")["tv_show_category"]
             .agg(lambda s: s.value_counts().index[0]))


def _filter_epg(epg: pd.DataFrame, cat: pd.Series) -> pd.DataFrame:
    e = epg.copy()
    e["category"] = e["tv_show_id"].map(cat)
    return e[e["category"].isin(ALLOWED_CATEGORIES)
             & (e["duration"] < MAX_DURATION)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",
                    choices=["repeat", "engagement", "test", "train", "blend"],
                    default="repeat",
                    help="channel-aware repeat engagement (default), the previous "
                         "engagement model, or a full-view exposure variant")
    ap.add_argument("--output", default="submission.csv")
    args = ap.parse_args()

    t0 = pd.Timestamp(TEST_START)
    t1 = pd.Timestamp(TEST_END)

    ov = pd.read_parquet(os.path.join(ART, "overlaps_airings.parquet"))
    epg = load_epg()
    sample = pd.read_csv(os.path.join(DATA, "submission.csv"),
                         dtype={"user_id": "uint64"})

    cat = _show_categories(epg)
    train_epg = _filter_epg(_prepare_epg(epg[epg["start"] < t0]), cat)
    test_epg = _filter_epg(
        _prepare_epg(epg[(epg["start"] >= t0) & (epg["start"] < t1)]), cat)
    candidates = set(_prepare_epg(
        epg[(epg["start"] >= t0) & (epg["start"] < t1)]
    )["tv_show_id"].unique().tolist())

    train_counts = train_epg.groupby("tv_show_id").size()
    train_recent = train_epg[
        train_epg["start"] >= t0 - pd.Timedelta(days=TRAIN_EXPOSURE_DAYS)]
    train_counts_recent = train_recent.groupby("tv_show_id").size()
    train_weeks = (train_epg.assign(week=train_epg["start"].dt.floor("7D"))
                   .groupby("tv_show_id")["week"].nunique())
    test_counts = test_epg.groupby("tv_show_id").size()

    # History: every observed overlap, graded by screen-time fraction.
    base = ov[(ov["air_start"] < t0)
              & ov["tv_show_id"].isin(candidates)].copy()
    base["category"] = base["tv_show_id"].map(cat)
    base = base[base["category"].isin(ALLOWED_CATEGORIES)
                & (base["air_dur"] < MAX_DURATION)].copy()
    assert len(base) > 0

    age_weeks = (t0 - base["air_start"]).dt.total_seconds() / (86400.0 * 7.0)
    base["decay"] = np.power(WEEKLY_DECAY, age_weeks)
    base["engagement"] = base["ratio"] * base["decay"]
    rankable = set(base["tv_show_id"].unique().tolist())

    # Full views are only needed for the older variants and for the fallback.
    full = base[base["ratio"] >= SCREEN_THRESHOLD].copy()
    full["rating"] = np.where(full["ratio"] >= 1.0,
                              5.0, 2.0 + 2.0 * full["ratio"] ** 2)
    full["rating"] *= full["decay"]

    if args.variant == "repeat":
        g = (base.groupby(["user_id", "tv_show_id"], as_index=False)
                 .agg(engagement=("engagement", "sum"),
                      global_cnt=("user_id", "nunique")))
        g["test_count"] = (g["tv_show_id"].map(test_counts)
                           .fillna(1).astype(float))
        g["train_count_recent"] = (g["tv_show_id"]
                                   .map(train_counts_recent)
                                   .fillna(1).astype(float))

        # How consistently the user watched this programme across its run.
        base["week"] = base["air_start"].dt.floor("7D")
        uw = (base.groupby(["user_id", "tv_show_id"], as_index=False)
                  .agg(n_weeks=("week", "nunique")))
        uw["week_cov"] = (uw["n_weeks"]
                          / uw["tv_show_id"].map(train_weeks)
                          .fillna(1).astype(float))
        g = g.merge(uw[["user_id", "tv_show_id", "week_cov"]],
                    on=["user_id", "tv_show_id"], how="left")
        g["week_cov"] = g["week_cov"].fillna(0.0)

        # Show-level repeatability: among full views, how often does a user
        # come back in a different week?  Shrink low-support shows to the mean.
        full["week"] = full["air_start"].dt.floor("7D")
        fuw = (full.groupby(["user_id", "tv_show_id"], as_index=False)
                   .agg(n_weeks=("week", "nunique")))
        fuw["multi"] = (fuw["n_weeks"] >= 2).astype(float)
        repeat_prior = float(fuw["multi"].mean()) if len(fuw) else 0.0
        show_rep = (fuw.groupby("tv_show_id")
                       .agg(n=("multi", "size"), s=("multi", "sum")))
        show_rep["repeat_sm"] = (
            (show_rep["s"] + REPEAT_PRIOR_WEIGHT * repeat_prior)
            / (show_rep["n"] + REPEAT_PRIOR_WEIGHT))
        g["repeat_sm"] = (g["tv_show_id"].map(show_rep["repeat_sm"])
                          .fillna(repeat_prior))

        # A show may air on several channels, but a user watches only a
        # subset.  Weight each channel by the user's engagement share for
        # this show, then sum that channel's test/train airings.
        test_by_channel = (test_epg.groupby(["tv_show_id", "channel_id"])
                           .size().rename("test_count_ch"))
        train_by_channel = (train_recent.groupby(["tv_show_id", "channel_id"])
                            .size().rename("train_count_ch"))
        uc = (base.groupby(["user_id", "tv_show_id", "channel_id"],
                           as_index=False)["engagement"].sum())
        uc["channel_share"] = (uc["engagement"]
                               / uc.groupby(["user_id", "tv_show_id"])
                               ["engagement"].transform("sum"))
        uc = (uc.join(test_by_channel, on=["tv_show_id", "channel_id"])
                .join(train_by_channel, on=["tv_show_id", "channel_id"]))
        uc[["test_count_ch", "train_count_ch"]] = (
            uc[["test_count_ch", "train_count_ch"]].fillna(0.0))
        ue = (uc.assign(user_test_count=uc["channel_share"] * uc["test_count_ch"],
                        user_train_count=uc["channel_share"] * uc["train_count_ch"])
                .groupby(["user_id", "tv_show_id"], as_index=False)
                [["user_test_count", "user_train_count"]].sum())
        g = g.merge(ue, on=["user_id", "tv_show_id"], how="left")
        g[["user_test_count", "user_train_count"]] = (
            g[["user_test_count", "user_train_count"]].fillna(0.0))

        # The show's dominant test channel, and the user's overall affinity
        # for that channel (not just for this particular show).
        channel_counts = (test_epg.groupby(["tv_show_id", "channel_id"])
                          .size().rename("n").reset_index())
        dominant = (channel_counts
                    .sort_values(["tv_show_id", "n", "channel_id"],
                                 ascending=[True, False, True])
                    .groupby("tv_show_id").head(1)
                    .set_index("tv_show_id")["channel_id"]
                    .rename("dominant_channel"))
        uct = (base.groupby(["user_id", "channel_id"])["engagement"]
               .sum().rename("e").reset_index())
        uct["affinity"] = (uct["e"]
                           / uct.groupby("user_id")["e"].transform("sum"))
        g = g.merge(dominant, left_on="tv_show_id", right_index=True,
                    how="left")
        g = g.merge(uct[["user_id", "channel_id", "affinity"]],
                    left_on=["user_id", "dominant_channel"],
                    right_on=["user_id", "channel_id"], how="left")
        g["channel_affinity"] = g["affinity"].fillna(0.0)

        # Recent target persistence: for each of the last 14 days, rank the
        # user's full views and add (6 - rank) / 5 for each Top-5 appearance.
        # This is a direct, leakage-free estimate of which programmes are
        # currently at the top of the user's target list.
        recent_users = set(sample["user_id"].astype("uint64").tolist())
        recent_rows = []
        for day in pd.date_range(
                t0 - pd.Timedelta(days=RECENT_DAYS - 1), t0, freq="D"):
            tt = top_targets(full, day, day + pd.Timedelta(days=1),
                             users=recent_users)
            for u, items in tt.items():
                for rank, sid in enumerate(items, start=1):
                    recent_rows.append(
                        (int(u), int(sid), float(6 - rank) / 5.0))
        if recent_rows:
            recent = (pd.DataFrame(recent_rows,
                                   columns=["user_id", "tv_show_id",
                                            "recent_gain"])
                      .groupby(["user_id", "tv_show_id"], as_index=False)
                      ["recent_gain"].sum())
            g = g.merge(recent, on=["user_id", "tv_show_id"], how="left")
        g["recent_gain"] = g.get("recent_gain", 0.0)
        g["recent_gain"] = g["recent_gain"].fillna(0.0)

        # Category affinity: the share of the user's engagement that goes to
        # the show's category.  This is a light prior for programmes with
        # little personal history and smooths the per-show engagement.
        g["category"] = g["tv_show_id"].map(cat)
        ucat = (base.groupby(["user_id", "category"], as_index=False)
                    ["engagement"].sum())
        ucat["category_affinity"] = (
            ucat["engagement"]
            / ucat.groupby("user_id")["engagement"].transform("sum"))
        g = g.merge(ucat[["user_id", "category", "category_affinity"]],
                    on=["user_id", "category"], how="left")
        g["category_affinity"] = g["category_affinity"].fillna(0.0)

        v_eff = g["user_test_count"]
        t_eff = g["user_train_count"]
        schedule_ratio = ((v_eff + SCHEDULE_SMOOTH)
                          / (t_eff + SCHEDULE_SMOOTH))
        g["score"] = (g["engagement"]
                      * np.log10(np.maximum(v_eff, 0.5))
                      * np.power(schedule_ratio, SCHEDULE_ALPHA)
                      * (1.0 + REPEAT_BETA * g["repeat_sm"])
                      * (1.0 + CONSISTENCY_GAMMA * g["week_cov"])
                      * (1.0 + CHANNEL_AFFINITY_BETA * g["channel_affinity"])
                      * (1.0 + RECENT_BETA * g["recent_gain"])
                      * (1.0 + CATEGORY_AFFINITY_BETA
                         * g["category_affinity"]))
    elif args.variant == "engagement":
        g = (base.groupby(["user_id", "tv_show_id"], as_index=False)
                 .agg(engagement=("engagement", "sum"),
                      global_cnt=("user_id", "nunique")))
        g["test_count"] = (g["tv_show_id"].map(test_counts)
                           .fillna(1).astype(float))
        g["score"] = g["engagement"] * np.log10(g["test_count"])
    else:
        g = (full.groupby(["user_id", "tv_show_id"], as_index=False)
                 .agg(count=("ratio", "size"),
                      mean_rating=("rating", "mean"),
                      global_cnt=("user_id", "nunique")))
        g["test_count"] = (g["tv_show_id"].map(test_counts)
                           .fillna(1).astype(float))
        g["train_count"] = (g["tv_show_id"].map(train_counts)
                            .fillna(1).astype(float))
        if args.variant == "test":
            g["score"] = (g["count"] * g["mean_rating"]
                          * np.log10(g["test_count"])
                          * (1.0 + 0.05 * np.log10(g["train_count"])))
        elif args.variant == "train":
            g["score"] = (g["count"] * g["mean_rating"]
                          * np.log10(g["train_count"]))
        else:
            g["score"] = (g["count"] * g["mean_rating"]
                          * np.log10((g["test_count"]
                                      + g["train_count"]) / 2.0))

    g = g.sort_values(["user_id", "score", "global_cnt", "tv_show_id"],
                      ascending=[True, False, False, True])

    preds = {int(u): grp["tv_show_id"].head(5).astype(int).tolist()
             for u, grp in g.groupby("user_id", sort=False)}

    # Popular fallback from the same eligible universe.  For the engagement
    # models this is total engagement, which is a better popularity measure
    # than a raw full-view count.
    if args.variant in ("repeat", "engagement"):
        pop = (base.groupby("tv_show_id")["engagement"].sum()
                   .sort_values(ascending=False))
    else:
        pop = full.groupby("tv_show_id").size().sort_values(ascending=False)
    pop10 = [int(s) for s in pop.index if int(s) in rankable][:10]

    # Score lookup for the long-form diagnostics.
    score_lookup = {(int(u), int(s)): float(v)
                    for u, s, v in g[["user_id", "tv_show_id", "score"]]
                    .itertuples(index=False, name=None)}

    out = sample.copy()
    rows = []
    missing = 0
    for i, u in enumerate(out["user_id"].astype("uint64")):
        p = preds.get(int(u), [])
        if len(p) < 5:
            missing += 1
            p = list(dict.fromkeys(p + pop10))[:5]
        if len(p) < 5:
            raise RuntimeError(f"cannot fill user {u}: {p}")
        out.at[i, "tv_show_id"] = " ".join(str(int(x)) for x in p)
        for r, sid in enumerate(p, start=1):
            rows.append({"user_id": int(u), "rank": r, "tv_show_id": int(sid),
                         "score": score_lookup.get((int(u), int(sid)), np.nan)})

    # Keep a backup of the previous submission before replacing it.
    old = os.path.join(ROOT, "submission.csv")
    backup = os.path.join(ROOT, "submission_personal_previous.csv")
    if args.output == "submission.csv" and os.path.exists(old) and not os.path.exists(backup):
        shutil.copy2(old, backup)
        print("backup:", backup)

    path = os.path.join(ROOT, args.output)
    out.to_csv(path, index=False)
    out.to_csv(os.path.join(RESULTS, f"submission_personal_{args.variant}.csv"),
               index=False)
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(
        RESULTS, f"final_top5_long_personal_{args.variant}.csv"), index=False)

    print("variant:", args.variant)
    print("wrote", path, out.shape)
    print("users with <5 personal recommendations filled from popularity:",
          missing)
    print(out.head(5).to_string(index=False))
    ok = validate_submission(path, epg)
    if not ok:
        raise SystemExit(1)
    print("long-form:", long.shape)
    print("unique recommended shows:", long["tv_show_id"].nunique())


if __name__ == "__main__":
    main()
