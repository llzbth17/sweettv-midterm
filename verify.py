"""Hand-checkable verification log for the report (§4 Verification).

Run after the pipeline:
    python3 midterm/verify.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, RESULTS
from core import average_precision_at_k, load_split
from data_prep import add_screen_time, load_epg, load_sessions, top_targets

OUT = []


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    OUT.append(line)


def main():
    ov = add_screen_time(pd.read_parquet(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "midterm_artifacts", "overlaps_clean.parquet")))
    sess = load_sessions()
    epg = load_epg()

    say("=" * 78)
    say("SWEET.TV midterm — verification log")
    say("=" * 78)

    # 1. One overlap recomputed by hand from raw timestamps -----------------
    u, sid = 6519574208115912303, 5200303
    row = ov[(ov.user_id == u) & (ov.tv_show_id == sid) &
             (ov.air_start == pd.Timestamp("2020-03-11 22:00:00"))].iloc[0]
    srow = sess[(sess.user_id == u) & (sess.channel_id == 4) &
                (sess.start == pd.Timestamp("2020-03-11 22:00:04"))].iloc[0]
    prow = epg[(epg.channel_id == 4) & (epg.start == pd.Timestamp("2020-03-11 22:00:00")) &
               (epg.tv_show_id == sid)].iloc[0]
    overlap = min(srow.stop, prow.end) - max(srow.start, prow.start)
    overlap_s = overlap / np.timedelta64(1, "s")
    ratio = overlap_s / prow.duration
    say("\n[1] Interval join, one concrete row")
    say(f"    session: user={u}, channel=4, {srow.start} -> {srow.stop}")
    say(f"    airing : show={sid}, channel=4, {prow.start} + {prow.duration}s "
        f"(end {prow.end})")
    say(f"    overlap = min(stop, end) - max(start, start) = {overlap_s:.0f}s")
    say(f"    screen time = {overlap_s:.0f}/{prow.duration} = {ratio:.6f}")
    say(f"    table value: seconds={row.seconds:.0f}, ratio={row.ratio:.6f}")
    assert abs(float(row.seconds) - overlap_s) < 1e-3
    assert abs(float(row.ratio) - ratio) < 1e-6
    say("    MATCH")

    # 2. Official target recipe for one user --------------------------------
    say("\n[2] Target recipe for user 6519574208115912303, validation window")
    t = top_targets(ov, "2020-05-18", "2020-07-27", users={u})[u]
    d = ov[(ov.user_id == u) & (ov.air_start >= "2020-05-18") &
           (ov.air_start < "2020-07-27") & (ov.ratio >= 0.8)]
    g = (d.groupby("tv_show_id").agg(freq=("air_start", "size"),
                                     secs=("seconds", "sum"))
           .sort_values(["freq", "secs"], ascending=False).head(7))
    say("    full-view frequency (frequency, seconds):")
    for sid2, r in g.iterrows():
        say(f"      {int(sid2):>9d}  freq={int(r.freq):>2d}  secs={r.secs:>7.0f}")
    say(f"    top-5 programme ids: {t}")
    assert t == g.index[:5].astype(int).tolist()
    say("    MATCH")

    # 3. Duplicate-key integrity -------------------------------------------
    key = ["user_id", "channel_id", "tv_show_id", "air_start", "sess_start"]
    say("\n[3] Join integrity")
    say(f"    overlap rows: {len(ov):,}")
    say(f"    duplicate session-airing keys: {ov.duplicated(key).sum()}")
    say(f"    rows with screen time > 1: {(ov.ratio > 1 + 1e-6).sum()}")
    assert ov.duplicated(key).sum() == 0
    assert (ov.ratio <= 1 + 1e-6).all()
    say("    MATCH")

    # 4. Candidate pool / target containment --------------------------------
    s = load_split("main")
    target_ids = {x for v in s.targets.values() for x in v}
    say("\n[4] Fixed candidate pool")
    say(f"    validation candidates: {s.n_candidates:,}")
    say(f"    target programmes: {len(target_ids):,}")
    say(f"    target programmes inside pool: {len(target_ids & set(s.candidates.tolist())):,}")
    assert target_ids <= set(s.candidates.tolist())
    say("    MATCH (100% target containment)")

    # 5. Hand-computed AP@5 for user B -------------------------------------
    say("\n[5] Hand-computed AP@5 for user 14874594108965844718")
    u2 = 14874594108965844718
    rel = set(s.targets[u2])
    # Hybrid Top-5 copied from case_studies.csv (rank order).
    cs = pd.read_csv(os.path.join(RESULTS, "case_studies.csv"))
    pred = cs[(cs.user_id == u2) & (cs.method == "hybrid")].sort_values("rank")["tv_show_id"].tolist()
    hits, ap, terms = 0, 0.0, []
    for i, x in enumerate(pred, 1):
        if int(x) in rel:
            hits += 1
            ap += hits / i
            terms.append(f"{hits}/{i}")
            say(f"    rank {i}: {int(x)} HIT  -> cumulative hits={hits}, add {hits}/{i}")
        else:
            say(f"    rank {i}: {int(x)} miss")
    ap /= 5.0
    say(f"    AP@5 = ({' + '.join(terms)}) / 5 = {ap:.4f}")
    say(f"    AP@5 = {ap:.4f}")
    check = average_precision_at_k(rel, [int(x) for x in pred], 5)
    say(f"    map_at_k implementation = {check:.4f}")
    assert abs(ap - check) < 1e-12
    say("    MATCH")

    # 6. Submission format ---------------------------------------------------
    sub = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "submission.csv"), dtype={"user_id": "uint64"})
    sample = pd.read_csv(os.path.join(DATA, "submission.csv"), dtype={"user_id": "uint64"})
    say("\n[6] submission.csv")
    say(f"    rows={len(sub)}, users_match={np.array_equal(sub.user_id.values, sample.user_id.values)}")
    ids = sub.tv_show_id.astype(str).str.split()
    say(f"    exactly 5 ids per row: {bool((ids.str.len() == 5).all())}")
    say(f"    unique ids per row: {bool((ids.apply(lambda x: len(set(x)) == 5)).all())}")
    say(f"    all ids positive: {bool(ids.apply(lambda x: all(int(y) > 0 for y in x)).all())}")

    path = os.path.join(RESULTS, "verification.log")
    with open(path, "w") as f:
        f.write("\n".join(OUT) + "\n")
    print("\nwrote", path)


if __name__ == "__main__":
    main()
