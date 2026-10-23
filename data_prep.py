"""Data preparation: sessions x EPG interval join and official target labelling.

Matching rule
-------------
A viewing session (user, channel, start, stop) is joined to every EPG programme
on the *same channel* whose half-open interval [programme_start,
programme_start + duration) intersects the session.  The interaction strength
is the overlap in seconds.  A programme airing is counted as a full view when

    screen_time = overlap_seconds / programme_duration >= 0.8

Programmes with ``tv_show_id == 0`` ("programme missing") are not identifiable
and are excluded from the item universe.  They still count as airtime in the
raw EPG, which is why we report their share explicitly.

The join is implemented as a vectorised interval sweep per channel.  For each
session we compute the first and last programme index that can overlap, then
expand offsets 0..n-1.  The same (session, airing) pair is emitted exactly once;
this is asserted in ``run_all``.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (ART, DATA, EPG_FILES, SCREEN_THRESHOLD, TRAIN_END)


EPG_COLS = [
    "channel_id", "channel_title", "start_time", "duration", "tv_show_title",
    "tv_show_id", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
    "tv_show_genre_3", "year_of_production",
]


def _read_epg_file(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=EPG_COLS, low_memory=False)
    d["start"] = pd.to_datetime(d["start_time"], format="%d.%m.%Y %H:%M:%S")
    d["source_file"] = os.path.basename(path)
    return d


def load_epg(cache: bool = True) -> pd.DataFrame:
    """All EPG rows (including tv_show_id == 0) with parsed timestamps."""
    path = os.path.join(ART, "epg_all.parquet")
    if cache and os.path.exists(path):
        return pd.read_parquet(path)
    frames = [_read_epg_file(os.path.join(DATA, f)) for f in EPG_FILES]
    epg = pd.concat(frames, ignore_index=True)
    epg["duration"] = pd.to_numeric(epg["duration"], errors="coerce")
    epg = epg[epg["duration"].notna() & (epg["duration"] > 0)].copy()
    epg["duration"] = epg["duration"].astype("int64")
    epg["end"] = epg["start"] + pd.to_timedelta(epg["duration"], unit="s")
    epg["year"] = pd.to_numeric(epg["year_of_production"], errors="coerce")
    epg.to_parquet(path, index=False)
    return epg


def load_sessions(cache: bool = True) -> pd.DataFrame:
    path = os.path.join(ART, "sessions.parquet")
    if cache and os.path.exists(path):
        return pd.read_parquet(path)
    d = pd.read_csv(
        os.path.join(DATA, "dataset11-30.csv"),
        dtype={"user_id": "uint64", "vsetv_id": "int64", "duraton": "int64"},
        parse_dates=["start_time", "stop_time"],
    )
    d = d.rename(columns={"vsetv_id": "channel_id"})
    d["start"] = d["start_time"]
    d["stop"] = d["stop_time"]
    # A handful of sessions have stop < start; treat them as zero-length.
    bad = d["stop"] < d["start"]
    d.loc[bad, "stop"] = d.loc[bad, "start"]
    d = d[["user_id", "channel_id", "start", "stop", "duraton"]]
    # 122 sessions share (user, channel, start) and differ by a few seconds in
    # stop time (duplicate heartbeat rows).  Keep the longest stop so each
    # session-airing pair is counted once.
    d = (d.sort_values(["user_id", "channel_id", "start", "stop"])
           .drop_duplicates(["user_id", "channel_id", "start"], keep="last")
           .reset_index(drop=True))
    d.to_parquet(path, index=False)
    return d


def _channel_overlaps(sess: pd.DataFrame, prog: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    """Return (session_pos, programme_pos, overlap_seconds) for one channel."""
    # Sort programmes; drop exact duplicate airing rows (same channel/start/id).
    prog = prog.drop_duplicates(["start", "tv_show_id"]).sort_values("start")
    if len(prog) == 0 or len(sess) == 0:
        return (np.array([], dtype="int64"),) * 3

    def ns(x):
        return x.values.astype("datetime64[ns]").astype("int64")

    es = ns(prog["start"])
    ee = ns(prog["end"])
    ss = ns(sess["start"])
    se = ns(sess["stop"])
    good_sess = se > ss
    sess_idx_all = np.nonzero(good_sess)[0]
    ss, se = ss[good_sess], se[good_sess]

    # First programme that starts at or before the session start.
    lo = np.searchsorted(es, ss, side="right") - 1
    # Programmes with start < session end.
    hi = np.searchsorted(es, se, side="left")
    lo = np.maximum(lo, 0)
    n = np.maximum(hi - lo, 0)

    out_s, out_p, out_ov = [], [], []
    max_n = int(n.max()) if len(n) else 0
    # Sanity guard for pathological sessions (e.g. a week-long session).
    # They are handled one by one below, so the offset loop stays bounded.
    offset_limit = min(max_n, 500)
    for off in range(offset_limit):
        sel = np.nonzero(n > off)[0]
        if len(sel) == 0:
            break
        k = lo[sel] + off
        ov = np.minimum(se[sel], ee[k]) - np.maximum(ss[sel], es[k])
        keep = ov > 0
        out_s.append(sess_idx_all[sel[keep]])
        out_p.append(k[keep])
        out_ov.append(ov[keep].astype("float64") / 1e9)

    heavy = np.nonzero(n > offset_limit)[0]
    for h in heavy:
        ks = np.arange(lo[h], hi[h])
        ov = np.minimum(se[h], ee[ks]) - np.maximum(ss[h], es[ks])
        keep = ov > 0
        out_s.append(np.full(int(keep.sum()), sess_idx_all[h], dtype="int64"))
        out_p.append(ks[keep])
        out_ov.append(ov[keep].astype("float64") / 1e9)

    if not out_s:
        return (np.array([], dtype="int64"),) * 3
    return (
        np.concatenate(out_s).astype("int64"),
        np.concatenate(out_p).astype("int64"),
        np.concatenate(out_ov),
    )


def build_overlaps(cache: bool = True) -> pd.DataFrame:
    """Session x identified-programme overlap table (one row per pair)."""
    path = os.path.join(ART, "overlaps_clean.parquet")
    if cache and os.path.exists(path):
        return pd.read_parquet(path)

    sess = load_sessions()
    epg = load_epg()
    epg_id = epg[epg["tv_show_id"].fillna(0).astype("int64") != 0].copy()
    epg_id["tv_show_id"] = epg_id["tv_show_id"].astype("int64")
    print(f"sessions={len(sess):,}  EPG identified rows={len(epg_id):,}")

    frames = []
    for ch, prog in epg_id.groupby("channel_id", sort=False):
        s = sess[sess["channel_id"] == ch]
        if len(s) == 0:
            continue
        si, pi, ov = _channel_overlaps(s, prog)
        if len(si) == 0:
            continue
        pp = prog.iloc[pi]
        frames.append(pd.DataFrame({
            "user_id": s["user_id"].values[si],
            "tv_show_id": pp["tv_show_id"].values,
            "seconds": ov,
            "air_start": pp["start"].values,
            "air_dur": pp["duration"].values,
            "sess_start": s["start"].values[si],
            "sess_dur": (s["stop"].values[si] - s["start"].values[si]) / np.timedelta64(1, "s"),
            "channel_id": ch,
        }))
    ov = pd.concat(frames, ignore_index=True)
    ov["user_id"] = ov["user_id"].astype("uint64")
    ov["tv_show_id"] = ov["tv_show_id"].astype("int64")
    ov["seconds"] = ov["seconds"].astype("float32")
    ov["air_dur"] = ov["air_dur"].astype("int32")
    ov.to_parquet(path, index=False)
    print(f"overlaps={len(ov):,}  users={ov.user_id.nunique():,}  shows={ov.tv_show_id.nunique():,}")
    return ov


def add_screen_time(ov: pd.DataFrame) -> pd.DataFrame:
    ov = ov.copy()
    ov["ratio"] = (ov["seconds"] / ov["air_dur"]).clip(upper=1.0)
    return ov


def build_airings(ov: pd.DataFrame, cache: bool = True) -> pd.DataFrame:
    """Collapse repeated sessions of the same programme airing.

    The official recipe counts a programme airing once per user: if a viewer
    zaps away and returns, the watched seconds belong to the same airing.  The
    raw overlap table has one row per session-airing pair, so we sum seconds by
    ``(user, show, air_start, channel, air_duration)`` before applying the
    screen-time threshold.  This is the table used by the final personal model.
    """
    path = os.path.join(ART, "overlaps_airings.parquet")
    if cache and os.path.exists(path):
        return pd.read_parquet(path)
    key = ["user_id", "tv_show_id", "air_start", "channel_id", "air_dur"]
    a = (ov.groupby(key, as_index=False)
           .agg(seconds=("seconds", "sum"),
                sess_start=("sess_start", "min")))
    a["seconds"] = np.minimum(a["seconds"], a["air_dur"].astype("float32"))
    a["ratio"] = (a["seconds"] / a["air_dur"]).clip(upper=1.0)
    a["user_id"] = a["user_id"].astype("uint64")
    a["tv_show_id"] = a["tv_show_id"].astype("int64")
    a["seconds"] = a["seconds"].astype("float32")
    a["air_dur"] = a["air_dur"].astype("int32")
    a.to_parquet(path, index=False)
    return a


def top_targets(ov: pd.DataFrame, t0, t1, users=None, top_n: int = 5,
                threshold: float = SCREEN_THRESHOLD) -> Dict[int, list]:
    """Official recipe: full airings -> frequency -> exactly top_n programmes."""
    d = ov[(ov["air_start"] >= pd.Timestamp(t0)) & (ov["air_start"] < pd.Timestamp(t1))]
    if users is not None:
        d = d[d["user_id"].isin(users)]
    d = d[d["ratio"] >= threshold]
    if len(d) == 0:
        return {}
    g = (d.groupby(["user_id", "tv_show_id"], as_index=False)
           .agg(freq=("air_start", "size"), secs=("seconds", "sum")))
    g = g.sort_values(["user_id", "freq", "secs", "tv_show_id"],
                      ascending=[True, False, False, True])
    out = {}
    for u, grp in g.groupby("user_id", sort=False):
        out[int(u)] = grp["tv_show_id"].head(top_n).astype(int).tolist()
    return out


def build_metadata(epg: pd.DataFrame) -> pd.DataFrame:
    """One metadata row per tv_show_id (mode/first across all airings)."""
    e = epg[epg["tv_show_id"].fillna(0).astype("int64") != 0].copy()
    e["tv_show_id"] = e["tv_show_id"].astype("int64")
    # A show can have several textual variants; use the most frequent non-null.
    def mode_or_first(s):
        s = s.dropna().astype(str)
        if len(s) == 0:
            return ""
        return s.value_counts().index[0]

    e["title"] = e["tv_show_title"].fillna("")
    g = e.groupby("tv_show_id", as_index=False).agg(
        title=("title", mode_or_first),
        category=("tv_show_category", mode_or_first),
        genre1=("tv_show_genre_1", mode_or_first),
        genre2=("tv_show_genre_2", mode_or_first),
        genre3=("tv_show_genre_3", mode_or_first),
        year=("year", "median"),
        channel_id=("channel_id", lambda s: int(s.mode().iat[0])),
        n_air=("start", "size"),
    )
    g["year"] = g["year"].fillna(0).astype(int)
    g.to_parquet(os.path.join(ART, "show_metadata.parquet"), index=False)
    return g


def run_all(force: bool = False):
    if force:
        for f in ["epg_all.parquet", "sessions.parquet", "overlaps_clean.parquet",
                  "show_metadata.parquet"]:
            p = os.path.join(ART, f)
            if os.path.exists(p):
                os.remove(p)
    epg = load_epg(cache=not force)
    sess = load_sessions(cache=not force)
    ov = add_screen_time(build_overlaps(cache=not force))
    airings = build_airings(ov, cache=not force)
    meta = build_metadata(epg)

    # ---- integrity checks -------------------------------------------------
    key = ["user_id", "channel_id", "tv_show_id", "air_start", "sess_start"]
    dup = ov.duplicated(key).sum()
    assert dup == 0, f"duplicate session-airing rows: {dup}"
    assert (ov["seconds"] > 0).all(), "non-positive overlap"
    assert (ov["seconds"] <= ov["air_dur"] + 1e-6).all(), "overlap longer than programme"
    assert (ov["ratio"] <= 1.0 + 1e-6).all(), "screen time > 1"

    # Coverage diagnostics used in the report.
    sess_total = len(sess)
    matched_sessions = ov[["user_id", "sess_start"]].drop_duplicates().shape[0]
    id0_share = float((epg["tv_show_id"].fillna(0).astype("int64") == 0).mean())
    print("\n--- data integrity ---")
    print(f"EPG rows with tv_show_id==0: {id0_share:.2%}")
    print(f"sessions: {sess_total:,}; with >=1 identified overlap: "
          f"{matched_sessions:,} ({matched_sessions/sess_total:.1%})")
    print(f"overlap rows: {len(ov):,}; duplicate keys: {dup}")
    print(f"airing rows after collapsing repeated sessions: {len(airings):,}; "
          f"full views: {(airings['ratio'] >= SCREEN_THRESHOLD).sum():,}")
    print(f"screen-time quantiles: {ov['ratio'].quantile([.5,.8,.9,.95,.99]).round(3).to_dict()}")

    targets = top_targets(ov, TRAIN_END, "2020-07-27")
    print(f"validation targets: users={len(targets):,}; exactly-5 users="
          f"{sum(len(v)==5 for v in targets.values()):,}")
    return ov, meta


if __name__ == "__main__":
    run_all(force="--force" in sys.argv)
