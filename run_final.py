"""Fit the selected hybrid on all labelled logs and write submission.csv."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, RESULTS, ROOT, TEST_START, TEST_END
from core import (evaluate_predictions, load_split, make_inference_split,
                  rrf_fuse, topk_from_scores)
from data_prep import add_screen_time, load_epg
from methods import run_all


def validate_submission(path: str, epg: pd.DataFrame) -> bool:
    sample = pd.read_csv(os.path.join(DATA, "submission.csv"), dtype={"user_id": "uint64"})
    out = pd.read_csv(path, dtype={"user_id": "uint64"})
    problems = []
    if list(out.columns) != ["user_id", "tv_show_id"]:
        problems.append(f"columns={list(out.columns)}")
    if len(out) != len(sample):
        problems.append(f"rows={len(out)} expected={len(sample)}")
    if not np.array_equal(out["user_id"].values, sample["user_id"].values):
        problems.append("user_id values/order differ from sample")
    valid = set(epg[(epg["start"] >= pd.Timestamp(TEST_START)) &
                    (epg["start"] < pd.Timestamp(TEST_END)) &
                    (epg["tv_show_id"].fillna(0).astype("int64") != 0)]["tv_show_id"].astype(int))
    for i, v in enumerate(out["tv_show_id"].astype(str)):
        parts = v.split()
        if len(parts) != 5:
            problems.append(f"row {i}: {len(parts)} ids")
            continue
        try:
            ids = [int(x) for x in parts]
        except ValueError:
            problems.append(f"row {i}: non-integer id")
            continue
        if any(x <= 0 for x in ids):
            problems.append(f"row {i}: non-positive id")
        if len(set(ids)) != 5:
            problems.append(f"row {i}: duplicate id")
        if not set(ids) <= valid:
            problems.append(f"row {i}: id not in test EPG")
    print("\nsubmission validation:", "OK" if not problems else "FAILED")
    for p in problems[:20]:
        print("  -", p)
    return not problems


def main():
    ov = add_screen_time(pd.read_parquet(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "midterm_artifacts", "overlaps_clean.parquet")))
    epg = load_epg()
    meta = pd.read_parquet(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "midterm_artifacts", "show_metadata.parquet"))
    sample = pd.read_csv(os.path.join(DATA, "submission.csv"), dtype={"user_id": "uint64"})
    s = make_inference_split(ov, epg, meta, sample["user_id"].values)
    print(json.dumps(s.diagnostics, indent=2))
    scores, info = run_all(s)

    wpath = os.path.join(RESULTS, "hybrid_weights_inner.json")
    with open(wpath) as f:
        weights = json.load(f)
    print("hybrid weights:", weights)
    fused = rrf_fuse(scores, weights)
    fb = s.candidates[np.argsort(-s.pop[s.cand_idx], kind="stable")]
    preds = topk_from_scores(fused, s.candidates, fallback=fb)

    # Long-form top-5 with metadata and score, useful for the report / QA.
    rows = []
    for i, u in enumerate(s.eval_users):
        for r, sid in enumerate(preds[i], start=1):
            m = s.meta.loc[int(sid)] if int(sid) in s.meta.index else None
            rows.append({
                "user_id": int(u), "rank": r, "tv_show_id": int(sid),
                "title": "" if m is None else m["title"],
                "category": "" if m is None else m["category"],
                "hybrid_score": float(fused[i, np.where(s.candidates == sid)[0][0]]),
            })
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(RESULTS, "final_top5_long.csv"), index=False)

    out = sample.copy()
    out["tv_show_id"] = [" ".join(str(int(x)) for x in preds[i])
                         for i in range(len(s.eval_users))]
    path = os.path.join(ROOT, "submission.csv")
    out.to_csv(path, index=False)
    out.to_csv(os.path.join(RESULTS, "submission_final.csv"), index=False)
    print("wrote", path, out.shape)
    print(out.head(3).to_string(index=False))
    ok = validate_submission(path, epg)
    with open(os.path.join(RESULTS, "final_info.json"), "w") as f:
        json.dump({"diagnostics": s.diagnostics, "methods": info,
                   "hybrid_weights": weights, "valid": ok}, f, indent=2, default=str)
    # Additional sanity: every user got 5 distinct candidates.
    assert len(long) == len(s.eval_users) * 5
    assert long.groupby("user_id")["tv_show_id"].nunique().min() == 5
    print("long-form top5:", long.shape)


if __name__ == "__main__":
    main()
