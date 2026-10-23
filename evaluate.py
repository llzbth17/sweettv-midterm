"""Validation harness: one split, one candidate pool, one top-k function.

Usage:
    python3 midterm/evaluate.py --split inner
    python3 midterm/evaluate.py --split main
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (HYBRID_WEIGHTS, RESULTS)
from core import (Split, evaluate_predictions, load_split, per_user_ap,
                  rrf_fuse, topk_from_scores)
from methods import run_all


def popularity_fallback(s: Split) -> np.ndarray:
    return s.candidates[np.argsort(-s.pop[s.cand_idx], kind="stable")]


def score_table(s: Split, scores: Dict[str, np.ndarray], info: Dict[str, dict],
                weights: Dict[str, float], tag: str) -> pd.DataFrame:
    fb = popularity_fallback(s)
    rows = []
    hybrid = rrf_fuse(scores, weights)
    all_scores = {**scores, "hybrid": hybrid}
    for name, sc in all_scores.items():
        t = time.time()
        preds = topk_from_scores(sc, s.candidates, fallback=fb)
        met = evaluate_predictions(s.targets, s.eval_users, preds, s.candidates,
                                   pop_share=s.pop_share)
        ap = per_user_ap(s.targets, s.eval_users, preds)
        met.update({
            "split": tag,
            "method": name,
            "Median_AP": float(np.median(ap)),
            "Users_with_hit": int((ap > 0).sum()),
            "Seconds": info.get(name, {}).get("wall_seconds", 0.0),
        })
        if name == "hybrid":
            met["Seconds"] = round(sum(info.get(n, {}).get("wall_seconds", 0.0)
                                       for n in scores), 2)
        rows.append(met)
    df = pd.DataFrame(rows)
    cols = ["split", "method", "MAP@5", "NDCG@5", "Recall@5", "Coverage",
            "Novelty", "Users_with_hit", "Median_AP", "Seconds", "Users"]
    return df[cols].sort_values("MAP@5", ascending=False).reset_index(drop=True)


def tune_hybrid(s: Split, scores: Dict[str, np.ndarray], n_samples: int = 300,
                seed: int = 42) -> Dict[str, float]:
    """Random Dirichlet search on the *inner* split only."""
    rng = np.random.default_rng(seed)
    names = list(scores)
    fb = popularity_fallback(s)
    best_w, best = None, -1.0
    # Include the hand-set prior as a candidate.
    cands = [np.array([HYBRID_WEIGHTS[n] for n in names], dtype="float64")]
    for _ in range(n_samples):
        w = rng.dirichlet(np.ones(len(names)) * 0.8)
        cands.append(w)
    for w in cands:
        wd = dict(zip(names, w))
        preds = topk_from_scores(rrf_fuse(scores, wd), s.candidates, fallback=fb)
        ap = per_user_ap(s.targets, s.eval_users, preds).mean()
        if ap > best:
            best, best_w = ap, wd
    return {k: float(round(v, 4)) for k, v in best_w.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["inner", "main"], default="inner")
    ap.add_argument("--tune", action="store_true", help="tune hybrid weights on this split")
    args = ap.parse_args()

    s = load_split(args.split)
    print(f"\n=== {args.split} split: {s.t0.date()} -> {s.t1.date()} ===")
    print(json.dumps(s.diagnostics, indent=2))
    scores, info = run_all(s)

    weights = HYBRID_WEIGHTS
    if args.split == "main":
        inner_path = os.path.join(RESULTS, "hybrid_weights_inner.json")
        if os.path.exists(inner_path):
            with open(inner_path) as f:
                weights = json.load(f)
            print("using inner-split weights:", weights)
    if args.tune:
        weights = tune_hybrid(s, scores)
        print("tuned weights:", weights)
        with open(os.path.join(RESULTS, f"hybrid_weights_{args.split}.json"), "w") as f:
            json.dump(weights, f, indent=2)

    tbl = score_table(s, scores, info, weights, args.split)
    print("\n" + tbl.to_string(index=False))
    tbl.to_csv(os.path.join(RESULTS, f"validation_{args.split}.csv"), index=False)

    with open(os.path.join(RESULTS, f"validation_{args.split}_info.json"), "w") as f:
        json.dump({"diagnostics": s.diagnostics, "methods": info,
                   "hybrid_weights": weights}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
