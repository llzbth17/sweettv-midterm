"""Shared data structures, metrics and scoring utilities.

The protocol is deliberately boring: one temporal cut, one candidate pool, one
user set, one top-k implementation.  Every method receives exactly the same
``Split`` object and returns a dense score matrix over the same candidates.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (ART, DATA, K, SCREEN_THRESHOLD, TRAIN_END, VAL_END,
                    INNER_TRAIN_END, INNER_VAL_END)
from data_prep import add_screen_time, load_epg, top_targets


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def average_precision_at_k(rel: set, pred: Sequence[int], k: int = K) -> float:
    hits, score = 0, 0.0
    for i, item in enumerate(pred[:k], start=1):
        if item in rel:
            hits += 1
            score += hits / i
    return score / min(len(rel), k) if rel else 0.0


def ndcg_at_k(rel: set, pred: Sequence[int], k: int = K) -> float:
    dcg, hits = 0.0, 0
    for i, item in enumerate(pred[:k], start=1):
        if item in rel:
            hits += 1
            dcg += 1.0 / np.log2(i + 1)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg else 0.0


def recall_at_k(rel: set, pred: Sequence[int], k: int = K) -> float:
    if not rel:
        return 0.0
    return len(set(pred[:k]) & rel) / min(len(rel), k)


def topk_from_scores(scores: np.ndarray, candidates: np.ndarray,
                     k: int = K, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    """Top-k item ids per row; deterministic tie-break by candidate order.

    If ``fallback`` is supplied, rows with fewer than k positive scores are
    completed from the fallback ordering (popularity).
    """
    n, m = scores.shape
    kk = min(k, m)
    # argpartition on -score; stable sort of the small selected block.
    part = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
    sel = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-sel, axis=1, kind="stable")
    top = np.take_along_axis(part, order, axis=1)
    conf = np.take_along_axis(sel, order, axis=1) > 0

    if fallback is None:
        return candidates[top]
    out = np.empty((n, kk), dtype=np.int64)
    fb = np.asarray(fallback)
    for i in range(n):
        valid = top[i][conf[i]]
        if len(valid) >= kk:
            out[i] = candidates[valid[:kk]]
            continue
        chosen = list(candidates[valid])
        for j in fb:
            if len(chosen) >= kk:
                break
            if int(j) not in chosen:
                chosen.append(int(j))
        # If the pool itself is smaller than k, pad deterministically.
        if len(chosen) < kk:
            for j in candidates:
                if len(chosen) >= kk:
                    break
                if int(j) not in chosen:
                    chosen.append(int(j))
        out[i] = chosen[:kk]
    return out


def evaluate_predictions(targets: Dict[int, list], users: Sequence[int],
                         preds: np.ndarray, candidates: np.ndarray,
                         pop_share: Optional[Dict[int, float]] = None) -> Dict[str, float]:
    rel = {int(u): set(targets.get(int(u), [])) for u in users}
    aps, ndcgs, recalls = [], [], []
    rec_items = []
    for row, u in enumerate(users):
        p = [int(x) for x in preds[row]]
        aps.append(average_precision_at_k(rel[int(u)], p))
        ndcgs.append(ndcg_at_k(rel[int(u)], p))
        recalls.append(recall_at_k(rel[int(u)], p))
        rec_items.extend(p)
    out = {
        "MAP@5": float(np.mean(aps)),
        "NDCG@5": float(np.mean(ndcgs)),
        "Recall@5": float(np.mean(recalls)),
        "Coverage": float(len(set(rec_items)) / len(candidates)),
        "Users": int(len(users)),
    }
    if pop_share:
        nov = []
        for it in rec_items:
            s = max(pop_share.get(int(it), 1e-12), 1e-12)
            nov.append(-np.log2(s))
        out["Novelty"] = float(np.mean(nov))
    return out


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(values)
    means = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_user_ap(targets: Dict[int, list], users: Sequence[int], preds: np.ndarray) -> np.ndarray:
    return np.asarray([average_precision_at_k(set(targets.get(int(u), [])),
                                               [int(x) for x in preds[i]])
                       for i, u in enumerate(users)])


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
@dataclass
class Split:
    name: str
    t0: pd.Timestamp
    t1: pd.Timestamp
    targets: Dict[int, list]
    eval_users: np.ndarray                 # users with exactly 5 targets
    candidates: np.ndarray                 # fixed pool, sorted
    train_users: np.ndarray
    items: np.ndarray                      # train items + candidates, sorted
    meta: pd.DataFrame                     # indexed by tv_show_id
    X: sp.csr_matrix                       # train users x items, full-view counts
    cand_idx: np.ndarray
    eval_rows: np.ndarray
    pop: np.ndarray                        # popularity over items
    pop_share: Dict[int, float]
    n_train_events: int
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_users(self): return self.X.shape[0]
    @property
    def n_items(self): return self.X.shape[1]
    @property
    def n_candidates(self): return len(self.candidates)


def make_split(name: str, t0: str, t1: str, ov: pd.DataFrame, epg: pd.DataFrame,
               meta: pd.DataFrame, submission_users: np.ndarray,
               require_exactly_five: bool = True) -> Split:
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    train = ov[ov["air_start"] < t0]
    val = ov[(ov["air_start"] >= t0) & (ov["air_start"] < t1)]

    # Candidate pool: every identified programme airing in the window.
    air = epg[(epg["start"] >= t0) & (epg["start"] < t1) &
              (epg["tv_show_id"].fillna(0).astype("int64") != 0)]
    candidates = np.array(sorted(air["tv_show_id"].astype("int64").unique()))

    targets = top_targets(ov, t0, t1, users=set(submission_users.tolist()))
    if require_exactly_five:
        eval_users = np.array(sorted([u for u, v in targets.items() if len(v) == 5]),
                              dtype="uint64")
    else:
        eval_users = np.array(sorted([u for u, v in targets.items() if len(v) > 0]),
                              dtype="uint64")

    train_users = np.array(sorted(train["user_id"].unique()), dtype="uint64")
    train_items = np.array(sorted(train["tv_show_id"].unique()), dtype="int64")
    items = np.array(sorted(set(train_items.tolist()) | set(candidates.tolist())), dtype="int64")
    uidx = {int(u): i for i, u in enumerate(train_users)}
    iidx = {int(s): i for i, s in enumerate(items)}

    # Full-view count matrix, exactly the signal used by the official target.
    full = train[train["ratio"] >= SCREEN_THRESHOLD]
    ui = full["user_id"].map(uidx).to_numpy()
    si = full["tv_show_id"].map(iidx).to_numpy()
    m = np.isfinite(ui) & np.isfinite(si)
    X = sp.coo_matrix((np.ones(int(m.sum()), dtype=np.float32),
                       (ui[m].astype("int64"), si[m].astype("int64"))),
                      shape=(len(train_users), len(items))).tocsr()
    # A user may complete the same show several times; cap at 5 to avoid a
    # single marathon re-run dominating every model.
    X.data = np.minimum(X.data, 5.0)

    cand_idx = np.array([iidx[int(s)] for s in candidates])
    # All submission users should have some training history; if not, append
    # zero rows so the cold-start fallback path is exercised.
    eval_rows = np.array([uidx.get(int(u), -1) for u in eval_users], dtype="int64")
    if (eval_rows < 0).any():
        cold_mask = eval_rows < 0
        cold = eval_users[cold_mask]
        old_n = X.shape[0]
        X = sp.vstack([X, sp.csr_matrix((len(cold), len(items)), dtype=np.float32)]).tocsr()
        train_users = np.concatenate([train_users, cold.astype("uint64")])
        cold_pos = {int(u): old_n + j for j, u in enumerate(cold)}
        eval_rows = np.array([uidx[int(u)] if int(u) in uidx else cold_pos[int(u)]
                              for u in eval_users], dtype="int64")
        uidx = {int(u): i for i, u in enumerate(train_users)}

    pop = np.asarray(X.sum(axis=0)).ravel().astype("float64")
    tot = max(pop.sum(), 1.0)
    pop_share = {int(items[i]): float(pop[i] / tot) for i in range(len(items))}
    diag = {
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "train_users": int(len(train_users)),
        "eval_users": int(len(eval_users)),
        "candidates": int(len(candidates)),
        "candidates_with_train_history": int((np.asarray(X[:, cand_idx].sum(axis=0)).ravel() > 0).sum()),
        "target_shows": int(len({s for v in targets.values() for s in v})),
        "target_shows_in_pool": int(len({s for v in targets.values() for s in v} & set(candidates.tolist()))),
        "density_pct": float(100.0 * X.nnz / (X.shape[0] * X.shape[1])),
    }
    return Split(name, t0, t1, targets, eval_users, candidates, train_users, items,
                 meta.set_index("tv_show_id"), X, cand_idx, eval_rows, pop, pop_share,
                 int(X.nnz), diag)


def make_inference_split(ov: pd.DataFrame, epg: pd.DataFrame, meta: pd.DataFrame,
                         submission_users: np.ndarray) -> Split:
    """Final fit on all labelled logs; candidates = programmes in weeks 31-42."""
    from config import TEST_START, TEST_END
    t0, t1 = pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)
    train = ov[ov["air_start"] < t0]
    air = epg[(epg["start"] >= t0) & (epg["start"] < t1) &
              (epg["tv_show_id"].fillna(0).astype("int64") != 0)]
    candidates = np.array(sorted(air["tv_show_id"].astype("int64").unique()))
    users = np.array(sorted(submission_users.tolist()), dtype="uint64")
    train_users = np.array(sorted(train["user_id"].unique()), dtype="uint64")
    train_items = np.array(sorted(train["tv_show_id"].unique()), dtype="int64")
    items = np.array(sorted(set(train_items.tolist()) | set(candidates.tolist())), dtype="int64")
    uidx = {int(u): i for i, u in enumerate(train_users)}
    iidx = {int(s): i for i, s in enumerate(items)}

    full = train[train["ratio"] >= SCREEN_THRESHOLD]
    ui = full["user_id"].map(uidx).to_numpy()
    si = full["tv_show_id"].map(iidx).to_numpy()
    m = np.isfinite(ui) & np.isfinite(si)
    X = sp.coo_matrix((np.ones(int(m.sum()), dtype=np.float32),
                       (ui[m].astype("int64"), si[m].astype("int64"))),
                      shape=(len(train_users), len(items))).tocsr()
    X.data = np.minimum(X.data, 5.0)

    eval_rows = np.array([uidx.get(int(u), -1) for u in users], dtype="int64")
    if (eval_rows < 0).any():
        cold_mask = eval_rows < 0
        cold = users[cold_mask]
        old_n = X.shape[0]
        X = sp.vstack([X, sp.csr_matrix((len(cold), len(items)), dtype=np.float32)]).tocsr()
        train_users = np.concatenate([train_users, cold])
        cold_pos = {int(u): old_n + j for j, u in enumerate(cold)}
        eval_rows = np.array([uidx[int(u)] if int(u) in uidx else cold_pos[int(u)]
                              for u in users], dtype="int64")

    pop = np.asarray(X.sum(axis=0)).ravel().astype("float64")
    tot = max(pop.sum(), 1.0)
    pop_share = {int(items[i]): float(pop[i] / tot) for i in range(len(items))}
    diag = {
        "train_rows": int(len(train)),
        "train_users": int(len(train_users)),
        "eval_users": int(len(users)),
        "candidates": int(len(candidates)),
        "candidates_with_train_history": int((np.asarray(X[:, np.array([iidx[int(s)] for s in candidates])].sum(axis=0)).ravel() > 0).sum()),
        "density_pct": float(100.0 * X.nnz / (X.shape[0] * X.shape[1])),
    }
    return Split("submission", t0, t1, {}, users, candidates, train_users, items,
                 meta.set_index("tv_show_id"), X,
                 np.array([iidx[int(s)] for s in candidates]),
                 eval_rows, pop, pop_share, int(X.nnz), diag)


def load_split(name: str = "main", force: bool = False) -> Split:
    """Load the fixed main or inner split (cached as npz + parquet metadata)."""
    if name == "main":
        t0, t1 = TRAIN_END, VAL_END
    elif name == "inner":
        t0, t1 = INNER_TRAIN_END, INNER_VAL_END
    else:
        raise ValueError(name)

    ov = add_screen_time(pd.read_parquet(os.path.join(ART, "overlaps_clean.parquet")))
    epg = load_epg()
    meta = pd.read_parquet(os.path.join(ART, "show_metadata.parquet"))
    sub = pd.read_csv(os.path.join(DATA, "submission.csv"), dtype={"user_id": "uint64"})
    return make_split(name, t0, t1, ov, epg, meta, sub["user_id"].values)


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------
def row_normalize(scores: np.ndarray) -> np.ndarray:
    """Scale each row to sum 1 over positive entries (for diagnostics only)."""
    s = np.asarray(scores, dtype="float64")
    pos = np.clip(s, 0, None)
    z = pos.sum(axis=1, keepdims=True)
    return np.divide(pos, z, out=np.zeros_like(pos), where=z > 0)


def rank_matrix(scores: np.ndarray) -> np.ndarray:
    """0 = best; ties broken by ascending candidate index (stable argsort)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(scores.shape[1])[None, :]
    return ranks


def rrf_fuse(scores_by_method: Dict[str, np.ndarray], weights: Dict[str, float],
             k_rrf: int = 60) -> np.ndarray:
    """Reciprocal-rank fusion with per-user weight renormalisation.

    A method is unavailable for a user when its row has no positive score; its
    weight is ignored for that user instead of injecting an arbitrary ranking.
    """
    names = list(scores_by_method)
    M = [scores_by_method[n] for n in names]
    n, m = M[0].shape
    ranks = np.stack([rank_matrix(s) for s in M], axis=0)   # methods x users x items
    avail = np.stack([(s > 0).any(axis=1) for s in M], axis=0)  # methods x users
    w = np.array([weights.get(n, 0.0) for n in names], dtype="float64")
    w = w[:, None] * avail
    wsum = w.sum(axis=0)
    w = np.divide(w, wsum[None, :], out=np.zeros_like(w), where=wsum[None, :] > 0)
    fused = np.zeros((n, m), dtype="float64")
    for j in range(len(names)):
        fused += w[j][:, None] * (1.0 / (k_rrf + ranks[j]))
    return fused
