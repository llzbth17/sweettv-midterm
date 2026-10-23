"""Five recommenders on one fixed split.

Implemented methods
-------------------
* Popularity             -- reference baseline (full-view count)
* CBF                    -- content-based filtering on programme metadata
* AssociationRules       -- pairwise association rules (support/confidence/lift)
* MatrixFactorization     -- PureSVD latent-factor CF
* PersonalizedPageRank   -- random walk with restart on the user-item graph

Each ``fit_*`` function returns a dense (eval_users x candidates) score matrix.
Zero means "no evidence"; the evaluation wrapper fills missing top-k slots with
the popularity ordering.  This keeps cold-start behaviour identical across
methods and makes the comparison honest.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (AR_MIN_CONFIDENCE, AR_MIN_SUPPORT, AR_TOP_RULES_PER_ITEM,
                    CBF_MAX_FEATURES, CBF_META_WEIGHT, CBF_TITLE_WEIGHT,
                    MF_FACTORS, PPR_ALPHA, PPR_ITERS, RANDOM_SEED)
from core import Split


# ---------------------------------------------------------------------------
# Popularity
# ---------------------------------------------------------------------------
def fit_popularity(s: Split) -> np.ndarray:
    """Same global top-5 for everyone; a hard reference point."""
    if s.eval_rows.max() >= s.n_users:
        raise RuntimeError("eval row index out of range")
    p = s.pop[s.cand_idx].astype("float64")
    return np.tile(p, (len(s.eval_users), 1))


# ---------------------------------------------------------------------------
# Content-based filtering
# ---------------------------------------------------------------------------
def _metadata_features(s: Split) -> sp.csr_matrix:
    meta = s.meta.reindex(s.items).fillna("")
    title = meta["title"].astype(str).str.lower().str.replace(r"[^0-9a-zа-яёіїєґ\s]", " ", regex=True)
    title = title.str.replace(r"\s+", " ", regex=True).str.strip()

    tv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                         min_df=2, max_features=CBF_MAX_FEATURES,
                         sublinear_tf=True)
    Ft = tv.fit_transform(title.values)

    # Categorical metadata, including a coarse year bucket and the channel.
    rows = []
    for _, r in meta.iterrows():
        year = int(r["year"]) if pd.notna(r["year"]) else 0
        bucket = "unknown" if year == 0 else f"{(year // 10) * 10}s"
        rows.append({
            "category": str(r["category"]),
            "genre1": str(r["genre1"]),
            "genre2": str(r["genre2"]),
            "genre3": str(r["genre3"]),
            "year": bucket,
            "channel": str(int(r["channel_id"])),
        })
    dv = DictVectorizer(sparse=True)
    Fm = dv.fit_transform(rows)

    F = sp.hstack([CBF_TITLE_WEIGHT * Ft, CBF_META_WEIGHT * Fm], format="csr")
    return normalize(F, norm="l2", axis=1, copy=False)


def fit_cbf(s: Split) -> Tuple[np.ndarray, dict]:
    t = time.time()
    F = _metadata_features(s)
    # User profile: full-view-weighted average of the programmes they completed.
    P = s.X @ F
    P = normalize(P, norm="l2", axis=1, copy=False)
    R = P[s.eval_rows] @ F[s.cand_idx].T
    scores = np.asarray(R.todense(), dtype="float32")
    scores[~np.isfinite(scores)] = 0.0
    return scores, {"features": F.shape[1], "seconds": round(time.time() - t, 2)}


# ---------------------------------------------------------------------------
# Association rules
# ---------------------------------------------------------------------------
def fit_association_rules(s: Split) -> Tuple[np.ndarray, dict]:
    """Mine X -> Y rules from user baskets of completed programmes.

    Only 2-item rules are kept.  With ~5k users and ~6.6k items the full
    itemset lattice is unnecessary: the co-occurrence matrix is exact for
    pairs, and pair rules are the ones that produce item-to-item scores.
    """
    t = time.time()
    B = s.X.copy()
    B.data = np.ones_like(B.data, dtype=np.float32)
    B.eliminate_zeros()
    n_users, n_items = B.shape
    sup = np.asarray(B.sum(axis=0)).ravel().astype("float64")   # users per item
    cooc = (B.T @ B).tocsr()
    cand_set = set(s.cand_idx.tolist())
    col_of = {int(ci): j for j, ci in enumerate(s.cand_idx)}

    rows, cols, vals = [], [], []
    for i in range(n_items):
        start, end = cooc.indptr[i], cooc.indptr[i + 1]
        if start == end or sup[i] < AR_MIN_SUPPORT:
            continue
        js = cooc.indices[start:end]
        vs = cooc.data[start:end].astype("float64")
        if len(js) == 0:
            continue
        conf = vs / max(sup[i], 1.0)
        # lift = conf / P(j) = (v / sup_i) / (sup_j / n_users)
        lift = conf * n_users / np.maximum(sup[js], 1.0)
        good = (vs >= AR_MIN_SUPPORT) & (conf >= AR_MIN_CONFIDENCE) & (js != i)
        # candidate consequents only
        cand_mask = np.array([int(j) in cand_set for j in js], dtype=bool)
        good &= cand_mask
        if not good.any():
            continue
        w = conf[good] * np.log1p(np.maximum(lift[good], 1.0))
        jj = js[good]
        if len(jj) > AR_TOP_RULES_PER_ITEM:
            keep = np.argpartition(-w, AR_TOP_RULES_PER_ITEM)[:AR_TOP_RULES_PER_ITEM]
            jj, w = jj[keep], w[keep]
        rows.extend([i] * len(jj))
        cols.extend([col_of[int(j)] for j in jj])
        vals.extend(w)

    R = sp.coo_matrix((np.asarray(vals, dtype="float32"),
                       (np.asarray(rows, dtype="int64"), np.asarray(cols, dtype="int64"))),
                      shape=(n_items, len(s.cand_idx))).tocsr()
    S = B @ R
    scores = np.asarray(S[s.eval_rows].todense(), dtype="float32")
    return scores, {"rules": int(R.nnz), "items_with_rules": int((np.diff(R.tocsr().indptr) > 0).sum()),
                    "seconds": round(time.time() - t, 2)}


# ---------------------------------------------------------------------------
# Matrix factorization (PureSVD)
# ---------------------------------------------------------------------------
def fit_matrix_factorization(s: Split) -> Tuple[np.ndarray, dict]:
    t = time.time()
    A = s.X.copy().astype("float32")
    # log1p damps marathon re-watches; IDF prevents wall-to-wall channels from
    # dominating the latent space; row L2 puts users on a comparable scale.
    A.data = np.log1p(A.data)
    df = np.asarray((A > 0).sum(axis=0)).ravel().astype("float32")
    idf = np.log((A.shape[0] + 1.0) / (df + 1.0)) + 1.0
    A = A @ sp.diags(idf)
    A = normalize(A, norm="l2", axis=1, copy=False)
    # Randomized SVD on a matrix with all-zero cold-user rows emits harmless
    # BLAS divide/overflow warnings on some macOS Accelerate builds; the
    # resulting factors are finite (asserted below).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        svd = TruncatedSVD(n_components=MF_FACTORS, n_iter=7, random_state=RANDOM_SEED)
        U = svd.fit_transform(A).astype("float32")
        V = svd.components_.astype("float32")
        S = U[s.eval_rows] @ V[:, s.cand_idx]
    assert np.isfinite(S).all(), "non-finite MF scores"
    return np.asarray(S, dtype="float32"), {
        "factors": MF_FACTORS,
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
        "seconds": round(time.time() - t, 2),
    }


# ---------------------------------------------------------------------------
# Personalized PageRank
# ---------------------------------------------------------------------------
def fit_personalized_pagerank(s: Split) -> Tuple[np.ndarray, dict]:
    """Random walk with restart on the bipartite user-programme graph.

    p_{t+1} = alpha * seed + (1-alpha) * R^T C p, where R is user->item and C
    is item->user.  The item-item transition is never materialised; each
    iteration is two sparse products, which keeps the 6.6k-item test pool
    tractable.
    """
    t = time.time()
    X = s.X.astype("float32")
    row = np.asarray(X.sum(axis=1)).ravel()
    col = np.asarray(X.sum(axis=0)).ravel()
    R = sp.diags(1.0 / np.maximum(row, 1e-9)) @ X          # user -> item
    C = X @ sp.diags(1.0 / np.maximum(col, 1e-9))          # item -> user
    R = R.tocsr(); C = C.tocsr()

    seed = R[s.eval_rows].T.toarray().astype("float32")    # items x users
    P = seed.copy()
    for _ in range(PPR_ITERS):
        q = C @ P                                           # users x users
        P = PPR_ALPHA * seed + (1.0 - PPR_ALPHA) * (R.T @ q)
    scores = P[s.cand_idx].T                                # users x candidates
    return np.asarray(scores, dtype="float32"), {
        "alpha": PPR_ALPHA, "iterations": PPR_ITERS,
        "seconds": round(time.time() - t, 2),
    }


METHODS = {
    "popularity": fit_popularity,
    "cbf": fit_cbf,
    "ar": fit_association_rules,
    "mf": fit_matrix_factorization,
    "ppr": fit_personalized_pagerank,
}


def run_all(s: Split, methods=None) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    methods = methods or list(METHODS)
    scores, info = {}, {}
    for name in methods:
        t = time.time()
        out = METHODS[name](s)
        if isinstance(out, tuple):
            sc, extra = out
        else:
            sc, extra = out, {}
        scores[name] = np.asarray(sc, dtype="float32")
        info[name] = {**extra, "wall_seconds": round(time.time() - t, 2)}
        print(f"  {name:12s} shape={scores[name].shape} "
              f"nonzero_rows={int((scores[name] > 0).any(axis=1).sum()):4d} "
              f"time={info[name]['wall_seconds']:.1f}s {extra}")
    return scores, info


if __name__ == "__main__":
    from core import load_split
    s = load_split("inner")
    print(s.diagnostics)
    run_all(s)
