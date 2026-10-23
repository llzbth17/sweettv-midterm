"""Post-hoc analysis for the report: confidence intervals, diversity, cold-start."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FIGURES, RESULTS, HYBRID_WEIGHTS
from core import (bootstrap_ci, load_split, per_user_ap, rrf_fuse,
                  topk_from_scores)
from evaluate import popularity_fallback
from methods import run_all


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    s = load_split("main")
    scores, info = run_all(s)
    wpath = os.path.join(RESULTS, "hybrid_weights_inner.json")
    with open(wpath) as f:
        weights = json.load(f)
    fused = rrf_fuse(scores, weights)
    fb = popularity_fallback(s)
    all_scores = {**scores, "hybrid": fused}
    preds = {k: topk_from_scores(v, s.candidates, fallback=fb)
             for k, v in all_scores.items()}

    order = ["popularity", "cbf", "ar", "mf", "ppr", "hybrid"]
    rows, aps = [], {}
    for name in order:
        ap = per_user_ap(s.targets, s.eval_users, preds[name])
        aps[name] = ap
        lo, hi = bootstrap_ci(ap, n_boot=1000)
        items = preds[name].ravel()
        cold = np.array([s.pop[s.cand_idx][np.where(s.candidates == it)[0][0]] == 0
                         for it in items])
        cats = Counter()
        for it in items:
            cats[s.meta.loc[int(it), "category"] if int(it) in s.meta.index else "?"] += 1
        rows.append({
            "method": name,
            "MAP@5": ap.mean(),
            "CI_low": lo,
            "CI_high": hi,
            "NDCG@5": np.nan,
            "Recall@5": np.nan,
            "Coverage": len(set(items.tolist())) / s.n_candidates,
            "Unique_items": len(set(items.tolist())),
            "Novelty": np.mean([-np.log2(max(s.pop_share.get(int(x), 1e-12), 1e-12))
                                for x in items]),
            "Cold_share": cold.mean(),
            "Top_category": cats.most_common(1)[0][0],
            "Top_category_share": cats.most_common(1)[0][1] / len(items),
            "Seconds": info.get(name, {}).get("wall_seconds",
                                              sum(info[n]["wall_seconds"] for n in scores)),
        })

    # Fill NDCG / Recall from the existing validation table to keep one source.
    vt = pd.read_csv(os.path.join(RESULTS, "validation_main.csv")).set_index("method")
    for r in rows:
        r["NDCG@5"] = float(vt.loc[r["method"], "NDCG@5"])
        r["Recall@5"] = float(vt.loc[r["method"], "Recall@5"])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "analysis_main.csv"), index=False)

    # Paired bootstrap for the two headline comparisons.
    diffs = {}
    for a, b in [("hybrid", "ppr"), ("hybrid", "popularity"), ("ppr", "popularity"),
                 ("mf", "popularity")]:
        d = aps[a] - aps[b]
        lo, hi = bootstrap_ci(d, n_boot=2000)
        diffs[f"{a}_minus_{b}"] = {
            "mean": float(d.mean()), "ci_low": lo, "ci_high": hi,
            "share_users_better": float((d > 0).mean()),
            "share_users_worse": float((d < 0).mean()),
        }

    # Top-5 overlap between methods (average over users).
    overlap = pd.DataFrame(index=order, columns=order, dtype="float64")
    for a in order:
        for b in order:
            overlap.loc[a, b] = np.mean([
                jaccard(preds[a][i], preds[b][i]) for i in range(len(s.eval_users))
            ])
    overlap.to_csv(os.path.join(RESULTS, "method_overlap_main.csv"))

    # Candidate pool diagnostics.
    cold_candidates = int((s.pop[s.cand_idx] == 0).sum())
    diag = dict(s.diagnostics)
    diag["cold_candidates"] = cold_candidates
    diag["cold_candidate_share"] = cold_candidates / s.n_candidates
    diag["train_full_view_events"] = int(s.X.nnz)
    diag["train_distinct_users"] = int((np.asarray(s.X.sum(axis=1)).ravel() > 0).sum())
    diag["method_seconds"] = {k: v.get("wall_seconds", 0) for k, v in info.items()}
    with open(os.path.join(RESULTS, "analysis_main.json"), "w") as f:
        json.dump({"diagnostics": diag, "paired_bootstrap": diffs,
                   "weights": weights,
                   "overlap_mean": overlap.mean(axis=1).to_dict()}, f, indent=2)

    # ------------------------------------------------------------------
    # Figure 1: MAP@5 with 95% bootstrap CI
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    d = df.set_index("method").loc[order]
    colors = ["#9aa5b1", "#c6a15b", "#5b8db8", "#7a6faf", "#4c9a6a", "#c0504d"]
    y = np.arange(len(d))
    ax.barh(y, d["MAP@5"], xerr=[d["MAP@5"] - d["CI_low"], d["CI_high"] - d["MAP@5"]],
            color=colors, capsize=3, height=0.62)
    for i, (v, lo, hi) in enumerate(zip(d["MAP@5"], d["CI_low"], d["CI_high"])):
        ax.text(hi + 0.008, i, f"{v:.3f}  [{lo:.3f}, {hi:.3f}]", va="center", fontsize=7.6)
    ax.set_yticks(y); ax.set_yticklabels([n.upper() if n != "ppr" else "PPR" for n in d.index])
    ax.invert_yaxis(); ax.set_xlim(0, 0.45)
    ax.set_xlabel("MAP@5 on the fixed 10-week validation split (1124 users)")
    ax.set_title("Validation comparison: one split, one candidate pool, one protocol",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_map_ci.png"), dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: method top-5 overlap heatmap
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    im = ax.imshow(overlap.loc[order, order].values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(order)):
            v = overlap.loc[order[i], order[j]]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.55 else "black", fontsize=8)
    ax.set_title("Mean Jaccard overlap of user Top-5 lists")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_overlap.png"), dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: pipeline diagram
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 2.7))
    ax.axis("off")
    boxes = [
        (0.01, "Viewing log\n5.82M sessions"),
        (0.205, "EPG\n3.44M airings"),
        (0.40, "Channel-time\ninterval join"),
        (0.595, "Full-view matrix\nscreen time $\\geq$ 0.8"),
        (0.79, "5 rankers +\nRRF hybrid"),
    ]
    for i, (x, txt) in enumerate(boxes):
        ax.add_patch(plt.Rectangle((x, 0.38), 0.17, 0.34, facecolor="#eef3f8",
                                   edgecolor="#365f7d", lw=1.2))
        ax.text(x + 0.085, 0.55, txt, ha="center", va="center", fontsize=7.6)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 0.197, 0.55), xytext=(x + 0.172, 0.55),
                        arrowprops=dict(arrowstyle="->", lw=1.1, color="#365f7d"))
    ax.text(0.02, 0.20, "Drop tv_show_id = 0: 88.9% of EPG rows; no imputation",
            ha="left", fontsize=7.2, color="#8b1a1a")
    ax.text(0.52, 0.20, "One temporal cut; identical candidate pool for all methods",
            ha="left", fontsize=7.2, color="#8b1a1a")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_pipeline.png"), dpi=220)
    plt.close(fig)

    print(df.to_string(index=False))
    print("\nPaired bootstrap:")
    for k, v in diffs.items():
        print(f"  {k:28s} {v['mean']:+.4f}  [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]  "
              f"better={v['share_users_better']:.1%} worse={v['share_users_worse']:.1%}")
    print("\nOverlap:\n", overlap.round(3).to_string())
    print("\nDiagnostics:", json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
