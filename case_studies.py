"""Top-5 case studies for two fixed users, identical across every method."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS
from core import load_split, per_user_ap, rrf_fuse, topk_from_scores
from evaluate import popularity_fallback
from methods import run_all

USERS = [6519574208115912303, 14874594108965844718]
ROLES = {
    6519574208115912303: "News/film generalist",
    14874594108965844718: "Family and serial viewer",
}
METHOD_ORDER = ["popularity", "cbf", "ar", "mf", "ppr", "hybrid"]


def clean(x):
    return str(x).replace("|", "/").strip()


def main():
    s = load_split("main")
    scores, _ = run_all(s)
    with open(os.path.join(RESULTS, "hybrid_weights_inner.json")) as f:
        weights = json.load(f)
    scores["hybrid"] = rrf_fuse(scores, weights)
    fb = popularity_fallback(s)
    preds = {k: topk_from_scores(v, s.candidates, fallback=fb)
             for k, v in scores.items()}

    rows = []
    md = ["# Top-5 case studies", "",
          "Two users were fixed before looking at model outputs. Both have exactly five "
          "validation targets. User A is a news/film generalist; User B is a mixed "
          "family/serial viewer. Every method sees the same candidate pool.", ""]
    summary = {}
    for u in USERS:
        i = int(np.where(s.eval_users == u)[0][0])
        rel = set(s.targets[u])
        row = s.X[s.eval_rows[i]]
        hist = sorted([(float(row[0, j]), int(s.items[j])) for j in row.indices],
                      reverse=True)
        hist_cats = Counter(s.meta.loc[x, "category"] for _, x in hist)
        tgt_cats = Counter(s.meta.loc[x, "category"] for x in s.targets[u])
        md += [f"## {u} — {ROLES[u]}", "",
               f"Training history: {len(hist)} distinct programmes, "
               f"{int(row.sum())} full-view airings. History mix: "
               f"{', '.join(f'{k} {v}' for k, v in hist_cats.most_common())}.", "",
               "**Target (official recipe, ordered by frequency):**", "",
               "| # | tv_show_id | Title | Category |", "|---:|---:|---|---|"]
        for r, x in enumerate(s.targets[u], 1):
            md.append(f"| {r} | {x} | {clean(s.meta.loc[x,'title'])} | {s.meta.loc[x,'category']} |")
            rows.append(dict(user_id=u, role=ROLES[u], section="target", method="target",
                             rank=r, tv_show_id=x, title=s.meta.loc[x, "title"],
                             category=s.meta.loc[x, "category"], hit=True))
        md += ["", "**Most-watched training programmes:**", "",
               "| # | tv_show_id | Title | Category | Full views |",
               "|---:|---:|---|---|---:|"]
        for r, (w, x) in enumerate(hist[:8], 1):
            md.append(f"| {r} | {x} | {clean(s.meta.loc[x,'title'])} | "
                      f"{s.meta.loc[x,'category']} | {w:.0f} |")
            rows.append(dict(user_id=u, role=ROLES[u], section="history", method="history",
                             rank=r, tv_show_id=x, title=s.meta.loc[x, "title"],
                             category=s.meta.loc[x, "category"], hit=False))
        md += ["", "**Recommendations (hit = item in the official target):**", "",
               "| Method | # | tv_show_id | Title | Category | Hit |",
               "|---|---:|---:|---|---|:---:|"]
        for m in METHOD_ORDER:
            for r, x in enumerate(preds[m][i], 1):
                hit = bool(int(x) in rel)
                md.append(f"| {m} | {r} | {int(x)} | {clean(s.meta.loc[int(x),'title'])} | "
                          f"{s.meta.loc[int(x),'category']} | {'yes' if hit else ''} |")
                rows.append(dict(user_id=u, role=ROLES[u], section="recommendation",
                                 method=m, rank=r, tv_show_id=int(x),
                                 title=s.meta.loc[int(x), "title"],
                                 category=s.meta.loc[int(x), "category"], hit=hit))
        ap = per_user_ap(s.targets, [u], preds["hybrid"][i:i + 1])[0]
        summary[str(u)] = {
            "role": ROLES[u], "history_distinct": len(hist),
            "history_full_views": float(row.sum()),
            "history_categories": hist_cats.most_common(),
            "target_categories": tgt_cats.most_common(),
            "hybrid_ap": float(ap),
            "hits": {m: int(sum(int(x) in rel for x in preds[m][i]))
                     for m in METHOD_ORDER},
        }
        md.append("")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "case_studies.csv"), index=False)
    with open(os.path.join(RESULTS, "case_studies.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(RESULTS, "case_studies.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("wrote", os.path.join(RESULTS, "case_studies.csv"))


if __name__ == "__main__":
    main()
