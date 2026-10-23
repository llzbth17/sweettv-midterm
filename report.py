"""Build the midterm report as a PDF (and a Markdown mirror).

The PDF is intentionally compact: IEEE-style sections, one results table, one
case-study table, one pipeline diagram, and explicit placeholders for the
GitHub URL and the Kaggle score screenshot.
"""
from __future__ import annotations

import json
import os
import sys
from xml.sax.saxutils import escape

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                LongTable, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FIGURES, REPORT, RESULTS

FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", os.path.join(FONT_DIR, "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", os.path.join(FONT_DIR, "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK = colors.HexColor("#1f2d3d")
MUTED = colors.HexColor("#5b6770")
ACCENT = colors.HexColor("#365f7d")
LIGHT = colors.HexColor("#eef3f8")
RULE = colors.HexColor("#c7d2dc")

S = {}
S["title"] = ParagraphStyle("title", fontName="DV-B", fontSize=15.5, leading=18.5,
                            textColor=INK, spaceAfter=2, alignment=TA_LEFT)
S["subtitle"] = ParagraphStyle("subtitle", fontName="DV", fontSize=8.6, leading=11,
                               textColor=MUTED, spaceAfter=1)
S["meta"] = ParagraphStyle("meta", fontName="DV", fontSize=8.0, leading=10.4,
                           textColor=INK)
S["h1"] = ParagraphStyle("h1", fontName="DV-B", fontSize=10.8, leading=13,
                         textColor=ACCENT, spaceBefore=7, spaceAfter=2.5)
S["h2"] = ParagraphStyle("h2", fontName="DV-B", fontSize=8.9, leading=11,
                         textColor=INK, spaceBefore=4, spaceAfter=1.5)
S["body"] = ParagraphStyle("body", fontName="DV", fontSize=8.25, leading=10.55,
                           textColor=INK, alignment=TA_JUSTIFY, spaceAfter=2.8)
S["bullet"] = ParagraphStyle("bullet", parent=S["body"], leftIndent=10, bulletIndent=2,
                             spaceAfter=1.4)
S["small"] = ParagraphStyle("small", fontName="DV", fontSize=6.6, leading=8.0,
                            textColor=INK)
S["tiny"] = ParagraphStyle("tiny", fontName="DV", fontSize=5.8, leading=6.9,
                           textColor=INK)
S["tinyb"] = ParagraphStyle("tinyb", fontName="DV-B", fontSize=5.8, leading=6.9,
                            textColor=INK)
S["caption"] = ParagraphStyle("caption", fontName="DV-I", fontSize=7.1, leading=8.8,
                              textColor=MUTED, spaceBefore=2, spaceAfter=4)
S["code"] = ParagraphStyle("code", fontName="DVM", fontSize=6.5, leading=8.1,
                           textColor=INK, leftIndent=4, spaceAfter=0)
S["box"] = ParagraphStyle("box", fontName="DV", fontSize=7.2, leading=9.0,
                          textColor=INK, alignment=TA_LEFT)


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def fmt(x, n=3):
    return f"{x:.{n}f}"


def pct(x, n=1):
    return f"{100*x:.{n}f}%"


def load_data():
    vt = pd.read_csv(os.path.join(RESULTS, "validation_main.csv")).set_index("method")
    an = pd.read_csv(os.path.join(RESULTS, "analysis_main.csv")).set_index("method")
    with open(os.path.join(RESULTS, "analysis_main.json")) as f:
        aj = json.load(f)
    cs = pd.read_csv(os.path.join(RESULTS, "case_studies.csv"))
    with open(os.path.join(RESULTS, "case_studies.json")) as f:
        csj = json.load(f)
    return vt, an, aj, cs, csj


def results_table(vt, an):
    order = ["popularity", "cbf", "ar", "mf", "ppr", "hybrid"]
    head = ["Method", "MAP@5", "95% CI", "NDCG@5", "Recall@5", "Coverage",
            "Items", "Novelty", "Fit (s)"]
    rows = [head]
    for m in order:
        r = vt.loc[m]
        a = an.loc[m]
        rows.append([
            m.upper() if m != "ppr" else "PPR",
            fmt(r["MAP@5"]), f"[{a.CI_low:.3f}, {a.CI_high:.3f}]",
            fmt(r["NDCG@5"]), fmt(r["Recall@5"]),
            pct(r["Coverage"], 2), f"{int(a.Unique_items)}",
            fmt(r["Novelty"], 2), fmt(r["Seconds"], 2),
        ])
    t = LongTable(rows, colWidths=[2.0*cm, 1.25*cm, 2.35*cm, 1.25*cm, 1.35*cm,
                                   1.35*cm, 1.05*cm, 1.25*cm, 1.15*cm],
                  repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "DV-B"),
        ("FONTNAME", (0, 1), (-1, -1), "DV"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
    ]))
    return t


def cell(text, style="tiny", bold=False):
    """Paragraph cell so long titles wrap instead of overflowing a column."""
    txt = escape(str(text))
    if bold:
        txt = f"<b>{txt}</b>"
    return Paragraph(txt, S[style])


def history_target_table(cs, user):
    rows = [[cell("Type", bold=True), cell("#", bold=True), cell("ID", bold=True),
             cell("Title", bold=True), cell("Category", bold=True)]]
    hist = cs[(cs.user_id == user) & (cs.section == "history")].sort_values("rank").head(5)
    tgt = cs[(cs.user_id == user) & (cs.section == "target")].sort_values("rank")
    for _, r in hist.iterrows():
        rows.append([cell("History"), cell(int(r["rank"])), cell(int(r["tv_show_id"])),
                     cell(r["title"]), cell(r["category"])])
    for _, r in tgt.iterrows():
        rows.append([cell("Target"), cell(int(r["rank"])), cell(int(r["tv_show_id"])),
                     cell(r["title"]), cell(r["category"])])
    t = LongTable(rows, colWidths=[1.3*cm, 0.55*cm, 1.45*cm, 11.3*cm, 3.3*cm],
                  repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "DV-B"), ("FONTNAME", (0, 1), (-1, -1), "DV"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.2),
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("LINEBELOW", (0, 0), (-1, -1), 0.2, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 1), (2, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def case_table(cs):
    ua, ub = 6519574208115912303, 14874594108965844718
    order = ["popularity", "cbf", "ar", "mf", "ppr", "hybrid"]
    rows = [[cell("Method", bold=True), cell("#", bold=True), cell("A: ID", bold=True),
             cell("A: Title", bold=True), cell("A: Cat.", bold=True), cell("✓", bold=True),
             cell("B: ID", bold=True), cell("B: Title", bold=True),
             cell("B: Cat.", bold=True), cell("✓", bold=True)]]
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "DV-B"),
        ("FONTNAME", (0, 1), (-1, -1), "DV"),
        ("FONTSIZE", (0, 0), (-1, -1), 5.4),
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 1), (1, -1), "CENTER"),
        ("ALIGN", (5, 1), (5, -1), "CENTER"),
        ("ALIGN", (9, 1), (9, -1), "CENTER"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.2, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
    ]
    for mi, m in enumerate(order):
        for rank in range(1, 6):
            ra = cs[(cs.user_id == ua) & (cs.section == "recommendation") &
                    (cs.method == m) & (cs["rank"] == rank)].iloc[0]
            rb = cs[(cs.user_id == ub) & (cs.section == "recommendation") &
                    (cs.method == m) & (cs["rank"] == rank)].iloc[0]
            rows.append([
                cell(m if rank == 1 else ""), cell(rank),
                cell(int(ra["tv_show_id"])), cell(ra["title"]),
                cell(ra["category"]), cell("✓" if bool(ra["hit"]) else ""),
                cell(int(rb["tv_show_id"])), cell(rb["title"]),
                cell(rb["category"]), cell("✓" if bool(rb["hit"]) else ""),
            ])
        style_cmds.append(("BACKGROUND", (0, mi * 5 + 1), (-1, mi * 5 + 5),
                           colors.white if mi % 2 == 0 else LIGHT))
    t = LongTable(rows, colWidths=[1.15*cm, 0.35*cm, 1.45*cm, 4.47*cm, 1.95*cm,
                                   0.30*cm, 1.45*cm, 4.47*cm, 1.95*cm, 0.30*cm],
                  repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle(style_cmds))
    return t


def md_table(rows):
    """Render a list of rows (first row = header) as a GitHub Markdown table."""
    head = "| " + " | ".join(str(c) for c in rows[0]) + " |"
    rule = "|" + "|".join(["---"] * len(rows[0])) + "|"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join([head, rule] + body)


def results_table_md(vt, an):
    order = ["popularity", "cbf", "ar", "mf", "ppr", "hybrid"]
    rows = [["Method", "MAP@5", "95% CI", "NDCG@5", "Recall@5", "Coverage",
             "Items", "Novelty", "Fit (s)"]]
    for m in order:
        r = vt.loc[m]
        a = an.loc[m]
        rows.append([
            m.upper() if m != "ppr" else "PPR",
            fmt(r["MAP@5"]), f"[{a.CI_low:.3f}, {a.CI_high:.3f}]",
            fmt(r["NDCG@5"]), fmt(r["Recall@5"]),
            pct(r["Coverage"], 2), f"{int(a.Unique_items)}",
            fmt(r["Novelty"], 2), fmt(r["Seconds"], 2),
        ])
    return md_table(rows)


def history_target_table_md(cs, user):
    rows = [["Type", "#", "ID", "Title", "Category"]]
    hist = cs[(cs.user_id == user) & (cs.section == "history")].sort_values("rank").head(5)
    tgt = cs[(cs.user_id == user) & (cs.section == "target")].sort_values("rank")
    for _, r in hist.iterrows():
        rows.append(["History", int(r["rank"]), int(r["tv_show_id"]),
                     str(r["title"]).replace("|", "/"), r["category"]])
    for _, r in tgt.iterrows():
        rows.append(["Target", int(r["rank"]), int(r["tv_show_id"]),
                     str(r["title"]).replace("|", "/"), r["category"]])
    return md_table(rows)


def case_table_md(cs):
    ua, ub = 6519574208115912303, 14874594108965844718
    order = ["popularity", "cbf", "ar", "mf", "ppr", "hybrid"]
    rows = [["Method", "#", "A: ID", "A: Title", "A: Cat.", "Hit",
             "B: ID", "B: Title", "B: Cat.", "Hit"]]
    for m in order:
        for rank in range(1, 6):
            ra = cs[(cs.user_id == ua) & (cs.section == "recommendation") &
                    (cs.method == m) & (cs["rank"] == rank)].iloc[0]
            rb = cs[(cs.user_id == ub) & (cs.section == "recommendation") &
                    (cs.method == m) & (cs["rank"] == rank)].iloc[0]
            rows.append([
                m if rank == 1 else "", str(rank),
                str(int(ra["tv_show_id"])), str(ra["title"]).replace("|", "/"),
                ra["category"], "Y" if bool(ra["hit"]) else "",
                str(int(rb["tv_show_id"])), str(rb["title"]).replace("|", "/"),
                rb["category"], "Y" if bool(rb["hit"]) else "",
            ])
    return md_table(rows)


def build_markdown(vt, an, aj, cs, csj):
    order = ["popularity", "cbf", "ar", "mf", "ppr", "hybrid"]
    lines = [
        "# Leakage-Safe Hybrid TV Programme Recommendation on SWEET.TV",
        "",
        "**Student:** Elizaveta Gringruz  |  **Team:** Individual  |  **Email:** emgringruz@edu.hse.ru  ",
        "**Date:** 2026-10-22  |  **Assignment:** LLM4Rec Midterm Exam  ",
        "**Kaggle:** elizavetagringruz  |  **GitHub:** https://github.com/llzbth17/sweettv-midterm",
        "",
        "## Abstract",
        "",
        "SWEET.TV must rank five programmes for 1,260 users, yet only 11.1% of EPG airings carry a usable programme identifier. We join 5.82M viewing sessions to identified airings by same-channel time overlap, apply the official screen-time >= 0.8 rule, and compare a popularity baseline with content-based filtering, pair association rules, PureSVD matrix factorization, and personalized PageRank on one fixed 10-week temporal split (6,376 candidates, 1,124 users). A reciprocal-rank hybrid reaches MAP@5 = 0.318 [95% CI 0.304-0.332], versus 0.273 for popularity and 0.304 for PageRank; the +0.014 gain over PageRank is small but its paired bootstrap interval excludes zero. The binding constraint is coverage rather than model class: 71.3% of validation candidates have no training interaction, and a mixed family account still fails when one profile must serve children, news, and serials. The submitted ranker is a channel-aware repeat model with recent Top-5 persistence; on the same fixed split it reaches MAP@5 = 0.398 [0.383, 0.412], so the hybrid remains the interpretable benchmark rather than the deployed model.",
        "",
        "**Index Terms** — recommender systems, implicit feedback, hybrid ranking, personalized PageRank, TV EPG.",
        "",
        "## 1. Introduction",
        "",
        "SWEET.TV has 5.82M viewing sessions for 4,914 users and a programme schedule (EPG) covering weeks 11-42 of 2020. The task is to produce a ranked Top-5 list of programmes for each of 1,260 test users. The central difficulty is not the ranking model: 88.91% of EPG rows have `tv_show_id = 0` (\"programme missing\"), so only 381,892 of 3.44M airings can be linked to an item. Any recommender must first decide how to join a session to a schedule without inventing items.",
        "",
        "A good recommender increases tune-in and retention by putting the right news, kids, film, or serial programme in front of the right viewer. The business value is highest where the schedule is fragmented: a household that watches children's animation in the morning and serials at night is poorly served by a single global popularity list.",
        "",
        "Concrete example: user `6519574208115912303` completed `ТСН.` nine times and `Утро с Интером.` four times in the validation window. A popularity list gives four of the five targets, but it always puts the same national news first. The content model recovers `Утро с Интером.` from the morning-show profile; the hybrid keeps the safe news items and adds the morning show, reaching AP@5 = 1.0.",
        "",
        "**Contributions.** (i) A leakage-safe, reproducible session-to-EPG pipeline with an explicit matching rule and a documented missing-ID policy. (ii) A fixed temporal split and candidate pool used unchanged by five methods and one hybrid. (iii) A validation comparison, two fixed-user Top-5 case studies, and a reproducible submission.",
        "",
        "## 2. Related Work",
        "",
        "Content-based filtering represents items by metadata and users by the items they consumed [5]. Item-based collaborative filtering replaces item content with co-consumption similarity [4], while implicit-feedback matrix factorization models confidence rather than ratings [3]. Association rules provide an interpretable item-to-item mapping through support, confidence, and lift [1]. Personalized PageRank propagates relevance over a user-item graph with a restart to the target user [2]; BPR is the corresponding pairwise-ranking alternative for implicit data [6]. The competition page defines the task and the official target recipe [7].",
        "",
        "**Alternatives considered.** Item-kNN instead of PureSVD: rejected because its Top-5 was almost a popularity list on this sparse matrix; association rules kept as the interpretable co-occurrence model. BPR instead of PureSVD: rejected because no GPU was available and the deterministic SVD was easier to reproduce. Higher-order association rules: pair rules were exact, fast, and sufficient at 0.29% density. A sequence or LLM recommender: promising but out of scope for a one-week midterm and impossible to validate without more temporal folds.",
        "",
        "## 3. Method",
        "",
        "### 3.1 Session-to-schedule join and target",
        "",
        "The log gives `(user_id, vsetv_id, start_time, stop_time)`. `vsetv_id` is the EPG `channel_id`; 197 of the 202 log channels appear in the EPG (five channels are absent). A session is joined to every identified programme on the **same channel** whose half-open interval `[start, start + duration)` intersects the session. The interaction strength is `overlap_seconds`; the official screen time is `overlap_seconds / programme_duration`.",
        "",
        "Programmes with `tv_show_id = 0` are dropped before the join and are **not imputed**. They are 88.91% of EPG rows. A synthetic ID would create false interactions and contaminate both training and the target, so the pipeline keeps only the 381,892 identified airings. The join yields 3,847,592 session-airing rows, 2,963,956 matched sessions (50.9% of 5.82M), 4,895 users, and 4,363 programmes. 122 duplicate session starts were collapsed to the longest stop before joining, so each session-airing pair is emitted once.",
        "",
        "The target follows the organisers' recipe exactly: drop `tv_show_id = 0`, keep airings with screen time >= 0.8, count full airings per `(user, programme)`, break ties by total seconds and then programme ID, and take the five most frequent programmes. A candidate pool is every identified programme airing in the evaluation window; it contains 100% of validation targets (815/815).",
        "",
        "### 3.2 Fixed protocol and leakage control",
        "",
        "The fixed validation cut is **2020-05-18**: training uses all log rows before the cut (10 weeks, 2.38M overlap rows) and validation uses the next 10 weeks (1.46M rows, 6,376 candidates, 1,124 users with exactly five targets). The inner split for hybrid weights is 2020-04-20 to 2020-05-18. The final model trains on all labelled rows through 2020-07-26 and ranks the 6,646 programmes that air in weeks 31-42.",
        "",
        "No post-cutoff viewing behaviour enters any model. Programme metadata (title, category, genres, year, channel) and the evaluation-window schedule are allowed because they are published in advance. Popularity, user profiles, baskets, the SVD, and the PageRank graph are all built from `air_start < cut`. The hybrid weights are selected on the inner split, not on the reported validation split. All methods receive the same `Split` object and the same candidate array; cold items are filled with the same popularity ordering.",
        "",
        "### 3.3 Methods",
        "",
        "**Popularity (reference).** `p_i = sum_u X_ui`, where `X` counts full-view airings. Every user receives the same Top-5. It is not one of the two required methods.",
        "",
        "**Content-based filtering (CBF).** Each programme is a sparse vector: character 3-5 gram TF-IDF of the title plus one-hot category, genre 1-3, decade, and channel. A user profile is the L2-normalised, full-view-weighted sum of the vectors of programmes they completed. The score is cosine similarity between profile and candidate vector. CBF is the only method that can score the 71.3% of candidates with no training interaction.",
        "",
        "**Association rules.** Baskets are the sets of programmes each user completed in training. We mine exact 2-item rules `i -> j` with support >= 3 users, confidence >= 0.02, and lift > 1. The score for user `u` and candidate `j` is `sum_i X_ui * conf(i->j) * log1p(lift(i->j))`, keeping the 50 strongest consequents per antecedent. Pair rules are the right granularity at this density; triples are too sparse.",
        "",
        "**Matrix factorization (PureSVD).** `A = L2_normalise(log1p(X) * IDF)`. A 32-factor truncated SVD gives `U S V^T`; the score is the reconstruction `U_u . V_j`. Log1p damps marathon re-watches, IDF prevents wall-to-wall channels from dominating, and row normalisation puts users on a comparable scale. The fit explains 29.1% of the variance on the validation split.",
        "",
        "**Personalized PageRank (PPR).** On the bipartite user-programme graph, `R` is user->item and `C` is item->user. With restart `alpha = 0.15` and 20 iterations, `p_(t+1) = alpha * seed_u + (1-alpha) * R^T C p_t`; the seed is the user's training history. The item-item transition is never materialised: each iteration is two sparse products, which keeps the 6,646-item test pool tractable.",
        "",
        "**Hybrid.** Scores are fused by reciprocal-rank fusion (RRF, `k = 60`) with weights tuned by 300 Dirichlet draws on the inner split: popularity 0.059, CBF 0.018, AR 0.224, MF 0.410, PPR 0.290. RRF is used instead of raw-score blending because the five score scales are not comparable. A method with no positive score for a user is excluded and the remaining weights are renormalised for that user.",
        "",
        "![Leakage-safe pipeline](midterm_figures/fig_pipeline.png)",
        "",
        "**Figure 1.** Leakage-safe pipeline. Every stage before the five rankers is shared, so differences in the results table come from the model, not from data preparation.",
        "",
        "## 4. Experiments",
        "",
        "### 4.1 Setup",
        "",
        "All experiments run on an Apple M4 Pro laptop (CPU only, Python 3.9.6). The validation protocol is fixed: 1,124 users with exactly five relevant programmes, 6,376 candidates, MAP@5 as the primary metric, and NDCG@5, Recall@5, catalog coverage, and novelty as secondary diagnostics. Coverage is the share of the candidate pool that appears in any user's Top-5; novelty is the mean `-log2(p_i)` of recommended items. Fit times are wall-clock seconds for the 1,124 x 6,376 score matrix; the hybrid adds about 1.2 s of rank fusion.",
        "",
        "",
        results_table_md(vt, an),
        "",
        "**Table 1.** Validation comparison. All rows use the same split, candidate pool, users, fallback, and Top-5 function. Confidence intervals are 1,000-user bootstraps. The full table is in `midterm_results/validation_main.csv`.",
        "",
    ]
    # Add figure after markdown table
    lines += [
        f"![Validation MAP@5](midterm_figures/fig_map_ci.png)",
        "",
        "**Figure 2.** MAP@5 with 95% bootstrap intervals on the fixed validation split.",
        "",
        "### 4.2 Comparison with the baseline",
        "",
        "The hybrid reaches MAP@5 = 0.318 [0.304, 0.332], a **+16.7% relative lift** over popularity (0.273) and **+4.7%** over PPR (0.304). The paired bootstrap difference hybrid - PPR is +0.0144 [0.0051, 0.0227]; 37.3% of users improve, 28.2% get worse, and the rest tie. PPR alone is +11.4% over popularity. Association rules are statistically close to popularity (0.267), MF is 20.8% below popularity (0.216), and CBF is 40.5% below (0.162). The ranking is stable when users with fewer than five targets are included: hybrid 0.305, PPR 0.295, popularity 0.265 on 1,236 users.",
        "",
        "### 4.3 Verification",
        "",
        "Every claim below is reproduced by `python3 midterm/verify.py` and stored in `midterm_results/verification.log`.",
    ]
    lines += [
        "- One interval row was recomputed from raw timestamps: session 2020-03-11 22:00:04 -> 2020-03-12 00:04:12 on channel 4, programme 5200303 (22:00:00 + 6300 s). Overlap = 6296 s, screen time = 6296/6300 = 0.999365; the table stores 6296 s and 0.999365.",
        "- The target recipe was recomputed for user 6519574208115912303: full-view frequencies 9, 4, 3, 2, 2, 1, ... produce `[2400480, 200337, 20088, 240081, 2400508]`, exactly the stored target.",
        "- AP@5 for user 14874594108965844718 was hand-computed from the hybrid list: hits at ranks 1, 2, 4 -> `(1/1 + 2/2 + 3/4) / 5 = 0.55`, identical to the implementation.",
        "- Join integrity: 0 duplicate session-airing keys and 0 rows with screen time > 1. Target containment is 815/815 (100%). The submission has 1,260 rows, exactly five unique positive IDs per row, and every ID airs in the test window.",
    ]
    lines += [
        "",
        "### 4.4 Top-5 case studies",
        "",
        "Two users were fixed before inspecting model outputs. User A (`6519574208115912303`) is a news/film generalist with 23 distinct training programmes and 41 full views. User B (`14874594108965844718`) is a mixed family/serial viewer with 30 distinct programmes and 44 full views. Both have exactly five validation targets.",
        "",
        "**Table 2.** Training history and validation target for user A.",
        "",
        history_target_table_md(cs, 6519574208115912303),
        "",
        "**Table 3.** Training history and validation target for user B.",
        "",
        history_target_table_md(cs, 14874594108965844718),
        "",
        "**Table 4.** Method Top-5 for the same two users. `Y` marks a hit against the official target. IDs are programme IDs; titles and categories are from the EPG.",
        "",
        case_table_md(cs),
        "",
        "**User A.** All methods except CBF recover at least four of the five news/morning targets. Popularity and PPR both pick `Викна-Новости` (2400467), a national news programme the user did not complete; CBF replaces it with `Утро с Украиной` (5900256), another morning show. MF recovers `Утро с Интером.` (200337), the user's second most frequent target. The hybrid keeps the four safe hits and adds `Утро с Интером.`, so AP@5 = 1.0. The example shows why content and collaborative signals are complementary: CBF knows the morning-show genre, MF knows the co-view pattern.",
        "",
        "**User B.** This is the informative failure. The history mixes films (10), children's animation (5), news (5), and serials (5); the target includes two serials (`Отчаянные домохозяйки`, `Однажды в Одессе`), one children's show, one news, and one entertainment show. CBF recommends five children's shows and hits only `Губка Боб`; MF recovers both serials and hits 3/5; AR and popularity recommend almost only news and hit 1/5 and 2/5. The hybrid reaches 3/5 (AP@5 = 0.55) but still misses both serials, because AR, PPR, and popularity outvote MF. This is a household-profile problem, not a scoring bug: one latent vector cannot represent a viewer who switches between children's animation and evening serials.",
        "",
        "## 5. Discussion",
        "",
        "### 5.1 Business and algorithmic analysis",
        "",
        "**Selected strategy.** The submitted ranker is a channel-aware repeat model with recent Top-5 persistence and category affinity, not the five-way hybrid. On the fixed method split (1,124 users) it reaches MAP@5 = 0.398 [0.383, 0.412], +0.0027 over the channel-only model (0.395), +0.0071 over repeat-only (0.391), and far above the hybrid (0.318). On the 12-week airing-level mirror it reaches 0.409. The cost is a few groupbys over 3.0M airing-level overlap rows plus a Top-5 sort; there is no iterative training. The same 6,376-candidate pool and temporal cut are used as in the comparison. A post-hoc lead-in residual reranker (`submission_stable2_h.csv`) raised the latest 12-week holdout from 0.4089 to 0.4124 (+0.0035, 95% CI [+0.0005, +0.0065]), but the gain did not transfer to Kaggle: the public score was worse than the confirmed 0.44224, so the candidate is rejected.",
        "",
        "**Why this strategy.** The target counts complete airings in the next 12 weeks, so the user's own recent viewing is the strongest signal. Engagement weights every observed overlap by screen-time fraction and `0.9^(age in weeks)`; a schedule ratio rewards shows that become more frequent in the test window; repeat and consistency factors reward programmes the user returns to across weeks. Exposure is channel-weighted: airings on channels the user watches are counted fully, airings on channels they never use are discounted, and a small affinity term rewards the show's dominant channel. Recent persistence adds the user's daily Top-5 target ranks over the last 14 days, and category affinity adds a light prior from the user's category mix. The hybrid's collaborative models remain useful as cold-start and explanation components, but they dilute the personal signal: on the same split the best hybrid is 0.318 and PPR 0.304, while the personal ranker is 0.398. The personal model uses only identified programmes in the six allowed categories and fills 8 users with fewer than five eligible shows from the popularity head.",
        "",
        "**User experience.** The final submission recommends 154 distinct programmes across 1,260 users, versus 5 for pure popularity; 49.6% of final slots are news, 20.3% children, 14.8% entertainment, 9.4% serials, 4.7% educational, and 1.2% other. That is closer to the validation target mix than the old hybrid but still news-heavy. A production system should add a diversity or exploration quota (for example, at most three news items per user and at least one item from a second category) and monitor household-level satisfaction rather than only MAP@5.",
        "",
        "**Algorithmic differences.** Mean Top-5 Jaccard overlap is 0.708 between popularity and PPR, 0.652 between AR and PPR, 0.482 between MF and hybrid, and 0.202 between CBF and hybrid. PPR and popularity agree on the news head; MF and CBF contribute the long tail. The hybrid's 0.448 overlap with popularity is therefore not a collapse to the baseline: it keeps the safe head while admitting 20% of slots from other models.",
        "",
        "### 5.2 Failure case: duplicated join rows",
        "",
        "The first implementation of the interval join emitted the best-overlapping programme twice: once in the forward expansion over programmes starting inside the session, and again in the `best` array. On the old artifact this produced 5,948,563 overlap rows with 1,343,480 duplicate `(user, programme, airing)` groups and 2,467,243 extra rows. The duplicate seconds inflated screen time above 1.0 and over-counted the frequency of exactly the programmes a user watched most, which is the worst possible bias for a Top-5 target.",
        "",
        "**Root cause.** The forward loop appended every overlapping programme, then a second pass appended `best` for every session. There was no deduplication key and no invariant check. A second, smaller cause was 122 duplicate session starts in the raw log, which created duplicate overlaps even after the first bug was fixed.",
        "",
        "**Fix and verification.** The join is now a single offset expansion per session; the `best` append was removed; sessions are deduplicated by `(user, channel, start)` keeping the longest stop; and `data_prep.py` asserts 0 duplicate session-airing keys and `screen_time <= 1`. The corrected table has 3,847,592 rows, 0 duplicates, and 0 ratios above 1.0. `verify.py` also recomputes one row from raw timestamps (0.999365) and the target frequencies for a fixed user. This changed the local MAP scale but not the method ranking, which is why the report uses only the corrected artifacts.",
        "",
        "### 5.3 What worked, what surprised me",
        "",
        "**Worked.** (i) Aligning the training matrix with the official full-view target instead of using all seconds. (ii) PPR: a simple graph walk was the strongest single personalised model and the cheapest strong model. (iii) Association rules: pair rules are interpretable and nearly match popularity without any content. (iv) A single popularity fallback for every method made cold-start behaviour comparable instead of hiding it inside each model.",
        "",
        "**Surprised.** (i) CBF has the highest novelty (12.62) and the worst MAP (0.162); novelty is not value. (ii) PureSVD is below the popularity baseline (0.216 vs 0.273) at 0.29% density, even though it is the best model for the serial-heavy User B. (iii) The hybrid still leaves 71.3% of validation candidates unscored by four of five methods; coverage, not model class, is the binding constraint, and the personal ranker avoids this by scoring every show the user has already watched and falling back to popularity. (iv) The local MAP scale (0.16-0.40) is not directly comparable to the Kaggle leaderboard (top around 0.43) because the windows, users, and hidden target differ; the report only claims the measured local gains.",
        "",
        "### 5.4 Limitations",
        "",
        "- **Schedule coverage.** 88.91% of EPG rows are unidentified; 49.1% of sessions never match an identified programme. A programme that only airs with `tv_show_id = 0` can never be recommended or evaluated.",
        "- **Cold candidates.** 71.3% of validation candidates and 57.5% of test candidates have no training interaction; only CBF can score them, and its ranking quality is low.",
        "- **Target definition.** The 0.8 threshold discards partial viewers; users with fewer than five full views are excluded from the main table (112 of 1,236 users with any target). The official test labels are hidden and may use a different tie-break.",
        "- **Distribution shift.** Validation is May-July; the test is August-October, with a new school season and schedule changes. One 10-week split cannot measure seasonal drift.",
        "- **Model limits.** PureSVD is untuned (32 factors, no negative sampling); association rules are pair-only; the hybrid weights come from a four-week inner split with 1,004 users and may not transfer to every segment.",
        "- **No online evidence.** There is no A/B test, impression log, or revenue measurement; all business claims are offline and associational.",
        "",
        "### 5.5 Next improvement",
        "",
        "The concrete next step is **household/session decomposition**: cluster a user's sessions by time-of-day and content, learn 2-3 profiles per heavy account, and route each request to the matching profile. User B is the motivating case: a single vector averages children's animation with evening serials. A second step is to add partial-view and time-decay signals as auxiliary confidence, not as target labels, and to evaluate on three rolling temporal folds. A third is a content embedding (or an LLM-generated genre/style vector) for the 71% cold candidates, blended into PPR rather than used as a standalone ranker.",
        "",
        "### 5.6 Submission and reproducibility",
        "",
        "The final `submission.csv` has 1,260 rows, five unique positive IDs per row, and only programmes airing in weeks 31-42. The final model is the channel-aware repeat ranker with recent Top-5 persistence and category affinity; it trains on all labelled logs through 2020-07-26 and ranks candidates airing in weeks 31-42. The exact commands are listed below; the repository layout is described in `README.md`.",
        "",
        "```",
        "python3 midterm/data_prep.py --force        # rebuild sessions, EPG, overlaps",
        "python3 midterm/evaluate.py --split inner --tune",
        "python3 midterm/evaluate.py --split main",
        "python3 midterm/run_final_personal.py --variant repeat  # writes submission.csv",
        "python3 midterm/analysis.py                # tables, CIs, figures",
        "python3 midterm/case_studies.py",
        "python3 midterm/verify.py",
        "python3 midterm/report.py                  # this PDF",
        "```",
        "",
        "**GitHub:** `https://github.com/llzbth17/sweettv-midterm`. The repository contains `midterm/` (code), `midterm_results/` (tables, logs, case studies), `midterm_figures/` (figures), and the final `submission.csv`.",
        "",
        "![Kaggle submission score](midterm_figures/kaggle_score.png)",
        "",
        "**Kaggle score.** Submission page for the designated final `submission.csv`: public score **0.44048**, private score **0.43478**, user elizavetagringruz. A separate channel-variant submission scored 0.44224 public / 0.43453 private and is kept as `submission_personal_043453.csv`.",
        "",
        "## 6. AI Usage Disclosure",
        "",
        "**Tools.** OpenCode 1.18.30 with DeepSeek V4.1 Flash Free; Python 3.9.6; pandas 2.3.3; NumPy 2.0.2; SciPy 1.13.1; scikit-learn 1.6.1; ReportLab 5.0.1; matplotlib 3.9.4.",
        "",
        "**How AI was used.** Code generation and refactoring for the join, the five models, the evaluation harness, and the report generator; debugging the duplicate-join failure; figure specifications; and a first draft of the report text. The AI did not have access to any post-cutoff viewing behaviour and did not tune on the reported validation labels.",
        "",
        "**What I personally verified.** I re-ran the full pipeline from raw CSVs and checked: (i) one overlap row by hand (6296 s / 6300 s = 0.999365); (ii) the full-view target frequencies for user 6519574208115912303; (iii) AP@5 for user 14874594108965844718 by hand (0.55); (iv) the duplicate-key and screen-time invariants; (v) the 815/815 target containment; and (vi) the 1,260 x 5 submission format. The verification log is generated by `midterm/verify.py`.",
        "",
        "**What I trusted without independent verification.** The official Kaggle MAP@5 implementation and the hidden private-test distribution; the exact organiser tie-break for users with fewer than five full views; and the assumption that EPG metadata for the evaluation window is available at recommendation time. These are stated as assumptions, not verified facts.",
        "",
        "**Session log.** The original, unmodified AI-session export is attached separately (`<student>_midterm_session.json`). It was not edited. The report claims and the session should be read together.",
        "",
        "## References",
        "",
        "[1] R. Agrawal, T. Imieliński, and A. Swami, \"Mining association rules between sets of items in large databases,\" in *Proc. 1993 ACM SIGMOD Int. Conf. Management of Data*, 1993, pp. 207-216. [Online]. Available: https://doi.org/10.1145/170035.170072. [Accessed: 2026-10-22].",
        "",
        "[2] S. Brin and L. Page, \"The anatomy of a large-scale hypertextual Web search engine,\" *Computer Networks and ISDN Systems*, vol. 30, no. 1-7, pp. 107-117, 1998. [Online]. Available: https://doi.org/10.1016/S0169-7552(98)00110-X. [Accessed: 2026-10-22].",
        "",
        "[3] Y. Hu, Y. Koren, and C. Volinsky, \"Collaborative filtering for implicit feedback datasets,\" in *Proc. 8th IEEE Int. Conf. Data Mining (ICDM)*, 2008, pp. 263-272. [Online]. Available: https://doi.org/10.1109/ICDM.2008.22. [Accessed: 2026-10-22].",
        "",
        "[4] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, \"Item-based collaborative filtering recommendation algorithms,\" in *Proc. 10th Int. Conf. World Wide Web (WWW)*, 2001, pp. 285-295. [Online]. Available: https://doi.org/10.1145/371920.372071. [Accessed: 2026-10-22].",
        "",
        "[5] M. J. Pazzani and D. Billsus, \"Content-based recommendation systems,\" in *The Adaptive Web*, LNCS 4321, Springer, 2007, pp. 325-341. [Online]. Available: https://doi.org/10.1007/978-3-540-72079-9_10. [Accessed: 2026-10-22].",
        "",
        "[6] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, \"BPR: Bayesian personalized ranking from implicit feedback,\" in *Proc. 25th Conf. Uncertainty in Artificial Intelligence (UAI)*, 2009. [Online]. Available: https://arxiv.org/abs/1205.2618. [Accessed: 2026-10-22].",
        "",
        "[7] Kaggle, \"sweet.tv - TV program recommender,\" competition page, 2020. [Online]. Available: https://www.kaggle.com/competitions/sweettv-tv-program-recommender. [Accessed: 2026-10-22].",
        "",
    ]
    with open(os.path.join(REPORT, "report.md"), "w") as f:
        f.write("\n".join(lines))


def build_pdf(vt, an, aj, cs, csj):
    path = os.path.join(REPORT, "midterm_report.pdf")
    doc = BaseDocTemplate(path, pagesize=A4,
                          leftMargin=1.55*cm, rightMargin=1.55*cm,
                          topMargin=1.35*cm, bottomMargin=1.35*cm,
                          title="Leakage-Safe Hybrid TV Programme Recommendation on SWEET.TV",
                          author="Elizaveta Gringruz")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")

    def on_page(canvas, doc_):
        canvas.saveState()
        canvas.setFont("DV", 6.3)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, 0.72*cm,
                          "LLM4Rec Midterm — SWEET.TV TV programme recommender")
        canvas.drawRightString(A4[0] - doc.rightMargin, 0.72*cm,
                               f"page {doc_.page}")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.3)
        canvas.line(doc.leftMargin, 0.98*cm, A4[0] - doc.rightMargin, 0.98*cm)
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=on_page)])

    story = []
    story.append(P("Leakage-Safe Hybrid TV Programme Recommendation on SWEET.TV", "title"))
    story.append(P("LLM4Rec Midterm — Kaggle <i>sweettv-tv-program-recommender</i>", "subtitle"))
    story.append(Spacer(1, 2))
    meta = Table([
        [P("<b>Student:</b> Elizaveta Gringruz", "meta"),
         P("<b>Team:</b> Individual", "meta"),
         P("<b>Kaggle:</b> elizavetagringruz", "meta")],
        [P("<b>Email:</b> emgringruz@edu.hse.ru", "meta"),
         P("<b>Date:</b> 2026-10-22", "meta"),
         P("<b>Assignment:</b> Midterm Exam", "meta")],
    ], colWidths=[7.1*cm, 4.2*cm, 6.2*cm])
    meta.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(meta)
    story.append(Spacer(1, 2))
    story.append(P("<b>Code &amp; reproducibility:</b> "
                   "<font color='#8b1a1a'>https://github.com/llzbth17/sweettv-midterm</font>", "subtitle"))
    story.append(Spacer(1, 4))

    story.append(P("<b>Abstract</b>", "h2"))
    story.append(P(
        "SWEET.TV must rank five programmes for 1,260 users, yet only 11.1% of EPG airings "
        "carry a usable programme identifier. We join 5.82M viewing sessions to identified "
        "airings by same-channel time overlap, apply the official screen-time &gt;= 0.8 rule, "
        "and compare a popularity baseline with content-based filtering, pair association "
        "rules, PureSVD matrix factorization, and personalized PageRank on one fixed 10-week "
        "temporal split (6,376 candidates, 1,124 users). A reciprocal-rank hybrid reaches "
        "MAP@5 = 0.318 [95% CI 0.304-0.332], versus 0.273 for popularity and 0.304 for "
        "PageRank; the +0.014 gain over PageRank is small but its paired bootstrap interval "
        "excludes zero. The binding constraint is coverage rather than model class: 71.3% of "
        "validation candidates have no training interaction, and a mixed family account still "
        "fails when one profile must serve children, news, and serials. The submitted ranker is "
        "a channel-aware repeat model with recent Top-5 "
        "persistence; on the same fixed split it reaches MAP@5 = 0.398 [0.383, 0.412], so the hybrid remains the interpretable benchmark rather "
        "than the deployed model."))
    story.append(P("<b>Index Terms</b> — recommender systems, implicit feedback, hybrid ranking, "
                   "personalized PageRank, TV EPG.", "subtitle"))

    # ---------------- 1
    story.append(P("1. Introduction", "h1"))
    story.append(P(
        "SWEET.TV has 5.82M viewing sessions for 4,914 users and a programme schedule (EPG) "
        "covering weeks 11-42 of 2020. The task is to produce a ranked Top-5 list of programmes "
        "for each of 1,260 test users. The central difficulty is not the ranking model: 88.91% "
        "of EPG rows have <font face='DVM'>tv_show_id = 0</font> (\"programme missing\"), so only "
        "381,892 of 3.44M airings can be linked to an item. Any recommender must first decide "
        "how to join a session to a schedule without inventing items."))
    story.append(P(
        "A good recommender increases tune-in and retention by putting the right news, kids, "
        "film, or serial programme in front of the right viewer. The business value is highest "
        "where the schedule is fragmented: a household that watches children's animation in the "
        "morning and serials at night is poorly served by a single global popularity list."))
    story.append(P(
        "<b>Concrete example.</b> User <font face='DVM'>6519574208115912303</font> completed "
        "<i>ТСН.</i> nine times and <i>Утро с Интером.</i> four times in the validation window. "
        "A popularity list gives four of the five targets, but it always puts the same national "
        "news first. The content model recovers <i>Утро с Интером.</i> from the morning-show "
        "profile; the hybrid keeps the safe news items and adds the morning show, reaching "
        "AP@5 = 1.0."))
    story += bullets([
        "<b>Contributions.</b> (i) A leakage-safe, reproducible session-to-EPG pipeline with an "
        "explicit matching rule and a documented missing-ID policy.",
        "(ii) A fixed temporal split, candidate pool, user set, and Top-5 function used unchanged "
        "by five methods and one hybrid.",
        "(iii) A validation comparison, two fixed-user Top-5 case studies, and a final submission.",
    ])

    # ---------------- 2
    story.append(P("2. Related Work", "h1"))
    story.append(P(
        "Content-based filtering represents items by metadata and users by the items they "
        "consumed [5]. Item-based collaborative filtering replaces item content with "
        "co-consumption similarity [4], while implicit-feedback matrix factorization models "
        "confidence rather than ratings [3]. Association rules provide an interpretable "
        "item-to-item mapping through support, confidence, and lift [1]. Personalized PageRank "
        "propagates relevance over a user-item graph with a restart to the target user [2]; BPR "
        "is the corresponding pairwise-ranking alternative for implicit data [6]. The competition "
        "page defines the task and the official target recipe [7]."))
    story.append(P(
        "<b>Alternatives considered.</b> Item-kNN instead of PureSVD: rejected because its Top-5 "
        "was almost a popularity list on this sparse matrix; association rules kept as the "
        "interpretable co-occurrence model. BPR instead of PureSVD: rejected because no GPU was "
        "available and the deterministic SVD was easier to reproduce. Higher-order association "
        "rules: pair rules were exact, fast, and sufficient at 0.29% density. A sequence or LLM "
        "recommender: promising but out of scope for a one-week midterm and impossible to "
        "validate without more temporal folds."))

    # ---------------- 3
    story.append(P("3. Method", "h1"))
    story.append(P("3.1 Session-to-schedule join and target", "h2"))
    story.append(P(
        "The log gives <font face='DVM'>(user_id, vsetv_id, start_time, stop_time)</font>. "
        "<font face='DVM'>vsetv_id</font> is the EPG <font face='DVM'>channel_id</font>; 197 of "
        "the 202 log channels appear in the EPG (five channels are absent). A session is joined "
        "to every identified programme on the <b>same channel</b> whose half-open interval "
        "<font face='DVM'>[start, start + duration)</font> intersects the session. The interaction "
        "strength is overlap seconds; the official screen time is "
        "<font face='DVM'>overlap_seconds / programme_duration</font>."))
    story.append(P(
        "Programmes with <font face='DVM'>tv_show_id = 0</font> are dropped before the join and "
        "are <b>not imputed</b>. They are 88.91% of EPG rows. A synthetic ID would create false "
        "interactions and contaminate both training and the target, so the pipeline keeps only "
        "the 381,892 identified airings. The join yields 3,847,592 session-airing rows, 2,963,956 "
        "matched sessions (50.9% of 5.82M), 4,895 users, and 4,363 programmes. 122 duplicate "
        "session starts were collapsed to the longest stop before joining, so each session-airing "
        "pair is emitted once."))
    story.append(P(
        "The target follows the organisers' recipe exactly: drop <font face='DVM'>tv_show_id = 0</font>, "
        "keep airings with screen time &gt;= 0.8, count full airings per (user, programme), break "
        "ties by total seconds and then programme ID, and take the five most frequent programmes. "
        "The candidate pool is every identified programme airing in the evaluation window; it "
        "contains 100% of validation targets (815/815)."))
    story.append(P("3.2 Fixed protocol and leakage control", "h2"))
    story.append(P(
        "The fixed validation cut is <b>2020-05-18</b>: training uses all log rows before the cut "
        "(10 weeks, 2.38M overlap rows) and validation uses the next 10 weeks (1.46M rows, 6,376 "
        "candidates, 1,124 users with exactly five targets). The inner split for hybrid weights "
        "is 2020-04-20 to 2020-05-18. The final model trains on all labelled rows through "
        "2020-07-26 and ranks the 6,646 programmes that air in weeks 31-42."))
    story.append(P(
        "No post-cutoff viewing behaviour enters any model. Programme metadata (title, category, "
        "genres, year, channel) and the evaluation-window schedule are allowed because they are "
        "published in advance. Popularity, user profiles, baskets, the SVD, and the PageRank graph "
        "are all built from <font face='DVM'>air_start &lt; cut</font>. The hybrid weights are "
        "selected on the inner split, not on the reported validation split. All methods receive "
        "the same <font face='DVM'>Split</font> object and the same candidate array; cold items "
        "are filled with the same popularity ordering."))
    story.append(P("3.3 Methods", "h2"))
    story.append(P(
        "<b>Popularity (reference).</b> <font face='DVM'>p_i = sum_u X_ui</font>, where X counts "
        "full-view airings. Every user receives the same Top-5. It is not one of the two required "
        "methods."))
    story.append(P(
        "<b>Content-based filtering (CBF).</b> Each programme is a sparse vector: character 3-5 "
        "gram TF-IDF of the title plus one-hot category, genre 1-3, decade, and channel. A user "
        "profile is the L2-normalised, full-view-weighted sum of the vectors of programmes they "
        "completed. The score is cosine similarity between profile and candidate vector. CBF is "
        "the only method that can score the 71.3% of candidates with no training interaction."))
    story.append(P(
        "<b>Association rules.</b> Baskets are the sets of programmes each user completed in "
        "training. We mine exact 2-item rules <font face='DVM'>i -&gt; j</font> with support "
        "&gt;= 3 users, confidence &gt;= 0.02, and lift &gt; 1. The score for user u and candidate "
        "j is <font face='DVM'>sum_i X_ui * conf(i-&gt;j) * log1p(lift(i-&gt;j))</font>, keeping "
        "the 50 strongest consequents per antecedent. Pair rules are the right granularity at "
        "this density; triples are too sparse."))
    story.append(P(
        "<b>Matrix factorization (PureSVD).</b> <font face='DVM'>A = L2(log1p(X) * IDF)</font>. "
        "A 32-factor truncated SVD gives <font face='DVM'>U S V^T</font>; the score is the "
        "reconstruction <font face='DVM'>U_u . V_j</font>. Log1p damps marathon re-watches, IDF "
        "prevents wall-to-wall channels from dominating, and row normalisation puts users on a "
        "comparable scale. The fit explains 29.1% of the variance on the validation split."))
    story.append(P(
        "<b>Personalized PageRank (PPR).</b> On the bipartite user-programme graph, R is "
        "user-&gt;item and C is item-&gt;user. With restart alpha = 0.15 and 20 iterations, "
        "<font face='DVM'>p_(t+1) = alpha * seed_u + (1-alpha) * R^T C p_t</font>; the seed is "
        "the user's training history. The item-item transition is never materialised: each "
        "iteration is two sparse products, which keeps the 6,646-item test pool tractable."))
    story.append(P(
        "<b>Hybrid.</b> Scores are fused by reciprocal-rank fusion (RRF, k = 60) with weights "
        "tuned by 300 Dirichlet draws on the inner split: popularity 0.059, CBF 0.018, AR 0.224, "
        "MF 0.410, PPR 0.290. RRF is used instead of raw-score blending because the five score "
        "scales are not comparable. A method with no positive score for a user is excluded and "
        "the remaining weights are renormalised for that user."))

    story.append(Spacer(1, 2))
    img = Image(os.path.join(FIGURES, "fig_pipeline.png"), width=17.0*cm, height=5.6*cm)
    story.append(img)
    story.append(P("Figure 1. Leakage-safe pipeline. Every stage before the five rankers is shared, "
                   "so differences in Table 2 come from the model, not from data preparation.", "caption"))

    story.append(P("3.4 Tools and configuration", "h2"))
    cfg = Table([
        [P("<b>Component</b>", "tinyb"), P("<b>Value</b>", "tinyb")],
        [P("Environment", "tiny"), P("Python 3.9.6; pandas 2.3.3; NumPy 2.0.2; SciPy 1.13.1; "
                                     "scikit-learn 1.6.1; matplotlib 3.9.4; ReportLab 5.0.1; "
                                     "OpenCode 1.18.30 / DeepSeek V4.1 Flash Free", "tiny")],
        [P("Validation cut", "tiny"), P("train &lt; 2020-05-18; validation 2020-05-18 to 2020-07-27; "
                                        "1,124 users; 6,376 candidates", "tiny")],
        [P("Final cut", "tiny"), P("train &lt; 2020-07-27; test 2020-07-27 to 2020-10-20; "
                                   "1,260 users; 6,646 candidates", "tiny")],
        [P("CBF", "tiny"), P("char_wb 3-5 grams, max 20,000 features; category/genre/decade/channel "
                             "one-hot; L2 cosine", "tiny")],
        [P("AR", "tiny"), P("2-item rules; support &gt;= 3; confidence &gt;= 0.02; top 50 per antecedent", "tiny")],
        [P("MF", "tiny"), P("PureSVD, 32 factors, log1p + IDF + row L2, seed 42", "tiny")],
        [P("PPR", "tiny"), P("alpha = 0.15; 20 iterations; bipartite sparse walk", "tiny")],
        [P("Hybrid", "tiny"), P("RRF k = 60; weights 0.059 / 0.018 / 0.224 / 0.410 / 0.290 "
                                "(pop / CBF / AR / MF / PPR)", "tiny")],
    ], colWidths=[2.6*cm, 14.4*cm])
    cfg.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.2, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
    ]))
    story.append(cfg)
    story.append(P("Table 1. Configuration. Hyperparameters were fixed a priori except the hybrid "
                   "weights, which were selected on the inner split.", "caption"))

    # ---------------- 4
    story.append(P("4. Experiments", "h1"))
    story.append(P("4.1 Setup", "h2"))
    story.append(P(
        "All experiments run on an Apple M4 Pro laptop (CPU only). The validation protocol is "
        "fixed: 1,124 users with exactly five relevant programmes, 6,376 candidates, MAP@5 as the "
        "primary metric, and NDCG@5, Recall@5, catalog coverage, and novelty as secondary "
        "diagnostics. Coverage is the share of the candidate pool that appears in any user's "
        "Top-5; novelty is the mean <font face='DVM'>-log2(p_i)</font> of recommended items. Fit "
        "times are wall-clock seconds for the 1,124 x 6,376 score matrix; the hybrid adds about "
        "1.2 s of rank fusion."))
    story.append(results_table(vt, an))
    story.append(P("Table 2. Validation comparison. All rows use the same split, candidate pool, "
                   "users, fallback, and Top-5 function. Confidence intervals are 1,000-user "
                   "bootstraps. Full table: <font face='DVM'>midterm_results/validation_main.csv</font>.",
                   "caption"))
    story.append(Image(os.path.join(FIGURES, "fig_map_ci.png"), width=16.2*cm, height=7.2*cm))
    story.append(P("Figure 2. MAP@5 with 95% bootstrap intervals on the fixed validation split.",
                   "caption"))

    story.append(P("4.2 Comparison with the baseline", "h2"))
    story.append(P(
        "The hybrid reaches MAP@5 = 0.318 [0.304, 0.332], a <b>+16.7% relative lift</b> over "
        "popularity (0.273) and <b>+4.7%</b> over PPR (0.304). The paired bootstrap difference "
        "hybrid - PPR is +0.0144 [0.0051, 0.0227]; 37.3% of users improve, 28.2% get worse, and "
        "the rest tie. PPR alone is +11.4% over popularity. Association rules are statistically "
        "close to popularity (0.267), MF is 20.8% below popularity (0.216), and CBF is 40.5% "
        "below (0.162). The ranking is stable when users with fewer than five targets are "
        "included: hybrid 0.305, PPR 0.295, popularity 0.265 on 1,236 users."))

    story.append(P("4.3 Verification", "h2"))
    story.append(P("Every claim below is reproduced by <font face='DVM'>python3 midterm/verify.py</font> "
                   "and stored in <font face='DVM'>midterm_results/verification.log</font>."))
    story += bullets([
        "One interval row was recomputed from raw timestamps: session 2020-03-11 22:00:04 to "
        "2020-03-12 00:04:12 on channel 4, programme 5200303 (22:00:00 + 6300 s). Overlap = 6296 s, "
        "screen time = 6296/6300 = 0.999365; the table stores 6296 s and 0.999365.",
        "The target recipe was recomputed for user 6519574208115912303: full-view frequencies "
        "9, 4, 3, 2, 2, 1, ... produce [2400480, 200337, 20088, 240081, 2400508], exactly the "
        "stored target.",
        "AP@5 for user 14874594108965844718 was hand-computed from the hybrid list: hits at ranks "
        "1, 2, 4 -> (1/1 + 2/2 + 3/4) / 5 = 0.55, identical to the implementation.",
        "Join integrity: 0 duplicate session-airing keys and 0 rows with screen time &gt; 1. "
        "Target containment is 815/815 (100%). The submission has 1,260 rows, exactly five unique "
        "positive IDs per row, and every ID airs in the test window.",
    ])

    story.append(P("4.4 Top-5 case studies", "h2"))
    story.append(P(
        "Two users were fixed before inspecting model outputs. User A (6519574208115912303) is a "
        "news/film generalist with 23 distinct training programmes and 41 full views. User B "
        "(14874594108965844718) is a mixed family/serial viewer with 30 distinct programmes and "
        "44 full views. Both have exactly five validation targets."))
    story.append(P("<b>Table 3.</b> Training history (top five) and the full validation target for "
                   "user A.", "caption"))
    story.append(history_target_table(cs, 6519574208115912303))
    story.append(Spacer(1, 2))
    story.append(P("<b>Table 4.</b> Training history (top five) and the full validation target for "
                   "user B.", "caption"))
    story.append(history_target_table(cs, 14874594108965844718))
    story.append(Spacer(1, 3))
    story.append(P("<b>Table 5.</b> Method Top-5 for the same two users. A check mark is a hit "
                   "against the official target. IDs are programme IDs; titles and categories are "
                   "from the EPG.", "caption"))
    story.append(case_table(cs))
    story.append(Spacer(1, 2))
    story.append(P(
        "<b>User A.</b> All methods except CBF recover at least four of the five news/morning "
        "targets. Popularity and PPR both pick <i>Викна-Новости</i> (2400467), a national news "
        "programme the user did not complete; CBF replaces it with <i>Утро с Украиной</i> "
        "(5900256), another morning show. MF recovers <i>Утро с Интером.</i> (200337), the user's "
        "second most frequent target. The hybrid keeps the four safe hits and adds <i>Утро с "
        "Интером.</i>, so AP@5 = 1.0. The example shows why content and collaborative signals are "
        "complementary: CBF knows the morning-show genre, MF knows the co-view pattern."))
    story.append(P(
        "<b>User B.</b> This is the informative failure. The history mixes films (10), children's "
        "animation (5), news (5), and serials (5); the target includes two serials (<i>Отчаянные "
        "домохозяйки</i>, <i>Однажды в Одессе</i>), one children's show, one news, and one "
        "entertainment show. CBF recommends five children's shows and hits only <i>Губка Боб</i>; "
        "MF recovers both serials and hits 3/5; AR and popularity recommend almost only news and "
        "hit 1/5 and 2/5. The hybrid reaches 3/5 (AP@5 = 0.55) but still misses both serials, "
        "because AR, PPR, and popularity outvote MF. This is a household-profile problem, not a "
        "scoring bug: one latent vector cannot represent a viewer who switches between children's "
        "animation and evening serials."))

    # ---------------- 5
    story.append(P("5. Discussion", "h1"))
    story.append(P("5.1 Business and algorithmic analysis", "h2"))
    story.append(P(
        "<b>Selected strategy.</b> The submitted ranker is a channel-aware repeat model with recent "
        "Top-5 persistence and category affinity, not the five-way hybrid. On the fixed method "
        "split (1,124 users) it reaches MAP@5 = 0.398 [0.383, 0.412], +0.0027 over the channel-only "
        "model (0.395), +0.0071 over the repeat-only model (0.391), and far above the hybrid "
        "(0.318). On the 12-week airing-level mirror it reaches 0.409. The cost is a few groupbys "
        "over 3.0M airing-level "
        "overlap rows plus a Top-5 sort; there is no iterative training. The same 6,376-candidate "
        "pool and temporal cut are used as in the comparison. A post-hoc lead-in residual reranker "
        "(<font face='DVM'>submission_stable2_h.csv</font>) raised the latest 12-week holdout from "
        "0.4089 to 0.4124 (+0.0035, 95% CI [+0.0005, +0.0065]), but the gain did not transfer: "
        "the public score was worse than the confirmed 0.44224, so the candidate is rejected."))
    story.append(P(
        "<b>Why this strategy.</b> The target counts complete airings in the next 12 weeks, so the "
        "user's own recent viewing is the strongest signal. Engagement weights every observed "
        "overlap by screen-time fraction and 0.9^(age in weeks); a schedule ratio rewards shows "
        "that become more frequent in the test window; repeat and consistency factors reward "
        "programmes the user returns to across weeks. Exposure is channel-weighted: airings on "
        "channels the user watches are counted fully, airings on channels they never use are "
        "discounted, and a small affinity term rewards the show's dominant channel. Recent "
        "persistence adds the user's daily Top-5 target ranks over the last 14 days, and category "
        "affinity adds a light prior from the user's category mix. The hybrid's "
        "collaborative models remain useful as cold-start and explanation components, but they "
        "dilute the personal signal: on the same split the best hybrid is 0.318 and PPR 0.304, "
        "while the personal ranker is 0.398. The personal model uses only identified programmes "
        "in the six allowed categories and fills 8 users with fewer than five eligible shows from "
        "the popularity head."))
    story.append(P(
        "<b>User experience.</b> The final submission recommends 154 distinct programmes across "
        "1,260 users, versus 5 for pure popularity; 49.6% of final slots are news, 20.3% children, "
        "14.8% entertainment, 9.4% serials, 4.7% educational, and 1.2% other. That is closer to "
        "the validation target mix than the old hybrid but still news-heavy. A production system "
        "should add a diversity or exploration quota (for example, at most three news items per "
        "user and at least one item from a second category) and monitor household-level "
        "satisfaction rather than only MAP@5."))
    story.append(P(
        "<b>Algorithmic differences.</b> Mean Top-5 Jaccard overlap is 0.708 between popularity "
        "and PPR, 0.652 between AR and PPR, 0.482 between MF and hybrid, and 0.202 between CBF and "
        "hybrid. PPR and popularity agree on the news head; MF and CBF contribute the long tail. "
        "The hybrid's 0.448 overlap with popularity is therefore not a collapse to the baseline: "
        "it keeps the safe head while admitting about 20% of slots from other models."))

    story.append(P("5.2 Failure case: duplicated join rows", "h2"))
    story.append(P(
        "The first implementation of the interval join emitted the best-overlapping programme "
        "twice: once in the forward expansion over programmes starting inside the session, and "
        "again in the <font face='DVM'>best</font> array. On the old artifact this produced "
        "5,948,563 overlap rows with 1,343,480 duplicate (user, programme, airing) groups and "
        "2,467,243 extra rows. The duplicate seconds inflated screen time above 1.0 and "
        "over-counted the frequency of exactly the programmes a user watched most, which is the "
        "worst possible bias for a Top-5 target."))
    story.append(P(
        "<b>Root cause.</b> The forward loop appended every overlapping programme, then a second "
        "pass appended <font face='DVM'>best</font> for every session. There was no deduplication "
        "key and no invariant check. A second, smaller cause was 122 duplicate session starts in "
        "the raw log, which created duplicate overlaps even after the first bug was fixed."))
    story.append(P(
        "<b>Fix and verification.</b> The join is now a single offset expansion per session; the "
        "<font face='DVM'>best</font> append was removed; sessions are deduplicated by "
        "(user, channel, start) keeping the longest stop; and <font face='DVM'>data_prep.py</font> "
        "asserts 0 duplicate session-airing keys and screen time &lt;= 1. The corrected table has "
        "3,847,592 rows, 0 duplicates, and 0 ratios above 1.0. <font face='DVM'>verify.py</font> "
        "also recomputes one row from raw timestamps (0.999365) and the target frequencies for a "
        "fixed user. This changed the local MAP scale but not the method ranking, which is why the "
        "report uses only the corrected artifacts."))

    story.append(P("5.3 What worked, what surprised me", "h2"))
    story.append(P(
        "<b>Worked.</b> (i) Aligning the training matrix with the official full-view target "
        "instead of using all seconds. (ii) PPR: a simple graph walk was the strongest single "
        "personalised model and the cheapest strong model. (iii) Association rules: pair rules are "
        "interpretable and nearly match popularity without any content. (iv) A single popularity "
        "fallback for every method made cold-start behaviour comparable instead of hiding it "
        "inside each model."))
    story.append(P(
        "<b>Surprised.</b> (i) CBF has the highest novelty (12.62) and the worst MAP (0.162); "
        "novelty is not value. (ii) PureSVD is below the popularity baseline (0.216 vs 0.273) at "
        "0.29% density, even though it is the best model for the serial-heavy User B. (iii) The "
        "hybrid still leaves 71.3% of validation candidates unscored by four of five methods; "
        "coverage, not model class, is the binding constraint, and the personal ranker avoids this "
        "by scoring every show the user has already watched and falling back to popularity. "
        "(iv) The local MAP scale (0.16-0.40) is not directly comparable to the Kaggle leaderboard "
        "(top around 0.43) because the windows, users, and hidden target differ; the report only "
        "claims the measured local gains."))

    story.append(P("5.4 Limitations", "h2"))
    lim = [
        "<b>Schedule coverage.</b> 88.91% of EPG rows are unidentified; 49.1% of sessions never "
        "match an identified programme. A programme that only airs with tv_show_id = 0 can never "
        "be recommended or evaluated.",
        "<b>Cold candidates.</b> 71.3% of validation candidates and 57.5% of test candidates have "
        "no training interaction; only CBF can score them, and its ranking quality is low.",
        "<b>Target definition.</b> The 0.8 threshold discards partial viewers; users with fewer "
        "than five full views are excluded from the main table (112 of 1,236 users with any "
        "target). The official test labels are hidden and may use a different tie-break.",
        "<b>Distribution shift.</b> Validation is May-July; the test is August-October, with a new "
        "school season and schedule changes. One 10-week split cannot measure seasonal drift.",
        "<b>Model limits.</b> PureSVD is untuned (32 factors, no negative sampling); association "
        "rules are pair-only; the hybrid weights come from a four-week inner split with 1,004 "
        "users and may not transfer to every segment.",
        "<b>No online evidence.</b> There is no A/B test, impression log, or revenue measurement; "
        "all business claims are offline and associational.",
    ]
    story += bullets(lim)

    story.append(P("5.5 Next improvement", "h2"))
    story.append(P(
        "The concrete next step is <b>household/session decomposition</b>: cluster a user's "
        "sessions by time-of-day and content, learn 2-3 profiles per heavy account, and route each "
        "request to the matching profile. User B is the motivating case: a single vector averages "
        "children's animation with evening serials. A second step is to add partial-view and "
        "time-decay signals as auxiliary confidence, not as target labels, and to evaluate on "
        "three rolling temporal folds. A third is a content embedding (or an LLM-generated "
        "genre/style vector) for the 71% cold candidates, blended into PPR rather than used as a "
        "standalone ranker."))

    story.append(P("5.6 Submission, reproducibility, and placeholders", "h2"))
    story.append(P(
        "The final <font face='DVM'>submission.csv</font> has 1,260 rows, five unique positive IDs "
        "per row, and only programmes airing in weeks 31-42. The final model is the channel-aware "
        "repeat ranker with recent Top-5 persistence and category affinity; it trains on all "
        "labelled logs through 2020-07-26 and ranks "
        "candidates airing in weeks 31-42."))
    repro = ("python3 midterm/data_prep.py --force\n"
             "python3 midterm/evaluate.py --split inner --tune\n"
             "python3 midterm/evaluate.py --split main\n"
             "python3 midterm/run_final_personal.py --variant repeat\n"
             "python3 midterm/analysis.py\n"
             "python3 midterm/case_studies.py\n"
             "python3 midterm/verify.py\n"
             "python3 midterm/report.py")
    box = Table([[P(repro.replace("\n", "<br/>"), "code")]], colWidths=[17.0*cm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f9fb")),
        ("BOX", (0, 0), (-1, -1), 0.3, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(box)
    story.append(P(
        "<b>GitHub:</b> <font color='#8b1a1a'>https://github.com/llzbth17/sweettv-midterm</font>. "
        "The repository contains <font face='DVM'>midterm/</font> "
        "(code), <font face='DVM'>midterm_results/</font> (tables, logs, case studies), "
        "<font face='DVM'>midterm_figures/</font> (figures), and the final "
        "<font face='DVM'>submission.csv</font>."))
    # Kaggle score screenshot
    iw, ih = 7.4*cm, 7.4*cm*940.0/1235.0
    shot_img = Image(os.path.join(FIGURES, "kaggle_score.png"), width=iw, height=ih)
    shot_txt = P(
        "<b>Kaggle score.</b> Submission page for the designated final "
        "<font face='DVM'>submission.csv</font> (user elizavetagringruz): "
        "public score <b>0.44048</b>, private score <b>0.43478</b>. A separate "
        "channel-variant submission scored 0.44224 public / 0.43453 private and is "
        "kept as <font face='DVM'>submission_personal_043453.csv</font>.", "caption")
    shot = Table([[shot_img, shot_txt]], colWidths=[iw+0.3*cm, 17.0*cm-iw-0.3*cm])
    shot.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 0.2*cm)]))
    story.append(Spacer(1, 3))
    story.append(shot)

    # ---------------- 6
    story.append(P("6. AI Usage Disclosure", "h1"))
    story.append(P(
        "<b>Tools.</b> OpenCode 1.18.30 with DeepSeek V4.1 Flash Free; Python 3.9.6; pandas "
        "2.3.3; NumPy 2.0.2; SciPy 1.13.1; scikit-learn 1.6.1; ReportLab 5.0.1; matplotlib 3.9.4."))
    story.append(P(
        "<b>How AI was used.</b> Code generation and refactoring for the join, the five models, "
        "the evaluation harness, and the report generator; debugging the duplicate-join failure; "
        "figure specifications; and a first draft of the report text. The AI did not have access "
        "to any post-cutoff viewing behaviour and did not tune on the reported validation labels."))
    story.append(P(
        "<b>What I personally verified.</b> I re-ran the full pipeline from raw CSVs and checked: "
        "(i) one overlap row by hand (6296 s / 6300 s = 0.999365); (ii) the full-view target "
        "frequencies for user 6519574208115912303; (iii) AP@5 for user 14874594108965844718 by "
        "hand (0.55); (iv) the duplicate-key and screen-time invariants; (v) the 815/815 target "
        "containment; and (vi) the 1,260 x 5 submission format. The verification log is generated "
        "by <font face='DVM'>midterm/verify.py</font>."))
    story.append(P(
        "<b>What I trusted without independent verification.</b> The official Kaggle MAP@5 "
        "implementation and the hidden private-test distribution; the exact organiser tie-break "
        "for users with fewer than five full views; and the assumption that EPG metadata for the "
        "evaluation window is available at recommendation time. These are stated as assumptions, "
        "not verified facts."))
    story.append(P(
        "<b>Session log.</b> The original, unmodified AI-session export is attached separately "
        "(<font face='DVM'>&lt;student&gt;_midterm_session.json</font>). It was not edited. The "
        "report claims and the session should be read together."))

    # ---------------- refs
    story.append(P("References", "h1"))
    refs = (
        "[1] R. Agrawal, T. Imieliński, and A. Swami, \"Mining association rules between sets of items in "
        "large databases,\" in <i>Proc. ACM SIGMOD</i>, 1993, pp. 207-216. doi:10.1145/170035.170072.   "
        "[2] S. Brin and L. Page, \"The anatomy of a large-scale hypertextual Web search engine,\" "
        "<i>Computer Networks and ISDN Systems</i>, vol. 30, no. 1-7, pp. 107-117, 1998. "
        "doi:10.1016/S0169-7552(98)00110-X.   "
        "[3] Y. Hu, Y. Koren, and C. Volinsky, \"Collaborative filtering for implicit feedback datasets,\" "
        "in <i>Proc. IEEE ICDM</i>, 2008, pp. 263-272. doi:10.1109/ICDM.2008.22.   "
        "[4] B. Sarwar, K. Karypis, J. Konstan, and J. Riedl, \"Item-based collaborative filtering "
        "recommendation algorithms,\" in <i>Proc. WWW</i>, 2001, pp. 285-295. "
        "doi:10.1145/371920.372071.   "
        "[5] M. J. Pazzani and D. Billsus, \"Content-based recommendation systems,\" in <i>The Adaptive "
        "Web</i>, LNCS 4321, Springer, 2007, pp. 325-341. doi:10.1007/978-3-540-72079-9_10.   "
        "[6] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, \"BPR: Bayesian personalized "
        "ranking from implicit feedback,\" in <i>Proc. UAI</i>, 2009. arXiv:1205.2618.   "
        "[7] Kaggle, \"sweet.tv - TV program recommender,\" competition page, 2020. "
        "kaggle.com/competitions/sweettv-tv-program-recommender."
    )
    story.append(P(refs, "small"))

    doc.build(story)
    print("wrote", path)


def main():
    vt, an, aj, cs, csj = load_data()
    build_markdown(vt, an, aj, cs, csj)
    build_pdf(vt, an, aj, cs, csj)


if __name__ == "__main__":
    main()
