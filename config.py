"""Central configuration for the SWEET.TV midterm pipeline.

All dates are inclusive at the start and exclusive at the end.
The validation split is fixed for every model and every candidate pool.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ART = os.path.join(ROOT, "midterm_artifacts")
RESULTS = os.path.join(ROOT, "midterm_results")
FIGURES = os.path.join(ROOT, "midterm_figures")
REPORT = os.path.join(ROOT, "midterm_report")

os.makedirs(ART, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed protocol (do not change between methods)
# ---------------------------------------------------------------------------
K = 5
SCREEN_THRESHOLD = 0.8          # official "screen time >= 0.8" rule
RANDOM_SEED = 42

TRAIN_END = "2020-05-18 00:00:00"   # 10 weeks of log (weeks 11-20)
VAL_END = "2020-07-27 00:00:00"     # 10-week validation window (weeks 21-30)
INNER_TRAIN_END = "2020-04-20 00:00:00"
INNER_VAL_END = "2020-05-18 00:00:00"

TEST_START = "2020-07-27 00:00:00"  # weeks 31-42
TEST_END = "2020-10-20 00:00:00"

# Fixed candidate pool size is not truncated: every identified programme that
# airs in the evaluation window is a candidate.  Cold-start candidates are kept.
POPULARITY_FALLBACK = True

# Method hyperparameters (fixed a priori / selected on the inner split only).
CBF_MAX_FEATURES = 20000
CBF_TITLE_WEIGHT = 1.0
CBF_META_WEIGHT = 1.0
AR_MIN_SUPPORT = 3
AR_MIN_CONFIDENCE = 0.02
AR_TOP_RULES_PER_ITEM = 50
MF_FACTORS = 32
MF_ALPHA = 20.0
MF_REG = 0.08
MF_ITERS = 12
PPR_ALPHA = 0.15
PPR_ITERS = 20
HYBRID_WEIGHTS = {
    "popularity": 0.10,
    "cbf": 0.20,
    "ar": 0.20,
    "mf": 0.25,
    "ppr": 0.25,
}

EPG_FILES = [
    "export_arh_11-20-final.csv",
    "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
]
