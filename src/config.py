"""Central settings: paths, seeds and candidate-generation parameters.

The data root is configurable through the PATHSNAP_DATA_DIR environment variable
so the evaluator trips can be dropped into any folder on the server.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_DIR", REPO_ROOT / "data" / "Student folder"))
TRAIN_DIR = DATA_ROOT / "train"
STUDENT_TEST_DIR = DATA_ROOT / "student_test"

RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MAPS_DIR = REPORTS_DIR / "maps"
CACHE_DIR = REPO_ROOT / "cache"          # processed feature tables (gitignored)

SEED = 42

# Candidate generation
CANDIDATE_RADIUS_M = 60.0     # keep arcs whose segment is within this distance of the GPS point
MAX_CANDIDATES = 25          # keep at most this many nearest arcs per point (dense networks need many)

# Reference labels derived by alignment are trusted only if the aligned arc is this close to the point
LABEL_OK_DIST_M = 50.0
