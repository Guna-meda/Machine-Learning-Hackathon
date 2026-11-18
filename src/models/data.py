"""Load the cached candidate/feature table and prepare train/eval subsets."""
import numpy as np
import pandas as pd
from src.config import CACHE_DIR, SEED
from src.features.build import FEATURE_COLUMNS


def load_table():
    t = pd.read_pickle(CACHE_DIR / "candidates_features.pkl")
    # one global id per (trip, GPS point) so we can sample whole points, never single candidates
    t["pid"] = t.groupby(["trip_id", "point_idx"], sort=False).ngroup()
    return t


def sample_points(table, frac, seed=SEED):
    """Keep all candidate rows of a random fraction of GPS points (used to keep training fast)."""
    if frac >= 1.0:
        return table
    rng = np.random.default_rng(seed)
    pids = table["pid"].unique()
    keep = rng.choice(pids, size=int(len(pids) * frac), replace=False)
    return table[table["pid"].isin(keep)]


def xy(table):
    return table[FEATURE_COLUMNS].values.astype(np.float32), table["label"].values
