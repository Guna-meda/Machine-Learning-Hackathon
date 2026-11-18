"""Step 2: align labels, generate candidates, compute features for train + student_test."""
import sys, time
sys.path.insert(0, ".")
from multiprocessing import Pool
import pandas as pd
from src.config import TRAIN_DIR, STUDENT_TEST_DIR, CACHE_DIR, RESULTS_DIR
from src.data.loader import list_trip_dirs
from src.data.dataset import build_trip_table


def work(args):
    trip_dir, split = args
    feats, quality, _, _ = build_trip_table(trip_dir, split)
    return feats, quality


if __name__ == "__main__":
    CACHE_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)
    jobs = [(d, "train") for d in list_trip_dirs(TRAIN_DIR)] + [(d, "student_test") for d in list_trip_dirs(STUDENT_TEST_DIR)]
    # biggest trips first so the pool stays busy
    jobs.sort(key=lambda j: -(j[0] / f"{j[0].name}.arcs").stat().st_size)
    t0 = time.time()
    with Pool(4) as p:
        out = p.map(work, jobs, chunksize=1)
    table = pd.concat([o[0] for o in out], ignore_index=True)
    qual = pd.DataFrame([o[1] for o in out]).sort_values(["split", "trip_id"])
    table.to_pickle(CACHE_DIR / "candidates_features.pkl")
    qual.to_csv(RESULTS_DIR / "alignment_and_candidate_quality.csv", index=False)
    print("seconds", round(time.time() - t0), "rows", len(table))
    print(qual.groupby("split")[["align_mean_dist_m", "align_frac_gt_30m", "route_coverage", "points_without_candidates", "candidate_miss_rate", "candidate_miss_rate_reliable", "frac_label_ok",
                                 "mean_candidates", "miss_radius_20m", "miss_radius_30m", "miss_radius_50m", "miss_radius_80m", "miss_radius_120m"]].mean().round(4).T.to_string())
    print(qual.sort_values("candidate_miss_rate").tail(5)[["trip_id", "split", "candidate_miss_rate", "candidate_miss_rate_reliable", "frac_label_ok", "align_frac_gt_30m", "align_p95_dist_m"]].to_string())
