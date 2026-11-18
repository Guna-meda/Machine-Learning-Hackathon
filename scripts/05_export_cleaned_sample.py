"""Export a small, human-readable sample of the cleaned/processed data for demos.

Writes two CSVs for one trip (default: first student_test trip):
  results/cleaned_points_<trip>.csv      one row per GPS point: lon, lat, projected x/y (m), time, aligned reference arc, distance to it
  results/cleaned_candidates_<trip>.csv  first 300 points x their candidate arcs, with all features and the 0/1 label
"""
import sys
sys.path.insert(0, ".")
import pandas as pd
from src.config import STUDENT_TEST_DIR, RESULTS_DIR
from src.data.loader import load_trip, list_trip_dirs
from src.data.dataset import build_trip_table

trip_dir = list_trip_dirs(STUDENT_TEST_DIR)[0] if len(sys.argv) < 2 else STUDENT_TEST_DIR / sys.argv[1]
trip = load_trip(trip_dir)
feats, quality, ref_arc, ref_dist = build_trip_table(trip_dir, "student_test")

pts = trip.track.copy()
pts.insert(0, "point_idx", range(len(pts)))
pts["x_m"], pts["y_m"] = trip.track_xy[:, 0].round(2), trip.track_xy[:, 1].round(2)
pts["ref_arc"] = ref_arc
pts["dist_to_ref_arc_m"] = ref_dist.round(2)
pts["label_ok"] = (ref_dist <= 50).astype(int)
pts.to_csv(RESULTS_DIR / f"cleaned_points_{trip.trip_id}.csv", index=False)

cand = feats[feats.point_idx < 300].drop(columns=["split", "label_ok"]).round(3)
cand.to_csv(RESULTS_DIR / f"cleaned_candidates_{trip.trip_id}.csv", index=False)
print(trip.trip_id, "points", len(pts), "candidate rows", len(cand), "label_ok share", pts.label_ok.mean().round(4))
