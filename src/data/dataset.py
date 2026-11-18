"""Build (and cache) the labelled candidate table of one trip."""
import numpy as np
import pandas as pd
from src.config import CACHE_DIR, LABEL_OK_DIST_M
from src.data.loader import load_trip
from src.data.alignment import align_points_to_route
from src.data.candidates import generate_candidates
from src.features.build import build_features


def build_trip_table(trip_dir, split):
    trip = load_trip(trip_dir)
    pos, ref_arc, ref_dist = align_points_to_route(trip)
    cands = generate_candidates(trip)
    feats = build_features(trip, cands)
    feats["trip_id"] = trip.trip_id
    feats["split"] = split
    feats["ref_arc"] = ref_arc[feats["point_idx"].values]
    feats["label"] = (feats["ref_arc"] == feats["arc_id"]).astype(int)
    # label_ok: the aligned reference arc is close to the point, so the label is believable
    feats["label_ok"] = (ref_dist[feats["point_idx"].values] <= LABEL_OK_DIST_M).astype(int)
    n = len(trip.track)
    has_pos = feats.groupby("point_idx")["label"].max().reindex(range(n), fill_value=0)
    quality = dict(
        trip_id=trip.trip_id, split=split, n_points=n, n_route_arcs=len(trip.route),
        align_mean_dist_m=float(ref_dist.mean()), align_median_dist_m=float(np.median(ref_dist)),
        align_p95_dist_m=float(np.percentile(ref_dist, 95)), align_frac_gt_30m=float((ref_dist > 30).mean()),
        route_coverage=float(len(set(pos)) / len(trip.route)),
        points_without_candidates=int(n - feats["point_idx"].nunique()),
        candidate_miss_rate=float(1 - has_pos.mean()),
        candidate_miss_rate_reliable=float(1 - has_pos[ref_dist <= LABEL_OK_DIST_M].mean()),
        frac_label_ok=float((ref_dist <= LABEL_OK_DIST_M).mean()),
        mean_candidates=float(feats.groupby("point_idx").size().mean()) if len(feats) else 0.0,
    )
    # miss rate if only a radius limit were applied (no cap on candidate count)
    for r in (20, 30, 50, 80, 120):
        quality[f"miss_radius_{r}m"] = float((ref_dist > r).mean())
    return feats, quality, ref_arc, ref_dist
