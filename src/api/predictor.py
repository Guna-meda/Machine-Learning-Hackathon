"""Prediction pipeline used by the API: raw trip folder -> predicted route.

This module NEVER reads the .route file (ground truth).  Scoring against the reference
happens in src/api/evaluator.py, after the prediction is finished.

Steps: load trip -> clean GPS -> candidate arcs (KD-tree) -> features -> classifier probabilities
       -> HMM/Viterbi -> stitched connected route.
"""
import json
import time
import numpy as np
import joblib
from src.config import CANDIDATE_RADIUS_M, MAX_CANDIDATES
from src.data.loader import load_trip, set_projection
from src.data.candidates import generate_candidates
from src.features.build import build_features, FEATURE_COLUMNS
from src.hmm.network import RoadNetwork, stitch_route
from src.hmm.viterbi import decode_trip, HMMConfig


def clean_track(trip):
    """Keep valid GPS rows only, sorted by time.  Returns the original row index of every kept row.

    Dropped: NaN values, impossible coordinates, (0,0) points.  Time order is repaired with a stable sort.
    """
    tr = trip.track
    ok = (tr.notna().all(axis=1) & tr.lon.between(-180, 180) & tr.lat.between(-90, 90) & ~((tr.lon == 0) & (tr.lat == 0)))
    kept = tr[ok].sort_values("t", kind="stable")
    orig_idx = kept.index.values.copy()
    trip.track = kept.reset_index(drop=True)
    set_projection(trip)
    return orig_idx


class Predictor:
    def __init__(self, artifacts_dir):
        from pathlib import Path
        artifacts_dir = Path(artifacts_dir)
        self.model = joblib.load(artifacts_dir / "model.joblib")
        self.model_cfg = json.load(open(artifacts_dir / "model_config.json"))
        self.hmm_cfg = HMMConfig(**json.load(open(artifacts_dir / "hmm_config.json")))
        assert self.model_cfg["feature_columns"] == FEATURE_COLUMNS, "feature list in code and artifact differ"
        self.radius = self.model_cfg.get("candidate_radius_m", CANDIDATE_RADIUS_M)
        self.max_cands = self.model_cfg.get("max_candidates", MAX_CANDIDATES)

    def predict_trip_dir(self, trip_dir, use_hmm=True):
        """Returns a dict with everything the API needs plus private objects (keys starting with '_')."""
        timings = {}
        t0 = time.time()
        trip = load_trip(trip_dir, with_route=False)           # ground truth is never opened here
        n_raw = len(trip.track)
        orig_idx = clean_track(trip)
        n = len(trip.track)
        timings["load_s"] = time.time() - t0

        result = dict(trip_id=trip.trip_id, n_points_raw=n_raw, n_points_used=n, n_points_dropped=n_raw - n)
        empty = dict(point_arcs_hmm=np.full(n, -1), point_arcs_clf=np.full(n, -1), route=[], breaks=[], info={})
        if n == 0 or len(trip.arcs) == 0 or len(trip.nodes) == 0:
            result.update(empty, warning="no usable GPS points or empty road network", timings=timings, _trip=trip, _orig_idx=orig_idx)
            return result

        t0 = time.time()
        try:
            cands = generate_candidates(trip, radius=self.radius, max_cands=self.max_cands)
        except ValueError:                                     # e.g. no arc anywhere near the track
            cands = None
        timings["candidates_s"] = time.time() - t0
        if cands is None or len(cands) == 0:
            result.update(empty, warning="no candidate roads near the GPS points", timings=timings, _trip=trip, _orig_idx=orig_idx)
            return result

        t0 = time.time()
        feats = build_features(trip, cands)
        timings["features_s"] = time.time() - t0

        t0 = time.time()
        feats["score"] = self.model.predict_proba(feats[FEATURE_COLUMNS].values.astype(np.float32))[:, 1]
        timings["classifier_s"] = time.time() - t0

        t0 = time.time()
        net = RoadNetwork(trip)
        idx = feats.groupby("point_idx")["score"].idxmax()
        clf = np.full(n, -1, dtype=np.int64)
        clf[feats.loc[idx, "point_idx"].values] = feats.loc[idx, "arc_id"].values
        if use_hmm:
            hmm, info = decode_trip(net, trip.track_xy, trip.track["t"].values, feats, self.hmm_cfg)
        else:
            hmm, info = clf, {}
        route, breaks = stitch_route(net, hmm)
        timings["hmm_s"] = time.time() - t0

        result.update(point_arcs_hmm=hmm, point_arcs_clf=clf, route=route, breaks=breaks, info=info,
                      mean_candidates=float(len(feats) / max(feats.point_idx.nunique(), 1)), timings=timings,
                      _trip=trip, _orig_idx=orig_idx, _net=net)
        return result


def route_polylines(trip, route, breaks):
    """Predicted route as lists of [lat, lon] points, split where the route has an unrepairable break."""
    lines, cur = [], []
    cut = set(breaks)
    for k, a in enumerate(route):
        u, v = trip.nodes[trip.arcs[a, 0]], trip.nodes[trip.arcs[a, 1]]
        if k in cut and cur:
            lines.append(cur); cur = []
        if not cur:
            cur.append([float(u[1]), float(u[0])])
        cur.append([float(v[1]), float(v[0])])
    if cur:
        lines.append(cur)
    return lines
