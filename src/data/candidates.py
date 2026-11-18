"""Candidate road generation with a KD-tree.

Each arc is a straight segment between two nodes.  To find arcs near a GPS point
without comparing against all (up to ~300k) arcs, we:
  1. keep only arcs inside the track's bounding box (+ margin),
  2. sample points every SAMPLE_STEP_M along each remaining arc,
  3. put the samples in a cKDTree and ask for samples within radius + step/2 of each GPS point,
  4. compute the exact point-to-segment distance for the arcs found and keep the nearest ones.
"""
import numpy as np
from scipy.spatial import cKDTree
from src.config import CANDIDATE_RADIUS_M, MAX_CANDIDATES
from src.features.geometry import point_segment

SAMPLE_STEP_M = 20.0


def _arcs_near_track(trip, margin):
    """Arc ids whose segment bounding box overlaps the (padded) track bounding box."""
    lo = trip.track_xy.min(axis=0) - margin
    hi = trip.track_xy.max(axis=0) + margin
    A = trip.nodes_xy[trip.arcs[:, 0]]
    B = trip.nodes_xy[trip.arcs[:, 1]]
    seg_lo = np.minimum(A, B)
    seg_hi = np.maximum(A, B)
    keep = (seg_hi >= lo).all(axis=1) & (seg_lo <= hi).all(axis=1)
    return np.where(keep)[0]


def generate_candidates(trip, radius=CANDIDATE_RADIUS_M, max_cands=MAX_CANDIDATES):
    """Return DataFrame-like dict with arrays: point_idx, arc_id, dist (sorted by point, then distance)."""
    import pandas as pd
    arcs_local = _arcs_near_track(trip, radius + SAMPLE_STEP_M)
    A = trip.nodes_xy[trip.arcs[arcs_local, 0]]
    B = trip.nodes_xy[trip.arcs[arcs_local, 1]]
    length = np.hypot(*(B - A).T)
    n_samp = np.maximum(2, np.ceil(length / SAMPLE_STEP_M).astype(int) + 1)
    owner = np.repeat(np.arange(len(arcs_local)), n_samp)
    start = np.cumsum(n_samp) - n_samp
    frac = (np.arange(n_samp.sum()) - np.repeat(start, n_samp)) / np.repeat(n_samp - 1, n_samp)
    samples = A[owner] + (B[owner] - A[owner]) * frac[:, None]
    tree = cKDTree(samples)

    hits = tree.query_ball_point(trip.track_xy, r=radius + SAMPLE_STEP_M / 2)
    counts = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
    if counts.sum() == 0:
        return pd.DataFrame({"point_idx": [], "arc_id": [], "dist": []})
    pt = np.repeat(np.arange(len(hits)), counts)
    sm = np.fromiter((s for h in hits for s in h), dtype=np.int64, count=counts.sum())
    local = owner[sm]
    # drop duplicate (point, arc) pairs coming from several samples of the same arc
    key = np.unique(pt.astype(np.int64) * len(arcs_local) + local)
    pt, local = key // len(arcs_local), key % len(arcs_local)
    d, _, _ = point_segment(trip.track_xy[pt], A[local], B[local])
    df = pd.DataFrame({"point_idx": pt, "arc_id": arcs_local[local], "dist": d})
    df = df[df.dist <= radius].sort_values(["point_idx", "dist"], kind="stable")
    df = df.groupby("point_idx", sort=True).head(max_cands).reset_index(drop=True)
    return df
