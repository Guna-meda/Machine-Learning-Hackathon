"""Turn (trip, candidates) into a feature table with one row per (GPS point, candidate arc).

Features are computed from the GPS track and the road network only, never from
the route, so the same code is used at training time and inside the API.
"""
import numpy as np
import pandas as pd
from src.features.geometry import point_segment, signed_lateral

CTX_OFFSETS = (-3, -2, -1, 1, 2, 3)      # neighbouring GPS points used as context
MOVING_SPEED = 1.0                       # m/s below which heading is considered unreliable

FEATURE_COLUMNS = [
    "dist", "dist_rank", "dist_minus_min", "n_cands", "frac_along", "lateral", "arc_len",
    "speed", "turn_rate", "moving", "heading_cos", "heading_cos_ctx",
    "ctx_dist_mean", "ctx_dist_max", "ctx_dist_min", "progress_along", "progress_ratio",
    "from_out_deg", "to_out_deg", "from_in_deg", "has_reverse", "dt_prev", "dt_next",
]


def _headings(xy, t):
    """Per-point travel direction (unit vector) and speed from a centred window of +-2 points."""
    n = len(xy)
    lo = np.clip(np.arange(n) - 2, 0, n - 1)
    hi = np.clip(np.arange(n) + 2, 0, n - 1)
    d = xy[hi] - xy[lo]
    dt = np.maximum(t[hi] - t[lo], 1e-3)
    dist = np.hypot(d[:, 0], d[:, 1])
    speed = dist / dt
    unit = d / np.maximum(dist, 1e-6)[:, None]
    return unit, speed


def build_features(trip, cands):
    """cands has columns point_idx, arc_id, dist.  Returns a DataFrame with features + ids."""
    xy = trip.track_xy
    t = trip.track["t"].values
    n = len(xy)
    unit, speed = _headings(xy, t)
    # turn rate: change of direction between the points before and after (radians per step)
    ang = np.arctan2(unit[:, 1], unit[:, 0])
    turn = np.abs(np.angle(np.exp(1j * (np.roll(ang, -2) - np.roll(ang, 2)))))
    turn[:2] = 0; turn[-2:] = 0

    pi = cands["point_idx"].values
    aid = cands["arc_id"].values
    fa, ta = trip.arcs[aid, 0], trip.arcs[aid, 1]
    A, B = trip.nodes_xy[fa], trip.nodes_xy[ta]

    dist, s, arc_len = point_segment(xy[pi], A, B)
    lateral = signed_lateral(xy[pi], A, B)
    arc_dir = (B - A) / np.maximum(arc_len, 1e-6)[:, None]
    heading_cos = (unit[pi] * arc_dir).sum(axis=1)
    moving = (speed[pi] > MOVING_SPEED).astype(float)
    heading_cos = heading_cos * moving                      # unreliable when standing still

    df = pd.DataFrame({"point_idx": pi, "arc_id": aid, "dist": dist})
    g = df.groupby("point_idx")["dist"]
    df["dist_rank"] = g.rank(method="first").values
    df["dist_minus_min"] = dist - g.transform("min").values
    df["n_cands"] = g.transform("size").values
    df["frac_along"] = np.clip(s / np.maximum(arc_len, 1e-6), 0, 1)
    df["lateral"] = lateral
    df["arc_len"] = arc_len
    df["speed"] = speed[pi]
    df["turn_rate"] = turn[pi]
    df["moving"] = moving
    df["heading_cos"] = heading_cos

    # context: how do the neighbouring GPS points relate to the same candidate arc?
    ctx_d, ctx_cos, ctx_s = [], [], {}
    for o in CTX_OFFSETS:
        nb = np.clip(pi + o, 0, n - 1)
        d_o, s_o, _ = point_segment(xy[nb], A, B)
        ctx_d.append(d_o)
        ctx_cos.append(heading_cos_for(unit[nb], arc_dir, speed[nb]))
        ctx_s[o] = s_o
    ctx_d = np.vstack(ctx_d)
    df["ctx_dist_mean"] = ctx_d.mean(axis=0)
    df["ctx_dist_max"] = ctx_d.max(axis=0)
    df["ctx_dist_min"] = ctx_d.min(axis=0)
    df["heading_cos_ctx"] = np.vstack(ctx_cos).mean(axis=0)
    # how far did the vehicle move along this arc's direction between point i-3 and i+3?
    progress = ctx_s[3] - ctx_s[-3]
    df["progress_along"] = progress
    path_len = np.hypot(*(xy[np.clip(pi + 3, 0, n - 1)] - xy[np.clip(pi - 3, 0, n - 1)]).T)
    df["progress_ratio"] = progress / np.maximum(path_len, 1.0)

    # network topology around the arc
    n_nodes = len(trip.nodes)
    out_deg = np.bincount(trip.arcs[:, 0], minlength=n_nodes)
    in_deg = np.bincount(trip.arcs[:, 1], minlength=n_nodes)
    df["from_out_deg"] = out_deg[fa]
    df["to_out_deg"] = out_deg[ta]
    df["from_in_deg"] = in_deg[fa]
    pair_key = set((trip.arcs[:, 1].astype(np.int64) * n_nodes + trip.arcs[:, 0]).tolist())
    df["has_reverse"] = np.fromiter(((int(b) * n_nodes + int(a)) in pair_key for a, b in zip(fa, ta)), dtype=float, count=len(fa))
    # sampling-irregularity context
    dt_prev = np.r_[0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0]
    df["dt_prev"] = dt_prev[pi]
    df["dt_next"] = dt_next[pi]
    return df


def heading_cos_for(unit_nb, arc_dir, speed_nb):
    return (unit_nb * arc_dir).sum(axis=1) * (speed_nb > MOVING_SPEED)
