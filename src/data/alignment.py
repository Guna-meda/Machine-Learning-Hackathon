"""Derive per-GPS-point labels from the ordered reference route.

The .route file lists the arcs the vehicle used, in order, but says nothing
about which GPS point belongs to which arc.  We recover that with a monotone
dynamic-programming alignment (a restricted Viterbi):

  * state j  = position in the route list (so a repeated arc is a different state)
  * point i may stay on route position j, or move forward to j+1, j+2 ...
  * going backwards is not allowed (the route is chronological)
  * cost of putting point i on route position j = (robust) distance^2 from the point to that arc
  * jumping over k>1 arcs costs SKIP_PENALTY per skipped arc (short arcs can be passed
    between two 1 Hz samples, so skipping is allowed but discouraged)
  * after a time gap > GAP_S the skip is free (the vehicle drove unobserved)

Limitations: the labels are only as good as the route; points far from the
route (GPS outliers) are still forced onto some arc; the straight arc geometry
has no curve shape; route breaks (non-connected consecutive arcs) are not repaired.
"""
import numpy as np
from src.features.geometry import point_segment

SIGMA_M = 15.0        # scale of the distance cost
CAP_M = 60.0          # distances above this are treated alike (robust to outliers)
SKIP_PENALTY = 1.5    # cost per skipped route arc
MAX_SKIP = 25         # largest forward jump allowed without a time gap
GAP_S = 5.0           # a time step larger than this counts as a gap


def _distance_matrix(track_xy, A, B, chunk=2000):
    """(n_points, n_route_arcs) distances from each GPS point to each route arc."""
    n, m = len(track_xy), len(A)
    out = np.empty((n, m), dtype=np.float32)
    for s in range(0, n, chunk):
        P = track_xy[s:s + chunk]
        k = len(P)
        d, _, _ = point_segment(np.repeat(P, m, axis=0), np.tile(A, (k, 1)), np.tile(B, (k, 1)))
        out[s:s + k] = d.reshape(k, m)
    return out


def align_points_to_route(trip):
    """Return (route_pos, arc_id, dist) arrays, one entry per GPS point."""
    r = trip.route
    A = trip.nodes_xy[trip.arcs[r, 0]]
    B = trip.nodes_xy[trip.arcs[r, 1]]
    D = _distance_matrix(trip.track_xy, A, B)
    cost = (np.minimum(D, CAP_M) / SIGMA_M) ** 2
    n, m = cost.shape
    t = trip.track["t"].values

    back = np.zeros((n, m), dtype=np.int16)    # how many positions we moved forward
    prev = cost[0].copy()                      # free start: any route position
    skip_pen = np.arange(MAX_SKIP + 1, dtype=np.float64)
    skip_pen[1:] = (skip_pen[1:] - 1) * SKIP_PENALTY   # k=0 stay, k=1 next arc: free
    for i in range(1, n):
        if t[i] - t[i - 1] > GAP_S:
            # unobserved stretch: best predecessor is anything at or before j, no penalty
            run_min = np.minimum.accumulate(prev)
            arg = np.zeros(m, dtype=np.int64)
            # position of the running minimum (needed to recover the path)
            best_idx = 0
            for j in range(m):
                if prev[j] <= prev[best_idx]:
                    best_idx = j
                arg[j] = best_idx
            back[i] = np.minimum(np.arange(m) - arg, 32000)
            prev = run_min + cost[i]
            continue
        best = prev.copy()                     # k = 0 (stay)
        best_k = np.zeros(m, dtype=np.int16)
        for k in range(1, MAX_SKIP + 1):
            cand = np.full(m, np.inf)
            cand[k:] = prev[:m - k] + skip_pen[k]
            better = cand < best
            best = np.where(better, cand, best)
            best_k = np.where(better, k, best_k)
        back[i] = best_k
        prev = best + cost[i]

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))             # free end
    for i in range(n - 1, 0, -1):
        pos[i - 1] = pos[i] - back[i, pos[i]]
    arc_id = r[pos]
    dist = D[np.arange(n), pos]
    return pos, arc_id, dist
