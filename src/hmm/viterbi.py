"""HMM / Viterbi route reconstruction on top of the classifier scores.

Hidden states  : the candidate road arcs of each GPS point (the state set changes per point).
Emission       : log of the classifier probability that the candidate is the true road (times a weight).
Transition     : how plausible is it to go from candidate a (previous point) to candidate b (this point)?
                 route_dist = driving distance along the network from a's position to b's position,
                 gps_dist   = straight distance between the two GPS points.
                 log P = -|route_dist - gps_dist| / beta        (a vehicle drives about as far as the GPS moved)
                 - u-turn (b is the reverse of a): extra penalty
                 - not reachable in the network within `reach_bound`: fixed `disconnect_penalty` (finite, so the
                   decoder can still continue after a map/route break instead of crashing)
Gaps           : if the time gap is > gap_s, or the points are further apart than reach_bound, we do not trust
                 the network any more: the chain restarts (transition scores are all 0, only emissions decide).
No candidates  : the point is skipped by the chain and gets the previous decoded arc afterwards (forward fill),
                 or the next one if it is at the start.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


@dataclass
class HMMConfig:
    beta: float = 30.0                 # metres; larger = more tolerance for route/GPS distance mismatch
    emission_weight: float = 1.0       # scales the classifier log-probability
    disconnect_penalty: float = 12.0   # log-penalty when the candidate is not reachable in the network
    uturn_penalty: float = 4.0         # log-penalty for flipping onto the reverse arc
    reach_bound: float = 400.0         # metres searched along the network from the previous arc
    gap_s: float = 10.0                # time gap that restarts the chain
    max_states: int = 8                # keep the best-scored candidates per point
    min_prob: float = 1e-4             # candidates below this probability are dropped (unless it empties the set)

    def to_dict(self):
        return asdict(self)


def decode_trip(net, track_xy, t, cand, cfg: HMMConfig):
    """Decode one trip.

    cand: DataFrame with columns point_idx, arc_id, score, frac_along (one row per candidate).
    Returns (arc_per_point int array, info dict).  arc_per_point is -1 where nothing could be decided.
    """
    n = len(track_xy)
    out = np.full(n, -1, dtype=np.int64)
    info = dict(n_points=n, n_no_candidates=0, n_restarts=0, n_disconnected=0, n_transitions=0)
    if len(cand) == 0:
        info["n_no_candidates"] = n
        return out, info

    # keep the top candidates per point
    c = cand[cand.score >= cfg.min_prob].copy()
    missing = np.setdiff1d(cand.point_idx.unique(), c.point_idx.unique())
    if len(missing):
        c = pd.concat([c, cand[cand.point_idx.isin(missing)]])
    c = c.sort_values(["point_idx", "score"], ascending=[True, False])
    c = c.groupby("point_idx", sort=True).head(cfg.max_states)
    pts = c.point_idx.values
    bounds = np.flatnonzero(np.r_[True, pts[1:] != pts[:-1], True])
    point_ids = pts[bounds[:-1]]
    info["n_no_candidates"] = int(n - len(point_ids))
    arcs_all = c.arc_id.values
    emis_all = cfg.emission_weight * np.log(np.clip(c.score.values, 1e-6, 1.0))
    frac_all = c.frac_along.values

    back, scores, slices = [], [], []
    prev_log = None
    for k, pid in enumerate(point_ids):
        sl = slice(bounds[k], bounds[k + 1])
        a, e, f = arcs_all[sl], emis_all[sl], frac_all[sl]
        slices.append(sl)
        if prev_log is None:
            scores.append(e); back.append(np.zeros(len(a), dtype=np.int64)); prev_log = e; prev_k = k; prev_pid = pid
            continue
        psl = slices[prev_k]
        pa, pf = arcs_all[psl], frac_all[psl]
        gps_dist = float(np.hypot(*(track_xy[pid] - track_xy[prev_pid])))
        dt = float(t[pid] - t[prev_pid])
        if dt > cfg.gap_s or gps_dist > cfg.reach_bound:
            trans = np.zeros((len(pa), len(a)))                 # restart: network gives no information
            info["n_restarts"] += 1
        else:
            trans = np.empty((len(pa), len(a)))
            for i, ai in enumerate(pa):
                head_i = net.head[ai]
                dist_map, _ = net.reach(head_i, cfg.reach_bound)
                rem = (1.0 - pf[i]) * net.length[ai]            # metres left on the previous arc
                for j, bj in enumerate(a):
                    if ai == bj:
                        rd = abs(f[j] - pf[i]) * net.length[ai]
                        pen = 0.0
                    else:
                        d_nodes = dist_map.get(net.tail[bj])
                        if d_nodes is None:
                            trans[i, j] = -cfg.disconnect_penalty
                            info["n_disconnected"] += 1
                            continue
                        rd = rem + d_nodes + f[j] * net.length[bj]
                        pen = cfg.uturn_penalty if (net.tail[ai] == net.head[bj] and net.head[ai] == net.tail[bj]) else 0.0
                    trans[i, j] = -abs(rd - gps_dist) / cfg.beta - pen
            info["n_transitions"] += trans.size
        total = prev_log[:, None] + trans                         # (prev states, current states)
        bp = np.argmax(total, axis=0)
        cur = total[bp, np.arange(len(a))] + e
        scores.append(cur); back.append(bp)
        prev_log = cur; prev_k = k; prev_pid = pid

    # backtrack the best path
    choice = np.empty(len(point_ids), dtype=np.int64)
    choice[-1] = int(np.argmax(scores[-1]))
    for k in range(len(point_ids) - 1, 0, -1):
        choice[k - 1] = back[k][choice[k]]
    decoded = np.array([arcs_all[slices[k]][choice[k]] for k in range(len(point_ids))])
    out[point_ids] = decoded
    # forward-fill points that had no candidates (backward-fill at the very start)
    last = -1
    for i in range(n):
        if out[i] >= 0:
            last = out[i]
        elif last >= 0:
            out[i] = last
    first = out[out >= 0]
    if len(first):
        out[:np.argmax(out >= 0)] = first[0]
    return out, info
