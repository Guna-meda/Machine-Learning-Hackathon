"""Compare classifier-only picks with classifier + HMM/Viterbi picks on scored candidate tables."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.config import TRAIN_DIR, STUDENT_TEST_DIR
from src.data.loader import load_trip
from src.hmm.network import RoadNetwork, stitch_route
from src.hmm.viterbi import decode_trip
from src.models.evaluate import route_continuity

_TRIPS = {}


def get_trip(trip_id, split):
    """Load (and cache) trip + road network + reference arcs for evaluation."""
    key = (trip_id, split)
    if key not in _TRIPS:
        trip = load_trip((TRAIN_DIR if split == "train" else STUDENT_TEST_DIR) / trip_id)
        _TRIPS[key] = (trip, RoadNetwork(trip))
    return _TRIPS[key]


def classifier_top1(cand):
    """Highest-scoring candidate per point -> Series indexed by point_idx."""
    idx = cand.groupby("point_idx")["score"].idxmax()
    return cand.loc[idx].set_index("point_idx")["arc_id"]


def route_set_scores(pred_route, ref_route):
    """Sequence-level score: overlap of the set of arcs in the stitched route and the reference route."""
    p, r = set(map(int, pred_route)), set(map(int, ref_route))
    inter = len(p & r)
    prec = inter / max(len(p), 1); rec = inter / max(len(r), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1


def evaluate_trip(trip_id, split, cand_t, point_ref, cfg, want_decoded=False):
    """cand_t: scored candidates of ONE trip (cols point_idx, arc_id, score, frac_along).
    point_ref: DataFrame(point_idx, ref_arc, label_ok) for every GPS point of the trip.
    """
    trip, net = get_trip(trip_id, split)
    n = len(trip.track)
    clf = classifier_top1(cand_t).reindex(range(n)).fillna(-1).astype(int).values
    hmm, info = decode_trip(net, trip.track_xy, trip.track["t"].values, cand_t, cfg)
    ref = point_ref.set_index("point_idx")["ref_arc"].reindex(range(n)).values
    ok = point_ref.set_index("point_idx")["label_ok"].reindex(range(n)).values.astype(bool)

    def f1(pred, mask):
        labels = np.union1d(ref[mask], pred[mask])
        return f1_score(ref[mask], pred[mask], labels=labels, average="macro", zero_division=0)

    res = dict(trip_id=trip_id, n_points=int(ok.sum()))
    for name, pred in (("clf", clf), ("hmm", hmm)):
        res[f"{name}_top1"] = float((pred[ok] == ref[ok]).mean())
        res[f"{name}_macro_f1"] = float(f1(pred, ok))
        res[f"{name}_continuity"], res[f"{name}_changes"] = route_continuity(pred, net.arcs)
        route, breaks = stitch_route(net, pred)
        res[f"{name}_route_breaks"] = len(breaks)
        res[f"{name}_route_len_arcs"] = len(route)
        if trip.route is not None:
            res[f"{name}_route_prec"], res[f"{name}_route_rec"], res[f"{name}_route_f1"] = route_set_scores(route, trip.route)
    res.update({f"hmm_{k}": v for k, v in info.items() if k != "n_points"})
    if want_decoded:
        return res, clf, hmm
    return res


def evaluate_split(scored, split, cfg, trip_ids=None):
    """scored: candidate table with columns trip_id, point_idx, arc_id, score, frac_along, ref_arc, label_ok."""
    rows = []
    for tid, g in scored.groupby("trip_id", sort=True):
        if trip_ids is not None and tid not in trip_ids:
            continue
        pr = g[["point_idx", "ref_arc", "label_ok"]].drop_duplicates("point_idx")
        trip, _ = get_trip(tid, split)
        # points without any candidate still count (as wrong); give them a reference label too
        full = pd.DataFrame({"point_idx": range(len(trip.track))}).merge(pr, on="point_idx", how="left")
        full["ref_arc"] = full["ref_arc"].fillna(-2).astype(int); full["label_ok"] = full["label_ok"].fillna(0)
        rows.append(evaluate_trip(tid, split, g, full, cfg))
    return pd.DataFrame(rows)


def summarise(df):
    """Point-weighted means of the per-trip metrics (so big trips count more, like observation-level metrics)."""
    w = df["n_points"].values
    out = {}
    for c in df.columns:
        if c in ("trip_id", "n_points") or df[c].dtype == object:
            continue
        out[c] = float(np.average(df[c].fillna(0), weights=w))
    out["n_trips"] = len(df); out["n_points"] = int(w.sum())
    return out
