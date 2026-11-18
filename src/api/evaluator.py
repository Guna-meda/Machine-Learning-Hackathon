"""Server-side scoring of a prediction against the reference route (ground truth).

The reference never leaves the server: this module returns numbers only, no reference geometry.
Per-point reference labels are derived with the same monotone alignment used for training.
"""
import numpy as np
from sklearn.metrics import f1_score
from src.config import LABEL_OK_DIST_M
from src.data.loader import load_trip, trip_prefix
from src.data.alignment import align_points_to_route
from src.api.predictor import clean_track
from src.hmm.evaluate_hmm import route_set_scores
from src.models.evaluate import route_continuity


def score_prediction(trip_dir, pred):
    """pred: result dict from Predictor.predict_trip_dir.  Returns a metrics dict (or {'available': False})."""
    from pathlib import Path
    route_file = Path(trip_dir) / f"{trip_prefix(trip_dir)}.route"
    if not route_file.exists() or route_file.stat().st_size == 0:
        return {"available": False, "reason": "no reference route on the server for this trip"}
    trip = load_trip(trip_dir, with_route=True)
    clean_track(trip)                                   # same cleaning as the predictor -> same row order
    if len(trip.track) == 0 or trip.route is None or len(trip.route) == 0:
        return {"available": False, "reason": "reference or track empty"}
    _, ref, ref_dist = align_points_to_route(trip)
    ok = ref_dist <= LABEL_OK_DIST_M
    out = {"available": True, "n_points_scored": int(len(ref)), "n_reference_arcs": int(len(trip.route))}
    for name, key in (("hmm", "point_arcs_hmm"), ("classifier_only", "point_arcs_clf")):
        p = pred[key]
        out[f"{name}_top1_accuracy"] = float((p == ref).mean())
        out[f"{name}_top1_accuracy_reliable_labels"] = float((p[ok] == ref[ok]).mean()) if ok.any() else None
        labels = np.union1d(ref, p)
        out[f"{name}_macro_f1"] = float(f1_score(ref, p, labels=labels, average="macro", zero_division=0))
        out[f"{name}_route_continuity"], _ = route_continuity(p, trip.arcs)
    prec, rec, f1 = route_set_scores(pred["route"], trip.route)
    out.update(route_precision=prec, route_recall=rec, route_f1=f1)
    out["reliable_label_share"] = float(ok.mean())
    return out
