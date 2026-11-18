"""Metrics for candidate scoring and route reconstruction.

All metrics work on a table of candidate rows with columns
trip_id, point_idx, arc_id, ref_arc, label and a score column.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from src.config import TRAIN_DIR, STUDENT_TEST_DIR
from src.data.loader import _read_table

_ARC_CACHE = {}


def trip_arcs(trip_id, split):
    key = (trip_id, split)
    if key not in _ARC_CACHE:
        d = (TRAIN_DIR if split == "train" else STUDENT_TEST_DIR) / trip_id
        _ARC_CACHE[key] = _read_table(d / f"{trip_id}.arcs", 2, np.int64)
    return _ARC_CACHE[key]


def top1_per_point(df, score_col="score"):
    """Highest-scoring candidate for every GPS point (one row per point)."""
    idx = df.groupby(["trip_id", "point_idx"], sort=False)[score_col].idxmax()
    return df.loc[idx]


def route_continuity(arc_ids, arcs):
    """Share of arc changes a->b where a really leads into b (head of a == tail of b).

    Also returns the number of changes.  Staying on the same arc is not a change.
    """
    a, b = arc_ids[:-1], arc_ids[1:]
    change = (a != b) & (a >= 0) & (b >= 0)
    if change.sum() == 0:
        return 1.0, 0
    ok = arcs[a[change], 1] == arcs[b[change], 0]
    return float(ok.mean()), int(change.sum())


def evaluate_predictions(cand, all_points, score_col="score"):
    """cand: candidate rows with scores.  all_points: DataFrame(trip_id, point_idx, ref_arc, split)
    listing EVERY evaluated GPS point (points without candidates count as wrong).
    """
    top = top1_per_point(cand, score_col)[["trip_id", "point_idx", "arc_id"]].rename(columns={"arc_id": "pred_arc"})
    pts = all_points.merge(top, on=["trip_id", "point_idx"], how="left")
    pts["pred_arc"] = pts["pred_arc"].fillna(-1).astype(int)
    pts["correct"] = pts["pred_arc"] == pts["ref_arc"]
    res = {"top1_acc": float(pts["correct"].mean()), "n_points": len(pts)}

    f1s, conts, nchg = [], [], []
    for tid, g in pts.groupby("trip_id", sort=False):
        labels = np.union1d(g["ref_arc"].values, g["pred_arc"].values)
        f1s.append(f1_score(g["ref_arc"], g["pred_arc"], labels=labels, average="macro", zero_division=0))
        g = g.sort_values("point_idx")
        c, n = route_continuity(g["pred_arc"].values, trip_arcs(tid, g["split"].iloc[0]))
        conts.append(c); nchg.append(n)
    res["macro_f1_arc"] = float(np.mean(f1s))            # arc classes, averaged over trips
    res["route_continuity"] = float(np.average(conts, weights=np.maximum(nchg, 1)))

    # candidate-level binary view: top-1 candidate = predicted road, everything else = rejected
    sel = cand.copy()
    best = cand.groupby(["trip_id", "point_idx"], sort=False)[score_col].transform("max")
    sel["pred"] = (cand[score_col] == best).astype(int)
    res["macro_f1_binary"] = float(f1_score(sel["label"], sel["pred"], average="macro"))
    p = np.clip(cand[score_col].values, 1e-6, 1 - 1e-6)
    res["log_loss"] = float(log_loss(cand["label"], p, labels=[0, 1]))
    return res
