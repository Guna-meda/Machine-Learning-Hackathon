"""Step 11: offline static version of the Folium map layers (GPS, network, reference route, predicted route) for the report.

Used because Folium HTML needs internet for tiles.  Development only (shows the reference route).  Output: reports/figures/map_static_pred_student.png
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from src.config import STUDENT_TEST_DIR, MODEL_ARTIFACTS_DIR, FIGURES_DIR
from src.api.predictor import Predictor
from src.data.loader import load_trip

pred = Predictor(MODEL_ARTIFACTS_DIR)
fig, axes = plt.subplots(1, 2, figsize=(15, 7.4))
for ax, tid in zip(axes, ["00000000", "00000077"]):
    d = STUDENT_TEST_DIR / tid
    res = pred.predict_trip_dir(d); trip = res["_trip"]; ref = load_trip(d).route
    xy = trip.nodes_xy; A, B = trip.arcs[:, 0], trip.arcs[:, 1]
    mid = (xy[A] + xy[B]) / 2
    dist, _ = cKDTree(trip.track_xy[::3]).query(mid)
    # zoom: a 700 m window around one GPS point (otherwise 50 km trips are just a thin line)
    k = {"00000000": len(trip.track) // 2, "00000077": len(trip.track) // 3}[tid]
    cx, cy = trip.track_xy[k]; half = 350.0
    inwin = lambda p, q: (min(abs(p[0] - cx), abs(q[0] - cx)) < half) and (min(abs(p[1] - cy), abs(q[1] - cy)) < half)
    for i in np.where(dist <= 120)[0]:
        if not inwin(xy[A[i]], xy[B[i]]):
            continue
        ax.plot(*zip(xy[A[i]], xy[B[i]]), color="#b0b5bb", lw=0.8, zorder=1)
    for a in ref:
        if not inwin(xy[A[a]], xy[B[a]]):
            continue
        ax.plot(*zip(xy[A[a]], xy[B[a]]), color="#1a9850", lw=9, alpha=0.5, zorder=2)
    for a in res["route"]:
        if not inwin(xy[A[a]], xy[B[a]]):
            continue
        ax.plot(*zip(xy[A[a]], xy[B[a]]), color="#d73027", lw=2.2, ls=(0, (4, 3)), zorder=3)
    w = (np.abs(trip.track_xy[:, 0] - cx) < half) & (np.abs(trip.track_xy[:, 1] - cy) < half)
    ax.scatter(*trip.track_xy[w].T, s=16, c="#2166ac", zorder=4)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal"); ax.set_title(f"student_test trip {tid} - 700 m window ({len(trip.track)} GPS points in trip)"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#2166ac", label="GPS points"), Line2D([], [], color="#b0b5bb", label="road network near track"),
                    Line2D([], [], color="#1a9850", lw=5, alpha=.55, label="reference route"), Line2D([], [], color="#d73027", lw=1.8, ls="--", label="predicted route")],
           loc="lower center", ncol=4)
plt.tight_layout(rect=(0, 0.05, 1, 0.96)); plt.savefig(FIGURES_DIR / "map_static_pred_student.png", dpi=110)
print("ok")
