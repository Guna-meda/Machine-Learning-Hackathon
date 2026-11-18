"""Step 1b: Folium maps for a few training trips (GPS + network + reference route) and EDA plots."""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import TRAIN_DIR, STUDENT_TEST_DIR, MAPS_DIR, FIGURES_DIR
from src.data.loader import load_trip
from src.viz.maps import trip_map

MAPS_DIR.mkdir(parents=True, exist_ok=True); FIGURES_DIR.mkdir(parents=True, exist_ok=True)
stats = pd.read_csv("results/eda_trip_stats.csv", dtype={"trip_id": str})
tr = stats[stats.split == "train"].sort_values("n_points")
picks = [tr.iloc[len(tr) // 2].trip_id, tr.iloc[5].trip_id]          # a median trip and a small one
for tid in picks:
    trip = load_trip(TRAIN_DIR / tid)
    trip_map(trip, ref_arcs=trip.route, title=f"EDA: train trip {tid}").save(MAPS_DIR / f"eda_train_{tid}.html")
    print("saved map", tid)

# trajectory plots (projected metres) for 6 train trips, GPS vs reference route
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, tid in zip(axes.ravel(), tr.iloc[np.linspace(0, len(tr) - 1, 6).astype(int)].trip_id):
    t = load_trip(TRAIN_DIR / tid)
    for aid in t.route:
        u, v = t.nodes_xy[t.arcs[aid]]
        ax.plot([u[0], v[0]], [u[1], v[1]], color="#1a9850", lw=2.5, zorder=1)
    ax.scatter(*t.track_xy.T, s=2, c="#2166ac", zorder=2)
    ax.set_title(f"train trip {tid}  ({len(t.track)} pts)"); ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
plt.suptitle("GPS points (blue) vs reference route (green), projected metres")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "eda_trajectories.png", dpi=100)

# time-gap and speed distribution over all trips
dts, spd = [], []
for sp, d in [("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)]:
    for tid in stats[stats.split == sp].trip_id:
        t = load_trip(d / tid)
        dt = np.diff(t.track.t.values); st = np.linalg.norm(np.diff(t.track_xy, axis=0), axis=1)
        dts.append(dt); spd.append(st[dt > 0] / dt[dt > 0])
dts, spd = np.concatenate(dts), np.concatenate(spd)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(np.clip(dts, 0, 10), bins=np.arange(0.5, 11.5, 1), color="#d98c1f"); ax[0].set_yscale("log")
ax[0].set_title("Time between GPS points (s, clipped at 10)")
ax[1].hist(np.clip(spd, 0, 60), bins=60, color="#2a9d6f"); ax[1].set_title("Speed between points (m/s, clipped at 60)")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "eda_gaps_speed.png", dpi=100)
print("share dt==1: %.3f  share dt>5: %.4f  share speed>50 m/s: %.4f" % ((dts == 1).mean(), (dts > 5).mean(), (spd > 50).mean()))
