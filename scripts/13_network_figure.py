"""Step 13: road-network visualisation for the report: whole network of two trips with GPS points and reference route.

Left: one train trip.  Right: one student_test trip.  Output: reports/figures/eda_road_network.png
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from src.config import TRAIN_DIR, STUDENT_TEST_DIR, FIGURES_DIR
from src.data.loader import load_trip

stats = pd.read_csv("results/eda_trip_stats.csv", dtype={"trip_id": str})
pick = []
for split, d in (("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)):
    s = stats[(stats.split == split) & (stats.n_arcs.between(3000, 25000))].sort_values("n_arcs")
    pick.append((split, d, s.iloc[len(s) // 2].trip_id))

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, (split, d, tid) in zip(axes, pick):
    t = load_trip(d / tid)
    seg = np.stack([t.nodes_xy[t.arcs[:, 0]], t.nodes_xy[t.arcs[:, 1]]], axis=1)
    ax.add_collection(LineCollection(seg, colors="#b0b5bb", linewidths=0.4, zorder=1))
    rseg = seg[t.route]
    ax.add_collection(LineCollection(rseg, colors="#1a9850", linewidths=2.2, zorder=2))
    ax.scatter(*t.track_xy.T, s=2, c="#2166ac", zorder=3)
    lo, hi = t.track_xy.min(axis=0) - 1500, t.track_xy.max(axis=0) + 1500
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_title(f"{split} trip {tid}: {len(t.arcs):,} arcs, {len(t.nodes):,} nodes, {len(t.track):,} GPS points")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color="#b0b5bb", label="road network (all arcs)"), Line2D([], [], color="#1a9850", lw=2.2, label="reference route"),
                    Line2D([], [], marker="o", ls="", color="#2166ac", label="GPS points")], loc="lower center", ncol=3)
plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.savefig(FIGURES_DIR / "eda_road_network.png", dpi=110)
print("trips:", pick)
