"""Step 1: dataset inventory and data-quality statistics for train + student_test."""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from src.config import TRAIN_DIR, STUDENT_TEST_DIR, RESULTS_DIR, FIGURES_DIR
from src.data.loader import load_trip, list_trip_dirs

RESULTS_DIR.mkdir(exist_ok=True); FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def trip_stats(split, trip_dir):
    t = load_trip(trip_dir)
    tr, a, r = t.track, t.arcs, t.route
    dt = np.diff(tr["t"].values)
    xy = t.track_xy
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    n_nodes = len(t.nodes)
    # weakly connected components of the road graph
    g = coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(g, directed=False)
    sizes = np.bincount(labels)
    used = np.zeros(n_nodes, bool); used[a.ravel()] = True
    route_len = np.linalg.norm(t.nodes_xy[a[r, 1]] - t.nodes_xy[a[r, 0]], axis=1).sum()
    files = {ext: (trip_dir / f"{trip_dir.name}.{ext}").stat().st_size for ext in ["track", "nodes", "arcs", "route"]}
    return dict(
        split=split, trip_id=trip_dir.name, n_points=len(tr), n_nodes=n_nodes, n_arcs=len(a), n_route_arcs=len(r),
        missing_values=int(tr.isna().sum().sum()),
        invalid_coords=int(((tr.lon.abs() > 180) | (tr.lat.abs() > 90) | ((tr.lon == 0) & (tr.lat == 0))).sum()),
        non_increasing_t=int((dt <= 0).sum()), duplicate_points=int((step == 0).sum()),
        frac_dt_1s=float((dt == 1).mean()), max_dt_s=float(dt.max()), n_gaps_gt_5s=int((dt > 5).sum()),
        duration_s=float(tr.t.iloc[-1] - tr.t.iloc[0]),
        extent_x_km=float(np.ptp(xy[:, 0]) / 1000), extent_y_km=float(np.ptp(xy[:, 1]) / 1000),
        gps_path_km=float(step.sum() / 1000), route_km=float(route_len / 1000),
        max_step_m=float(step.max()), n_jumps_gt_150m=int((step > 150).sum()),
        isolated_nodes=int((~used).sum()), n_components=int(n_comp), largest_component_frac=float(sizes.max() / n_nodes),
        route_connected=bool((a[r[:-1], 1] == a[r[1:], 0]).all()),
        route_has_repeats=bool(len(set(r.tolist())) < len(r)),
        lon0=t.lon0, lat0=t.lat0, size_kb=sum(files.values()) / 1024,
    )


rows = []
for split, d in [("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)]:
    for td in list_trip_dirs(d):
        rows.append(trip_stats(split, td))
df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "eda_trip_stats.csv", index=False)
summ = df.groupby("split").agg(trips=("trip_id", "count"), points=("n_points", "sum"), med_points=("n_points", "median"),
                               med_arcs=("n_arcs", "median"), max_arcs=("n_arcs", "max"), med_route_arcs=("n_route_arcs", "median"),
                               missing=("missing_values", "sum"), invalid=("invalid_coords", "sum"),
                               nonmono=("non_increasing_t", "sum"), gaps5=("n_gaps_gt_5s", "sum"), max_gap=("max_dt_s", "max"),
                               jumps=("n_jumps_gt_150m", "sum"), isolated=("isolated_nodes", "sum"),
                               multi_comp_trips=("n_components", lambda s: int((s > 1).sum())),
                               route_conn=("route_connected", "sum"), route_rep=("route_has_repeats", "sum"))
summ.to_csv(RESULTS_DIR / "eda_summary.csv")
print(summ.T.to_string())
print(df[["n_points", "n_arcs", "gps_path_km", "route_km", "max_dt_s", "n_gaps_gt_5s", "n_jumps_gt_150m", "largest_component_frac"]].describe().round(2).to_string())

# figures
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
ax[0].hist(df.n_points, bins=25, color="#2a6fbb"); ax[0].set_title("GPS points per trip")
ax[1].hist(np.log10(df.n_arcs), bins=25, color="#2a9d6f"); ax[1].set_title("Road arcs per trip (log10)")
ax[2].hist(df.frac_dt_1s, bins=25, color="#d98c1f"); ax[2].set_title("Share of 1-second steps")
ax[3].scatter(df.route_km, df.gps_path_km, s=12, c="#aa3355"); ax[3].plot([0, df.route_km.max()], [0, df.route_km.max()], "k--", lw=1)
ax[3].set_xlabel("route length km"); ax[3].set_ylabel("GPS path km"); ax[3].set_title("GPS path vs route length")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "eda_overview.png", dpi=110)
