"""Phase 2 / step 10: robustness tests.

Part 1: every trip served by a running API (default http://localhost:8000) is predicted; status, time and metrics are logged.
Part 2: edge cases are generated from one student trip in a temporary folder and sent through the same API code
        (FastAPI TestClient).  Expected: a sensible answer or a clean 4xx error, never a crash (500).

Outputs: results/api_robustness_all_trips.csv, results/api_edge_cases.csv
Usage:   python3 scripts/10_robustness_test.py [api_url]
"""
import os, sys, shutil, tempfile, time
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from src.config import RESULTS_DIR, STUDENT_TEST_DIR

API = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def part1():
    trips = requests.get(f"{API}/trips", timeout=30).json()["trips"]
    rows = []
    for t in trips:
        t0 = time.time()
        r = requests.post(f"{API}/predict", json={"trip_id": t["trip_id"]}, timeout=600)
        row = dict(trip_id=t["trip_id"], n_gps_points=t["n_gps_points"], http_status=r.status_code, wall_s=round(time.time() - t0, 2))
        if r.ok:
            d = r.json(); ev = d["evaluation"]
            row.update(route_arcs=len(d["predicted_route_arcs"]), route_breaks=d["route_breaks"], predict_s=d["timings"]["total_predict_s"])
            if ev.get("available"):
                row.update({k: round(ev[k], 4) for k in ("hmm_top1_accuracy", "classifier_only_top1_accuracy", "hmm_macro_f1", "hmm_route_continuity", "route_f1")})
        rows.append(row); print(row, flush=True)
    df = pd.DataFrame(rows); df.to_csv(RESULTS_DIR / "api_robustness_all_trips.csv", index=False)
    ok = (df.http_status == 200).all()
    print("PART 1:", "all trips returned 200" if ok else "SOME TRIPS FAILED", "| mean HMM top-1 %.4f" % df.hmm_top1_accuracy.mean())
    return ok


def make_edge_cases(root: Path, source: Path):
    """Create modified copies of one trip.  Returns {case_name: expectation}."""
    tid = source.name
    base_track = np.loadtxt(source / f"{tid}.track")
    cases = {}

    def new_case(name, track=None, drop_route=False, arcs=None, raw_track=None, expect="200"):
        d = root / name; d.mkdir(parents=True)
        for ext in ("nodes", "arcs", "route"):
            if ext == "route" and drop_route:
                continue
            shutil.copy(source / f"{tid}.{ext}", d / f"{name}.{ext}")
        if arcs is not None:
            (d / f"{name}.arcs").write_text(arcs)
        if raw_track is not None:
            (d / f"{name}.track").write_text(raw_track)
        else:
            np.savetxt(d / f"{name}.track", track if track is not None else base_track, fmt="%.6f")
        cases[name] = expect

    new_case("normal_copy")
    new_case("one_gps_point", base_track[:1])
    new_case("two_gps_points", base_track[:2])
    new_case("empty_track_file", raw_track="")
    bad = base_track.copy(); bad[5::20, 0] = np.nan; bad[7::25, 1] = 999.0; bad[11::40, :2] = 0.0
    new_case("nan_and_invalid_rows", bad)
    rng = np.random.default_rng(0); shuffled = base_track[rng.permutation(len(base_track))]
    new_case("shuffled_time_order", shuffled)
    dup = base_track.copy(); dup[:, 2] = np.floor(dup[:, 2] / 3)           # many identical timestamps
    new_case("duplicate_timestamps", dup)
    far = base_track.copy(); far[:, 0] += 1.5
    new_case("track_far_from_network", far)
    gap = base_track.copy(); gap[len(gap) // 2:, 2] += 3000
    new_case("large_time_gap_midway", gap)
    new_case("missing_route_file", drop_route=True)
    new_case("garbage_track_file", raw_track="this is not a gps file\nabc def ghi\n", expect="422")
    new_case("empty_arcs_file", arcs="", expect="200")
    return cases


def part2():
    tmp = Path(tempfile.mkdtemp(prefix="pathsnap_edge_"))
    source = sorted(STUDENT_TEST_DIR.iterdir())[4]                         # a small trip
    cases = make_edge_cases(tmp, source)
    os.environ["PATHSNAP_EVAL_DIR"] = str(tmp)                              # point the API at the edge-case folder
    from fastapi.testclient import TestClient
    import importlib, src.config, src.api.app as app_module
    importlib.reload(src.config); app_module = importlib.reload(app_module)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    rows = []
    for name, expect in cases.items():
        r = client.post("/predict", json={"trip_id": name})
        row = dict(case=name, expected_status=expect, http_status=r.status_code)
        if r.is_success:
            d = r.json(); ev = d["evaluation"]
            row.update(points_used=d["n_points_used"], points_dropped=d["n_points_dropped"], route_arcs=len(d["predicted_route_arcs"]),
                       warning=d.get("warning"), hmm_top1=round(ev["hmm_top1_accuracy"], 3) if ev.get("available") else "n/a (" + ev.get("reason", "") + ")")
        else:
            row["detail"] = str(r.json().get("detail"))[:90]
        row["passed"] = str(r.status_code) == expect
        rows.append(row); print(row, flush=True)
    r = client.post("/predict", json={"trip_id": "does_not_exist"}); rows.append(dict(case="unknown_trip_id", expected_status="404", http_status=r.status_code, passed=r.status_code == 404))
    r = client.post("/predict", json={"trip_id": "../../etc"}); rows.append(dict(case="path_traversal_id", expected_status="404", http_status=r.status_code, passed=r.status_code == 404))
    r = client.post("/predict", json={}); rows.append(dict(case="missing_trip_id_field", expected_status="422", http_status=r.status_code, passed=r.status_code == 422))
    df = pd.DataFrame(rows); df.to_csv(RESULTS_DIR / "api_edge_cases.csv", index=False)
    shutil.rmtree(tmp, ignore_errors=True)
    print("PART 2:", f"{int(df.passed.sum())}/{len(df)} edge cases behaved as expected")
    return bool(df.passed.all())


if __name__ == "__main__":
    ok1 = part1()
    ok2 = part2()
    sys.exit(0 if ok1 and ok2 else 1)
