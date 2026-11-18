"""FastAPI backend (Laptop 1).

Run:  uvicorn src.api.app:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_EVAL_DIR       folder containing the trips to serve (default: student_test folder)
      PATHSNAP_ARTIFACTS_DIR  folder with model.joblib, model_config.json, hmm_config.json (default: artifacts/)

Endpoints
  GET  /health            service + model status
  GET  /trips             ids of the trips found in the data folder (no ground truth)
  POST /predict           body {"trip_id": "..."}  -> predicted route + server-side accuracy
  GET  /map/{trip_id}     Folium HTML map (GPS + network + predicted route; never the reference route)
Every request is logged to the console and to logs/api_requests.log.
"""
import logging
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.config import EVAL_DIR, MODEL_ARTIFACTS_DIR, REPO_ROOT
from src.api.predictor import Predictor, route_polylines
from src.api.evaluator import score_prediction
from src.data.loader import trip_prefix
from src.viz.maps import trip_map

LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
log = logging.getLogger("pathsnap")
log.setLevel(logging.INFO)
log.propagate = False
log.handlers.clear()
_fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
for h in (logging.StreamHandler(), logging.FileHandler(LOG_DIR / "api_requests.log")):
    h.setFormatter(_fmt)
    log.addHandler(h)

app = FastAPI(title="PathSnap map-matching API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = Predictor(MODEL_ARTIFACTS_DIR)
_last = {}          # trip_id -> last prediction, only used so /map does not have to recompute


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    client = request.client.host if request.client else "?"
    try:
        response = await call_next(request)
    except Exception:
        log.exception("%s %s from %s -> 500 (unhandled error)", request.method, request.url.path, client)
        raise
    log.info("%s %s from %s -> %s in %.0f ms", request.method, request.url.path, client, response.status_code, 1000 * (time.time() - t0))
    return response


def trip_folders():
    """Trip folders under EVAL_DIR: folders that contain a .track file, one or two levels down
    (an unzipped archive often adds a wrapper folder).  The folder name is the trip id."""
    root = Path(EVAL_DIR)
    found = {}
    if not root.exists():
        return found
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        candidates = [p] if any(p.glob("*.track")) else [c for c in sorted(p.iterdir()) if c.is_dir() and any(c.glob("*.track"))]
        for c in candidates:
            found.setdefault(c.name, c)
    return found


def resolve_trip(trip_id):
    """Only ids that exist as folders are accepted (also blocks path tricks like '../x')."""
    folders = trip_folders()
    if trip_id not in folders:
        raise HTTPException(status_code=404, detail=f"unknown trip id '{trip_id}'")
    return folders[trip_id]


class PredictRequest(BaseModel):
    trip_id: str


def count_points(track_file):
    try:
        with open(track_file, "rb") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return None


@app.get("/health")
def health():
    return {"status": "ok", "data_dir": str(EVAL_DIR), "n_trips": len(trip_folders()),
            "model": predictor.model_cfg.get("model_name"), "hmm": predictor.hmm_cfg.to_dict()}


@app.get("/trips")
def trips():
    out = [{"trip_id": tid, "n_gps_points": count_points(p / f"{trip_prefix(p)}.track")} for tid, p in trip_folders().items()]
    log.info("listing %d trips", len(out))
    return {"trips": out}


def run_prediction(trip_id):
    folder = resolve_trip(trip_id)
    log.info("PREDICT trip=%s : starting pipeline", trip_id)
    t0 = time.time()
    try:
        pred = predictor.predict_trip_dir(folder)
    except Exception as e:                                   # malformed files etc.
        log.exception("trip %s failed while predicting", trip_id)
        raise HTTPException(status_code=422, detail=f"could not process trip {trip_id}: {type(e).__name__}: {e}")
    pred["timings"]["total_predict_s"] = time.time() - t0
    log.info("PREDICT trip=%s : %d points, route of %d arcs, %.2fs", trip_id, pred["n_points_used"], len(pred["route"]), pred["timings"]["total_predict_s"])
    return folder, pred


@app.post("/predict")
def predict(req: PredictRequest):
    folder, pred = run_prediction(req.trip_id)
    trip = pred["_trip"]
    t0 = time.time()
    try:
        evaluation = score_prediction(folder, pred)           # ground truth is used only here, on the server
    except Exception as e:
        log.exception("scoring failed for %s", req.trip_id)
        evaluation = {"available": False, "reason": f"scoring error: {type(e).__name__}: {e}"}
    pred["timings"]["scoring_s"] = time.time() - t0
    _last[req.trip_id] = pred
    if evaluation.get("available"):
        log.info("SCORE trip=%s : hmm_top1=%.4f classifier_top1=%.4f route_f1=%.4f", req.trip_id,
                 evaluation["hmm_top1_accuracy"], evaluation["classifier_only_top1_accuracy"], evaluation["route_f1"])
    # what did the HMM change?  the classifier's raw per-point picks are drawn as separate arcs (no joining)
    clf, hmm = pred["point_arcs_clf"], pred["point_arcs_hmm"]
    clf_arcs = sorted({int(a) for a in clf if a >= 0})[:8000]
    clf_lines = [[[float(trip.nodes[trip.arcs[a, 0]][1]), float(trip.nodes[trip.arcs[a, 0]][0])],
                  [float(trip.nodes[trip.arcs[a, 1]][1]), float(trip.nodes[trip.arcs[a, 1]][0])]] for a in clf_arcs]
    changed = int(((clf != hmm) & (clf >= 0)).sum())
    step = max(1, len(trip.track) // 4000)                    # thin the GPS trace for the browser
    gps = trip.track.iloc[::step][["lat", "lon"]].values.tolist()
    return {
        "trip_id": req.trip_id,
        "n_points_raw": pred["n_points_raw"], "n_points_used": pred["n_points_used"], "n_points_dropped": pred["n_points_dropped"],
        "warning": pred.get("warning"),
        "predicted_route_arcs": [int(a) for a in pred["route"]],
        "predicted_route_polylines": route_polylines(trip, pred["route"], pred["breaks"]),
        "route_breaks": len(pred["breaks"]),
        "predicted_arc_per_point": [int(a) for a in pred["point_arcs_hmm"]],
        "classifier_only_polylines": clf_lines,
        "hmm_changed_points": changed,
        "gps_points": gps,
        "diagnostics": {k: int(v) for k, v in pred["info"].items()},
        "evaluation": evaluation,
        "timings": {k: round(v, 3) for k, v in pred["timings"].items()},
    }


@app.get("/map/{trip_id}", response_class=HTMLResponse)
def map_view(trip_id: str):
    folder = resolve_trip(trip_id)
    pred = _last.get(trip_id) or run_prediction(trip_id)[1]
    m = trip_map(pred["_trip"], ref_arcs=None, pred_arcs=pred["route"], title=f"Trip {trip_id}: GPS vs predicted route")
    return HTMLResponse(m.get_root().render())
