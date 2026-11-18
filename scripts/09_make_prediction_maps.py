"""Phase 2 / step 9: Folium maps for every student_test trip: GPS + network + REFERENCE route + PREDICTED route.

Development only: these maps show the reference route, so they are made offline from student_test data and are
never served by the API (the API map shows GPS + predicted route only).
"""
import sys
sys.path.insert(0, ".")
from src.config import STUDENT_TEST_DIR, MAPS_DIR, MODEL_ARTIFACTS_DIR
from src.api.predictor import Predictor
from src.data.loader import list_trip_dirs, load_trip
from src.viz.maps import trip_map

MAPS_DIR.mkdir(parents=True, exist_ok=True)
pred = Predictor(MODEL_ARTIFACTS_DIR)
for d in list_trip_dirs(STUDENT_TEST_DIR):
    res = pred.predict_trip_dir(d)
    ref_trip = load_trip(d, with_route=True)
    m = trip_map(res["_trip"], ref_arcs=ref_trip.route, pred_arcs=res["route"], title=f"student_test trip {d.name}: reference vs predicted")
    m.save(MAPS_DIR / f"pred_student_{d.name}.html")
    print("map", d.name, "route arcs", len(res["route"]))
