"""Phase 2 / step 7: export the final model and decoder settings used by the API (artifacts/).

The final model = HistGradientBoosting with the Optuna-best parameters, refit on ALL reliable train rows
(80 train trips).  student_test and evaluator data are never used for fitting.
"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
import joblib, sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from src.config import RESULTS_DIR, ARTIFACTS_DIR, SEED, CANDIDATE_RADIUS_M, MAX_CANDIDATES
from src.features.build import FEATURE_COLUMNS
from src.models.data import load_table
from src.hmm.viterbi import HMMConfig

if __name__ == "__main__":
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    best = json.load(open(RESULTS_DIR / "optuna_best_params.json"))["best_params"]
    hmm_path = RESULTS_DIR / "hmm_best_config.json"
    hmm_cfg = HMMConfig(**json.load(open(hmm_path))) if hmm_path.exists() else HMMConfig()
    table = load_table()
    train = table[(table.split == "train") & (table.label_ok == 1)]
    t0 = time.time()
    model = HistGradientBoostingClassifier(random_state=SEED, **best).fit(train[FEATURE_COLUMNS].values.astype(np.float32), train["label"].values)
    fit_s = time.time() - t0
    joblib.dump(model, ARTIFACTS_DIR / "model.joblib", compress=3)
    json.dump(dict(model_name="HistGradientBoosting (Optuna-tuned)", params=best, feature_columns=FEATURE_COLUMNS,
                   candidate_radius_m=CANDIDATE_RADIUS_M, max_candidates=MAX_CANDIDATES, seed=SEED,
                   trained_on="80 train trips, label_ok rows only", n_training_rows=int(len(train)), fit_seconds=round(fit_s, 1),
                   sklearn_version=sklearn.__version__), open(ARTIFACTS_DIR / "model_config.json", "w"), indent=2)
    json.dump(hmm_cfg.to_dict(), open(ARTIFACTS_DIR / "hmm_config.json", "w"), indent=2)
    # sanity check of the exported model on the development trips (classifier only, no HMM)
    from src.models.evaluate import evaluate_predictions
    student = table[table.split == "student_test"].copy()
    t0 = time.time()
    student["score"] = model.predict_proba(student[FEATURE_COLUMNS].values.astype(np.float32))[:, 1]
    inf_s = time.time() - t0
    pts = student[["trip_id", "point_idx", "ref_arc", "split"]].drop_duplicates(["trip_id", "point_idx"])
    m = evaluate_predictions(student, pts)
    json.dump(dict(student_top1=m["top1_acc"], student_macro_f1=m["macro_f1_arc"], student_log_loss=m["log_loss"],
                   student_route_continuity=m["route_continuity"], inference_s=inf_s, note="final model refit on all reliable train rows, classifier only"),
              open(RESULTS_DIR / "final_model_student.json", "w"), indent=2)
    print("final model on student_test:", {k: round(v, 4) for k, v in m.items()})
    print("saved artifacts:", [p.name for p in ARTIFACTS_DIR.iterdir()], f"fit {fit_s:.0f}s")
