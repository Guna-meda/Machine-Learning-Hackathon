"""Phase 2 / step 6: Bayesian hyperparameter search (Optuna TPE) for HistGradientBoosting.

Objective: mean CV macro F1 (arc level) over 5 GroupKFold folds (grouped by trip) on the 80 train trips.
Outputs: results/optuna_trials.csv, results/optuna_best_params.json, results/optuna_study.db,
         results/tuned_vs_untuned.csv, reports/figures/optuna_*.png
Usage: python3 scripts/06_tune_optuna.py [n_trials]
"""
import sys, time, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from src.config import RESULTS_DIR, FIGURES_DIR, SEED
from src.features.build import FEATURE_COLUMNS
from src.models.data import load_table, sample_points
from src.models.evaluate import evaluate_predictions

FIT_FRAC = 0.25       # same fit sample as the Phase 1 comparison, so untuned vs tuned is fair
N_FOLDS = 5
UNTUNED = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31)       # = Phase 1 configuration


def to_xy(df):
    return df[FEATURE_COLUMNS].values.astype(np.float32), df["label"].values


def points_of(df):
    return df[["trip_id", "point_idx", "ref_arc", "split"]].drop_duplicates(["trip_id", "point_idx"])


def make_hgb(params):
    return HistGradientBoostingClassifier(random_state=SEED, **params)


def cv_eval(params, folds):
    """Fit on each fold's training part, score the held-out trips, average the metrics."""
    ms, fit_t = [], 0.0
    for tr, va in folds:
        t0 = time.time()
        m = make_hgb(params).fit(*tr)
        fit_t += time.time() - t0
        v = va["frame"].copy()
        v["score"] = m.predict_proba(va["X"])[:, 1]
        ms.append(evaluate_predictions(v, va["points"]))
    d = pd.DataFrame(ms)
    return d.mean().to_dict(), d["macro_f1_arc"].std(), fit_t / len(folds)


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    timeout_s = float(sys.argv[2]) if len(sys.argv) > 2 else None      # stop starting new trials after this many seconds
    table = load_table()
    train = table[(table.split == "train") & (table.label_ok == 1)]
    student = table[table.split == "student_test"].copy()
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = []
    for tr_idx, va_idx in gkf.split(train, groups=train["trip_id"].values):
        trd = sample_points(train.iloc[tr_idx], FIT_FRAC)
        vad = train.iloc[va_idx]
        folds.append((to_xy(trd), dict(frame=vad, X=vad[FEATURE_COLUMNS].values.astype(np.float32), points=points_of(vad))))
    print("folds ready", flush=True)

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 400, step=50),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 95),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 400, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            max_features=trial.suggest_float("max_features", 0.5, 1.0),
        )
        t0 = time.time()
        m, std, fit_s = cv_eval(params, folds)
        trial.set_user_attr("cv_top1", m["top1_acc"]); trial.set_user_attr("cv_log_loss", m["log_loss"])
        trial.set_user_attr("cv_route_continuity", m["route_continuity"]); trial.set_user_attr("cv_f1_std", std)
        trial.set_user_attr("mean_fit_s", fit_s); trial.set_user_attr("trial_s", time.time() - t0)
        print(f"trial {trial.number}: f1={m['macro_f1_arc']:.4f} top1={m['top1_acc']:.4f} ll={m['log_loss']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        return m["macro_f1_arc"]

    db = RESULTS_DIR / "optuna_study.db"
    if db.exists():
        db.unlink()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                storage=f"sqlite:///{db}", study_name="hgb_cv_macro_f1")
    study.enqueue_trial({"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                         "l2_regularization": 0.001, "max_features": 1.0})   # start from (near) the untuned default
    t_start = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)
    tune_s = time.time() - t_start
    study.trials_dataframe().to_csv(RESULTS_DIR / "optuna_trials.csv", index=False)
    best = study.best_params
    json.dump(dict(best_params=best, best_cv_macro_f1=study.best_value, n_trials=len(study.trials), tuning_seconds=tune_s, seed=SEED,
                   search_space=dict(learning_rate="log-uniform 0.02-0.3", max_iter="int 100-400 step 50", max_leaf_nodes="int 15-95",
                                     min_samples_leaf="log-int 20-400", l2_regularization="log-uniform 1e-3-10", max_features="uniform 0.5-1.0"),
                   objective="mean 5-fold GroupKFold CV macro F1 (arc level)"), open(RESULTS_DIR / "optuna_best_params.json", "w"), indent=2)
    print("best", study.best_value, best, f"tuning took {tune_s:.0f}s", flush=True)

    # ---- untuned vs tuned: identical protocol (CV folds + student_test, same fit sample)
    fit_all = sample_points(train, FIT_FRAC)
    rows = []
    for name, params in [("untuned", UNTUNED), ("tuned", best)]:
        m, std, _ = cv_eval(params, folds)
        t0 = time.time(); model = make_hgb(params).fit(*to_xy(fit_all)); fit_s = time.time() - t0
        st = student.copy()
        t0 = time.time(); st["score"] = model.predict_proba(st[FEATURE_COLUMNS].values.astype(np.float32))[:, 1]; inf_s = time.time() - t0
        sm = evaluate_predictions(st, points_of(st))
        rows.append(dict(model=name, params=json.dumps(params), cv_macro_f1=m["macro_f1_arc"], cv_macro_f1_std=std, cv_top1=m["top1_acc"],
                         cv_log_loss=m["log_loss"], student_top1=sm["top1_acc"], student_macro_f1=sm["macro_f1_arc"],
                         student_log_loss=sm["log_loss"], student_route_continuity=sm["route_continuity"],
                         fit_time_s=fit_s, inference_s_student=inf_s, inference_us_per_candidate=1e6 * inf_s / len(st)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tuned_vs_untuned.csv", index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = study.trials_dataframe()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(tr.number, tr.value, "o", alpha=.6, label="trial"); ax[0].plot(tr.number, tr.value.cummax(), "r-", label="best so far")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV macro F1"); ax[0].legend(); ax[0].set_title("Optuna (TPE) optimisation history")
    imp = optuna.importance.get_param_importances(study)
    ax[1].barh(list(imp)[::-1], list(imp.values())[::-1], color="#4c78a8"); ax[1].set_title("Hyperparameter importance")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "optuna_history.png", dpi=110)
