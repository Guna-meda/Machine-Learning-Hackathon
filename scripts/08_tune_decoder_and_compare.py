"""Phase 2 / step 8: tune the HMM decoder parameters and compare classifier-only vs classifier + HMM.

Rules followed: decoder parameters are tuned with TRAIN trips only (out-of-fold classifier scores, so the
classifier has never seen those trips) and then reported on student_test.  Nothing here touches evaluator data.

Outputs: results/hmm_tuning_trials.csv, results/hmm_best_config.json, results/hmm_comparison_train_oof.csv,
         results/hmm_comparison_student_per_trip.csv, results/hmm_comparison_summary.csv, reports/figures/hmm_*.png
"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from src.config import RESULTS_DIR, FIGURES_DIR, CACHE_DIR, SEED
from src.features.build import FEATURE_COLUMNS
from src.models.data import load_table, sample_points
from src.hmm.viterbi import HMMConfig
from src.hmm.evaluate_hmm import evaluate_split, summarise

N_TRIALS, TIMEOUT_S, N_TUNE_TRIPS = 100, 900, 40
optuna.logging.set_verbosity(optuna.logging.WARNING)


def fit_hgb(rows, params):
    return HistGradientBoostingClassifier(random_state=SEED, **params).fit(rows[FEATURE_COLUMNS].values.astype(np.float32), rows["label"].values)


def score(model, df):
    out = df.copy()
    out["score"] = model.predict_proba(df[FEATURE_COLUMNS].values.astype(np.float32))[:, 1]
    return out


if __name__ == "__main__":
    best_clf = json.load(open(RESULTS_DIR / "optuna_best_params.json"))["best_params"]
    table = load_table()
    train = table[(table.split == "train") & (table.label_ok == 1)]
    student = table[table.split == "student_test"]

    # 1. out-of-fold classifier scores on the train trips (each trip scored by a model that never saw it)
    cache = CACHE_DIR / "oof_scores.pkl"
    if cache.exists():
        oof = pd.read_pickle(cache)
    else:
        parts = []
        for tr_idx, va_idx in GroupKFold(n_splits=5).split(train, groups=train["trip_id"].values):
            m = fit_hgb(sample_points(train.iloc[tr_idx], 0.25), best_clf)
            parts.append(score(m, train.iloc[va_idx]))
        oof = pd.concat(parts)
        oof.to_pickle(cache)
    print("OOF scores ready", len(oof), flush=True)

    # 2. choose a small, fixed set of train trips for tuning (the first N mid-sized trips by id, deterministic)
    sizes = oof.groupby("trip_id")["point_idx"].nunique()
    mid = sizes[(sizes >= 300) & (sizes <= 4000)].sort_index()
    tune_ids = list(mid.index[:N_TUNE_TRIPS])
    tune_df = oof[oof.trip_id.isin(tune_ids)]
    print("tuning decoder on", len(tune_ids), "train trips:", tune_ids, flush=True)

    def objective(trial):
        cfg = HMMConfig(beta=trial.suggest_float("beta", 3, 100, log=True),
                        emission_weight=trial.suggest_float("emission_weight", 0.1, 3.0),
                        disconnect_penalty=trial.suggest_float("disconnect_penalty", 2, 60),
                        uturn_penalty=trial.suggest_float("uturn_penalty", 0, 15))
        s = summarise(evaluate_split(tune_df, "train", cfg))
        trial.set_user_attr("hmm_route_f1", s["hmm_route_f1"]); trial.set_user_attr("hmm_continuity", s["hmm_continuity"])
        return s["hmm_top1"]                       # observation-level Top-1 is the task metric

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial({"beta": 30.0, "emission_weight": 1.0, "disconnect_penalty": 12.0, "uturn_penalty": 4.0})   # default decoder
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_S)
    study.trials_dataframe().to_csv(RESULTS_DIR / "hmm_tuning_trials.csv", index=False)
    best = HMMConfig(**study.best_params)
    json.dump(best.to_dict(), open(RESULTS_DIR / "hmm_best_config.json", "w"), indent=2)
    print(f"best decoder top1={study.best_value:.4f} after {len(study.trials)} trials ({time.time()-t0:.0f}s):", best.to_dict(), flush=True)

    # 3. classifier-only vs +HMM: default decoder vs tuned decoder on ALL train trips (out-of-fold) and on student_test
    rows, per_trip_student = [], None
    model = fit_hgb(sample_points(train, 0.25), best_clf)
    st_scored = score(model, student)
    for cname, cfg in (("default decoder", HMMConfig()), ("tuned decoder", best)):
        for split, scored in (("train (80 trips, out-of-fold)", oof), ("student_test (10 trips)", st_scored)):
            sp = "train" if split.startswith("train") else "student_test"
            per_trip = evaluate_split(scored, sp, cfg)
            s = summarise(per_trip)
            rows.append(dict(decoder=cname, partition=split, **s))
            if cname == "tuned decoder" and sp == "student_test":
                per_trip_student = per_trip
            if cname == "tuned decoder" and sp == "train":
                per_trip.to_csv(RESULTS_DIR / "hmm_comparison_train_oof.csv", index=False)
            print(cname, split, {k: round(v, 4) for k, v in s.items() if k.startswith(("clf_top1", "hmm_top1", "clf_cont", "hmm_cont", "clf_route_f1", "hmm_route_f1"))}, flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "hmm_comparison_summary.csv", index=False)
    per_trip_student.to_csv(RESULTS_DIR / "hmm_comparison_student_per_trip.csv", index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = per_trip_student
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(d)); w = 0.4
    for a, (c, h, title) in zip(ax, (("clf_top1", "hmm_top1", "Top-1 accuracy"), ("clf_continuity", "hmm_continuity", "Route continuity"), ("clf_route_f1", "hmm_route_f1", "Route arc F1 vs reference"))):
        a.bar(x - w / 2, d[c], w, label="classifier only", color="#9aa0a6"); a.bar(x + w / 2, d[h], w, label="classifier + HMM", color="#2a6fbb")
        a.set_xticks(x); a.set_xticklabels(d.trip_id, rotation=60, fontsize=7); a.set_title(title + " (student_test, per trip)")
    ax[0].legend(); plt.tight_layout(); plt.savefig(FIGURES_DIR / "hmm_vs_classifier_student.png", dpi=110)
