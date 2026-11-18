"""Step 3: compare 6 models with GroupKFold (by trip) on the 80 train trips and evaluate on student_test.

Outputs (results/): model_comparison.csv, cv_fold_metrics.csv, student_confusion_counts.csv, mlp_loss_curve.csv
"""
import sys, time, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from src.config import RESULTS_DIR, SEED
from src.features.build import FEATURE_COLUMNS
from src.models.data import load_table, sample_points
from src.models.zoo import make_model, MODEL_INFO
from src.models.evaluate import evaluate_predictions

FIT_FRAC = 0.25      # share of GPS points used for fitting (all their candidates are kept)
N_FOLDS = 5


def points_of(table):
    return table[["trip_id", "point_idx", "ref_arc", "split"]].drop_duplicates(["trip_id", "point_idx"])


def fit(name, rows):
    model = make_model(name)
    t0 = time.time()
    model.fit(rows[FEATURE_COLUMNS].values.astype(np.float32), rows["label"].values)
    return model, time.time() - t0


def score(model, rows):
    t0 = time.time()
    p = model.predict_proba(rows[FEATURE_COLUMNS].values.astype(np.float32))[:, 1]
    return p, time.time() - t0


if __name__ == "__main__":
    names = sys.argv[1:] or list(MODEL_INFO)
    table = load_table()
    train = table[(table.split == "train") & (table.label_ok == 1)]
    student = table[table.split == "student_test"]
    print("train rows", len(train), "student rows", len(student), flush=True)
    trips = train["trip_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(train, groups=trips))
    fit_all = sample_points(train, FIT_FRAC)

    comp_rows, fold_rows, conf_rows = [], [], []
    for name in names:
        fam, cfg = MODEL_INFO[name]
        print("=== ", name, flush=True)
        # ---- cross-validation on train trips (held-out trips per fold)
        cv = []
        for k, (tr_idx, va_idx) in enumerate(folds):
            tr = sample_points(train.iloc[tr_idx], FIT_FRAC)
            va = train.iloc[va_idx].copy()
            model, t_fit = fit(name, tr)
            va["score"], _ = score(model, va)
            m = evaluate_predictions(va, points_of(va))
            m.update(model=name, fold=k, fit_s=t_fit)
            cv.append(m); fold_rows.append(m)
            print(f"  fold {k}: top1={m['top1_acc']:.4f} f1={m['macro_f1_arc']:.4f} ll={m['log_loss']:.4f} fit={t_fit:.0f}s", flush=True)
        cv = pd.DataFrame(cv)
        # ---- final fit on all train trips, evaluate on student_test
        model, t_fit = fit(name, fit_all)
        st = student.copy()
        st["score"], t_inf = score(model, st)
        sm = evaluate_predictions(st, points_of(st))
        # confusion counts of the candidate-level binary view
        best = st.groupby(["trip_id", "point_idx"], sort=False)["score"].transform("max")
        pred = (st["score"] == best).astype(int)
        tn, fp = int(((st.label == 0) & (pred == 0)).sum()), int(((st.label == 0) & (pred == 1)).sum())
        fn, tp = int(((st.label == 1) & (pred == 0)).sum()), int(((st.label == 1) & (pred == 1)).sum())
        conf_rows.append(dict(model=name, tn=tn, fp=fp, fn=fn, tp=tp))
        # overfitting check: score on (a sample of) the rows it was fitted on
        tr_eval = fit_all[fit_all.trip_id.isin(fit_all.trip_id.unique()[:20])].copy()
        tr_eval["score"], _ = score(model, tr_eval)
        trm = evaluate_predictions(tr_eval, points_of(tr_eval))
        comp_rows.append(dict(
            model=name, family=fam, config=cfg,
            cv_macro_f1=cv.macro_f1_arc.mean(), cv_macro_f1_std=cv.macro_f1_arc.std(),
            cv_top1=cv.top1_acc.mean(), cv_top1_std=cv.top1_acc.std(), cv_log_loss=cv.log_loss.mean(),
            cv_route_continuity=cv.route_continuity.mean(),
            student_top1=sm["top1_acc"], student_macro_f1_arc=sm["macro_f1_arc"], student_macro_f1_binary=sm["macro_f1_binary"],
            student_log_loss=sm["log_loss"], student_route_continuity=sm["route_continuity"],
            train_top1_fitted_rows=trm["top1_acc"], train_log_loss_fitted_rows=trm["log_loss"],
            train_time_s=t_fit, cv_mean_fit_s=cv.fit_s.mean(), inference_s_student=t_inf,
            inference_us_per_candidate=1e6 * t_inf / len(st), n_fit_rows=len(fit_all), n_student_rows=len(st), seed=SEED,
        ))
        print(f"  STUDENT top1={sm['top1_acc']:.4f} f1={sm['macro_f1_arc']:.4f} ll={sm['log_loss']:.4f} cont={sm['route_continuity']:.4f} fit={t_fit:.0f}s inf={t_inf:.1f}s", flush=True)
        if name == "MLP":
            pd.DataFrame({"epoch": range(1, len(model[-1].loss_curve_) + 1), "train_loss": model[-1].loss_curve_,
                          "val_score": model[-1].validation_scores_}).to_csv(RESULTS_DIR / "mlp_loss_curve.csv", index=False)
        if name == "LightGBM":
            import joblib; (RESULTS_DIR.parent / "cache").mkdir(exist_ok=True)
        # write after every model so a crash loses nothing
        pd.DataFrame(comp_rows).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "cv_fold_metrics.csv", index=False)
        pd.DataFrame(conf_rows).to_csv(RESULTS_DIR / "student_confusion_counts.csv", index=False)
