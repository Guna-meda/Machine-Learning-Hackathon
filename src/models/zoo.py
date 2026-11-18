"""The candidate-scoring models we compare.  Each factory returns a fresh, unfitted estimator.

Models that care about feature scale (logistic regression, MLP) are wrapped in a
Pipeline with StandardScaler, so the scaler is fitted on training data only.
"""
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from src.config import SEED

MODEL_INFO = {
    "LogisticRegression": ("Linear / probabilistic", "StandardScaler + LogisticRegression(C=1.0, lbfgs, max_iter=300)"),
    "RandomForest": ("Bagged trees (ensemble)", "RandomForest(n_estimators=100, max_depth=16, min_samples_leaf=10, max_features=sqrt)"),
    "ExtraTrees": ("Randomised trees (ensemble)", "ExtraTrees(n_estimators=100, max_depth=16, min_samples_leaf=10, max_features=sqrt)"),
    "HistGradientBoosting": ("Gradient boosting (sklearn)", "HistGB(max_iter=200, learning_rate=0.1, max_leaf_nodes=31)"),
    "LightGBM": ("Gradient boosting (LightGBM)", "LGBM(n_estimators=300, learning_rate=0.1, num_leaves=31)"),
    "MLP": ("Neural network", "StandardScaler + MLP(hidden=(64,32), relu, adam, early_stopping, max_iter=60)"),
}


def make_model(name, **overrides):
    if name == "LogisticRegression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300, random_state=SEED))
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=10, max_features="sqrt", n_jobs=4, random_state=SEED)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=100, max_depth=16, min_samples_leaf=10, max_features="sqrt", n_jobs=4, random_state=SEED)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED)
    if name == "LightGBM":
        params = dict(n_estimators=300, learning_rate=0.1, num_leaves=31, random_state=SEED, n_jobs=4, verbose=-1)
        params.update(overrides)
        return lgb.LGBMClassifier(**params)
    if name == "MLP":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=60,
                                                            random_state=SEED))
    raise ValueError(name)
