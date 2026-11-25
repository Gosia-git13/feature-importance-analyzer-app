"""Measuring which features matter most for a trained model."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score

from src.modeling import RANDOM_SEED

N_REPEATS = 5  # how many times each column is shuffled (more = more stable, but slower)

# Same metrics as used to rank models. F1 uses the "weighted" average,
# so it works for both binary and multiclass targets.
METRIC_SCORERS = {
    "R2": r2_score,
    "Accuracy": accuracy_score,
    "F1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
}


def permutation_importance_by_column(experiment, model, metric: str) -> tuple[float, pd.DataFrame]:
    """Permutation importance for each original column, measured on the test set.

    Idea: shuffle one column so it carries no information, then check how much
    the model's score drops. A big drop means the model relies on that column.

    Returns the baseline test score and a table with columns:
    feature, importance (mean score drop), std (spread between repeats), share (0-1).
    """
    X_test = experiment.get_config("X_test")                # original columns
    y_test = experiment.get_config("y_test_transformed")   # target as the model sees it
    preprocessing = experiment.pipeline                    # encoding, imputation, ...
    score = METRIC_SCORERS[metric]
    rng = np.random.default_rng(RANDOM_SEED)

    def evaluate(X: pd.DataFrame) -> float:
        return score(y_test, model.predict(preprocessing.transform(X)))

    baseline = evaluate(X_test)

    rows = []
    for col in X_test.columns:
        drops = []
        for _ in range(N_REPEATS):
            X_shuffled = X_test.copy()
            X_shuffled[col] = rng.permutation(X_shuffled[col].to_numpy())
            drops.append(baseline - evaluate(X_shuffled))
        rows.append({"feature": col, "importance": np.mean(drops), "std": np.std(drops)})

    importance = pd.DataFrame(rows).sort_values("importance", ascending=False)

    # Share of total importance; negative values (random noise) count as zero
    positive = importance["importance"].clip(lower=0)
    total = positive.sum()
    importance["share"] = positive / total if total > 0 else 0.0

    return baseline, importance.reset_index(drop=True)
