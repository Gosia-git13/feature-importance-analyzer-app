"""Training and comparing machine learning models with PyCaret."""

import pandas as pd
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment

from src.problem_type import IMBALANCE_THRESHOLD

RANDOM_SEED = 123             # same seed -> same results every run
CV_FOLDS = 5                  # cross-validation folds (fewer = faster)
MAX_TRAIN_ROWS = 20_000       # larger files are sampled to keep training fast
ID_LIKE_UNIQUE_SHARE = 0.5    # text column with > 50% unique values -> probably names/IDs
SUSPICIOUS_SCORE = 0.99       # an almost perfect score usually means data leakage

# Only models that can report feature importance (needed for the feature chart)
CLASSIFICATION_MODELS = ["lr", "ridge", "lda", "dt", "rf", "et", "gbc", "ada", "lightgbm"]
REGRESSION_MODELS = ["lr", "ridge", "lasso", "en", "huber", "dt", "rf", "et", "gbr", "ada", "lightgbm"]


def suggest_columns_to_ignore(df: pd.DataFrame, target: str) -> list[str]:
    """Suggest feature columns that are useless for a model.

    - empty columns
    - text columns where most values are unique (names, IDs, free text)
    """
    suggestions = []
    for col in df.columns:
        if col == target:
            continue

        values = df[col].dropna()
        if values.empty:
            suggestions.append(col)
            continue

        is_text = not pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
        if is_text and values.nunique() / len(values) > ID_LIKE_UNIQUE_SHARE:
            suggestions.append(col)

    return suggestions


def prepare_training_data(
    df: pd.DataFrame, target: str, features: list[str]
) -> tuple[pd.DataFrame, bool]:
    """Keep only the chosen columns, drop rows without a target, sample large files.

    Returns the training data and a flag telling whether it was sampled.
    """
    data = df[features + [target]].dropna(subset=[target])

    if len(data) > MAX_TRAIN_ROWS:
        return data.sample(MAX_TRAIN_ROWS, random_state=RANDOM_SEED), True
    return data, False


def choose_sort_metric(target_values: pd.Series, problem_type: str) -> tuple[str, str]:
    """Choose the metric used to rank models, with a short reason.

    - classification, balanced classes -> Accuracy
    - classification, imbalanced classes -> F1
    - regression -> R2 (no units, so it works for any target)
    """
    if problem_type == "classification":
        smallest_class_share = target_values.value_counts(normalize=True).min()
        if smallest_class_share < IMBALANCE_THRESHOLD:
            return "F1", "classes are imbalanced, so accuracy could be misleading"
        return "Accuracy", "classes are reasonably balanced"

    return "R2", "it works for any numeric target, regardless of its units"


def train_best_model(data: pd.DataFrame, target: str, problem_type: str, sort_metric: str):
    """Compare several models with PyCaret.

    Returns a tuple: (experiment, best model, leaderboard with scores of all models).
    """
    if problem_type == "classification":
        experiment = ClassificationExperiment()
        models = CLASSIFICATION_MODELS
    else:
        experiment = RegressionExperiment()
        models = REGRESSION_MODELS

    # setup: prepares data (missing values, text encoding, train/test split)
    experiment.setup(
        data=data,
        target=target,
        session_id=RANDOM_SEED,
        fold=CV_FOLDS,
        verbose=False,
        html=False,
    )
    best_model = experiment.compare_models(include=models, sort=sort_metric, verbose=False)
    leaderboard = experiment.pull()  # scores of all compared models
    return experiment, best_model, leaderboard


def assess_model_quality(score: float) -> tuple[str, str]:
    """Rough, human-readable quality of a 0-1 score (R2, Accuracy, F1)."""
    if score >= SUSPICIOUS_SCORE:
        return "suspicious", (
            "The score is almost perfect, which is rare with real data. This often means "
            "that a feature 'gives away the answer' (data leakage). "
            "Check the most important features below, add the "
            "suspicious one to 'Columns to ignore' and train again."
        )
    if score >= 0.8:
        return "good", "Feature importance should be reliable."
    if score >= 0.5:
        return "moderate", "Treat feature importance as a useful hint, not a final answer."
    return "weak", "The model learned little, so feature importance may be mostly noise."
