"""Detecting whether a target means classification or regression."""
import pandas as pd

PROBLEM_TYPES = ["classification", "regression"]
CLASSIFICATION_MAX_UNIQUE = 20  # whole numbers with <= 20 distinct values -> classes
MIN_ROWS_PER_CLASS = 2          # model training needs at least 2 examples per class
IMBALANCE_THRESHOLD = 0.2       # smallest class below 20% of rows -> imbalanced
MAX_CLASSES = 50                # more classes makes no sense in this app


def detect_problem_type(target: pd.Series) -> tuple[str, str]:
    """Guess whether predicting `target` is classification or regression.

    Returns the problem type and a short, human-readable reason.
    """
    values = target.dropna()
    n_unique = values.nunique()

    # bool must be checked first: pandas treats True/False as numbers too
    if pd.api.types.is_bool_dtype(values) or not pd.api.types.is_numeric_dtype(values):
        return "classification", "it contains text or yes/no values"

    if (values % 1 != 0).any():
        return "regression", "it contains decimal numbers"

    if n_unique <= CLASSIFICATION_MAX_UNIQUE:
        return "classification", f"it has only {n_unique} distinct whole-number values"

    return "regression", f"it has many ({n_unique:,}) distinct numeric values"
