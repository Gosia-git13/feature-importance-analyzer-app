"""Turning model results into a plain-language summary."""

import pandas as pd

from src.formatting import format_share

TOP_FEATURES_IN_TEXT = 3
DOMINANT_FEATURE_SHARE = 0.8  # one feature above 80% of total importance -> mention it


def describe_score(metric: str, score: float, baseline_accuracy: float | None) -> str:
    """Explain the model score in one or two simple sentences."""
    if metric == "R2":
        if score <= 0:
            return (
                "It does **not** predict the target better than simply guessing the average "
                f"(R² = {score:.2f})."
            )
        return f"It explains about **{score:.0%}** of the differences in the target (R² = {score:.2f})."

    if metric == "Accuracy":
        text = f"It predicts the correct class in **{score:.0%}** of cases (accuracy = {score:.2f})."
        if baseline_accuracy is not None:
            text += (
                f" For comparison, always guessing the most common class would be right "
                f"in {baseline_accuracy:.0%} of cases."
            )
        return text

    # F1: used when classes are imbalanced
    text = (
        f"Its F1 score is **{score:.2f}** (0 = useless, 1 = perfect). F1 measures how well the "
        "model finds the rare class while avoiding false alarms."
    )
    if baseline_accuracy is not None:
        text += (
            f" Accuracy would be misleading here: a model that always guesses the most common "
            f"class would already be right in {baseline_accuracy:.0%} of cases, "
            "while never finding a single rare case."
        )
    return text


def describe_features(importance: pd.DataFrame) -> str:
    """Describe the most important features and those with no measurable effect."""
    top = importance[importance["share"] > 0].head(TOP_FEATURES_IN_TEXT)
    if top.empty:
        return "None of the features had a measurable effect on the model's predictions."

    first = top.iloc[0]
    text = (
        f"The most important feature is **{first['feature']}** "
        f"({format_share(first['share'])} of total importance)"
    )

    others = [
        f"**{feature}** ({format_share(share)})"
        for feature, share in zip(top["feature"].iloc[1:], top["share"].iloc[1:])
    ]
    if others:
        text += ", followed by " + " and ".join(others)
    text += "."

    if first["share"] > DOMINANT_FEATURE_SHARE:
        text += (
            f" {first['feature']} alone drives almost all predictions, so the other features "
            "add little on top of it."
        )

    no_effect = importance[importance["importance"] <= 0]["feature"].tolist()
    if no_effect:
        text += f" These features had no measurable effect: {', '.join(no_effect)}."

    return text


def build_description(
    *,
    target: str,
    problem_type: str,
    n_rows: int,
    n_features: int,
    n_models: int,
    best_model_name: str,
    metric: str,
    score: float,
    quality: str,
    quality_note: str,
    importance: pd.DataFrame,
    mae_text: str | None = None,
    n_classes: int | None = None,
    baseline_accuracy: float | None = None,
) -> str:
    """Build a short report (Markdown) describing what the analysis found."""
    if problem_type == "regression":
        task = "a **regression** problem: the model predicts a number"
    else:
        kind = "binary" if n_classes == 2 else "multiclass"
        task = f"a **{kind} classification** problem: the model predicts one of {n_classes} classes"

    paragraphs = [
        # 1. What was analysed
        f"We analysed **{n_rows:,} rows** and **{n_features} features** to find out what "
        f"influences **{target}**. This is {task}.",
        # 2. How good the model is
        f"Out of {n_models} compared models, the best one was **{best_model_name}**. "
        + describe_score(metric, score, baseline_accuracy)
        + (f" On average, its predictions are off by **{mae_text}**." if mae_text else ""),
        # 3. What matters most
        describe_features(importance),
        # 4. How much to trust it
        f"**How much to trust this:** model quality is *{quality}*. {quality_note} "
        "Keep in mind that feature importance shows what the model *relies on*, not cause and "
        "effect. A feature with low importance may still be related to the target if another "
        "feature carries the same information.",
    ]
    return "\n\n".join(paragraphs)
