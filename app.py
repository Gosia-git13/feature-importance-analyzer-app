import csv

import pandas as pd
import streamlit as st

from src.data_loading import convert_time_columns, load_csv
from src.formatting import format_duration, format_share
from src.modeling import (
    MAX_TRAIN_ROWS,
    assess_model_quality,
    choose_sort_metric,
    prepare_training_data,
    suggest_columns_to_ignore,
    train_best_model,
)
from src.problem_type import (
    IMBALANCE_THRESHOLD,
    MAX_CLASSES,
    MIN_ROWS_PER_CLASS,
    PROBLEM_TYPES,
    detect_problem_type,
)

# -------------------------------------------------------------------------------
# Page settings
# -------------------------------------------------------------------------------
st.set_page_config(page_title="Feature Importance Analyzer", layout="wide")

st.title("📊 :blue[Feature Importance] :grey[Analyzer]")
st.caption("Upload a CSV file to discover which features matter most.")

PREVIEW_ROWS = 10
TOP_CLASSES_IN_CHART = 20


# -------------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is None:
    st.info("👆 Upload a CSV file to get started.")
    st.stop()

file_bytes = uploaded_file.getvalue()

if not file_bytes.strip():
    st.error("The file is empty.")
    st.stop()

try:
    df = load_csv(file_bytes)
except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error, ValueError) as err:
    st.error(f"Could not read the file: {err}")
    st.stop()

if df.empty:
    st.warning("The file has no data records to display.")
    st.stop()

df, converted_time_columns = convert_time_columns(df)


# -------------------------------------------------------------------------------
# Data preview (v1)
# -------------------------------------------------------------------------------
st.success(f"File **{uploaded_file.name}** loaded successfully!")

if converted_time_columns:
    names = ", ".join(f"**{col}**" for col in converted_time_columns)
    message = (
        f"Converted {len(converted_time_columns)} time column(s) (HH:MM:SS) "
        f"to seconds: {names}."
    )
    n_failed = sum(converted_time_columns.values())
    if n_failed:
        message += (
            f" {n_failed:,} value(s) were not valid times (e.g. 'DNF') "
            "and are now treated as missing."
        )
    st.info(message)

col_rows, col_cols = st.columns(2)
col_rows.metric("Rows", f"{df.shape[0]:,}")
col_cols.metric("Columns", df.shape[1])

st.subheader(f"Preview: first {PREVIEW_ROWS} rows")
st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True, hide_index=True)


# -------------------------------------------------------------------------------
# Target column selection (v2)
# -------------------------------------------------------------------------------
st.divider()
st.subheader("🎯 Choose the target column")
st.caption(
    "The target is the column you want to predict. "
    "All other columns will be used as features."
)

if df.shape[1] < 2:
    st.error("The file needs at least 2 columns: one target and at least one feature.")
    st.stop()

target = st.selectbox(
    "Target column",
    options=df.columns.tolist(),
    index=None,
    placeholder="Select a column...",
)

if target is None:
    st.info("👆 Select the column you want to predict.")
    st.stop()

target_series = df[target]
n_unique = target_series.nunique(dropna=True)
n_missing = int(target_series.isna().sum())

col_type, col_unique, col_missing = st.columns(3)
col_type.metric("Data type", str(target_series.dtype))
col_unique.metric("Unique values", f"{n_unique:,}")
col_missing.metric("Missing values", f"{n_missing:,}")

if n_unique < 2:
    st.error(
        "The target column has fewer than 2 distinct values, so there is nothing "
        "to predict. Please choose another column."
    )
    st.stop()

if n_missing > 0:
    st.warning(
        f"Rows without a target value: {n_missing:,} ({format_share(n_missing / len(df))}). "
        "They will be removed before model training."
    )

feature_columns = [col for col in df.columns if col != target]

st.success(f"Target: **{target}** · Features: **{len(feature_columns)}** columns")

with st.expander("Show feature columns"):
    st.write(", ".join(feature_columns))


# -------------------------------------------------------------------------------
# Problem type: classification or regression (v3)
# -------------------------------------------------------------------------------
st.divider()
st.subheader("🔍 Problem type")

detected_type, reason = detect_problem_type(target_series)

problem_type = st.radio(
    "Problem type",
    options=PROBLEM_TYPES,
    index=PROBLEM_TYPES.index(detected_type),
    format_func=str.capitalize,
    horizontal=True,
    help="Detected automatically. Change it if the detection is wrong.",
)

st.info(f"Detected **{detected_type}** for **{target}** because {reason}.")

target_values = target_series.dropna()

if problem_type == "classification":
    class_counts = target_values.value_counts()
    n_classes = len(class_counts)
    kind = "Binary" if n_classes == 2 else "Multiclass"

    # Too many classes: probably an ID, free text, or a number stored as text
    if n_classes > MAX_CLASSES:
        st.error(
            f"Too many classes for classification ({n_classes:,}; the limit is {MAX_CLASSES}). "
            "This column looks like an ID or free text. If it holds numbers or times, "
            "check their format. Otherwise, please choose another target column."
        )
        st.stop()

    st.markdown(f"**{kind} classification**: the model will predict one of **{n_classes}** classes.")

    rare_classes = class_counts[class_counts < MIN_ROWS_PER_CLASS]
    if not rare_classes.empty:
        st.warning(
            f"{len(rare_classes)} class(es) have fewer than {MIN_ROWS_PER_CLASS} rows "
            "and may cause errors during training."
        )

    smallest_share = class_counts.min() / class_counts.sum()
    if smallest_share < IMBALANCE_THRESHOLD:
        st.warning(
            f"The classes are imbalanced: the smallest class covers only "
            f"{format_share(smallest_share)} of rows. Accuracy alone may be misleading."
        )

    chart_title = "Class distribution"
    if n_classes > TOP_CLASSES_IN_CHART:
        chart_title += f" (top {TOP_CLASSES_IN_CHART})"
    st.markdown(f"**{chart_title}**")
    st.bar_chart(class_counts.head(TOP_CLASSES_IN_CHART))

else:  # regression
    if not pd.api.types.is_numeric_dtype(target_values) or pd.api.types.is_bool_dtype(target_values):
        st.error("Regression needs a numeric target. Choose classification or another column.")
        st.stop()

    st.markdown("**Regression**: the model will predict a number.")

    # Times were converted to seconds for the model; show them back as HH:MM:SS
    is_time_target = target in converted_time_columns

    def fmt(value: float) -> str:
        return format_duration(value) if is_time_target else f"{value:,.2f}"

    col_min, col_median, col_mean, col_max = st.columns(4)
    col_min.metric("Min", fmt(target_values.min()))
    col_median.metric("Median", fmt(target_values.median()))
    col_mean.metric("Mean", fmt(target_values.mean()))
    col_max.metric("Max", fmt(target_values.max()))

    if is_time_target:
        st.caption("Times are shown as HH:MM:SS; the model works with seconds.")


# -------------------------------------------------------------------------------
# Model training (v4)
# -------------------------------------------------------------------------------
st.divider()
st.subheader("🤖 Train the model")

# --- 1. Choose features ---------------------------------------------------------
ignored_columns = st.multiselect(
    "Columns to ignore",
    options=feature_columns,
    default=suggest_columns_to_ignore(df, target),
    help=(
        "Pre-selected: empty columns and text columns where most values are unique "
        "(e.g. names, IDs). Also add columns that 'give away the answer', "
        "e.g. finishing place when predicting finishing time."
    ),
)

used_features = [col for col in feature_columns if col not in ignored_columns]
if not used_features:
    st.error("All feature columns are ignored. Keep at least one feature.")
    st.stop()

# --- 2. Prepare data and choose the ranking metric ------------------------------
train_df, was_sampled = prepare_training_data(df, target, used_features)
if was_sampled:
    st.caption(f"Large file: training on a random sample of {MAX_TRAIN_ROWS:,} rows to keep it fast.")

sort_metric, metric_reason = choose_sort_metric(train_df[target], problem_type)

st.markdown(
    f"Training on **{len(train_df):,}** rows with **{len(used_features)}** features. "
    f"Models will be ranked by **{sort_metric}** ({metric_reason})."
)

# --- 3. Train (only when the button is clicked) ---------------------------------
# Results are kept in session_state, because Streamlit reruns the whole script
# after every click. The key detects when inputs changed and old results are stale.
run_key = (hash(file_bytes), target, problem_type, tuple(used_features))

results = st.session_state.get("model_results")
if results is not None and results["key"] != run_key:
    results = None

if st.button("🚀 Train and compare models", type="primary"):
    with st.spinner(f"Training and comparing {problem_type} models... this can take a few minutes."):
        try:
            experiment, best_model, leaderboard = train_best_model(
                train_df, target, problem_type, sort_metric
            )
        except Exception as err:  # PyCaret can fail for many data-specific reasons
            st.error(f"Training failed: {err}")
            st.stop()

    results = {
        "key": run_key,
        "experiment": experiment,
        "best_model": best_model,
        "leaderboard": leaderboard,
        "sort_metric": sort_metric,
    }
    st.session_state["model_results"] = results

if results is None:
    st.info("👆 Click the button to train and compare models.")
    st.stop()

# --- 4. Show results ------------------------------------------------------------
leaderboard = results["leaderboard"]
best_row = leaderboard.iloc[0]
best_score = float(best_row[results["sort_metric"]])

st.success(f"Best model: **{best_row['Model']}**")

col_score, col_extra = st.columns(2)
col_score.metric(f"{results['sort_metric']} (cross-validation)", f"{best_score:.3f}")

if problem_type == "regression":
    mae = float(best_row["MAE"])
    mae_text = format_duration(mae) if target in converted_time_columns else f"{mae:,.2f}"
    col_extra.metric("MAE: average error", mae_text)
else:
    other_metric = "F1" if results["sort_metric"] == "Accuracy" else "Accuracy"
    col_extra.metric(other_metric, f"{float(best_row[other_metric]):.3f}")

quality, quality_note = assess_model_quality(best_score)
if quality == "good":
    st.success(f"Model quality: **{quality}**. {quality_note}")
elif quality == "moderate":
    st.info(f"Model quality: **{quality}**. {quality_note}")
else:
    st.warning(f"Model quality: **{quality}**. {quality_note}")

with st.expander("Compare all models"):
    st.caption("Average scores from cross-validation. TT (Sec) = training time in seconds.")
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)
