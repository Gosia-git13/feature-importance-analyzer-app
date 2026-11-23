import csv
import io

import pandas as pd
import streamlit as st


# -------------------------------------------------------------------------------
# Page settings
# -------------------------------------------------------------------------------
st.set_page_config(page_title="Feature Importance Analyzer", layout="wide")

st.title("📊 :blue[Feature Importance] :grey[Analyzer]")
st.caption("Upload a CSV file to discover which features matter most.")

PREVIEW_ROWS = 10
# "utf-8-sig" reads normal UTF-8 and also UTF-8 files saved by Excel (with a BOM);
# "latin-1" is a fallback for older European files.
ENCODINGS = ("utf-8-sig", "latin-1")

# Problem type detection (v3)
PROBLEM_TYPES = ["classification", "regression"]
CLASSIFICATION_MAX_UNIQUE = 20  # whole numbers with <= 20 distinct values -> classes
TOP_CLASSES_IN_CHART = 20
MIN_ROWS_PER_CLASS = 2          # model training needs at least 2 examples per class
IMBALANCE_THRESHOLD = 0.05      # smallest class below 5% of rows -> imbalanced


# -------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading file...")
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read CSV bytes into a DataFrame.
    - detects the column separator automatically (comma, semicolon, tab...)
    - tries UTF-8 first, then falls back to Latin-1
    """
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(
                io.BytesIO(file_bytes),
                sep=None,          # pandas to detect the separator
                engine="python",
                encoding=encoding,
            )
        except UnicodeDecodeError:
            continue  # if wrong encoding try the next one
    raise ValueError("Could not decode the file. Please save it as UTF-8.")


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


# -------------------------------------------------------------------------------
# Main flow
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# Data preview (v1)
# -------------------------------------------------------------------------------
st.success(f"File **{uploaded_file.name}** loaded successfully!")

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
        f"Rows without a target value: {n_missing:,} ({n_missing / len(df):.1%}). "
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

    st.markdown(f"**{kind} classification**: the model will predict one of **{n_classes}** classes.")

    # Looks like an ID or free text: almost every row has its own "class"
    if n_classes > 50 and n_classes / len(target_values) > 0.5:
        st.warning(
            "Almost every row has a different value. This column looks like an ID "
            "or free text and is probably not a good target."
        )

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
            f"{smallest_share:.1%} of rows. Accuracy alone may be misleading."
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

    col_min, col_median, col_mean, col_max = st.columns(4)
    col_min.metric("Min", f"{target_values.min():,.2f}")
    col_median.metric("Median", f"{target_values.median():,.2f}")
    col_mean.metric("Mean", f"{target_values.mean():,.2f}")
    col_max.metric("Max", f"{target_values.max():,.2f}")