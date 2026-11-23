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

TIME_PATTERN = r"^\d{1,2}:\d{2}:\d{2}$"  # HH:MM:SS, e.g. 01:04:59
TIME_MATCH_SHARE = 0.9  # >= 90% of values must look like a time

# "utf-8-sig" reads normal UTF-8 and also UTF-8 files saved by Excel (with a BOM);
# "latin-1" is a fallback for older European files.
ENCODINGS = ("utf-8-sig", "latin-1")

# -------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------
@st.cache_data(show_spinner="Checking column types...")
def convert_time_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert text columns that look like HH:MM:SS into seconds.

    Returns the converted DataFrame and a dict:
    {column name: number of values that could not be converted, e.g. "DNF"}.
    """
    df = df.copy()
    converted = {}

    for col in df.select_dtypes(include="object").columns:
        values = df[col].dropna().astype(str).str.strip() # pyright: ignore[reportAttributeAccessIssue]
        if values.empty:
            continue

        share_matching = values.str.match(TIME_PATTERN).mean()
        if share_matching < TIME_MATCH_SHARE:
            continue  # not a time column

        missing_before = df[col].isna().sum()
        df[col] = pd.to_timedelta(df[col], errors="coerce").dt.total_seconds()
        converted[col] = int(df[col].isna().sum() - missing_before)

    return df, converted


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
# Main flow
# -------------------------------------------------------------------------------

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
        f"Rows without a target value: {n_missing:,} ({n_missing / len(df):.1%}). "
        "They will be removed before model training."
    )

feature_columns = [col for col in df.columns if col != target]

st.success(f"Target: **{target}** · Features: **{len(feature_columns)}** columns")

with st.expander("Show feature columns"):
    st.write(", ".join(feature_columns))