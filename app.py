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
st.success(f"File **{uploaded_file.name}** loaded successfully!")

col_rows, col_cols = st.columns(2)
col_rows.metric("Rows", f"{df.shape[0]:,}")
col_cols.metric("Columns", df.shape[1])

st.subheader(f"Preview: first {PREVIEW_ROWS} rows")
st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True, hide_index=True)