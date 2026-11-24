"""Reading CSV files and fixing column types"""
import io

import pandas as pd
import streamlit as st

# "utf-8-sig" reads normal UTF-8 and also UTF-8 files saved by Excel (with a BOM);
# "latin-1" is a fallback for older European files.
ENCODINGS = ("utf-8-sig", "latin-1")

TIME_PATTERN = r"^\d{1,2}:\d{2}:\d{2}$"  # HH:MM:SS, e.g. 01:04:59
TIME_MATCH_SHARE = 0.9  # >= 90% of values must look like a time


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


@st.cache_data(show_spinner="Checking column types...")
def convert_time_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert text columns that look like HH:MM:SS into seconds.

    Returns the converted DataFrame and a dict:
    {column name: number of values that could not be converted, e.g. "DNF"}.
    """
    df = df.copy()
    converted = {}

    for col in df.select_dtypes(include="object").columns:
        values = df[col].dropna().astype(str).str.strip()  # pyright: ignore[reportAttributeAccessIssue]
        if values.empty:
            continue

        share_matching = values.str.match(TIME_PATTERN).mean()
        if share_matching < TIME_MATCH_SHARE:
            continue  # not a time column

        missing_before = df[col].isna().sum()
        df[col] = pd.to_timedelta(df[col], errors="coerce").dt.total_seconds()
        converted[col] = int(df[col].isna().sum() - missing_before)

    return df, converted
