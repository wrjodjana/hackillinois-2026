import streamlit as st
import pandas as pd
import subprocess
import os
import json
import sys

st.set_page_config(page_title="Refine Loop App", layout="wide")
st.title("Dataset Refinement App")

os.makedirs("temp", exist_ok=True)

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "cleaning", "script.py")

with st.sidebar:
    st.header("Cleaning Options")

    missing_strategy = st.radio(
        "Missing Values",
        ["Drop rows", "Impute (median / mode)"],
        help="Drop removes any row with a null. Impute fills numeric columns with median and categorical columns with mode.",
    )

    outlier_strategy = st.radio(
        "Outliers",
        ["Remove", "Cap (winsorize)", "Keep"],
        help="Remove deletes rows with extreme values. Cap clips them to the 1st/99th percentile. Keep leaves them as-is.",
    )

    typo_strategy = st.radio(
        "Typo / Categorical Cleanup",
        ["Auto-correct", "Skip"],
        help="Auto-correct uses AI to fix misspellings and standardize categorical values. Skip leaves them untouched.",
    )

options = {
    "missing": "drop" if missing_strategy == "Drop rows" else "impute",
    "outliers": outlier_strategy.lower().split(" ")[0],
    "typos": "auto" if typo_strategy == "Auto-correct" else "skip",
}

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded:
    input_path = os.path.abspath("temp/input.csv")
    output_path = os.path.abspath("temp/input_cleaned.csv")

    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())

    df = pd.read_csv(input_path)

    st.subheader("Preview of Uploaded Dataset")
    st.dataframe(df, width="stretch")
    st.caption(f"Rows: {len(df):,} | Columns: {df.shape[1]}")

    if st.button("Run Cleaning"):
        with st.spinner("Running cleaning agent..."):
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        SCRIPT_PATH,
                        input_path,
                        output_path,
                        json.dumps(options),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if result.returncode != 0:
                    st.error("Cleaning script failed.")
                    st.code(result.stderr, language="text")
                elif os.path.exists(output_path):
                    cleaned = pd.read_csv(output_path)

                    st.success("Cleaning completed!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Before")
                        st.caption(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
                        st.dataframe(df, width="stretch")
                    with col2:
                        st.subheader("After")
                        st.caption(f"Rows: {len(cleaned):,} | Columns: {cleaned.shape[1]}")
                        st.dataframe(cleaned, width="stretch")

                    rows_dropped = len(df) - len(cleaned)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Rows Dropped", f"{rows_dropped:,}")
                    m2.metric("Nulls Remaining", int(cleaned.isnull().sum().sum()))
                    m3.metric("Columns", cleaned.shape[1])

                    csv_data = cleaned.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Cleaned CSV",
                        csv_data,
                        file_name="cleaned.csv",
                        mime="text/csv",
                    )

                    st.session_state["cleaned_df"] = cleaned
                    st.divider()
                    if st.button("Proceed to Linear Regression"):
                        st.switch_page("pages/linreg.py")
                else:
                    st.error("Cleaned file not found.")

            except subprocess.TimeoutExpired:
                st.error("Cleaning script timed out (120s limit).")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
