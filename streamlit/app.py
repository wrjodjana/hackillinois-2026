import streamlit as st
import pandas as pd
import subprocess
import os

st.set_page_config(page_title="Refine Loop App", layout="wide")
st.title("Dataset Refinement App")

os.makedirs("temp", exist_ok=True)

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "cleaning", "script.py")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded:
    input_path = os.path.abspath("temp/input.csv")
    output_path = os.path.abspath("temp/input_cleaned.csv")

    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())

    df = pd.read_csv(input_path)

    st.subheader("Preview of Uploaded Dataset")
    st.dataframe(df, use_container_width=True)
    st.caption(f"Rows: {len(df):,} | Columns: {df.shape[1]}")

    if st.button("Run Cleaning"):
        with st.spinner("Running cleaning agent..."):
            try:
                result = subprocess.run(
                    ["python3", SCRIPT_PATH, input_path, output_path],
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
                        st.dataframe(df, use_container_width=True)
                    with col2:
                        st.subheader("After")
                        st.caption(f"Rows: {len(cleaned):,} | Columns: {cleaned.shape[1]}")
                        st.dataframe(cleaned, use_container_width=True)

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
                else:
                    st.error("Cleaned file not found.")

            except subprocess.TimeoutExpired:
                st.error("Cleaning script timed out (120s limit).")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
