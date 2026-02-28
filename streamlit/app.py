import streamlit as st
import pandas as pd
import subprocess
import os

st.set_page_config(page_title="Refine Loop App", layout="wide")
st.title("Dataset Refinement App")

# Make sure upload folder exists
os.makedirs("temp", exist_ok=True)

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded:
    # Save uploaded file
    input_path = "temp/input.csv"
    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())

    df = pd.read_csv(input_path)

    st.subheader("Preview of Uploaded Dataset")
    st.write(df.head(20))
    st.caption(f"Rows: {len(df):,} | Columns: {df.shape[1]}")

    # Button to trigger cleaning
    if st.button("Run Initial Cleaning"):
        with st.spinner("Running cleaning script..."):
            try:
                subprocess.run(
                    ["python", "../cleaning/script.py"],
                    check=True
                )

                cleaned_path = "../cleaning/test.csv"

                if os.path.exists(cleaned_path):
                    cleaned = pd.read_csv(cleaned_path)

                    st.success("Cleaning completed!")
                    st.subheader("Cleaned Dataset Preview")
                    st.write(cleaned.head(20))
                    st.caption(f"Rows: {len(cleaned):,} | Columns: {cleaned.shape[1]}")

                    st.session_state["cleaned_df"] = cleaned
                else:
                    st.error("Cleaned file not found.")

            except subprocess.CalledProcessError:
                st.error("Cleaning script failed.")