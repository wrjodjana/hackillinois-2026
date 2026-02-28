import streamlit as st
import pandas as pd
import requests
import base64
from io import BytesIO
from PIL import Image

API_BASE = "http://127.0.0.1:8000" #local

st.set_page_config(page_title="Linear Regression", layout="wide")
st.title("Linear Regression (FastAPI)")

def b64_to_image(b64_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64_str)))

# 0) Require cleaned_df from the cleaning page
cleaned_df = st.session_state.get("cleaned_df")
if cleaned_df is None:
    st.warning("No cleaned dataset found. Go to the Cleaning page first and run cleaning.")
    st.stop()

st.caption(f"Cleaned dataset in memory: {len(cleaned_df):,} rows × {cleaned_df.shape[1]} columns")

# 1) Upload cleaned_df to backend (once per session)
if "dataset_id" not in st.session_state:
    st.session_state["dataset_id"] = None
if "numeric_columns" not in st.session_state:
    st.session_state["numeric_columns"] = None

st.subheader("1) Send cleaned data to backend")

colA, colB = st.columns([1, 2])
with colA:
    if st.button("Upload cleaned dataset to FastAPI"):
        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        files = {"file": ("cleaned.csv", csv_bytes, "text/csv")}
        resp = requests.post(f"{API_BASE}/dataset/upload", files=files, timeout=60)
        if resp.status_code != 200:
            st.error(f"Upload failed: {resp.status_code} {resp.text}")
        else:
            data = resp.json()
            st.session_state["dataset_id"] = data["dataset_id"]
            st.session_state["numeric_columns"] = data["numeric_columns"]
            st.success(f"Uploaded! dataset_id = {data['dataset_id']}")

with colB:
    if st.session_state["dataset_id"]:
        st.info(f"Current dataset_id: {st.session_state['dataset_id']}")
    else:
        st.info("Not uploaded yet.")

dataset_id = st.session_state["dataset_id"]
numeric_cols = st.session_state["numeric_columns"] or []

st.divider()

# 2) Configure regression
st.subheader("2) Choose X and Y")

if not dataset_id:
    st.warning("Upload cleaned dataset first.")
    st.stop()

if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric columns. (Your backend only supports numeric regression for now.)")
    st.write("Numeric columns detected:", numeric_cols)
    st.stop()

left, right = st.columns(2)

with left:
    target = st.selectbox("Target (y)", options=numeric_cols, index=len(numeric_cols)-1)

with right:
    feature_options = [c for c in numeric_cols if c != target]
    default_features = feature_options[: min(4, len(feature_options))]
    features = st.multiselect("Features (X)", options=feature_options, default=default_features)

test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
standardize = st.checkbox("Standardize features", value=False)
dropna = st.checkbox("Drop rows with nulls in selected columns", value=True)

st.divider()

# 3) Run regression
st.subheader("3) Run")

if st.button("Run Linear Regression"):
    if not features:
        st.error("Select at least 1 feature.")
        st.stop()

    payload = {
        "dataset_id": dataset_id,
        "target": target,
        "features": features,
        "test_size": float(test_size),
        "random_state": 42,
        "standardize": bool(standardize),
        "dropna": bool(dropna),
    }

    with st.spinner("Running regression..."):
        resp = requests.post(f"{API_BASE}/regression/linear", json=payload, timeout=120)

    if resp.status_code != 200:
        st.error(f"Regression failed: {resp.status_code} {resp.text}")
        st.stop()

    out = resp.json()

    # Metrics
    m = out["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{m['r2']:.4f}")
    c2.metric("MAE", f"{m['mae']:.4f}")
    c3.metric("RMSE", f"{m['rmse']:.4f}")
    c4.metric("Rows used", f"{out['n_rows_used']}")

    st.caption(f"Rows dropped: {out['n_rows_dropped']} (input: {out['n_rows_input']})")

    if out.get("warnings"):
        st.warning(" | ".join(out["warnings"]))

    # Coefficients table
    coeffs_df = pd.DataFrame(
        [{"feature": k, "coefficient": v} for k, v in out["coefficients"].items()]
    ).sort_values("coefficient", ascending=False)

    st.subheader("Coefficients")
    st.dataframe(coeffs_df, use_container_width=True)

    # Plots
    st.subheader("Plots")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("**Actual vs Predicted**")
        st.image(b64_to_image(out["plot_actual_vs_pred"]), use_container_width=True)
    with p2:
        st.markdown("**Residuals vs Predicted**")
        st.image(b64_to_image(out["plot_residuals"]), use_container_width=True)
    with p3:
        st.markdown("**Coefficients**")
        st.image(b64_to_image(out["plot_coefficients"]), use_container_width=True)