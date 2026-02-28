from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from models import (
    UploadResponse,
    RegressionRequest,
    RegressionResponse,
    RegressionMetrics,
    ExplainRequest,
    ExplainResponse,
)
from storage import new_dataset_id, save_dataset_bytes, get_dataset_path
from regression import run_linear_regression

app = FastAPI(title="Regression Backend (Deterministic)")

# allow Streamlit localhost calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/dataset/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    dataset_id = new_dataset_id()
    path = save_dataset_bytes(dataset_id, file.filename, content)

    df = pd.read_csv(path)
    cols = list(df.columns)
    numeric_cols = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        # consider numeric if at least 90% of values can be parsed as numbers
        if s.notna().mean() >= 0.9:
            numeric_cols.append(c)

    return UploadResponse(
        dataset_id=dataset_id,
        filename=file.filename,
        columns=cols,
        numeric_columns=numeric_cols,
        n_rows=int(len(df)),
    )


@app.post("/regression/linear", response_model=RegressionResponse)
def linear_regression(req: RegressionRequest):
    path = get_dataset_path(req.dataset_id)
    df = pd.read_csv(path)

    result = run_linear_regression(
        df=df,
        target=req.target,
        features=req.features,
        test_size=req.test_size,
        random_state=req.random_state,
        standardize=req.standardize,
        dropna=req.dropna,
    )

    return RegressionResponse(
        dataset_id=req.dataset_id,
        target=req.target,
        features=req.features,
        n_rows_input=result.n_rows_input,
        n_rows_used=result.n_rows_used,
        n_rows_dropped=result.n_rows_dropped,
        metrics=RegressionMetrics(**result.metrics),
        intercept=result.intercept,
        coefficients=result.coefs,
        warnings=result.warnings,
        plot_actual_vs_pred=result.plots_b64["actual_vs_pred"],
        plot_residuals=result.plots_b64["residuals"],
        plot_coefficients=result.plots_b64["coefficients"],
    )


