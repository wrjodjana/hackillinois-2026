from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    columns: List[str]
    numeric_columns: List[str]
    n_rows: int


class RegressionRequest(BaseModel):
    dataset_id: str
    target: str
    features: List[str]
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    random_state: int = 42
    standardize: bool = False
    dropna: bool = True


class RegressionMetrics(BaseModel):
    r2: float
    mae: float
    mse: float
    rmse: float


class RegressionResponse(BaseModel):
    dataset_id: str
    target: str
    features: List[str]
    n_rows_input: int
    n_rows_used: int
    n_rows_dropped: int
    metrics: RegressionMetrics
    intercept: float
    coefficients: Dict[str, float]
    warnings: List[str] = Field(default_factory=list)

    # base64 PNGs
    plot_actual_vs_pred: str
    plot_residuals: str
    plot_coefficients: str


class ExplainRequest(BaseModel):
    # send only deterministic summary, not raw data
    summary: Dict[str, Any]


class ExplainResponse(BaseModel):
    narrative_md: str