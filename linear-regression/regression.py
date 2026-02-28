from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


@dataclass
class RegressionResult:
    n_rows_input: int
    n_rows_used: int
    n_rows_dropped: int
    metrics: Dict[str, float]
    intercept: float
    coefs: Dict[str, float]
    warnings: List[str]
    plots_b64: Dict[str, str]


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _validate_columns(df: pd.DataFrame, target: str, features: List[str]) -> List[str]:
    errors = []
    if target not in df.columns:
        errors.append(f"Target column '{target}' not found.")
    for c in features:
        if c not in df.columns:
            errors.append(f"Feature column '{c}' not found.")
    if target in features:
        errors.append("Target cannot be included as a feature.")
    if len(features) == 0:
        errors.append("At least 1 feature is required.")
    return errors


def _numeric_only(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    # Try convert to numeric; if too many NaNs introduced, warn.
    warnings = []
    out = df.copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(out[c]):
            before_nan = out[c].isna().mean()
            out[c] = pd.to_numeric(out[c], errors="coerce")
            after_nan = out[c].isna().mean()
            if after_nan > before_nan + 0.10:
                warnings.append(f"Column '{c}' became more null after numeric coercion (check values).")
    return out, warnings


def run_linear_regression(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    test_size: float,
    random_state: int,
    standardize: bool,
    dropna: bool,
) -> RegressionResult:
    warnings: List[str] = []

    errors = _validate_columns(df, target, features)
    if errors:
        raise ValueError(" | ".join(errors))

    # coerce to numeric where needed
    df2, w = _numeric_only(df, [target] + features)
    warnings.extend(w)

    n_rows_input = int(len(df2))

    # drop rows with NA in required columns
    if dropna:
        before = len(df2)
        df2 = df2.dropna(subset=[target] + features)
        after = len(df2)
        dropped = before - after
    else:
        # sklearn can't handle NaN, so we still have to drop; just make it explicit
        before = len(df2)
        df2 = df2.dropna(subset=[target] + features)
        after = len(df2)
        dropped = before - after
        if dropped > 0:
            warnings.append("dropna=false requested, but rows with NaN were dropped (LinearRegression cannot handle NaN).")

    n_rows_used = int(len(df2))
    n_rows_dropped = int(dropped)

    if n_rows_used < 20:
        warnings.append("Very small dataset after dropping nulls (<20 rows). Metrics may be unstable.")

    X = df2[features].to_numpy()
    y = df2[target].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if standardize:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))

    # extract coefficients
    if standardize:
        lr = model.named_steps["lr"]
        intercept = float(lr.intercept_)
        coef_arr = lr.coef_
    else:
        intercept = float(model.intercept_)
        coef_arr = model.coef_

    coefs = {features[i]: float(coef_arr[i]) for i in range(len(features))}

    # quick multicollinearity-ish warning
    if len(features) >= 2:
        corr = np.corrcoef(X.T)
        high = np.where(np.abs(corr) > 0.95)
        # ignore diagonal
        pairs = [(i, j) for i, j in zip(high[0], high[1]) if i < j]
        if pairs:
            warnings.append("Some features are highly correlated (>|0.95|). Coefficients may be unstable.")

    # --- Plots (matplotlib, deterministic) ---
    # 1) actual vs predicted
    fig1 = plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")

    # 2) residuals vs predicted
    residuals = y_test - y_pred
    fig2 = plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs Predicted")

    # 3) coefficients (sorted by abs magnitude)
    items = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig3 = plt.figure()
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), names, rotation=45, ha="right")
    plt.ylabel("Coefficient")
    plt.title("Linear Regression Coefficients")

    plots_b64 = {
        "actual_vs_pred": _fig_to_b64(fig1),
        "residuals": _fig_to_b64(fig2),
        "coefficients": _fig_to_b64(fig3),
    }

    return RegressionResult(
        n_rows_input=n_rows_input,
        n_rows_used=n_rows_used,
        n_rows_dropped=n_rows_dropped,
        metrics={"r2": r2, "mae": mae, "mse": mse, "rmse": rmse},
        intercept=intercept,
        coefs=coefs,
        warnings=warnings,
        plots_b64=plots_b64,
    )
