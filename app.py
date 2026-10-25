from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODULES_ROOT = ROOT / "modules"
for import_root in (ROOT, MODULES_ROOT):
    if import_root.is_dir() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from module1_data_matrix.module1_data_matrix.data_loader import load_sales_data
from module1_data_matrix.module1_data_matrix.config import DAY_MAP
from module1_data_matrix.module1_data_matrix.matrix_builder import construct_matrices
from module1_data_matrix.module1_data_matrix.matrix_operations import reduced_row_echelon_form
from module2_demand_prediction import (
    SingularMatrixError,
    calculate_metrics,
    calculate_residuals,
    fit_least_squares,
    predict,
    predict_from_matrix,
)
from module2_demand_prediction.data.sample_prediction_input import FEATURE_ORDER
from module3_pattern_analysis import (
    analyze_eigenvalues,
    calculate_covariance,
    detect_patterns,
    prepare_sales_matrix,
)
from run_project import run_workflow

MODULE1_SALES_PATH = next(
    path
    for path in (
        ROOT / "data" / "sample_sales.csv",
        ROOT / "module1_data_matrix" / "module1_data_matrix" / "data" / "sample_sales.csv",
        MODULES_ROOT / "module1_data_matrix" / "module1_data_matrix" / "data" / "sample_sales.csv",
    )
    if path.exists()
)
MODULE4_INVENTORY_PATH = next(
    path
    for path in (
        ROOT / "data" / "inventory.csv",
        ROOT / "module4_inventory_optimization" / "data" / "sample_inventory.csv",
        MODULES_ROOT / "module4_inventory_optimization" / "data" / "sample_inventory.csv",
    )
    if path.exists()
)
SALES_INPUT_PATHS = {
    "Use input_1.csv": ROOT / "data" / "input_1.csv",
    "Use input_2.csv": ROOT / "data" / "input_2.csv",
    "Use input_3_dependent.csv": ROOT / "data" / "input_3_dependent.csv",
}

PAGES = [
    "Dashboard",
    "Input Data",
    "Data Matrix Analysis",
    "Demand Prediction",
    "Pattern Analysis",
    "Inventory Optimization",
    "Complete Analysis",
    "Project Information",
]
INPUT_OPTIONS = [
    "Use supplied sample dataset",
    *SALES_INPUT_PATHS,
    "Upload a sales CSV",
    "Edit inventory values manually",
]

st.set_page_config(page_title="Retail Demand Dashboard", page_icon="📈", layout="wide")


@st.cache_data
def load_default_sales_df() -> pd.DataFrame:
    return pd.read_csv(MODULE1_SALES_PATH)


@st.cache_data
def load_default_inventory_df() -> pd.DataFrame:
    return pd.read_csv(MODULE4_INVENTORY_PATH)


def load_sales_input(path: str) -> pd.DataFrame:
    # Avoid caching by path only; a changed CSV must become active immediately.
    return pd.read_csv(path)


def initialize_session_state() -> None:
    if st.session_state.get("selected_page") not in PAGES:
        st.session_state["selected_page"] = "Dashboard"
    if "sales_df" not in st.session_state:
        st.session_state["sales_df"] = load_default_sales_df()
    if "inventory_df" not in st.session_state:
        st.session_state["inventory_df"] = load_default_inventory_df()
    st.session_state["inventory_df"] = normalize_inventory_df(
        st.session_state["inventory_df"]
    )
    if st.session_state.get("input_source") not in INPUT_OPTIONS:
        st.session_state["input_source"] = INPUT_OPTIONS[0]


def save_input_source() -> None:
    st.session_state["input_source"] = st.session_state["input_source_widget"]


def normalize_inventory_df(value) -> pd.DataFrame:
    required = {"product", "current_stock", "safety_stock"}
    if isinstance(value, pd.DataFrame):
        inventory = value.copy()
    elif isinstance(value, dict) and {"edited_rows", "added_rows", "deleted_rows"} <= value.keys():
        inventory = load_default_inventory_df()
        for row_index, changes in value["edited_rows"].items():
            if 0 <= int(row_index) < len(inventory):
                for column, cell_value in changes.items():
                    if column in inventory.columns:
                        inventory.at[int(row_index), column] = cell_value
        if value["deleted_rows"]:
            inventory = inventory.drop(index=value["deleted_rows"]).reset_index(drop=True)
        if value["added_rows"]:
            inventory = pd.concat(
                [inventory, pd.DataFrame(value["added_rows"])],
                ignore_index=True,
            )
    else:
        inventory = pd.DataFrame(value)
    if not required.issubset(inventory.columns):
        raise ValueError(
            "Inventory input must contain product, current_stock, and safety_stock columns."
        )
    return inventory

DEFAULT_CURRENT_STOCK = {
    "Milk": 30,
    "Bread": 50,
    "Rice": 60,
    "Oil": 20,
    "Tomatoes": 30,
    "Potatoes": 50,
    "Apples": 60,
    "Onions": 20,
}


def sync_inventory_products_to_sales(sales_df: pd.DataFrame) -> None:
    """Use the active sales products while retaining their stock inputs by name."""
    products = sales_df["product"].astype(str).str.strip().drop_duplicates().tolist()
    stock_by_product = dict(DEFAULT_CURRENT_STOCK)
    stock_by_product.update(st.session_state.get("inventory_stock_by_product", {}))

    current_inventory = normalize_inventory_df(
        st.session_state.get("inventory_df", load_default_inventory_df())
    )
    for row in current_inventory.to_dict(orient="records"):
        product = str(row["product"]).strip()
        try:
            stock_by_product[product] = float(row["current_stock"])
        except (TypeError, ValueError):
            pass

    fallback_stocks = [30, 50, 60, 20]
    rows = [
        {
            "product": product,
            "current_stock": stock_by_product.get(
                product, fallback_stocks[index % len(fallback_stocks)]
            ),
            "safety_stock": 10,
        }
        for index, product in enumerate(products)
        if product
    ]
    st.session_state["inventory_stock_by_product"] = stock_by_product
    st.session_state["inventory_df"] = pd.DataFrame(
        rows, columns=["product", "current_stock", "safety_stock"]
    )

def validate_inventory_inputs(inventory_df: pd.DataFrame) -> pd.DataFrame:
    inventory = normalize_inventory_df(inventory_df)
    if inventory.empty:
        raise ValueError("Inventory input contains no products. Add inventory values before optimizing.")

    missing_products = inventory["product"].isna() | inventory["product"].astype(str).str.strip().eq("")
    if missing_products.any():
        raise ValueError("Every inventory row must have a product name.")

    numeric = inventory[["current_stock", "safety_stock"]].apply(pd.to_numeric, errors="coerce")
    invalid_values = numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=float))
    if invalid_values.any().any():
        missing_products = inventory.loc[invalid_values.any(axis=1), "product"].astype(str).tolist()
        raise ValueError(
            "Enter valid current stock and safety stock values for: "
            + ", ".join(missing_products)
            + ". Select ‘Edit inventory values manually’ to complete them."
        )
    if (numeric < 0).any().any():
        raise ValueError("Current stock and safety stock values cannot be negative.")
    inventory[["current_stock", "safety_stock"]] = numeric
    return inventory


def _save_uploaded_csv(uploaded_file) -> Path:
    if uploaded_file is None:
        raise ValueError("No file uploaded.")
    with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as handle:
        handle.write(uploaded_file.getvalue())
        temp_path = Path(handle.name)
    return temp_path


def _read_sales_csv_from_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return load_default_sales_df()
    text = uploaded_file.getvalue().decode("utf-8-sig")
    sales_df = pd.read_csv(io.StringIO(text))
    # Validate the candidate before it can replace the active dataset. Passing
    # the resulting records through Module 1 also checks headers and row values.
    _module1_records_from_csv(sales_df)
    return sales_df


def _module1_records_from_csv(source: str | pd.DataFrame | None = None, uploaded_file=None):
    if uploaded_file is not None:
        temp_path = _save_uploaded_csv(uploaded_file)
        try:
            return load_sales_data(str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)
    if isinstance(source, pd.DataFrame):
        buffer = io.StringIO()
        source.to_csv(buffer, index=False)
        temp_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="sales_", mode="wb").name)
        temp_path.write_bytes(buffer.getvalue().encode("utf-8-sig"))
        try:
            return load_sales_data(str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)
    return load_sales_data(str(Path(source) if source else MODULE1_SALES_PATH))


def _module1_outputs(records):
    A, b = construct_matrices(records)
    matrix = np.asarray(A, dtype=float)
    target = np.asarray(b, dtype=float)
    transpose = matrix.T.copy()
    at_a = transpose @ matrix
    at_b = transpose @ target
    if matrix.shape[0] < matrix.shape[1]:
        return {
            "A": matrix,
            "b": target,
            "A_transpose": transpose,
            "AtA": at_a,
            "Atb": at_b,
            "rank": int(np.linalg.matrix_rank(matrix)),
            "fit_error": (
                f"A must have at least as many observations as features; "
                f"received {matrix.shape[0]} rows for {matrix.shape[1]} columns."
            ),
        }
    try:
        result = fit_least_squares(matrix, target)
    except SingularMatrixError as exc:
        return {
            "A": matrix,
            "b": target,
            "A_transpose": transpose,
            "AtA": at_a,
            "Atb": at_b,
            "rank": int(np.linalg.matrix_rank(matrix)),
            "fit_error": str(exc),
        }
    predictions = predict_from_matrix(matrix, result.coefficients)
    residuals = calculate_residuals(target, predictions)
    metrics = calculate_metrics(target, predictions)
    return {
        "A": matrix,
        "b": target,
        "coefficients": result.coefficients,
        "A_transpose": result.A_transpose,
        "AtA": result.AtA,
        "Atb": result.Atb,
        "rank": result.rank,
        "predicted": predictions,
        "residuals": residuals,
        "metrics": metrics,
        "fit_error": None,
    }


def current_analysis_report():
    input_error = st.session_state.get("sales_input_error")
    if input_error:
        raise ValueError(f"The current sales input is invalid: {input_error}")
    sales_df = st.session_state.get("sales_df", load_default_sales_df())
    sync_inventory_products_to_sales(sales_df)
    inventory_df = validate_inventory_inputs(
        st.session_state.get("inventory_df", load_default_inventory_df())
    )
    records = _module1_records_from_csv(sales_df)
    model_outputs = _module1_outputs(records)
    if model_outputs["fit_error"]:
        raise SingularMatrixError(model_outputs["fit_error"])
    return run_workflow(
        sales_records=records,
        inventory_records=inventory_df.to_dict(orient="records"),
    )


def model_fit_guidance(error: Exception) -> str:
    return (
        "A unique demand model needs sales from multiple weekdays, with price and "
        "promotion varying independently so the design matrix has rank 4. "
        f"Details: {error}"
    )


def metric_card(label: str, value: str, caption: str | None = None) -> None:
    html = (
        f"""
        <div style="border:1px solid #dfe3e8; border-radius:10px; padding:12px; margin-bottom:12px; background:#f8fafc;">
            <div style="font-size:0.75rem; color:#475467; text-transform:uppercase; letter-spacing:0.04em;">{label}</div>
            <div style="font-size:1.75rem; font-weight:700; color:#0f172a; margin-top:4px;">{value}</div>
            {f'<div style="font-size:0.8rem; color:#667085; margin-top:4px;">{caption}</div>' if caption else ''}
        </div>
        """
    )
    st.markdown(html, unsafe_allow_html=True)


def rounded_result_frame(frame: pd.DataFrame) -> pd.DataFrame:
    displayed = frame.copy()
    numeric_columns = displayed.select_dtypes(include="number").columns
    for column in numeric_columns:
        displayed[column] = displayed[column].round().astype("Int64")
    return displayed


def show_result_table(frame: pd.DataFrame, **kwargs) -> None:
    st.dataframe(rounded_result_frame(frame), **kwargs)


def select_page(page_name: str) -> None:
    st.session_state["selected_page"] = page_name


def dashboard_page() -> None:
    st.title("Demand Prediction and Inventory Optimization for Value Retailers Using Linear Algebra")
    st.caption("Combines sales data, demand forecasts, pattern analysis, and stock recommendations for retail products.")
    st.write(
        "This workflow loads historical retail sales, builds a linear-algebra model, predicts future demand, analyzes sales patterns, and produces inventory recommendations."
    )

    workflow = [
        "Historical Sales Data",
        "Matrix Construction",
        "Demand Prediction",
        "Pattern Analysis",
        "Inventory Recommendations",
    ]
    st.markdown(" → ".join(workflow))

    sales_df = st.session_state.get("sales_df", load_default_sales_df())
    inventory_df = st.session_state.get("inventory_df", load_default_inventory_df())
    pattern_day_count = sales_df["day"].nunique() if "day" in sales_df.columns else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Sales records", str(len(sales_df)), "Currently selected dataset")
    with c2:
        metric_card("Inventory products", str(len(inventory_df)), "Current inventory inputs")
    with c3:
        metric_card("Sales weekdays", str(pattern_day_count), "Pattern analysis from selected sales")

    st.subheader("Module overview")
    st.caption("Each module handles one stage of the retail analytics workflow.")
    cols = st.columns(4)
    module_cards = [
        ("Data Matrix Analysis", "Constructs the design matrix A and target vector b from historical sales data.", "Module 1"),
        ("Demand Prediction", "Fits a least-squares demand model and forecasts future demand.", "Module 2"),
        ("Pattern Analysis", "Calculates covariance and eigenstructure to identify dominant sales patterns.", "Module 3"),
        ("Inventory Optimization", "Turns demand forecasts and stock data into reorder recommendations.", "Module 4"),
    ]
    for idx, (title, text, label) in enumerate(module_cards):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="border:1px solid #e5e7eb; border-radius:12px; padding:16px; background:#ffffff; height:100%;">
                    <div style="font-size:0.8rem; color:#475467; margin-bottom:6px;">{label}</div>
                    <div style="font-size:1.2rem; font-weight:700; color:#111827; margin-bottom:8px;">{title}</div>
                    <div style="font-size:0.92rem; color:#4b5563; line-height:1.5;">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Project flow")
    st.caption("Historical sales are transformed into forecasts, then matched with stock data to recommend replenishment.")
    st.write("Historical sales data → Data matrix construction → Demand forecasting → Pattern analysis → Inventory recommendations")

    st.button(
        "Run complete analysis",
        use_container_width=True,
        on_click=select_page,
        args=("Complete Analysis",),
    )

    st.button(
        "Open input data",
        use_container_width=True,
        on_click=select_page,
        args=("Input Data",),
    )


def input_data_page() -> None:
    st.title("Input Data")
    st.caption("Select weekday-based sales data or review the product stock and safety-stock values.")
    st.write("The workflow expects historical sales records with the required columns used by Module 1, plus stock and safety stock values for Module 4.")

    if "input_source_widget" not in st.session_state:
        st.session_state["input_source_widget"] = st.session_state["input_source"]
    option = st.radio(
        "Choose an input source",
        INPUT_OPTIONS,
        key="input_source_widget",
        on_change=save_input_source,
    )

    if option == "Use supplied sample dataset":
        sales_df = load_default_sales_df()
        st.session_state["sales_df"] = sales_df
        st.session_state.pop("sales_input_error", None)
        st.success(f"Loaded sample dataset: {MODULE1_SALES_PATH.name}")
        st.dataframe(sales_df.head(10), use_container_width=True)
    elif option in SALES_INPUT_PATHS:
        try:
            sales_df = load_sales_input(str(SALES_INPUT_PATHS[option]))
            _module1_records_from_csv(sales_df)
            st.session_state["sales_df"] = sales_df
            st.session_state.pop("sales_input_error", None)
            st.success(f"Loaded dataset: {SALES_INPUT_PATHS[option].name}")
            st.dataframe(sales_df.head(10), use_container_width=True)
            model_outputs = _module1_outputs(_module1_records_from_csv(sales_df))
            if model_outputs["fit_error"]:
                st.warning(model_fit_guidance(ValueError(model_outputs["fit_error"])))
        except Exception as exc:
            st.session_state["sales_input_error"] = str(exc)
            st.error(f"The selected dataset could not be processed: {exc}")
    elif option == "Upload a sales CSV":
        uploaded = st.file_uploader("Upload a sales CSV", type=["csv"], key="sales_upload")
        if uploaded is not None:
            try:
                sales_df = _read_sales_csv_from_upload(uploaded)
                st.session_state["sales_df"] = sales_df
                st.session_state.pop("sales_input_error", None)
                st.success("CSV loaded successfully.")
                st.dataframe(sales_df.head(10), use_container_width=True)
            except Exception as exc:
                # Keep the last valid input available for an intentional
                # re-selection, but block every integrated result until a
                # valid input is selected again.
                st.session_state["sales_input_error"] = str(exc)
                st.error(f"The uploaded CSV could not be processed: {exc}")
        else:
            st.caption("Currently selected sales data:")
            st.dataframe(st.session_state["sales_df"].head(10), use_container_width=True)
    else:
        inventory_df = normalize_inventory_df(st.session_state["inventory_df"])
        edited = st.data_editor(
            inventory_df,
            use_container_width=True,
            num_rows="dynamic",
            key="inventory_editor_widget_v2",
        )
        st.session_state["inventory_df"] = normalize_inventory_df(edited)
        st.dataframe(edited, use_container_width=True)

    if st.session_state.get("sales_input_error"):
        st.error(
            "The current sales upload is invalid. Select a valid dataset before "
            "running analysis."
        )
    else:
        sales_df = st.session_state.get("sales_df", load_default_sales_df())
        try:
            records = _module1_records_from_csv(sales_df)
            st.info("Sales schema is valid for Module 1.")
            outputs = _module1_outputs(records)
            if outputs["fit_error"]:
                st.warning(model_fit_guidance(ValueError(outputs["fit_error"])))
        except Exception as exc:
            st.error(f"Sales schema validation failed: {exc}")


def data_matrix_page() -> None:
    st.title("Data Matrix Analysis")
    st.caption("Converts each sales record into numeric model inputs and a target value.")
    sales_df = st.session_state.get("sales_df", load_default_sales_df())
    records = _module1_records_from_csv(sales_df)
    outputs = _module1_outputs(records)
    A = outputs["A"]
    b = outputs["b"]

    st.subheader("Historical sales preview")
    st.caption("The sales rows currently selected for analysis.")
    st.dataframe(sales_df.head(10), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Matrix A")
        st.caption("Predictor matrix: each row is a sale, with an intercept, day, price, and promotion.")
        show_result_table(pd.DataFrame(A), use_container_width=True, hide_index=True)
    with col_b:
        st.write("Target vector b")
        st.caption("Observed sales values the model learns to predict.")
        show_result_table(pd.DataFrame(b), use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("A shape", f"{A.shape[0]} × {A.shape[1]}")
    with c2:
        metric_card("b shape", f"{b.shape[0]}")
    with c3:
        metric_card("AᵀA shape", f"{(A.T @ A).shape[0]} × {(A.T @ A).shape[1]}")
    with c4:
        metric_card("Rank", str(outputs["rank"]))

    if outputs["fit_error"]:
        st.warning(model_fit_guidance(ValueError(outputs["fit_error"])))

    st.subheader("Normal equations")
    st.caption("Matrix products used by least-squares regression to estimate model coefficients.")
    st.write("Aᵀ")
    show_result_table(pd.DataFrame(outputs["A_transpose"]), use_container_width=True, hide_index=True)
    st.write("AᵀA")
    show_result_table(pd.DataFrame(outputs["AtA"]), use_container_width=True, hide_index=True)
    st.write("Aᵀb")
    show_result_table(pd.DataFrame(outputs["Atb"]), use_container_width=True, hide_index=True)

    st.subheader("Reduced row echelon form example")
    st.caption("A row-reduced matrix form used to expose linear-system structure.")
    example = [[1.0, 2.0], [2.0, 4.0]]
    show_result_table(
        pd.DataFrame(reduced_row_echelon_form(example)),
        use_container_width=True,
        hide_index=True,
    )


def demand_prediction_page() -> None:
    st.title("Demand Prediction")
    st.caption("Fits a least-squares model to the selected sales data and compares predicted demand with observed sales.")
    sales_df = st.session_state.get("sales_df", load_default_sales_df())
    records = _module1_records_from_csv(sales_df)
    outputs = _module1_outputs(records)
    if outputs["fit_error"]:
        st.warning(model_fit_guidance(ValueError(outputs["fit_error"])))
        return
    target = outputs["b"]
    predictions = outputs["predicted"]
    residuals = outputs["residuals"]
    metrics = outputs["metrics"]
    coefficients = outputs["coefficients"]

    comparison = pd.DataFrame(
        {
            "Product": [record["product"] for record in records],
            "Actual demand": np.asarray(target).ravel(),
            "Predicted demand": np.asarray(predictions).ravel(),
            "Residual": np.asarray(residuals).ravel(),
        }
    )
    show_result_table(comparison, use_container_width=True)
    st.line_chart(rounded_result_frame(comparison[["Actual demand", "Predicted demand"]]))

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("MAE", f"{metrics.mae:.0f}")
    with c2:
        metric_card("MSE", f"{metrics.mse:.0f}")
    with c3:
        metric_card("RMSE", f"{metrics.rmse:.0f}")

    st.subheader("Future forecast")
    st.caption("Estimate demand for a chosen weekday, price, and promotion setting using the fitted coefficients.")
    day = st.slider("Day", min_value=1, max_value=14, value=7, key="forecast_day")
    price = st.slider("Price", min_value=20.0, max_value=200.0, value=55.0, step=0.5, key="forecast_price")
    promotion = st.selectbox("Promotion", [0, 1], index=0, key="forecast_promotion")
    future_features = {"day": float(day), "price": float(price), "promotion": float(promotion)}
    future_value = predict(coefficients, future_features, feature_order=FEATURE_ORDER)
    st.metric("Projected demand", f"{future_value:.0f} units")

    st.subheader("Model coefficients")
    st.caption("Learned weights showing how the intercept and each input feature contribute to predicted sales.")
    show_result_table(
        pd.DataFrame({"Coefficient": coefficients.ravel()}),
        use_container_width=True,
    )


def pattern_analysis_page() -> None:
    st.title("Pattern Analysis")
    st.caption("Examines how product sales from the currently selected sales data move together using covariance and eigen-analysis. Missing product/day rows count as zero recorded sales.")
    sales_df = st.session_state.get("sales_df", load_default_sales_df())
    records = _module1_records_from_csv(sales_df)
    product_columns = list(dict.fromkeys(record["product"] for record in records))
    weekday_numbers = sorted({DAY_MAP[record["day"].strip().lower()] for record in records})
    grouped_sales = {
        product: [
            sum(
                float(record["sales"])
                for record in records
                if record["product"] == product
                and DAY_MAP[record["day"].strip().lower()] == weekday
            )
            for weekday in weekday_numbers
        ]
        for product in product_columns
    }
    pattern_df = pd.DataFrame({"day": weekday_numbers, **grouped_sales})

    st.subheader("Prepared sales matrix")
    st.caption("Product sales observations arranged by day before calculating their relationships.")
    show_result_table(pattern_df, use_container_width=True)

    if len(weekday_numbers) < 2:
        st.warning("Pattern analysis needs sales recorded on at least two different weekdays.")
        return

    prepared = prepare_sales_matrix(grouped_sales, product_columns)
    covariance = calculate_covariance(prepared.centered_matrix)
    eigen = analyze_eigenvalues(covariance.covariance_matrix)
    pattern = detect_patterns(eigen, product_columns)

    st.subheader("Covariance matrix")
    st.caption("Measures how pairs of product sales vary together; larger magnitudes indicate stronger co-movement.")
    show_result_table(
        pd.DataFrame(
            covariance.covariance_matrix,
            columns=product_columns,
            index=product_columns,
        ),
        use_container_width=True,
    )

    st.subheader("Eigenvalues and explained variance")
    st.caption("Eigenvalues rank the dominant sales patterns; explained variance shows each pattern's share of total variation.")
    eigen_df = pd.DataFrame(
        {
            "Eigenvalue": eigen.eigenvalues,
            "Explained variance ratio": eigen.explained_variance,
            "Cumulative explained variance": eigen.cumulative_explained_variance,
            "Explained variance (%)": eigen.explained_variance_percentages,
            "Cumulative explained variance (%)": eigen.cumulative_explained_variance_percentages,
        }
    )
    show_result_table(eigen_df, use_container_width=True)
    st.bar_chart(
        rounded_result_frame(eigen_df.set_index("Eigenvalue")[["Explained variance (%)"]])
    )

    st.subheader("Dominant pattern summary")
    st.caption("Relative contribution of each product to the leading sales pattern.")
    st.write(
        {
            product: round(float(value))
            for product, value in pattern.product_contributions.items()
        }
    )


def inventory_optimization_page() -> None:
    st.title("Inventory Optimization")
    st.caption("Uses forecasts from the currently selected sales data and compares them with current stock and safety stock.")
    try:
        report = current_analysis_report()
    except SingularMatrixError as exc:
        st.warning(model_fit_guidance(exc))
        return
    except ValueError as exc:
        st.error(str(exc))
        return
    st.caption(f"Active sales input: {len(report['records'])} rows.")
    recommendations = report["recommendations"]

    summary_df = pd.DataFrame(
        [
            {
                "Product": rec.product_name,
                "Predicted demand": rec.predicted_demand,
                "Forecast basis": report["forecast_basis"][index],
                "Current stock": rec.current_stock,
                "Safety stock": rec.safety_stock,
                "Required stock": rec.required_stock,
                "Recommended order": rec.recommended_order_quantity,
                "Status": rec.stock_status,
            }
            for index, rec in enumerate(recommendations)
        ]
    )
    show_result_table(summary_df, use_container_width=True)
    st.bar_chart(
        rounded_result_frame(
            summary_df.set_index("Product")[
                ["Current stock", "Required stock", "Recommended order"]
            ]
        )
    )

    for rec in recommendations:
        st.info(
            f"{rec.product_name}: {rec.stock_status}. Predicted demand "
            f"{rec.predicted_demand:.0f}, required stock {rec.required_stock:.0f}, "
            f"recommended order {rec.recommended_order_quantity:.0f}."
        )


def complete_analysis_page() -> None:
    st.title("Complete Analysis")
    st.caption("Runs the selected sales and inventory inputs through the integrated module workflow.")
    if st.button("Execute integrated workflow", use_container_width=True):
        try:
            report = current_analysis_report()
            fit = report["fit"]
            metrics = report["metrics"]
            pattern = report["pattern"]
            recommendations = report["recommendations"]
            forecast = report["forecast"]

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Matrix rank", str(fit.rank))
            with c2:
                metric_card("MAE", f"{metrics.mae:.0f}")
            with c3:
                metric_card("Total variance", f"{report['eigen'].total_variance:.0f}")
            with c4:
                metric_card(
                    f"Forecast: {report['product_names'][0]}",
                    f"{forecast[0]:.0f} units",
                    "First product forecast; see table for all products",
                )

            st.write("### Inventory recommendations")
            st.caption("For each product, required stock is forecast demand plus safety stock; the order fills any shortfall.")
            forecast_df = pd.DataFrame(
                [
                    {
                        "Product": product,
                        "Forecast day": scenario["day"],
                        "Forecast price": scenario["price"],
                        "Promotion": int(scenario["promotion"]),
                        "Forecast basis": basis,
                        "Predicted demand": demand,
                    }
                    for product, scenario, basis, demand in zip(
                        report["product_names"],
                        report["future_scenarios"],
                        report["forecast_basis"],
                        forecast,
                    )
                ]
            )
            show_result_table(forecast_df, use_container_width=True)
            show_result_table(
                pd.DataFrame(
                    [
                        {
                            "Product": rec.product_name,
                            "Predicted demand": rec.predicted_demand,
                            "Current stock": rec.current_stock,
                            "Safety stock": rec.safety_stock,
                            "Required stock": rec.required_stock,
                            "Recommended order": rec.recommended_order_quantity,
                            "Status": rec.stock_status,
                        }
                        for rec in recommendations
                    ]
                ),
                use_container_width=True,
            )

            st.write("### Pattern summary")
            st.caption("Highlights the products contributing most to the leading sales-variation pattern.")
            show_result_table(
                pd.DataFrame(
                    {
                        "day": report["pattern_days"],
                        **report["pattern_sales_data"],
                    }
                ),
                use_container_width=True,
            )
            st.write(
                {
                    product: round(float(value))
                    for product, value in pattern.product_contributions.items()
                }
            )
            st.caption(f"Demand model input: {len(report['records'])} sales rows from the currently selected dataset.")
            st.success("Complete project workflow executed successfully using the real module implementations.")
        except SingularMatrixError as exc:
            st.error(f"Complete analysis failed. {model_fit_guidance(exc)}")
        except Exception as exc:
            st.error(f"Complete analysis failed: {exc}")
    else:
        st.info("Use the button above to execute the end-to-end workflow.")


def project_information_page() -> None:
    st.title("Project Information")
    st.caption("Overview of the project's purpose, dependencies, and local run commands.")
    st.markdown(
        """
        This dashboard presents an existing retail analytics project based on linear algebra. It uses the validated module implementations to:
        - load and validate historical sales records
        - build a design matrix and target vector
        - fit least-squares demand coefficients
        - analyze covariance and eigenstructure
        - generate inventory recommendations

        The project is designed to work offline using the included sample data or compatible CSV uploads.
        """
    )

    st.write("### Required dependencies")
    st.caption("Python packages needed to run the application and its tests.")
    st.code("Python 3.10+, NumPy, pandas, Streamlit, pytest")

    st.write("### Launch commands")
    st.caption("Run these commands from the project root in PowerShell.")
    st.code("cd \"C:\\Users\\ABHI\\Downloads\"\npython -m streamlit run app.py --server.headless true --server.port 8501")


def render_page(page_name: str) -> None:
    page_map = {
        "Dashboard": dashboard_page,
        "Input Data": input_data_page,
        "Data Matrix Analysis": data_matrix_page,
        "Demand Prediction": demand_prediction_page,
        "Pattern Analysis": pattern_analysis_page,
        "Inventory Optimization": inventory_optimization_page,
        "Complete Analysis": complete_analysis_page,
        "Project Information": project_information_page,
    }
    page_map.get(page_name, dashboard_page)()


if __name__ == "__main__":
    initialize_session_state()
    st.sidebar.title("Retail Analytics")
    st.sidebar.caption("Value retailers using linear algebra")
    selected = st.sidebar.radio("Navigation", PAGES, key="selected_page")
    render_page(selected)
