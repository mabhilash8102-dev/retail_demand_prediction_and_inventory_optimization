import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_run_workflow_uses_supplied_sales_records():
    script = """
import csv
from run_project import MODULE1_SALES_CSV, MODULE4_INVENTORY_CSV, run_workflow

with MODULE1_SALES_CSV.open("r", newline="", encoding="utf-8-sig") as handle:
    records = list(csv.DictReader(handle))
baseline = run_workflow()
updated_records = [
    {**record, "sales": float(record["sales"]) + 25.0}
    for record in records
]
with MODULE4_INVENTORY_CSV.open("r", newline="", encoding="utf-8-sig") as handle:
    inventory_records = list(csv.DictReader(handle))
inventory_records[0]["current_stock"] = 1000.0
report = run_workflow(
    sales_records=updated_records,
    inventory_records=inventory_records,
)

assert len(report["records"]) == len(updated_records)
assert all(
    actual["sales"] == expected["sales"]
    for actual, expected in zip(report["records"], updated_records)
)
assert report["forecast"] != baseline["forecast"]
assert report["inventory_result"].predicted_demand.tolist() == report["forecast"]
assert report["current_stock"][0] == 1000.0
assert report["recommendations"][0].current_stock == 1000.0
assert report["pattern_sales_data"] != baseline["pattern_sales_data"]

custom_products = [
    {**record, "product": "Custom-" + record["product"]}
    for record in updated_records
]
fallback_report = run_workflow(
    sales_records=custom_products,
    inventory_records=inventory_records,
)
assert fallback_report["pattern_product_names"] == list(
    dict.fromkeys(record["product"] for record in custom_products)
)
assert len(fallback_report["forecast"]) == len(inventory_records)
assert all(
    basis == "Latest sales row (product absent from sales input)"
    for basis in fallback_report["forecast_basis"]
)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_dashboard_module_3_module_4_and_complete_analysis_use_selected_inputs():
    script = """
import pandas as pd
from streamlit.testing.v1 import AppTest
from run_project import ROOT, run_workflow

sales = pd.read_csv(ROOT / "data" / "sample_sales.csv")
sales["sales"] += 25
inventory = pd.DataFrame({
    "product": ["Rice", "Oil", "Soap", "Milk"],
    "predicted_demand": [120, 90, 160, 180],
    "current_stock": [30, 50, 60, 20],
    "safety_stock": [10, 10, 10, 10],
})
inventory.loc[0, "current_stock"] = 1000
full_precision_report = run_workflow(
    sales_records=sales.to_dict(orient="records"),
    inventory_records=inventory.to_dict(orient="records"),
)
assert any(value != round(value) for value in full_precision_report["forecast"])

app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
app.session_state["sales_df"] = sales
app.session_state["inventory_df"] = inventory
app.session_state["selected_page"] = "Pattern Analysis"
app.run()
assert not app.exception, app.exception
pattern_matrix = app.dataframe[0].value
assert pattern_matrix["Milk"].sum() == sales.loc[sales["product"] == "Milk", "sales"].sum()

app.sidebar.radio[0].set_value("Inventory Optimization").run()
assert not app.exception, app.exception
inventory_output = app.dataframe[0].value
assert inventory_output.loc[0, "Current stock"] == 1000
assert inventory_output.loc[0, "Predicted demand"] != inventory.loc[0, "predicted_demand"]

app.sidebar.radio[0].set_value("Complete Analysis").run()
assert not app.exception, app.exception
next(button for button in app.button if button.label == "Execute integrated workflow").click().run()
assert not app.exception, app.exception
forecast_output = app.dataframe[0].value
recommendation_output = app.dataframe[1].value
complete_pattern = app.dataframe[2].value
assert forecast_output["Product"].tolist() == inventory["product"].tolist()
assert recommendation_output.loc[0, "Current stock"] == 1000
assert complete_pattern["Milk"].sum() == sales.loc[sales["product"] == "Milk", "sales"].sum()
for result_frame in (forecast_output, recommendation_output, complete_pattern):
    for column in result_frame.select_dtypes(include="number").columns:
        values = result_frame[column].dropna()
        assert (values % 1 == 0).all(), (column, values.tolist())
assert any("executed successfully" in item.value for item in app.success)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_rank_deficient_sales_show_guidance_without_crashing_pages():
    script = """
import pandas as pd
from streamlit.testing.v1 import AppTest
from run_project import ROOT

sales = pd.DataFrame({
    "day": ["Monday"] * 7,
    "product": ["Rice", "Milk", "Soap", "Oil", "Bread", "Tea", "Sugar"],
    "price": [60, 30, 25, 140, 40, 18, 22],
    "promotion": [0, 1, 0, 1, 0, 1, 0],
    "sales": [48, 42, 21, 35, 33, 55, 39],
})
app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
app.session_state["sales_df"] = sales
app.session_state["selected_page"] = "Data Matrix Analysis"
app.run()
assert not app.exception, app.exception
assert any("rank 4" in item.value for item in app.warning)

for page in ("Demand Prediction", "Pattern Analysis", "Inventory Optimization", "Complete Analysis"):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, (page, [error.message for error in app.exception])
    if page == "Complete Analysis":
        next(button for button in app.button if button.label == "Execute integrated workflow").click().run()
        assert not app.exception
        assert any("rank 4" in item.value for item in app.error)
    else:
        assert app.warning, page
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_manual_inventory_editor_survives_page_navigation():
    script = """
from streamlit.testing.v1 import AppTest
from run_project import ROOT

app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
app.session_state["selected_page"] = "Input Data"
app.session_state["input_source"] = "Edit inventory values manually"
app.session_state["input_source_widget"] = "Edit inventory values manually"
app.session_state["inventory_df"] = {
    "edited_rows": {0: {"current_stock": 20}},
    "added_rows": [],
    "deleted_rows": [],
}
app.run()
assert not app.exception, app.exception
assert app.session_state["inventory_df"].shape[0] > 0
assert app.session_state["inventory_df"].loc[0, "current_stock"] == 20

app.sidebar.radio[0].set_value("Dashboard").run()
assert not app.exception, app.exception
app.sidebar.radio[0].set_value("Input Data").run()
assert not app.exception, app.exception
assert app.session_state["inventory_df"].shape[0] > 0
assert app.session_state["inventory_df"].loc[0, "current_stock"] == 20
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
