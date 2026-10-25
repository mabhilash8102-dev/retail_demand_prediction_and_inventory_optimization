from __future__ import annotations

import pandas as pd
import subprocess
import sys
from pathlib import Path

app = None
run_workflow = None


class UploadedCsv:
    """Small UploadedFile stand-in that keeps a stable client filename."""

    name = "sales.csv"

    def __init__(self, text: str):
        self._content = text.encode("utf-8")

    def getvalue(self) -> bytes:
        return self._content


def sales_csv(prefix: str, offset: float = 0.0) -> str:
    rows = []
    scenarios = [
        ("Monday", 20, 0),
        ("Tuesday", 25, 1),
        ("Wednesday", 30, 0),
        ("Thursday", 35, 1),
        ("Friday", 40, 0),
        ("Saturday", 45, 1),
        ("Sunday", 50, 0),
        ("Monday", 55, 1),
    ]
    for index, (day, price, promotion) in enumerate(scenarios):
        product = f"{prefix}-{index % 2}"
        row = {
            "day": day,
            "product": product,
            "price": price,
            "promotion": promotion,
            "sales": 80 + index * 4 + promotion * 10 + offset,
        }
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False)


def run_uploaded(csv_text: str):
    frame = app._read_sales_csv_from_upload(UploadedCsv(csv_text))
    records = app._module1_records_from_csv(frame)
    inventory_df = app.validate_inventory_inputs(app.load_default_inventory_df())
    products = frame["product"].drop_duplicates().tolist()
    inventory_df = inventory_df.iloc[:len(products)].copy()
    inventory_df["product"] = products
    inventory = inventory_df.to_dict(orient="records")
    return frame, run_workflow(records, inventory)


def expect_value_error(call, expected_text: str) -> None:
    try:
        call()
    except ValueError as exc:
        assert expected_text in str(exc)
    else:
        raise AssertionError(f"Expected ValueError containing {expected_text!r}.")


def scenario_initial_and_sequential_dataset_switches_use_active_input():
    outputs = []
    for prefix, offset in (("Alpha", 0), ("Beta", 40), ("Gamma", 80)):
        frame, report = run_uploaded(sales_csv(prefix, offset))
        products = {item.product_name for item in report["recommendations"]}
        assert products == set(report["product_names"])
        assert {row["product"] for row in report["records"]} == set(frame["product"])
        outputs.append((frame, report))

        inventory = app.validate_inventory_inputs(app.load_default_inventory_df())
        app.st.session_state = {"sales_df": frame, "inventory_df": inventory}
        active_report = app.current_analysis_report()
        assert active_report["product_names"] == list(app.st.session_state["inventory_df"]["product"])
        assert {row["product"] for row in active_report["records"]} == set(frame["product"])
        assert active_report["forecast"] == report["forecast"]

    for (previous_frame, _), (active_frame, active_report) in zip(outputs, outputs[1:]):
        assert set(previous_frame["product"]).isdisjoint(
            item.product_name for item in active_report["recommendations"]
        )
    assert outputs[0][1]["forecast"] != outputs[1][1]["forecast"]
    assert outputs[1][1]["forecast"] != outputs[2][1]["forecast"]


def scenario_same_filename_with_changed_contents_is_reprocessed():
    first = app._read_sales_csv_from_upload(UploadedCsv(sales_csv("Old", 0)))
    second = app._read_sales_csv_from_upload(UploadedCsv(sales_csv("New", 25)))
    assert UploadedCsv.name == "sales.csv"
    assert set(first["product"]).isdisjoint(second["product"])
    assert not first.equals(second)
    assert len(app._module1_records_from_csv(second)) == len(second)


def scenario_invalid_upload_is_rejected_and_cannot_return_previous_results():
    _, valid_report = run_uploaded(sales_csv("Valid"))
    assert valid_report["recommendations"]
    expect_value_error(
        lambda: app._read_sales_csv_from_upload(
            UploadedCsv("product,current_stock\nOld,10\n")
        ),
        "Missing columns",
    )

    app.st.session_state = {
        "sales_df": pd.DataFrame({"product": ["Old"]}),
        "inventory_df": pd.DataFrame(
            [{"product": "Old", "current_stock": 10, "safety_stock": 5}]
        ),
        "sales_input_error": "Missing required field(s): ['day']",
    }
    expect_value_error(app.current_analysis_report, "current sales input is invalid")


def scenario_inventory_inputs_are_validated_separately_from_sales():
    incomplete_inventory = pd.DataFrame(
        [{"product": "Existing inventory item", "current_stock": pd.NA, "safety_stock": 20}]
    )
    expect_value_error(
        lambda: app.validate_inventory_inputs(incomplete_inventory),
        "Enter valid current stock and safety stock values",
    )


def scenario_provided_input_files():
    default_inventory = app.load_default_inventory_df()

    input_1 = app.load_sales_input(str(app.SALES_INPUT_PATHS["Use input_1.csv"]))
    inventory_1 = app.validate_inventory_inputs(default_inventory)
    report_1 = run_workflow(
        app._module1_records_from_csv(input_1), inventory_1.to_dict(orient="records")
    )
    assert report_1["product_names"] == inventory_1["product"].tolist()
    assert len(report_1["recommendations"]) == 4

    input_2 = app.load_sales_input(str(app.SALES_INPUT_PATHS["Use input_2.csv"]))
    report_2 = run_workflow(
        app._module1_records_from_csv(input_2),
        inventory_1.to_dict(orient="records"),
    )
    assert report_2["product_names"] == report_1["product_names"]
    assert {row["product"] for row in report_2["records"]} == set(input_2["product"])
    sales_products_2 = {product.casefold() for product in input_2["product"]}
    expected_basis_2 = [
        "Product history" if product.casefold() in sales_products_2
        else "Latest sales row (product absent from sales input)"
        for product in report_2["product_names"]
    ]
    assert report_2["forecast_basis"] == expected_basis_2
    assert report_2["forecast"] != report_1["forecast"]

    input_3 = app.load_sales_input(
        str(app.SALES_INPUT_PATHS["Use input_3_dependent.csv"])
    )
    model_3 = app._module1_outputs(app._module1_records_from_csv(input_3))
    assert "rank is 3" in model_3["fit_error"]


def test_inventory_input_switching_regressions():
    project_root = Path(__file__).resolve().parents[2]
    script = Path(__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--run-scenarios"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__" and "--run-scenarios" in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    # These scenarios exercise CSV parsing and the report guard without
    # starting a Streamlit server; a minimal module stub keeps them runnable in
    # environments that only have the project's numerical dependencies.
    import types

    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.cache_data = lambda function: function
    streamlit_stub.session_state = {}
    streamlit_stub.set_page_config = lambda **kwargs: None
    sys.modules.setdefault("streamlit", streamlit_stub)

    import app as app_module
    from run_project import run_workflow as workflow

    app = app_module
    run_workflow = workflow
    scenario_initial_and_sequential_dataset_switches_use_active_input()
    scenario_same_filename_with_changed_contents_is_reprocessed()
    scenario_invalid_upload_is_rejected_and_cannot_return_previous_results()
    scenario_inventory_inputs_are_validated_separately_from_sales()
    scenario_provided_input_files()





