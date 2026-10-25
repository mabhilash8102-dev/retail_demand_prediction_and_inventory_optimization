# Demand Prediction and Inventory Optimization Project

This workspace contains four independent but compatible linear-algebra modules for a retail demand and inventory workflow.

## Module inventory

1. `module1_data_matrix` — load and validate retail sales CSV data, construct the design matrix `A` and target vector `b`, and expose matrix operations such as transpose, matrix multiplication, rank, and RREF.
2. `module2_demand_prediction` — fit a least-squares demand model and predict future demand with explicit coefficient ordering.
3. `module3_pattern_analysis` — prepare centered sales data, compute covariance, and analyze eigenvalues/eigenvectors to interpret sales patterns.
4. `module4_inventory_optimization` — convert predicted demand into required stock, order quantities, and statuses.

## Data flow

`module1_data_matrix` loads the historical CSV and prepares the design matrix and target vector.
The matrix is passed to `module2_demand_prediction.fit_least_squares`, which computes coefficients and predictions.
`module3_pattern_analysis` receives the product-level sales matrix to compute pattern and variance information.
`module4_inventory_optimization` consumes the demand forecast and stock inputs to produce order recommendations.

## Root-level execution

From the parent directory containing the four module folders, run:

```powershell
cd C:\Users\ABHI\Downloads
python run_project.py
```

## GUI launch command

```powershell
cd C:\Users\ABHI\Downloads
python -m streamlit run app.py --server.headless true --server.port 8501
```

Open the browser to:

```text
http://localhost:8501
```

In the GUI, **Complete Analysis**, **Pattern Analysis**, and **Inventory Optimization** use the currently selected and validated sales dataset. Module 3 sums product sales by weekday from those rows; an unlisted product on a weekday is treated as zero recorded sales, and pattern analysis needs at least two weekdays. Demand prediction needs a full-rank design matrix: include multiple weekdays and vary price and promotion independently. Inventory Optimization aligns its product rows to distinct products in the active sales dataset, retaining current-stock values by product name where available and applying the module defaults for new products. Safety stock defaults to 10. The active sales input is read afresh so switching datasets or replacing a CSV cannot keep an old forecast active.

The Input Data page also includes the supplied `input_1.csv`, `input_2.csv`, and `input_3_dependent.csv` sales datasets. The inventory table remains a separate input as described above. `input_3_dependent.csv` is intentionally rank-deficient and cannot produce a unique demand forecast until its input features vary independently.

Numeric analysis results are displayed as whole numbers in both the GUI and command-line report. Calculations retain full precision internally; rounding is presentation-only.

## Test commands

```powershell
cd C:\Users\ABHI\Downloads
python -m pytest -q
```

## Module-level test commands

```powershell
cd C:\Users\ABHI\Downloads\module1_data_matrix
python -m pytest module1_data_matrix/tests -q

cd C:\Users\ABHI\Downloads\module2_demand_prediction
python -m pytest tests -q

cd C:\Users\ABHI\Downloads\module3_pattern_analysis
python -m pytest tests -q

cd C:\Users\ABHI\Downloads\module4_inventory_optimization
python -m pytest tests -q
```

## Required dependencies

- Python 3.10+
- NumPy
- pandas
- Streamlit
- pytest

Install if needed:

```powershell
cd C:\Users\ABHI\Downloads
python -m pip install -r requirements.txt
```

## Directory structure

```text
C:\Users\ABHI\Downloads
├── README.md
├── run_project.py
├── pytest.ini
├── module1_data_matrix/
│   └── module1_data_matrix/
├── module2_demand_prediction/
├── module3_pattern_analysis/
├── module4_inventory_optimization/
└── ...
```

## Notes and assumptions

- The project is split across four module folders in this workspace, so the parent `Downloads` directory acts as the integration root.
- Matrix and demand inputs are validated before use and the project uses explicit feature order for future prediction.
- The end-to-end demo uses the included CSV samples and does not invent unseen demand data.
