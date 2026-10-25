# Demand Prediction and Inventory Optimization for Value Retailers Using Linear Algebra

This project combines historical sales processing, least-squares demand forecasting, pattern analysis, and inventory optimization into a single end-to-end workflow.

## Modules

- `module1_data_matrix`: loads and validates sales data and constructs the design matrix A and target vector b.
- `module2_demand_prediction`: fits a least-squares model and predicts future demand using explicit feature ordering.
- `module3_pattern_analysis`: prepares product sales matrices, computes covariance, and analyzes eigenvalues/eigenvectors.
- `module4_inventory_optimization`: turns demand forecasts and stock inputs into reorder recommendations.

## Data flow

1. Historical sales data is loaded and validated.
2. Module 1 constructs the matrix A and vector b.
3. Module 2 fits a demand model and produces future-demand estimates.
4. Module 3 analyzes the sales pattern covariance structure.
5. Module 4 uses forecast demand and inventory values to produce recommendations.

## Running the project

```powershell
cd "C:\Users\ABHI\Downloads\Demand_Prediction_and_Inventory_Optimization_for_Value_Retailers_Using_Linear_Algebra"
python -m pytest -q
python run_project.py
python -m streamlit run app.py --server.headless true --server.port 8501
```
