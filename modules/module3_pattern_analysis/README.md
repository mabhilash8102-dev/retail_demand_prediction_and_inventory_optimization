# Member 3 Sales Pattern Analysis

Standalone sales pattern analysis using covariance and eigenvalues/eigenvectors. This version uses only the Python standard library: no NumPy, Pandas, Pytest, package installation, or virtual environment is needed. A Jacobi method computes eigenpairs for the symmetric covariance matrix.

## Run on Windows

Run the commands from the parent directory that contains the `module3_pattern_analysis` package folder, not from inside the package directory itself:

```powershell
cd C:\Users\ABHI\Downloads
python -m module3_pattern_analysis.demo
python -m pytest module3_pattern_analysis/tests -q
```

If `python` is unavailable, use `py` instead. The sample is `data/sample_sales_matrix.csv`; `day` is an observation ID and not a product column.

## Files

`data_processor.py` validates selected named columns and centers data (`D_c = D - mu`). `covariance.py` calculates sample covariance (`C = (1/(n-1))D_c.T D_c`) and checks symmetry. `eigen_analysis.py` calculates and sorts eigenvalues/vectors, explained variance (`lambda_i / sum(lambda_j)`), cumulative variance, and verifies `Cv = lambda v`. `pattern_detector.py` reports loadings for the dominant direction. Eigenvector signs are arbitrary; loadings indicate mathematical relationships, not causation. `models.py` defines results, `config.py` holds numerical tolerance, and `__init__.py` defines the public API. `demo.py` runs the example; `tests` contains standard-library unit tests.

The public API is `prepare_sales_matrix`, `calculate_covariance`, `is_symmetric`, `analyze_eigenvalues`, `verify_eigenpairs`, `detect_patterns`, `fit_demand_model`, `predict_historical_demand`, and `predict_future_demand`. The Module 2 compatibility functions are thin adapters to the existing `module2_demand_prediction` package and do not copy or replace that project. Member 1 may provide validated column lists to `prepare_sales_matrix`. Member 3 does not implement CSV ingestion as part of integration, least-squares demand prediction (Member 2), forecasting, or inventory and ordering (Member 4).
