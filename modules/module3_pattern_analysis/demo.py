"""Run the standalone Member 3 demonstration from the project root."""
from pathlib import Path
import csv
from module3_pattern_analysis import (
    prepare_sales_matrix,
    calculate_covariance,
    is_symmetric,
    analyze_eigenvalues,
    verify_eigenpairs,
    detect_patterns,
)

root = Path(__file__).parent
with (root / "data" / "sample_sales_matrix.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
data = {key: [row[key] for row in rows] for key in rows[0]}
products = ["Milk", "Bread", "Rice", "Oil"]
prepared = prepare_sales_matrix(data, products)
covariance = calculate_covariance(prepared.centered_matrix).covariance_matrix
analysis = analyze_eigenvalues(covariance)
pattern = detect_patterns(analysis, products)

print("=" * 58)
print("MEMBER 3: SALES PATTERN ANALYSIS")
print("=" * 58)
print(f"Input: {len(rows)} days of sales for {', '.join(products)}")

print("\n[CONCEPT 1: AVERAGE SALES]")
print("What it means: average units sold per day for each product.")
for name, mean in zip(products, prepared.means):
    print(f"  {name:<8} {mean:.3f} units/day")

print("\n[CONCEPT 2: COVARIANCE]")
print("What it means: how pairs of products' sales vary together.")
print("Rows and columns use this product order:", ", ".join(products))
for name, row in zip(products, covariance):
    print(f"  {name:<8} " + "  ".join(f"{value:9.3f}" for value in row))
print("Symmetry check:", "PASS" if is_symmetric(covariance) else "FAIL")

print("\n[CONCEPT 3: EIGENVALUES / VARIATION PATTERNS]")
print("What it means: each eigenvalue measures the variation captured by one pattern.")
for index, (value, share, cumulative) in enumerate(zip(
        analysis.eigenvalues, analysis.explained_variance,
        analysis.cumulative_explained_variance), start=1):
    perc = share * 100.0
    cum_pct = cumulative * 100.0
    print(f"  Pattern {index}: eigenvalue={value:.3f}; "
          f"explained variance ratio={share:.6f}; "
          f"variation={perc:.3f}%; cumulative={cum_pct:.3f}%")

print("\n[CONCEPT 4: DOMINANT SALES PATTERN]")
print("What it means: product loadings in the strongest pattern.")
print("Positive/negative signs show direction; size shows contribution strength.")
for name, loading in pattern.product_contributions.items():
    print(f"  {name:<8} {loading:+.3f}")
print("Interpretation: these are mathematical relationships in this sample; they do not prove cause or purchasing behavior.")

print("\n[CONCEPT 5: MATHEMATICAL CHECK]")
print("Checks whether each eigenvector satisfies C v = lambda v.")
print("Eigenpair check:", "PASS" if verify_eigenpairs(covariance, analysis) else "FAIL")
print("\nSales pattern analysis complete: the dominant variation pattern has been quantified and validated.")
print("=" * 58)
