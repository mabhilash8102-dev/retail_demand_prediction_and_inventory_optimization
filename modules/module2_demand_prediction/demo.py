"""Run the standalone Member 2 least-squares demonstration."""

from module2_demand_prediction import (
    calculate_metrics,
    calculate_residuals,
    fit_least_squares,
    predict,
    predict_from_matrix,
)
from module2_demand_prediction.data.sample_prediction_input import (
    FEATURE_ORDER,
    sample_prediction_input,
)


def main() -> None:
    A, b = sample_prediction_input()
    result = fit_least_squares(A, b)
    predicted = predict_from_matrix(A, result.coefficients)
    residuals = calculate_residuals(b, predicted)
    metrics = calculate_metrics(b, predicted)
    future_features = {"day": 11, "price": 30, "promotion": 1}
    future_demand = predict(
        result.coefficients,
        future_features,
        feature_order=FEATURE_ORDER,
    )

    print("=" * 40)
    print(" MEMBER 2 - DEMAND PREDICTION")
    print("=" * 40)
    print("\nInput matrix A:\n", A)
    print("\nSales vector b:\n", b)
    print("\nA transpose:\n", result.A_transpose)
    print("\nA.T @ A:\n", result.AtA)
    print("\nA.T @ b:\n", result.Atb)
    print("\nLeast-squares coefficients X:\n", result.coefficients)
    print("\nHistorical predictions A @ X:\n", predicted)
    print("\nResidual vector b - predicted:\n", residuals)
    print("\nA.T @ residual:\n", A.T @ residuals)
    print(f"\nMAE  = {metrics.mae:.6f}")
    print(f"MSE  = {metrics.mse:.6f}")
    print(f"RMSE = {metrics.rmse:.6f}")
    print("\nFuture input:")
    for name in FEATURE_ORDER:
        if name != "intercept":
            print(f"{name.capitalize():<10}= {future_features[name]}")
    print(f"\nPredicted demand = {future_demand:.2f} units")
    print("\nThe predicted historical vector lies in Col(A); the residual is")
    print("orthogonal to Col(A), as shown by A.T @ residual approximately 0.")
    print("\nMember 2 completed successfully.")


if __name__ == "__main__":
    main()
