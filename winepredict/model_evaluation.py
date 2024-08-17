import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluates the model on training and testing data.

    Args:
        model: The machine learning model to be evaluated.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Number of observations
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    # Number of predictors
    p = X_train.shape[1]

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Calculate adjusted R²
    train_adj_r2 = 1 - ((1 - train_r2) * (n_train - 1)) / (n_train - p - 1)
    test_adj_r2 = 1 - ((1 - test_r2) * (n_test - 1)) / (n_test - p - 1)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"Train Adjusted R²: {train_adj_r2:.5f}, Test Adjusted R²: {test_adj_r2:.5f}")
    print(f"Train RMSE: {train_rmse:.5f}, Test RMSE: {test_rmse:.5f}")
    print(f"Train MAE: {train_mae:.5f}, Test MAE: {test_mae:.5f}")

def cross_validate_model(model, X, y):
    """Performs cross-validation for model robustness.

    Args:
        model: The machine learning model to be cross-validated.
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.

    Returns:
        np.ndarray: Cross-validated R² scores.
    """
    cv_results = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validated R² scores: {cv_results}")
    print(f"Mean cross-validated R² score: {cv_results.mean():.5f}")
    return cv_results

def plot_actual_vs_predicted(y_test, y_pred_best):
    """Plots actual vs. predicted prices for the best model.

    Args:
        y_test (pd.Series): Actual target values.
        y_pred_best (np.ndarray): Predicted target values by the best model.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_best, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices - Best Model')
    plt.grid(True)
    plt.savefig('actual_vs_predicted.png')
    plt.show()

def visualize_residuals(y_test, y_pred_best):
    """Visualizes the residuals distribution and residuals vs. predicted prices.

    Args:
        y_test (pd.Series): Actual target values.
        y_pred_best (np.ndarray): Predicted target values by the best model.
    """
    residuals = y_test - y_pred_best

    # Residuals distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution - Best Model')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('residuals_distribution.png')
    plt.show()

    # Residuals vs Predicted Prices
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_best, residuals, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Prices - Best Model')
    plt.grid(True)
    plt.savefig('residuals_vs_predicted.png')
    plt.show()