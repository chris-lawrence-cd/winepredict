import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_feature_importance(feature_importance, title='Feature Importance'):
    """Plots feature importance.

    Args:
        feature_importance (pd.Series): Series containing feature importances.
        title (str): Title of the plot. Defaults to 'Feature Importance'.
    """
    plt.figure(figsize=(12, 8))
    feature_importance.sort_values().plot(kind='barh')
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.grid(True)
    plt.savefig('feature_importance.png')
    plt.show()