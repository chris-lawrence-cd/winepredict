import matplotlib.pyplot as plt
import seaborn as sns

def plot_actual_vs_predicted(y_test, y_pred_best, save_path='actual_vs_predicted.png'):
    """Plots actual vs. predicted prices for the best model.

    Args:
        y_test (pd.Series): Actual target values.
        y_pred_best (np.ndarray): Predicted target values by the best model.
        save_path (str): File path to save the plot. Defaults to 'actual_vs_predicted.png'.
    """
    if y_test.empty or len(y_pred_best) == 0:
        print("Error: Input data for plotting is empty.")
        return

    if len(y_test) != len(y_pred_best):
        print("Error: Mismatch in length between actual and predicted values.")
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices - Best Model')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def visualize_residuals(y_test, y_pred_best, save_path_residuals='residuals_distribution.png', save_path_vs_predicted='residuals_vs_predicted.png'):
    """Visualizes the residuals distribution and residuals vs. predicted prices.

    Args:
        y_test (pd.Series): Actual target values.
        y_pred_best (np.ndarray): Predicted target values by the best model.
        save_path_residuals (str): File path to save the residuals distribution plot. Defaults to 'residuals_distribution.png'.
        save_path_vs_predicted (str): File path to save the residuals vs. predicted plot. Defaults to 'residuals_vs_predicted.png'.
    """
    if y_test.empty or len(y_pred_best) == 0:
        print("Error: Input data for plotting is empty.")
        return

    if len(y_test) != len(y_pred_best):
        print("Error: Mismatch in length between actual and predicted values.")
        return

    residuals = y_test - y_pred_best

    # Residuals distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution - Best Model')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_residuals)
    plt.show()

    # Residuals vs Predicted Prices
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred_best, y=residuals, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Prices - Best Model')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_vs_predicted)
    plt.show()

def plot_feature_importance(feature_importance, title='Feature Importance', save_path='feature_importance.png'):
    """Plots feature importance for a given model.

    Args:
        feature_importance (pd.Series): Series containing feature importance scores.
        title (str): Title of the plot. Defaults to 'Feature Importance'.
        save_path (str): File path to save the plot. Defaults to 'feature_importance.png'.
    """
    plt.figure(figsize=(12, 8))
    feature_importance.sort_values().plot(kind='barh')
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
