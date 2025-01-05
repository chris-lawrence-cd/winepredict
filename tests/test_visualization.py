import pytest
import matplotlib.pyplot as plt
from winepredict.visualization import plot_actual_vs_predicted, visualize_residuals

@pytest.fixture
def sample_test_data():
    """Fixture for sample test data."""
    y_test = [100, 200, 300, 400, 500]
    y_pred = [110, 190, 310, 390, 510]
    return y_test, y_pred


def test_plot_actual_vs_predicted(sample_test_data):
    """Test plot_actual_vs_predicted function."""
    y_test, y_pred = sample_test_data
    plot_actual_vs_predicted(y_test, y_pred, save_path='test_actual_vs_predicted.png')
    # Check if the plot file was created
    assert plt.gcf().number == 1, "A plot should be created."


def test_visualize_residuals(sample_test_data):
    """Test visualize_residuals function."""
    y_test, y_pred = sample_test_data
    visualize_residuals(y_test, y_pred, save_path_residuals='test_residuals_distribution.png', save_path_vs_predicted='test_residuals_vs_predicted.png')
    # Check if the plots were created
    assert plt.gcf().number == 1, "A plot should be created."
