import pytest
from sklearn.linear_model import LinearRegression
from winepredict.model_evaluation import evaluate_model, cross_validate_model

@pytest.fixture
def sample_model_data():
    """Fixture for sample model data."""
    X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
    X_test = [[5, 6], [6, 7]]
    y_train = [1, 2, 3, 4]
    y_test = [5, 6]
    return X_train, X_test, y_train, y_test


def test_evaluate_model(sample_model_data):
    """Test evaluate_model function."""
    X_train, X_test, y_train, y_test = sample_model_data
    model = LinearRegression().fit(X_train, y_train)
    evaluate_model(model, X_train, X_test, y_train, y_test)
    # Add assertions based on expected output


def test_cross_validate_model(sample_model_data):
    """Test cross_validate_model function."""
    X_train, _, y_train, _ = sample_model_data
    model = LinearRegression()
    cross_validate_model(model, X_train, y_train)
    # Add assertions based on expected output
