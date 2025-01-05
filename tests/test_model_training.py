import pytest
import pandas as pd
from winepredict.model_training import train_and_evaluate_models, tune_and_evaluate_catboost

@pytest.fixture
def sample_scaled_df():
    """Fixture for a sample scaled DataFrame."""
    data = {
        'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature2': [0.5, 0.4, 0.3, 0.2, 0.1],
        'Average Wine Price': [1, 2, 3, 4, 5]
    }
    return pd.DataFrame(data)


def test_train_and_evaluate_models(sample_scaled_df):
    """Test train_and_evaluate_models function."""
    results = train_and_evaluate_models(sample_scaled_df)
    assert not results.empty, "Results should not be empty."
    assert 'Model' in results.columns, "Results should contain 'Model' column."


def test_tune_and_evaluate_catboost(sample_scaled_df):
    """Test tune_and_evaluate_catboost function."""
    X = sample_scaled_df.drop('Average Wine Price', axis=1)
    y = sample_scaled_df['Average Wine Price']
    X_train, X_test, y_train, y_test = X.iloc[:3], X.iloc[3:], y.iloc[:3], y.iloc[3:]
    results = tune_and_evaluate_catboost(X_train, y_train, X_test, y_test)
    assert 'Best Parameters' in results, "Results should contain 'Best Parameters'."
    assert 'R²' in results, "Results should contain 'R²'."
