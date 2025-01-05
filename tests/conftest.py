import pytest
import pandas as pd

@pytest.fixture
def sample_fred_data():
    """Fixture for sample FRED data."""
    data = {
        'date': pd.date_range(start='1/1/2020', periods=5, freq='M'),
        'value': [100, 200, 150, 300, 250]
    }
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df

@pytest.fixture
def sample_model_data():
    """Fixture for sample model data."""
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1]
    })
    y = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])
    return X, y
