import pytest
import pandas as pd
from unittest.mock import patch
from winepredict.data_processing import download_fred_data, process_fred_data, preprocess_and_analyze_data

@pytest.fixture
def sample_api_key():
    """Fixture for a sample API key."""
    return 'sample_api_key'


def test_download_fred_data(sample_api_key):
    """Test download_fred_data function."""
    with patch('winepredict.data_processing.requests.get') as mock_get:
        mock_get.return_value.json.return_value = {
            'observations': [{'date': '2020-01-01', 'value': '100'}, {'date': '2020-02-01', 'value': '200'}]
        }
        df = download_fred_data('GDP', sample_api_key)
        assert not df.empty, "DataFrame should not be empty."
        assert 'value' in df.columns, "DataFrame should contain 'value' column."


def test_process_fred_data(sample_api_key):
    """Test process_fred_data function."""
    with patch('winepredict.data_processing.download_fred_data') as mock_download:
        mock_download.return_value = pd.DataFrame({
            'date': pd.date_range(start='1/1/2020', periods=5, freq='ME'),
            'value': [100, 200, 300, 400, 500]
        }).set_index('date')
        process_fred_data(sample_api_key)
        # Check if the function executes without errors


def test_preprocess_and_analyze_data():
    """Test preprocess_and_analyze_data function."""
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'Average Wine Price': [1.5, 2.5, 3.5, 4.5, 5.5]
    }
    df = pd.DataFrame(data)
    result = preprocess_and_analyze_data(df, save_vif=False)
    assert not result.empty, "Resulting DataFrame should not be empty."
    assert 'Average Wine Price' in result.columns, "Resulting DataFrame should contain 'Average Wine Price' column."
