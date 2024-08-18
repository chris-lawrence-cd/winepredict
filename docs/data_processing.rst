data_processing Module
======================

The `data_processing` module is responsible for preparing and processing data that will be used in the modeling and analysis phases. This includes data retrieval, cleaning, preprocessing, and analysis to ensure that the data is in the optimal format for model training and evaluation.

The module contains several key functions that facilitate these tasks, ensuring that the data pipeline is robust and efficient.

Module Contents
---------------

.. automodule:: winepredict.data_processing
   :members:
   :undoc-members:
   :show-inheritance:

Functions Overview
------------------

The following functions are part of the `data_processing` module:

1. **calculate_vif**:
   Calculates the Variance Inflation Factor (VIF) for the features in a dataset to detect multicollinearity. A high VIF indicates that a feature is highly correlated with other features, which may lead to instability in the model.

2. **download_fred_data**:
   Downloads financial and economic data from the Federal Reserve Economic Data (FRED) API. This function handles API requests and ensures that the data is formatted correctly for subsequent analysis.

3. **preprocess_and_analyze_data**:
   Performs data preprocessing and exploratory data analysis (EDA). This function includes tasks like handling missing values, feature engineering, and generating initial insights into the data.

4. **process_fred_data**:
   Processes the data downloaded from the FRED API. This involves cleaning, transforming, and preparing the data for model training.

Detailed Function Descriptions
------------------------------

Below are detailed descriptions of each function within the `data_processing` module:

.. autofunction:: winepredict.data_processing.calculate_vif

   **Description**:
   - Calculates the Variance Inflation Factor (VIF) for each feature in the dataset.
   - Useful for identifying multicollinearity, which can negatively impact model performance.

   **Parameters**:
   - `df` (pandas.DataFrame): The dataframe containing the features.
   - `features` (list): List of feature names for which VIF should be calculated.

   **Returns**:
   - pandas.DataFrame: A dataframe containing features and their corresponding VIF values.

.. autofunction:: winepredict.data_processing.download_fred_data

   **Description**:
   - Downloads data from the FRED API for a given set of series IDs.
   - Handles API keys and ensures the data is stored efficiently.

   **Parameters**:
   - `series_ids` (list): List of FRED series IDs to download.
   - `start_date` (str): Start date for data retrieval in the format 'YYYY-MM-DD'.
   - `end_date` (str): End date for data retrieval in the format 'YYYY-MM-DD'.

   **Returns**:
   - pandas.DataFrame: A dataframe containing the downloaded FRED data.

.. autofunction:: winepredict.data_processing.preprocess_and_analyze_data

   **Description**:
   - Preprocesses the data and performs exploratory data analysis (EDA).
   - Includes tasks like normalization, encoding categorical variables, and generating summary statistics.

   **Parameters**:
   - `df` (pandas.DataFrame): The raw input dataframe.
   - `target` (str): The name of the target variable.

   **Returns**:
   - pandas.DataFrame: The preprocessed dataframe ready for model training.
   - dict: A dictionary containing EDA results and visualizations.

.. autofunction:: winepredict.data_processing.process_fred_data

   **Description**:
   - Cleans and processes the data retrieved from the FRED API.
   - Handles missing values, date formatting, and any necessary data transformations.

   **Parameters**:
   - `df` (pandas.DataFrame): The raw dataframe containing FRED data.
   - `columns` (list): List of columns to be processed.

   **Returns**:
   - pandas.DataFrame: The cleaned and processed dataframe.

Usage Examples
--------------

The following example demonstrates how to use the functions in the `data_processing` module:

```python
from winepredict.data_processing import (
    calculate_vif,
    download_fred_data,
    preprocess_and_analyze_data,
    process_fred_data
)

# Download data from FRED API
fred_data = download_fred_data(['GDP', 'CPI'], '2000-01-01', '2023-01-01')

# Process the downloaded data
processed_data = process_fred_data(fred_data, ['GDP', 'CPI'])

# Perform preprocessing and analysis
preprocessed_data, eda_results = preprocess_and_analyze_data(processed_data, 'target_variable')

# Calculate VIF to check for multicollinearity
vif_df = calculate_vif(preprocessed_data, ['feature1', 'feature2', 'feature3'])
