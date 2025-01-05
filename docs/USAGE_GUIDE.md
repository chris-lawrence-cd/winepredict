# WinePredict Usage Guide

Welcome to the WinePredict library! This guide will help you get started with using the library to predict wine prices using machine learning models.

## Installation

To install the WinePredict library, you can use the following command to install directly from the GitHub repository:

```bash
!python -m pip install -q -U git+https://github.com/chris-lawrence-cd/winepredict.git
```

## Getting Started

### Importing the Library

First, import the necessary modules from the WinePredict library:

```python
from winepredict import data_processing, model_training, model_evaluation, visualization
```

### Data Processing

1. **Download FRED Data**

   Use the `download_fred_data` function to fetch economic data from the FRED API:

   ```python
   api_key = 'your_fred_api_key'
   gdp_data = data_processing.download_fred_data('GDP', api_key)
   ```

   **Edge Case:** Ensure your API key is valid. Handle network errors gracefully.

2. **Process FRED Data**

   Process and save the data to an Excel file:

   ```python
   data_processing.process_fred_data(api_key)
   ```

   **Tip:** Verify that the Excel file is created successfully and contains the expected data.

3. **Preprocess and Analyze Data**

   Load and preprocess your data for analysis:

   ```python
   df = data_processing.preprocess_and_analyze_data('FRED_Data.xlsx')
   ```

   **Troubleshooting:** If the data contains NaN values, consider using `fillna()` to handle missing data.

### Model Training

1. **Train and Evaluate Models**

   Train multiple models and evaluate their performance:

   ```python
   results = model_training.train_and_evaluate_models(df)
   print(results)
   ```

   **Edge Case:** Ensure the DataFrame is not empty before training models.

2. **Tune and Evaluate CatBoost**

   Tune hyperparameters for the CatBoost model:

   ```python
   X = df.drop('Average Wine Price', axis=1)
   y = df['Average Wine Price']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   catboost_results = model_training.tune_and_evaluate_catboost(X_train, y_train, X_test, y_test)
   print(catboost_results)
   ```

   **Tip:** Use a validation set to prevent overfitting during hyperparameter tuning.

### Model Evaluation

1. **Evaluate a Model**

   Evaluate a trained model using test data:

   ```python
   model = SomeTrainedModel()
   model_evaluation.evaluate_model(model, X_train, X_test, y_train, y_test)
   ```

   **Troubleshooting:** Check for overfitting by comparing training and test performance.

2. **Cross-Validate a Model**

   Perform cross-validation to assess model robustness:

   ```python
   cross_val_scores = model_evaluation.cross_validate_model(model, X, y)
   print(cross_val_scores)
   ```

   **Edge Case:** Ensure the dataset is sufficiently large for cross-validation.

### Visualization

1. **Plot Actual vs. Predicted**

   Visualize the actual vs. predicted prices:

   ```python
   visualization.plot_actual_vs_predicted(y_test, y_pred)
   ```

   **Tip:** Use `tight_layout()` to ensure the plot elements are well-organized.

2. **Visualize Residuals**

   Visualize residuals to assess model fit:

   ```python
   visualization.visualize_residuals(y_test, y_pred)
   ```

   **Troubleshooting:** Large residuals may indicate model bias or variance issues.

## Running Tests

To run the tests for the WinePredict library, use the following command:

```bash
pytest tests/
```

This will execute all the tests in the `/tests/` directory and provide a summary of the results.

## Conclusion

This guide provides an overview of the key functionalities of the WinePredict library. For more detailed information, please refer to the module docstrings and source code. If you encounter any issues or have questions, feel free to reach out to the maintainers.

## Troubleshooting Tips

*   **API Key Issues:** Ensure your API key is valid and properly formatted.
*   **Data Quality Issues:** Check for missing or duplicate data, and handle accordingly.
*   **Model Performance Issues:** Check for overfitting or underfitting, and adjust hyperparameters as needed.
*   **Visualization Issues:** Check for plot formatting issues, and adjust as needed.

## Additional Examples

*   **Using Custom Models:** Train and evaluate custom machine learning models using the WinePredict library.
*   **Handling Imbalanced Data:** Use techniques such as oversampling or undersampling to handle imbalanced datasets.
*   **Using Ensemble Methods:** Combine multiple models to improve overall performance.

## Edge Cases

*   **Handling Missing Data:** Use techniques such as imputation or interpolation to handle missing data.
*   **Handling Outliers:**
    *   **Winsorization and Trimming:** These techniques can be used to limit the impact of extreme values by capping them at a certain percentile. However, they do not model the true Data Generating Process (DGP) and can lead to misleading results if not used carefully.
    *   **Robust Techniques:** Consider using robust statistical methods that are less sensitive to outliers, such as the median instead of the mean, or robust regression techniques. These methods aim to provide a more accurate representation of the underlying data without altering the original dataset.
    *   **Understanding the DGP:** Before applying any outlier handling technique, it's crucial to understand the context and the DGP. Outliers may contain valuable information or indicate errors in data collection. Analyzing the cause of outliers can provide insights into the data and guide the choice of technique.
*   **Handling Non-Numeric Data:** Use techniques such as encoding or normalization to handle non-numeric data.
