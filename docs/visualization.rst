visualization Module
====================

The `visualization` module provides functions for visualizing the results and insights from the WinePredict library. These visualizations include plotting actual vs. predicted values, analyzing feature importance, and examining residuals. These tools are essential for understanding model performance and identifying areas for improvement.

Module Contents
---------------

.. automodule:: winepredict.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Functions Overview
------------------

The `visualization` module includes the following key functions:

1. **plot_actual_vs_predicted**:
   Generates a plot comparing the actual vs. predicted values from a regression model, allowing for quick visual assessment of model performance.

2. **plot_feature_importance**:
   Visualizes the importance of each feature used by the model, aiding in understanding which features contribute most to the predictions.

3. **visualize_residuals**:
   Plots the residuals of the model predictions, helping to diagnose issues such as heteroscedasticity or non-linearity.

Detailed Function Descriptions
------------------------------

.. autofunction:: winepredict.visualization.plot_actual_vs_predicted

   **Description**:
   - Generates a scatter plot comparing the actual target values with the model's predicted values.
   - This plot helps to identify how well the model is performing, highlighting areas where the model may be under or over-predicting.

   **Parameters**:
   - `y_true` (array-like): The actual target values.
   - `y_pred` (array-like): The predicted values from the model.
   - `title` (str, optional): The title of the plot.
   - `save_path` (str, optional): Path to save the plot image.

.. autofunction:: winepredict.visualization.plot_feature_importance

   **Description**:
   - Creates a bar plot showing the importance of each feature in the model.
   - Useful for feature selection and understanding model behavior.

   **Parameters**:
   - `model` (object): A trained model with a `feature_importances_` attribute.
   - `feature_names` (list of str): The names of the features.
   - `title` (str, optional): The title of the plot.
   - `save_path` (str, optional): Path to save the plot image.

.. autofunction:: winepredict.visualization.visualize_residuals

   **Description**:
   - Generates a residuals plot, which is the difference between the actual and predicted values.
   - Helps in diagnosing model issues such as non-linearity, outliers, or patterns in the residuals that suggest a problem with the model fit.

   **Parameters**:
   - `y_true` (array-like): The actual target values.
   - `y_pred` (array-like): The predicted values from the model.
   - `title` (str, optional): The title of the plot.
   - `save_path` (str, optional): Path to save the plot image.
