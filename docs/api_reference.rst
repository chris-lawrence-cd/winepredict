API Reference
=============

This section provides detailed information about the modules and functions in the WinePredict library.

Overview
--------

The WinePredict library is a comprehensive toolkit designed for the analysis, training, evaluation, and visualization of models related to wine data prediction. This documentation covers the primary modules and key functions that are essential for working with the library.

Modules Overview
----------------

The library is divided into several core modules:

- **Data Processing**: Prepares and processes data for analysis and modeling.
- **Model Training**: Handles the training and tuning of predictive models.
- **Model Evaluation**: Evaluates the performance of trained models using various metrics.
- **Visualization**: Provides tools for visualizing data, predictions, and model performance.

Data Processing Module
----------------------

The Data Processing module contains functions that load, clean, and prepare data for analysis and model training.

.. automodule:: winepredict.data_processing
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions:
^^^^^^^^^^^^^^

- **process_fred_data**: Loads and processes data from the FRED API for analysis.
- **preprocess_and_analyze_data**: Performs preprocessing and exploratory analysis on the data.

Model Training Module
---------------------

The Model Training module contains functions that train machine learning models and tune their hyperparameters for optimal performance.

.. automodule:: winepredict.model_training
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions:
^^^^^^^^^^^^^^

- **train_and_evaluate_models**: Trains various models and evaluates their performance.
- **tune_and_evaluate_catboost**: Tunes and evaluates the CatBoost model for better accuracy.

Model Evaluation Module
-----------------------

The Model Evaluation module provides tools for assessing the performance of the trained models using cross-validation and other metrics.

.. automodule:: winepredict.model_evaluation
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions:
^^^^^^^^^^^^^^

- **evaluate_model**: Evaluates a model's performance on a given test dataset.
- **cross_validate_model**: Performs cross-validation to assess model stability and performance.

Visualization Module
--------------------

The Visualization module offers functions for plotting various aspects of the data and model predictions.

.. automodule:: winepredict.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions:
^^^^^^^^^^^^^^

- **plot_actual_vs_predicted**: Plots the actual vs. predicted values for a model.
- **visualize_residuals**: Visualizes residuals to assess the model's accuracy.
- **plot_feature_importance**: Displays feature importance as calculated by the model.

Usage Examples
--------------

For more details on how to use these functions, refer to the following sections:

- **Data Processing**: Loading and cleaning data before modeling.
- **Model Training**: Training and tuning models.
- **Model Evaluation**: Evaluating model performance.
- **Visualization**: Creating plots for model results.

Refer to the module-specific sections for comprehensive details on all available functions and classes.
