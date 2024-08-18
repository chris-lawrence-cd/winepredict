Introduction
============

WinePredict is a Python library designed to facilitate wine price prediction using advanced machine learning models. This library provides a comprehensive set of tools for data processing, model training, evaluation, and visualization, specifically tailored for the wine industry.

Key Features
------------

1. **Data Processing**: Automated collection and preprocessing of economic indicators from FRED (Federal Reserve Economic Data).
2. **Multiple Models**: Implementation of various machine learning models including:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - K-Nearest Neighbors
   - Neural Networks
   - Support Vector Machines
   - Decision Trees
   - Random Forests
   - Gradient Boosting
   - XGBoost
   - LightGBM
   - CatBoost

3. **Model Evaluation**: Comprehensive evaluation metrics including R², Adjusted R², RMSE, and MAE.
4. **Feature Importance**: Analysis and visualization of the most influential factors in wine price prediction.
5. **Visualization Tools**: Functions for plotting actual vs. predicted prices, residual analysis, and more.

Target Audience
---------------

WinePredict is designed for:

- Data scientists and analysts in the wine industry
- Economists studying wine market trends
- Wine producers and investors seeking data-driven insights
- Researchers in the field of agricultural economics

This documentation provides an overview of the library's features, installation instructions, usage examples, and a complete API reference to help you get started with WinePredict.


For the usage.rst file, you could include a basic example of how to use the library:

Usage
=====

This section provides examples of how to use the WinePredict library for wine price prediction.

Basic Usage
-----------

Here's a simple example of how to use WinePredict to download data, preprocess it, train models, and evaluate their performance:

.. code-block:: python

    from winepredict.data_processing import preprocess_and_analyze_data, process_fred_data
    from winepredict.model_training import train_and_evaluate_models, tune_and_evaluate_catboost
    from winepredict.model_evaluation import evaluate_model, cross_validate_model
    from winepredict.visualization import plot_actual_vs_predicted, visualize_residuals, plot_feature_importance

    # Define FRED API key
    api_key = 'your_api_key_here'

    # Download and process FRED data
    process_fred_data(api_key)

    # Preprocess and analyze data
    file_path = 'FRED_Data.xlsx'
    scaled_df = preprocess_and_analyze_data(file_path, output_file='correlation_matrix.png', save_vif=True)

    # Train and evaluate multiple models
    results_df = train_and_evaluate_models(scaled_df)

    # Tune and evaluate CatBoost model
    X = scaled_df.drop(['Average Wine Price'], axis=1)
    y = scaled_df['Average Wine Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
    catboost_results = tune_and_evaluate_catboost(X_train, y_train, X_test, y_test)
    best_catboost = catboost_results['best_estimator']

    # Visualize results
    plot_actual_vs_predicted(y_test, best_catboost.predict(X_test))
    visualize_residuals(y_test, best_catboost.predict(X_test))
    plot_feature_importance(catboost_results['ranked_feature_importance'])

This example demonstrates the basic workflow of using WinePredict, from data processing to model evaluation and visualization.

Advanced Usage
--------------

For more advanced usage and customization options, please refer to the API Reference section.
