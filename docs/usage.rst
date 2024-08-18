Usage
=====

This section provides detailed examples of how to use the WinePredict library for wine price prediction.

Data Processing
---------------

The first step in using WinePredict is to download and process the necessary economic data:

.. code-block:: python

    from winepredict.data_processing import process_fred_data, preprocess_and_analyze_data

    # Define your FRED API key
    api_key = 'your_api_key_here'

    # Download and process FRED data
    process_fred_data(api_key)

    # Preprocess and analyze the data
    file_path = 'FRED_Data.xlsx'
    scaled_df = preprocess_and_analyze_data(file_path, output_file='correlation_matrix.png', save_vif=True)

This will download the required economic indicators, preprocess the data, and generate a correlation matrix visualization.

Model Training and Evaluation
-----------------------------

Once the data is processed, you can train and evaluate multiple models:

.. code-block:: python

    from winepredict.model_training import train_and_evaluate_models
    from sklearn.model_selection import train_test_split

    # Split the data into features (X) and target (y)
    X = scaled_df.drop(['Average Wine Price'], axis=1)
    y = scaled_df['Average Wine Price']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate multiple models
    results_df = train_and_evaluate_models(scaled_df)

    print(results_df)

This will train multiple models and display their performance metrics.

Advanced Model Tuning
---------------------

For more advanced users, WinePredict offers hyperparameter tuning for the CatBoost model:

.. code-block:: python

    from winepredict.model_training import tune_and_evaluate_catboost

    # Tune and evaluate CatBoost model
    catboost_results = tune_and_evaluate_catboost(X_train, y_train, X_test, y_test)
    best_catboost = catboost_results['best_estimator']

    print(f"Best CatBoost parameters: {catboost_results['best_params']}")
    print(f"Best CatBoost R²: {catboost_results['R²']:.5f}")

Visualization
-------------

WinePredict provides several visualization tools to help interpret the results:

.. code-block:: python

    from winepredict.visualization import plot_actual_vs_predicted, visualize_residuals, plot_feature_importance

    # Plot actual vs. predicted prices
    plot_actual_vs_predicted(y_test, best_catboost.predict(X_test))

    # Visualize residuals
    visualize_residuals(y_test, best_catboost.predict(X_test))

    # Plot feature importance
    plot_feature_importance(catboost_results['ranked_feature_importance'])

These visualizations can help in understanding the model's performance and the importance of different features in predicting wine prices.
