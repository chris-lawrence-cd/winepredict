Usage
=====

This section provides examples of how to use the WinePredict library.

.. code-block:: python

    # Import necessary functions from the library
    from winepredict.data_processing import preprocess_and_analyze_data, process_fred_data
    from winepredict.model_training import train_and_evaluate_models, tune_and_evaluate_catboost
    from winepredict.model_evaluation import evaluate_model, cross_validate_model
    from winepredict.visualization import plot_actual_vs_predicted, visualize_residuals, plot_feature_importance
    from winepredict import info

    # Print library information
    info()

    # Define FRED API key
    api_key = 'your_api_key_here'

    # Step 1: Download and process FRED data
    process_fred_data(api_key)

    # Step 2: Preprocess and analyze data
    file_path = 'FRED_Data.xlsx'
    scaled_df = preprocess_and_analyze_data(file_path, output_file='correlation_matrix.png', save_vif=True)

    # Step 3: Train and evaluate multiple models
    results_df = train_and_evaluate_models(scaled_df)

    # Step 4: Tune and evaluate CatBoost model
    X = scaled_df.drop(['Average Wine Price'], axis=1)
    y = scaled_df['Average Wine Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
    catboost_results = tune_and_evaluate_catboost(X_train, y_train, X_test, y_test)
	best_catboost = catboost_results['best_estimator']

    # Plot actual vs. predicted prices for the best model
    plot_actual_vs_predicted(y_test, best_catboost.predict(X_test))

    # Cross-validation for model robustness
    cv_results = cross_validate_model(best_catboost, X, y)

    # Evaluate the best model on training and testing data
    evaluate_model(best_catboost, X_train, X_test, y_train, y_test)

    # Visualize residuals
    visualize_residuals(y_test, best_catboost.predict(X_test))

    # Plot feature importance
    plot_feature_importance(catboost_results['ranked_feature_importance'])