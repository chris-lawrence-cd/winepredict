Welcome to WinePredict's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   usage
   advanced_usage
   api_reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
Now, let's create the advanced_usage.rst file:

Advanced Usage
==============

This section covers more advanced topics and use cases for the WinePredict library.

Custom Model Integration
------------------------

While WinePredict comes with a variety of pre-implemented models, you may want to integrate your own custom model. Here's how you can do that:

.. code-block:: python

    from sklearn.base import BaseEstimator, RegressorMixin
    from winepredict.model_training import train_and_evaluate_models

    class CustomModel(BaseEstimator, RegressorMixin):
        def __init__(self, param1=1, param2=2):
            self.param1 = param1
            self.param2 = param2

        def fit(self, X, y):
            # Implement your custom fitting logic here
            return self

        def predict(self, X):
            # Implement your custom prediction logic here
            return predictions

    # Add your custom model to the list of models
    custom_models = {
        "Custom Model": CustomModel()
    }

    results_df = train_and_evaluate_models(scaled_df, additional_models=custom_models)

Feature Engineering
-------------------

WinePredict allows you to easily add custom features to your dataset. Here's an example of how to create interaction terms:

.. code-block:: python

    import pandas as pd
    from winepredict.data_processing import preprocess_and_analyze_data

    def add_interaction_terms(df):
        df['GDP_Unemployment'] = df['GDP'] * df['Unemployment Rate']
        df['CPI_RetailSales'] = df['Consumer Price Index (CPI)'] * df['Retail Sales']
        return df

    file_path = 'FRED_Data.xlsx'
    scaled_df = preprocess_and_analyze_data(file_path, custom_preprocessing=add_interaction_terms)

Time Series Cross-Validation
----------------------------

For time series data, it's often more appropriate to use time series cross-validation. Here's how you can implement this with WinePredict:

.. code-block:: python

    from sklearn.model_selection import TimeSeriesSplit
    from winepredict.model_evaluation import cross_validate_model

    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = cross_validate_model(best_catboost, X, y, cv=tscv)

Ensemble Methods
----------------

You can create ensemble models by combining predictions from multiple models:

.. code-block:: python

    from sklearn.ensemble import VotingRegressor
    from winepredict.model_training import train_and_evaluate_models

    # Train individual models
    results_df = train_and_evaluate_models(scaled_df)

    # Create an ensemble
    ensemble = VotingRegressor([
        ('catboost', results_df.loc['CatBoost', 'model']),
        ('xgboost', results_df.loc['XGBoost', 'model']),
        ('rf', results_df.loc['Random Forest', 'model'])
    ])

    # Evaluate the ensemble
    ensemble.fit(X_train, y_train)
    ensemble_score = ensemble.score(X_test, y_test)
    print(f"Ensemble R² score: {ensemble_score:.5f}")

Hyperparameter Optimization
---------------------------

For more advanced hyperparameter tuning, you can use techniques like Bayesian Optimization. Here's an example using the `scikit-optimize` library:

.. code-block:: python

    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    from winepredict.model_training import CatBoostRegressor

    # Define the search space
    search_spaces = {
        'learning_rate': Real(0.01, 1.0, 'log-uniform'),
        'depth': Integer(3, 10),
        'l2_leaf_reg': Real(1e-3, 10, 'log-uniform'),
        'iterations': Integer(50, 300)
    }

    # Create the BayesSearchCV object
    bayes_search = BayesSearchCV(
        CatBoostRegressor(verbose=0),
        search_spaces,
        n_iter=50,
        cv=5,
        n_jobs=-1,
        random_state=42
    )

    # Perform the search
    bayes_search.fit(X_train, y_train)

    print("Best parameters found: ", bayes_search.best_params_)
    print("Best cross-validation score: ", bayes_search.best_score_)

Feature Selection
-----------------

WinePredict can be extended with advanced feature selection techniques. Here's an example using Recursive Feature Elimination (RFE):

.. code-block:: python

    from sklearn.feature_selection import RFE
    from winepredict.model_training import RandomForestRegressor

    # Create the RFE object and specify the estimator
    rfe = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=10)

    # Fit RFE
    rfe = rfe.fit(X_train, y_train)

    # Get the selected features
    selected_features = X_train.columns[rfe.support_]
    print("Selected features: ", selected_features)

    # Use selected features for model training
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Train and evaluate model with selected features
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_selected, y_train)
    score = model.score(X_test_selected, y_test)
    print(f"Model R² score with selected features: {score:.5f}")

