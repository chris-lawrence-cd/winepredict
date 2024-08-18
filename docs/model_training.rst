model_training Module
=====================

The `model_training` module is responsible for training and fine-tuning various machine learning models used in the WinePredict library. It includes a wide range of regression models, as well as utilities for hyperparameter tuning and model evaluation.

Module Contents
---------------

.. automodule:: winepredict.model_training
   :members:
   :undoc-members:
   :show-inheritance:

Classes Overview
----------------

The following classes are available in the `model_training` module. They represent different machine learning models and utilities:

1. **CatBoostRegressor**:
   A regressor based on the CatBoost algorithm, which is particularly effective with categorical features and small datasets.

2. **ConvergenceWarning**:
   A warning class used when a model does not converge during training.

3. **DecisionTreeRegressor**:
   A decision tree regressor that can be used for predicting continuous values.

4. **GradientBoostingRegressor**:
   A regressor based on the gradient boosting algorithm, combining multiple weak models into a strong one.

5. **GridSearchCV**:
   A class for exhaustive search over specified hyperparameters for an estimator.

6. **KNeighborsRegressor**:
   A regressor that uses the k-nearest neighbors algorithm.

7. **LGBMRegressor**:
   A LightGBM regressor, known for its speed and efficiency in large datasets.

8. **Lasso**:
   A linear model with L1 regularization, useful for feature selection.

9. **LinearRegression**:
   A basic linear regression model for predicting continuous values.

10. **MLPRegressor**:
    A multi-layer perceptron regressor, suitable for complex, non-linear problems.

11. **RandomForestRegressor**:
    A regressor that fits multiple decision trees on various sub-samples of the dataset.

12. **Ridge**:
    A linear regression model with L2 regularization to prevent overfitting.

13. **SVR**:
    A support vector regressor for robust, non-linear predictions.

14. **XGBRegressor**:
    A regressor based on the XGBoost algorithm, effective for tabular data.

Class Documentation
-------------------

Here is a detailed breakdown of each class:

.. autoclass:: winepredict.model_training.CatBoostRegressor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: winepredict.model_training.ConvergenceWarning
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: winepredict.model_training.DecisionTreeRegressor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: winepredict.model_training.GradientBoostingRegressor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: winepredict.model_training.GridSearchCV
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: winepredict.model_training.KNeighborsRegressor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: winepredict.model_training.LGBMRegressor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: winepredict.model_training.Lasso
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: winepredict.model_training.LinearRegression
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: winepredict.model_training.MLPRegressor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: winepredict.model_training.RandomForestRegressor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: winepredict.model_training.Ridge
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: winepredict.model_training.SVR
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: winepredict.model_training.XGBRegressor
   :members:
   :undoc-members:
   :show-inheritance:

Functions Overview
------------------

The `model_training` module includes the following key functions:

1. **train_and_evaluate_models**:
   Trains and evaluates a set of machine learning models, comparing their performance using cross-validation and other metrics.

2. **tune_and_evaluate_catboost**:
   Tunes hyperparameters for the CatBoostRegressor using grid search and evaluates its performance.

Detailed Function Descriptions
------------------------------

Below are detailed descriptions of each function:

.. autofunction:: winepredict.model_training.train_and_evaluate_models

   **Description**:
   - Trains multiple machine learning models and evaluates them using cross-validation.
   - This function provides a comprehensive comparison of models' performance across different metrics.

   **Parameters**:
   - `X` (pandas.DataFrame): The feature matrix.
   - `y` (pandas.Series): The target variable.
   - `models` (list): A list of machine learning models to train.
   - `cv` (int): The number of cross-validation folds (default: 5).

   **Returns**:
   - dict: A dictionary containing the performance metrics for each model.

.. autofunction:: winepredict.model_training.tune_and_evaluate_catboost

   **Description**:
   - Performs hyperparameter tuning for the CatBoostRegressor and evaluates its performance on the validation set.
   - This function helps in finding the best set of hyperparameters for the CatBoost model.

   **Parameters**:
   - `X_train` (pandas.DataFrame): The training feature matrix.
   - `y_train` (pandas.Series): The training target variable.
   - `X_val` (pandas.DataFrame): The validation feature matrix.
   - `y_val` (pandas.Series): The validation target variable.

   **Returns**:
   - object: The tuned CatBoostRegressor model.

Usage Examples
--------------

Here’s an example of how to use the functions in the `model_training` module:

```python
from winepredict.model_training import train_and_evaluate_models, tune_and_evaluate_catboost
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Sample data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = [RandomForestRegressor(), GradientBoostingRegressor()]

# Train and evaluate models
results = train_and_evaluate_models(X_train, y_train, models, cv=5)
print("Model evaluation results:", results)

# Tune and evaluate CatBoost
best_catboost = tune_and_evaluate_catboost(X_train, y_train, X_val, y_val)
