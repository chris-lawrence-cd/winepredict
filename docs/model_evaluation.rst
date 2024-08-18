model_evaluation Module
=======================

The `model_evaluation` module provides functions to evaluate the performance of trained models. This includes cross-validation, performance metrics computation, and other evaluation techniques. These functions help in understanding how well the models are generalizing and where improvements can be made.

Module Contents
---------------

.. automodule:: winepredict.model_evaluation
   :members:
   :undoc-members:
   :show-inheritance:

Functions Overview
------------------

The following functions are part of the `model_evaluation` module:

1. **cross_validate_model**:
   Performs cross-validation on the model, providing a robust estimate of its performance on unseen data by splitting the data into multiple folds.

2. **evaluate_model**:
   Evaluates a trained model using various performance metrics, such as accuracy, precision, recall, and F1 score.

Detailed Function Descriptions
------------------------------

Below are detailed descriptions of each function within the `model_evaluation` module:

.. autofunction:: winepredict.model_evaluation.cross_validate_model

   **Description**:
   - Performs k-fold cross-validation on the specified model and dataset.
   - This function is useful for assessing the model's performance and variability across different subsets of data.

   **Parameters**:
   - `model` (object): The machine learning model to be evaluated.
   - `X` (pandas.DataFrame): The feature matrix.
   - `y` (pandas.Series): The target variable.
   - `cv` (int): The number of cross-validation folds (default: 5).

   **Returns**:
   - dict: A dictionary containing cross-validation scores for each fold, along with the mean and standard deviation.

.. autofunction:: winepredict.model_evaluation.evaluate_model

   **Description**:
   - Evaluates a trained model's performance using multiple metrics.
   - This function can be used to calculate accuracy, precision, recall, F1 score, and confusion matrix, among others.

   **Parameters**:
   - `model` (object): The trained machine learning model.
   - `X_test` (pandas.DataFrame): The test feature matrix.
   - `y_test` (pandas.Series): The true labels for the test set.

   **Returns**:
   - dict: A dictionary containing the calculated performance metrics.

Usage Examples
--------------

The following example demonstrates how to use the functions in the `model_evaluation` module:

```python
from winepredict.model_evaluation import cross_validate_model, evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Sample data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a model
model = RandomForestClassifier()

# Perform cross-validation
cv_results = cross_validate_model(model, X_train, y_train, cv=5)
print("Cross-validation results:", cv_results)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
evaluation_metrics = evaluate_model(model, X_test, y_test)
print("Evaluation metrics:", evaluation_metrics)
