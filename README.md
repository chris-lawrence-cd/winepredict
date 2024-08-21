# WinePredict: Forecasting Wine Prices

![Contributors](https://img.shields.io/github/contributors/chris-lawrence-cd/winepredict)
![Issues](https://img.shields.io/github/issues/chris-lawrence-cd/winepredict)
![Repo Size](https://img.shields.io/github/repo-size/chris-lawrence-cd/winepredict)
![Lines of Code](https://img.shields.io/tokei/lines/github/chris-lawrence-cd/winepredict)
![Last Commit](https://img.shields.io/github/last-commit/chris-lawrence-cd/winepredict)
![License](https://img.shields.io/github/license/chris-lawrence-cd/winepredict)







We explore the predictive power of various machine learning models in forecasting wine prices using a comprehensive dataset of economic indicators. Building on previous research that primarily utilized Lasso and Ridge regression models, we extend the analysis to include a wider range of models such as Linear Regression, Neural Networks, Support Vector Machines, Decision Trees, Random Forests, Gradient Boosting, XGBoost, LightGBM, and CatBoost. Our dataset spans from January 1996 to June 2024 and includes monthly data on key economic variables sourced from Federal Reserve Economic Data (FRED). The performance of each model is evaluated using metrics such as R², RMSE, and MAE. Our findings indicate that while traditional models like Ridge Regression perform well, other models such as Gradient Boosting also show strong predictive capabilities. Notably, the CatBoost model, after hyperparameter tuning, demonstrates significant potential with an R² score of 0.92763. These results highlight the importance of considering a diverse set of models for accurate wine price forecasting, providing valuable insights for investors, producers, and other industry participants.

Overleaf: https://www.overleaf.com/read/ydvfhffcycdw#7b39fa

## Installation

To install the WinePredict library on a Jupyter workbook, use the following command:

```python
!pip install git+https://github.com/chris-lawrence-cd/winepredict.git
```

Usage

Here’s an example of how to use the WinePredict library:

```python

# Import necessary libraries
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=seed)
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
```

License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

