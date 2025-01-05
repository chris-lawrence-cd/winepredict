<p align="center">
  <img src="images/image.png" alt="Header Image" width="800">
</p>


# WinePredict: Forecasting Wine Prices

![Contributors](https://img.shields.io/github/contributors/chris-lawrence-cd/winepredict)
![Issues](https://img.shields.io/github/issues/chris-lawrence-cd/winepredict)
![Repo Size](https://img.shields.io/github/repo-size/chris-lawrence-cd/winepredict)
![Last Commit](https://img.shields.io/github/last-commit/chris-lawrence-cd/winepredict)
![License](https://img.shields.io/github/license/chris-lawrence-cd/winepredict)

We explore the predictive power of various machine learning models in forecasting wine prices using a comprehensive dataset of economic indicators. Building on previous research that primarily utilized Lasso and Ridge regression models, we extend the analysis to include a wider range of models such as Linear Regression, Neural Networks, Support Vector Machines, Decision Trees, Random Forests, Gradient Boosting, XGBoost, LightGBM, and CatBoost. Our dataset spans from January 1996 to June 2024 and includes monthly data on key economic variables sourced from Federal Reserve Economic Data (FRED). The performance of each model is evaluated using metrics such as R², RMSE, and MAE. Our findings indicate that while traditional models like Ridge Regression perform well, other models such as Gradient Boosting also show strong predictive capabilities. Notably, the CatBoost model, after hyperparameter tuning, demonstrates significant potential with an R² score of 0.92763. These results highlight the importance of considering a diverse set of models for accurate wine price forecasting, providing valuable insights for investors, agents, and other industry participants.

Overleaf: https://www.overleaf.com/read/ydvfhffcycdw#7b39fa

## Installation

To install the WinePredict library on a Jupyter workbook, use the following command:

```python
!pip install git+https://github.com/chris-lawrence-cd/winepredict.git
```

## Usage

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
from winepredict.visualization import plot_actual_vs_predicted

# Load and preprocess the data
scaled_df = preprocess_and_analyze_data('FRED_Data.xlsx')

# Split the data and train models
X = scaled_df.drop(['Average Wine Price'], axis=1)
y = scaled_df['Average Wine Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=1)

# Train and evaluate multiple models
model_results = train_and_evaluate_models(scaled_df)
print("\nModel Performance Comparison:")
print(model_results)

# Tune and evaluate CatBoost model
catboost_results = tune_and_evaluate_catboost(X_train, y_train, X_test, y_test)

# The catboost_results dictionary contains:
# - "Best Parameters": The optimal hyperparameters found
# - "Best Estimator": The trained CatBoost model
# - "R²": R-squared score
# - "RMSE": Root Mean Square Error
# - "MAE": Mean Absolute Error
# - "Feature Importances": Importance scores for each feature

# Access the best model and make predictions
best_catboost = catboost_results['Best Estimator']
plot_actual_vs_predicted(y_test, best_catboost.predict(X_test))
```

## Model Performance

The CatBoost model, after hyperparameter tuning, typically achieves:
- R² (R-squared) score: ~0.89-0.90
- RMSE (Root Mean Square Error): ~0.90-0.95
- MAE (Mean Absolute Error): ~0.58-0.60

Feature importance analysis shows that economic indicators such as:
1. Producer Price Index
2. Personal Consumption Expenditures
3. Money Supply (M1)
4. Total Nonfarm Payrolls
5. Retail Sales

Have the strongest influence on wine price predictions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
