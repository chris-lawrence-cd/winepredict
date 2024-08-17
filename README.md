# Forecasting wine prices

We explore the predictive power of various machine learning models in forecasting wine prices using a comprehensive dataset of economic indicators. Building on previous research that primarily utilized Lasso and Ridge regression models, we extend the analysis to include a wider range of models such as Linear Regression, K-Nearest Neighbours, Neural Networks, Support Vector Machines, Decision Trees, Random Forests, Gradient Boosting, XGBoost, LightGBM, and CatBoost. Our dataset spans from January 1996 to June 2024 and includes monthly data on key economic variables sourced from FRED. The performance of each model is evaluated using metrics such as R², RMSE, and MAE. Our findings indicate that while traditional models like Ridge Regression perform well, more sophisticated models such as K-Nearest Neighbours and Gradient Boosting also show strong predictive capabilities. Notably, the CatBoost model, after hyperparameter tuning, demonstrates significant potential with an R² score of 0.92763. These results highlight the importance of considering a diverse set of models for accurate wine price forecasting, providing valuable insights for investors, producers, and policymakers.

Overleaf: https://www.overleaf.com/read/ydvfhffcycdw#7b39fa

# Directory Structure


winepredict/
├── winepredict/
│ ├── init.py/
│ ├── data_processing.py/
│ ├── model_training.py/
│ ├── model_evaluation.py/
│ └── visualization.py/
├── tests/
│ ├── init.py/
│ ├── test_data_processing.py/
│ ├── test_model_training.py/
│ ├── test_model_evaluation.py/
│ └── test_visualization.py/
├── pyproject.toml/
└── README.md/

