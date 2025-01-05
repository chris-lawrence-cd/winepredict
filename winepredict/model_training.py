import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import warnings

def train_and_evaluate_models(scaled_df):
    """Splits the dataset, trains multiple models, and evaluates their performance.

    Args:
        scaled_df (pd.DataFrame): The preprocessed and scaled DataFrame containing the features and target variable.

    Returns:
        pd.DataFrame: A DataFrame containing the performance metrics of each model.
    """
    if scaled_df.empty:
        print("Error: Input DataFrame is empty.")
        return pd.DataFrame()

    # Splitting dataset into X and y
    y = scaled_df['Average Wine Price']
    X = scaled_df.drop(['Average Wine Price'], axis=1)

    # Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=1)

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        # "K-Nearest Neighbors": KNeighborsRegressor(),
        "Neural Network": MLPRegressor(max_iter=2000),
        "Support Vector Machine (RBF Kernel)": SVR(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor(),
        "LightGBM": LGBMRegressor(verbose=-1),
        "CatBoost": CatBoostRegressor(verbose=0)
    }

    results = []

    for name, model in models.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            results.append({
                "Model": name,
                "R²": r2,
                "RMSE": rmse,
                "MAE": mae
            })
        except Exception as e:
            print(f"Error training {name}: {e}")

    return pd.DataFrame(results)

def tune_and_evaluate_catboost(X_train, y_train, X_test, y_test):
    """Tunes hyperparameters for CatBoost, evaluates the best model, and visualizes feature importance.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target variable.

    Returns:
        dict: A dictionary containing the best parameters, performance metrics, and ranked feature importance of the best CatBoost model.
    """
    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'iterations': [100, 200, 300]
    }

    catboost_model = CatBoostRegressor(verbose=0)

    grid_search = GridSearchCV(estimator=catboost_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)

    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    except Exception as e:
        print(f"Error during CatBoost hyperparameter tuning: {e}")
        return {}

    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    feature_importances = best_model.get_feature_importance(prettified=True)

    return {
        "Best Parameters": grid_search.best_params_,
        "R²": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Feature Importances": feature_importances
    }
