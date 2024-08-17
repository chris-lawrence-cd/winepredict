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
import warnings

def train_and_evaluate_models(scaled_df):
    """Splits the dataset, trains multiple models, and evaluates their performance.

    Args:
        scaled_df (pd.DataFrame): The preprocessed and scaled DataFrame containing the features and target variable.

    Returns:
        pd.DataFrame: A DataFrame containing the performance metrics of each model.
    """
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
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Neural Network": MLPRegressor(max_iter=2000),  # Increased max_iter to 2000
        "Support Vector Machine (RBF Kernel)": SVR(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor(),
        "LightGBM": LGBMRegressor(verbose=-1),  # Suppress LightGBM warnings
        "CatBoost": CatBoostRegressor(verbose=0)
    }

    # Number of observations
    n = X_test.shape[0]
    # Number of predictors
    p = X_test.shape[1]

    results = {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Calculate adjusted R²
        adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

        results[name] = {'R²': r2, 'Adjusted R²': adj_r2, 'RMSE': rmse, 'MAE': mae}
        print(f"{name} trained.")

    # Display results in a more readable format
    results_df = pd.DataFrame(results).T
    print("Model Performance:")
    print(results_df)

    return results_df

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
    # Hyperparameter tuning for CatBoost
    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [100, 200, 300]
    }
    catboost = CatBoostRegressor(verbose=0)
    grid_search = GridSearchCV(estimator=catboost, param_grid=param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_catboost = grid_search.best_estimator_
    print("Best CatBoost parameters:", grid_search.best_params_)

    # Evaluate best model
    y_pred_best = best_catboost.predict(X_test)
    r2_best = r2_score(y_test, y_pred_best)
    rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
    mae_best = mean_absolute_error(y_test, y_pred_best)

    print(f"Best CatBoost R²: {r2_best:.5f}")
    print(f"Best CatBoost RMSE: {rmse_best:.5f}")
    print(f"Best CatBoost MAE: {mae_best:.5f}")

    # Visualize feature importance for the best model
    feature_importance = pd.Series(best_catboost.feature_importances_, index=X_train.columns)
    plt.figure(figsize=(12, 8))
    feature_importance.sort_values().plot(kind='barh')
    plt.title('Feature Importance - Best CatBoost Model')
    plt.savefig('Feature_Importance.png')
    plt.show()

    # Rank feature importance
    ranked_feature_importance = feature_importance.sort_values(ascending=False)

    return {
        'best_estimator': best_catboost,
        'best_params': grid_search.best_params_,
        'R²': r2_best,
        'RMSE': rmse_best,
        'MAE': mae_best,
        'ranked_feature_importance': ranked_feature_importance
    }
