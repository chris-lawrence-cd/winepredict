"""
WinePredict Package

This package provides tools for predicting wine prices using machine learning models.
It includes modules for data processing, model training, evaluation, and visualization.

Modules:
- data_processing: Functions for downloading and preprocessing data.
- model_training: Functions for training and tuning machine learning models.
- model_evaluation: Functions for evaluating model performance.
- visualization: Functions for visualizing predictions and residuals.
- info: Provides package information and metadata.
"""

from .data_processing import download_fred_data, process_fred_data, preprocess_and_analyze_data
from .model_training import train_and_evaluate_models, tune_and_evaluate_catboost
from .model_evaluation import evaluate_model, cross_validate_model
from .visualization import plot_actual_vs_predicted, visualize_residuals
from .info import info
