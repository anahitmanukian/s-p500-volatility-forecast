import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # use file-based backend, no GUI window needed
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def run_model_diagnostics(model, model_name, X_train=None):

    # Tree models
    if hasattr(model, "feature_importances_") and X_train is not None:

        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        logger.info(f"{model_name} Top Features:\n{importance.head(10)}")

    # ARIMA diagnostics
    if hasattr(model, "aic"):

        logger.info(f"{model_name} AIC: {model.aic}")
        logger.info(f"{model_name} BIC: {model.bic}")

def evaluate_regression(model, y_true, y_pred, model_name, X_train=None):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # WRONG for volatility - vol is always positive, sign is always 1
    # directional_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    
    # CORRECT - did we predict the direction of change?
    
    # y_true is a pandas Series with the original date index, but pd.Series(y_pred) gets a default integer index (0, 1, 2...).
    # Fix by resetting the index on y_true before diffing: 
    directional_acc = np.mean(
    np.sign(y_true.reset_index(drop=True).diff()) == 
    np.sign(pd.Series(y_pred).diff())
    )

    logger.info(f"{model_name} MAE: {mae:.6f}")
    logger.info(f"{model_name} RMSE: {rmse:.6f}")
    logger.info(f"{model_name} R2: {r2:.4f}")
    logger.info(f"{model_name} Directional Accuracy: {directional_acc:.3f}")

    # Model-specific diagnostics
    run_model_diagnostics(model, model_name, X_train)

    return mae, rmse, r2, directional_acc

import matplotlib.dates as mdates

    
def plot_all_models(y_true, predictions: dict, dates=None,filename='model_comparison'):
    # predictions = {
    #     "Random Forest": y_pred_rf,
    #     "XGBoost": y_pred_xgb,
    #     "ARIMA": forecast_arima
    # }
    
    colors = ['royalblue', 'darkorange', 'green']
    x = dates if dates is not None else range(len(y_true))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # create ONCE outside loop
    fig.suptitle('Model Comparison — Volatility Forecast', fontsize=14)  
    
    axes[0].plot(x, y_true, label='Actual', color='black', linewidth=1.2)
    for i, (name, y_pred) in enumerate(predictions.items()):
        axes[0].plot(x, y_pred, label=name, color=colors[i], linewidth=1, alpha=0.8)
    axes[0].set_title('Actual vs Predicted')
    axes[0].legend()
    
    for i, (name, y_pred) in enumerate(predictions.items()):
        residuals = np.array(y_true) - np.array(y_pred)
        axes[1].plot(x, residuals, color=colors[i], linewidth=0.8, label=name)
        axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_title('Residuals (Actual − Predicted)')
    axes[1].legend()   
    
    for i, (name, y_pred) in enumerate(predictions.items()):
        axes[2].scatter(y_true, y_pred, label=name, alpha=0.3, s=10, color=colors[i])

    # diagonal goes AFTER the loop, once
    min_val = min(y_true)
    max_val = max(y_true)
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    axes[2].set_title('Predicted vs Actual (diagonal = perfect)')
    axes[2].set_xlabel('Actual')
    axes[2].set_ylabel('Predicted')
    axes[2].legend()
    
        
    plt.tight_layout()
    plt.savefig(f'reports/{filename}.png', dpi=150)   
        
        
    
      
    