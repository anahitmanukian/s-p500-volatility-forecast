import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import pandas as pd

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

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_predictions(y_true, y_pred, model_name, dates=None):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'{model_name} — Volatility Forecast Evaluation', fontsize=14)

    # Plot 1: Actual vs Predicted over time
    ax1 = axes[0]
    x = dates if dates is not None else range(len(y_true))
    ax1.plot(x, y_true, label='Actual', color='black', linewidth=1.2)
    ax1.plot(x, y_pred, label='Predicted', color='royalblue', linewidth=1, alpha=0.8)
    ax1.set_title('Actual vs Predicted Volatility')
    ax1.set_ylabel('Volatility')
    ax1.legend()

    # Plot 2: Residuals (errors) over time
    ax2 = axes[1]
    residuals = np.array(y_true) - np.array(y_pred)
    ax2.plot(x, residuals, color='blue', linewidth=0.8)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_title('Residuals (Actual − Predicted)')
    ax2.set_ylabel('Error')

    # Plot 3: Scatter — perfect model would be a diagonal line
    ax3 = axes[2]
    ax3.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    ax3.set_title('Predicted vs Actual (diagonal = perfect)')
    ax3.set_xlabel('Actual')
    ax3.set_ylabel('Predicted')

    plt.tight_layout()
    plt.savefig(f'reports/{model_name.lower().replace(" ", "_")}_evaluation.png', dpi=150)
    logger.info(f"Saved plot to reports/{model_name.lower().replace(' ', '_')}_evaluation.png")
    plt.show()