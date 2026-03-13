import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

def run_model_diagnostics(model, model_name, X_train=None):

    # Tree models
    if hasattr(model, "feature_importances_") and X_train is not None:

        import pandas as pd

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

    directional_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    logger.info(f"{model_name} MAE: {mae:.6f}")
    logger.info(f"{model_name} RMSE: {rmse:.6f}")
    logger.info(f"{model_name} R2: {r2:.4f}")
    logger.info(f"{model_name} Directional Accuracy: {directional_acc:.3f}")

    # Model-specific diagnostics
    run_model_diagnostics(model, model_name, X_train)

    return mae, rmse, r2, directional_acc