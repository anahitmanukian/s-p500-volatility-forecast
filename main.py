from src.models import run_random_forest
from src.evaluation import evaluate_regression, plot_all_models
from src.data_loader import download_sp500, load_data
from src.data_cleaning import clean_raw_data, save_cleaned_data
from src.feature_engineering import create_volatility_features, save_feature_engineering_csv
from src.models import split_data, run_random_forest
import logging
from src.logger import setup_logging
from src.config_loader import load_config
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


config = load_config()

setup_logging()

logger = logging.getLogger(__name__)
logger.info("Logging works!")


from pathlib import Path
Path('reports').mkdir(exist_ok=True)

data = download_sp500()
df = load_data()

clean_df = clean_raw_data(df)
save_cleaned_data(clean_df)

feature_engineering_df = create_volatility_features(clean_df)

save_feature_engineering_csv(feature_engineering_df)

final_df = load_data(config['paths']['feature_engineering_path'])


X_train, X_test, y_train, y_test = split_data(final_df)

print(f"Train: {X_train.index[0].date()} to {X_train.index[-1].date()} ({len(X_train)} rows)")
print(f"Test:  {X_test.index[0].date()} to {X_test.index[-1].date()} ({len(X_test)} rows)")
print(f"Train y range: {y_train.min():.4f} to {y_train.max():.4f}")
print(f"Test y range:  {y_test.min():.4f} to {y_test.max():.4f}")

import numpy as np
from sklearn.metrics import mean_absolute_error
# In main.py, after split_data
rolling_baseline = y_train.rolling(21).mean().iloc[-1]  # last 21-day mean from train
baseline_pred_rolling = np.full(len(y_test), rolling_baseline)
print(f"Rolling baseline MAE: {mean_absolute_error(y_test, baseline_pred_rolling):.6f}")

model, y_pred = run_random_forest(X_train, X_test, y_train)

evaluate_regression(
    model,
    y_test,
    y_pred,
    model_name="Random Forest",
    X_train=X_train
)

from src.models import split_data, run_random_forest, run_xgboost, run_arima

# XGBoost — same split, same features
model_xgb, y_pred_xgb = run_xgboost(X_train, X_test, y_train)
evaluate_regression(model_xgb, y_test, y_pred_xgb, model_name="XGBoost", X_train=X_train)

# ARIMA — uses its own internal split, only needs the full df
model_arima, forecast_arima, arima_dates = run_arima(final_df)

# len(y_test) == len(forecast_arima)
predictions = {
    "Random Forest": y_pred,
    "XGBoost": y_pred_xgb,
    "ARIMA": forecast_arima.values
}

plot_all_models(
    y_true=y_test.values,
    predictions=predictions,
    dates=y_test.index,
    filename='model_comparison_single_split'
)

from src.models import walk_forward_rf, walk_forward_xgboost
model_rf_wf, rf_wf_preds, rf_wf_true, rf_wf_dates = walk_forward_rf(final_df, fold_size=252)
model_xgb_wf, xgb_wf_preds, xgb_wf_true, xgb_wf_dates = walk_forward_xgboost(final_df, fold_size=252)

wf_predictions = {
    "RF Walk-forward":   rf_wf_preds,
    "XGB Walk-forward":  xgb_wf_preds,
}

plot_all_models(
    y_true=rf_wf_true,
    predictions=wf_predictions,
    dates=rf_wf_dates,
    filename='model_comparison_walk_forward'
)

