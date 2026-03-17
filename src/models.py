from sklearn.ensemble import RandomForestRegressor 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from src.config_loader import load_config
import logging

config = load_config()
logger = logging.getLogger(__name__)

# In Time Series, we must keep the order.
def split_data(df: pd.DataFrame):
    cols_to_drop = ['y', 'target_volatility', 'close', 'high', 'low', 'open', 'returns','volume']

    # Only keep the Lags and the Means
    X = df.drop(columns=cols_to_drop)
    y = df['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def run_random_forest(X_train, X_test, y_train):
    model = RandomForestRegressor(
        n_estimators=config['models']['random_forest']['n_estimators'],
        max_depth=config['models']['random_forest']['max_depth'],
        min_samples_leaf=10,      # prevents tiny leaf nodes
        max_features=0.5,         # use only 50% of features per split
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, y_pred

# 3. XGBoost Regressor
def run_xgboost(X_train, X_test, y_train, y_test):
    model = xgb.XGBRegressor(
        n_estimators=config['models']['xgboost']['n_estimators'],
        learning_rate=config['models']['xgboost']['learning_rate'],
        max_depth=config['models']['xgboost']['max_depth']
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, y_pred


def run_arima(df, max_steps=None):
    series = df['y'].reset_index(drop=True)
    train_size = int(len(series) * 0.8)
    
    train = series[:train_size]
    test  = series[train_size:]
    test_dates = df.index[train_size:]
    
    p = config['models']['arima']['p']
    d = config['models']['arima']['d']
    q = config['models']['arima']['q']
    
    steps = min(len(test), max_steps) if max_steps else len(test)
    logger.info(f"Starting ARIMA rolling forecast for {steps} steps...")
    
    predictions = []
    window = 252  # only use last 1 year of data — much faster, often better
    
    for i in range(steps):
        # Fixed window instead of ever-growing history
        history = series[max(0, train_size + i - window) : train_size + i].tolist()
        
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit(method_kwargs={"maxiter": 200})  # reduce maxiter too
        
        pred = model_fit.forecast(steps=1)
        predictions.append(pred[0])
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Rolling forecast: {i+1}/{steps} steps done")
    
    forecast = pd.Series(predictions)
    mae = mean_absolute_error(test[:steps], forecast)
    logger.info(f"ARIMA Rolling Forecast MAE: {mae:.6f}")
    
    return model_fit, forecast, test_dates[:steps]
# ```

# ---

# **Step by step what happens on each iteration:**
# ```
# i=0:
#   history = [day1, day2, ... day5020]       ← all training data
#   fit ARIMA on history
#   predict day 5021 → predictions = [0.0098]
#   reveal true day5021 = 0.0091
#   history = [day1, ... day5020, day5021]    ← grow by 1

# i=1:
#   history = [day1, day2, ... day5021]       ← one day longer
#   fit ARIMA on history
#   predict day 5022 → predictions = [0.0098, 0.0094]
#   reveal true day5022 = 0.0089
#   history = [day1, ... day5021, day5022]    ← grow by 1

# ... repeats 1256 times