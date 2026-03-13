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
    cols_to_drop = ['y', 'target_volatility', 'close', 'high', 'low', 'open']

    # Only keep the Lags and the Means
    X = df.drop(columns=cols_to_drop)
    y = df['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def run_random_forest(X_train, X_test, y_train):
    model = RandomForestRegressor(
        n_estimators=config['models']['random_forest']['n_estimators'],
        max_depth=config['models']['random_forest']['max_depth']
        # random_state=config['random_seed']
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, y_pred

# 3. XGBoost Regressor
def run_xgboost(X_train, X_test, y_train, y_test):
    # feature_cols = [
    #     'returns', 'return_lag_1', 'return_lag_5', 'return_lag_10',
    #     'vol_lag_1', 'vol_lag_5', 'vol_lag_10',
    #     'mean_5', 'mean_10', 'mean_30', 'volume'
    # ]
    
    # X = df[feature_cols]
    # y = df['y']
    model = xgb.XGBRegressor(
        n_estimators=config['models']['xgboost']['n_estimators'],
        learning_rate=config['models']['xgboost']['learning_rate'],
        max_depth=config['models']['xgboost']['max_depth']
        # random_state=config['random_seed']
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, y_pred

# 4. ARIMA Model
# Note: ARIMA usually only looks at the target variable itself (univariate)
def run_arima(df):
    # ARIMA needs a series, not a matrix. We use the 'target_volatility'
    series = df['target_volatility']
    train_size = int(len(series) * 0.8)
    train, test = series[0:train_size], series[train_size:len(series)]
    
    # Configuration from your YAML
    p = config['models']['arima']['p']
    d = config['models']['arima']['d']
    q = config['models']['arima']['q']
    
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()
    
    # Forecast
    forecast = model_fit.forecast(steps=len(test))
    mae = mean_absolute_error(test, forecast)
    logger.info(f"ARIMA Mean Absolute Error: {mae:.6f}")
    return model_fit, forecast