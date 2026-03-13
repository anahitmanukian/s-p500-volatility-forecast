import pandas as pd
import numpy as np
from src.config_loader import load_config 
import logging

logger = logging.getLogger(__name__)

def create_volatility_features(df, config=None):
    """ Open: The price when the market first opened for the day.
        High: The highest price reached during the trading day.
        Low: The lowest price reached during the trading day.
        Close: The final price when the market closed. This is the standard "benchmark" price used for most calculations and models.
        Volume: The total number of shares traded during that day. It measures the "strength" or "energy" behind a price movement.
        
        In financial data, the Close price is generally considered the most important metric 
        because it represents the final "valuation" of the asset at the end of the trading day.
        
    """
    logger.info('Creating new features!') 
            
    if config is None:
        config = load_config()

    # 1. Calculate Target: Daily Log Returns
    # We use log returns to normalize the S&P 500's growth over decades
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Calculate Realized Volatility (The value we want to predict)
    # Using 'volatility_window: 30' from your config
    vol_win = config['features']['volatility_window']
    df['target_volatility'] = df['returns'].rolling(window=vol_win).std()

    # 3. Create Rolling Mean Features
    # This helps the model see if the price is currently "high" or "low" relative to the past
    for window in config['features']['rolling_means']:
        df[f'mean_{window}'] = df['close'].rolling(window=window).mean()

    # 4. Create Lag Features (The "Memory")
    # This tells the model what happened 1, 5, and 10 days ago
    for lag in config['features']['lags']:
        df[f'vol_lag_{lag}'] = df['target_volatility'].shift(lag)
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)

    # 5. Shift the Target for Forecast
    # Since 'horizon: 1', we want today's row to predict tomorrow's volatility
    df['y'] = df['target_volatility'].shift(-config['forecast']['horizon'])
    
    logger.info('Features created succesfully!')
    # Drop NaNs created by rolling windows and lags
    return df.dropna()

# We use the logarithm because it makes the returns "additive" and helps normalize the data.
# If the S&P 500 goes from $100$ to $110$ (a $10\%$ gain) and then from $1,000$ to $1,100$ (also a $10\%$ gain),
# the log return treats these as identical movements. 
# This is essential for your model to generalize across different years as the S&P 500 grows.

# The Goal: 
    # We want to predict the Standard Deviation of Returns (the y column in your dataframe),
    # which tells us how much the price is likely to "swing" tomorrow, 
    # regardless of whether it goes up or down