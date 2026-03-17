import pandas as pd
import numpy as np
from src.config_loader import load_config 
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_volatility_features(df, config=None):
    """ Open: The price when the market first opened for the day.
        High: The highest price reached during the trading day.
        Low: The lowest price reached during the trading day.
        Close: The final price when the market closed. This is the standard "benchmark" price used for most calculations and models.
        Volume: The total number of shares traded during that day. It measures the "strength" or "energy" behind a price movement.
        
        In financial data, the Close price is generally considered the most important metric 
        because it represents the final "valuation" of the asset at the end of the trading day.
        
        Lags:
            1 — yesterday, captures immediate momentum
            5 — one trading week
            10 — two trading weeks
            21 — one trading month (markets have ~21 trading days per month)
            63 — one trading quarter (~63 trading days)
            252 - one trading year (~252 trading days)
      
    """
    
    # df['return'].shift(1)    # positive → shift DOWN → each row gets the value from 1 row ABOVE (the past)
    # df['return'].shift(-1)   # negative → shift UP  → each row gets the value from 1 row BELOW (the future)
    # ```

    # So visually:
    # ```
    # original:  [a, b, c, d, e]
    # shift(1):  [NaN, a, b, c, d]   ← everything moved down, row 0 gets past
    # shift(-1): [b, c, d, e, NaN]   ← everything moved up, row 0 gets future
    
    # df['close'] / df['rolling_mean_5']         close is 10% above MA → gives 1.10
    # df['close'] / df['rolling_mean_5'] - 1     close is 10% above MA → gives 0.10
    
    logger.info('Creating new features!') 
        
    if config is None:
        config = load_config()

    vol_win = config['features']['volatility_window']

    # 1. Returns
    df['returns'] = np.log(df['close'] / df['close'].shift(1))

    # 2. Rolling means (needed before dist_ma features)
    for window in config['features']['rolling_means']:
        df[f'mean_{window}'] = df['close'].rolling(window=window).mean()

    # 3. Target volatility (needed before vol_of_vol)
    df['target_volatility'] = df['returns'].rolling(window=vol_win).std()

    # 4. Derived features (all dependencies now exist)
    df['return_5']   = df['close'] / df['close'].shift(5) - 1
    df['return_30']  = df['close'] / df['close'].shift(30) - 1
    df['dist_ma10']  = df['close'] / df['mean_10'] - 1
    df['dist_ma30']  = df['close'] / df['mean_30'] - 1
    df['range']      = (df['high'] - df['low']) / df['close']
    df['volume_z']   = (df['volume'] - df['volume'].rolling(30).mean()) / df['volume'].rolling(30).std()
    df['vol_of_vol'] = df['target_volatility'].rolling(10).std()

    # 5. Lag features
    for lag in config['features']['lags']:
        df[f'vol_lag_{lag}']    = df['target_volatility'].shift(lag)
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

    # 6. Target (always last)
    df['y'] = df['returns'].shift(-vol_win).rolling(window=vol_win).std()

    logger.info('Features created successfully!')
    return df.dropna()

def save_feature_engineering_csv(df, feature_engineering_path=None):
    if feature_engineering_path is None:
        # Load from config
        config = load_config()
        feature_engineering_path = config['paths']['feature_engineering_path']
    
    # Create directory if it doesn't exist
    Path(feature_engineering_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the data
    df.to_csv(feature_engineering_path)
    logger.info(f"Saved cleaned data to {feature_engineering_path}")

# We use the logarithm because it makes the returns "additive" and helps normalize the data.
# If the S&P 500 goes from $100$ to $110$ (a $10\%$ gain) and then from $1,000$ to $1,100$ (also a $10\%$ gain),
# the log return treats these as identical movements. 
# This is essential for your model to generalize across different years as the S&P 500 grows.

# The Goal: 
    # We want to predict the Standard Deviation of Returns (the y column in your dataframe),
    # which tells us how much the price is likely to "swing" tomorrow, 
    # regardless of whether it goes up or down