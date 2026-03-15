# src/data_cleaning.py
import pandas as pd
import logging
from pathlib import Path
from src.config_loader import load_config, get_processed_data_path

logger = logging.getLogger(__name__)

def clean_raw_data(df):
    df = df.copy()
    df.sort_index(inplace=True)
    
    # Remove duplicate dates (can happen with yfinance)
    df = df[~df.index.duplicated(keep='first')]

    
    # Remove weekends if any slipped through
    df = df[df.index.dayofweek < 5]
    
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.columns = [col.lower() for col in df.columns]
    
    # Drop rows where close is 0 or negative (data corruption)
    df = df[df['close'] > 0]
    
    logger.info(f"Cleaned data: {df.shape}")
    return df

def save_cleaned_data(df, output_path=None):
    """Save cleaned data before feature engineering
    If output_path is not provided, uses path from config.yaml
    """
    if output_path is None:
        # Load from config
        config = load_config()
        output_path = config['paths']['clean_data']
    
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the data
    df.to_csv(output_path)
    logger.info(f"Saved cleaned data to {output_path}")
    


