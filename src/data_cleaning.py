# src/data_cleaning.py
import pandas as pd
import logging
from pathlib import Path
from src.config_loader import load_config, get_processed_data_path

logger = logging.getLogger(__name__)

def clean_raw_data(df):
    """Only basic cleaning - no feature creation for modeling"""
    # logger = logging.getLogger(__name__)
    
    df = df.copy()
    
    # Convert Date to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Handle missing values (basic)
    df.ffill(inplace=True)  # forward fill
    df.bfill(inplace=True)  # back fill for any remaining
    
    # Rename columns to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]
    
    logger.info(f"Cleaned data: {df.shape}")
    return df

def save_cleaned_data(df, output_path=None):
    """Save cleaned data before feature engineering
    If output_path is not provided, uses path from config.yaml
    """
    if output_path is None:
        # Load from config
        config = load_config()
        output_path = config['paths']['processed_data']
    
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the data
    df.to_csv(output_path)
    logger.info(f"Saved cleaned data to {output_path}")
    


