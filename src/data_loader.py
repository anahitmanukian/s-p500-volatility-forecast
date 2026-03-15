import logging
import pandas as pd
import yfinance as yf
from pathlib import Path
from src.config_loader import load_config  # Import the config loader

logger = logging.getLogger(__name__)

def download_sp500(start=None):
    """Download S&P 500 data and save to raw data folder"""
    
    # Load config
    config = load_config()
    
    # Use start date from config if not provided
    if start is None:
        start = config['data']['start_date']
    
    logger.info(f"Downloading S&P 500 data from {start}")
    
    # Download data
    data = yf.download("^GSPC", start=start, auto_adjust=True)
    
    # Flatten the MultiIndex columns
    data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    # Get raw data path from config
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / config['paths']['raw_data']
    
    # Create directory if it doesn't exist
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving data to {raw_path}")
    logger.info(f"Data columns: {data.columns.tolist()}")
    
    # Save CSV
    data.to_csv(raw_path)
    
    return data

def load_data(path=None):
    """Load data from specified path or from config default"""
    logger.info("Loading ")
    
    if path is None:
        # Load from config if no path provided
        config = load_config()
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / config['paths']['raw_data']
    
    logger.info(f"Loading data from {path}")
    
    # df = pd.read_csv(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    return df