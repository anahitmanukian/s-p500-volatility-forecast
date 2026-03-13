from src.models import run_random_forest
from src.evaluation import evaluate_regression
from src.data_loader import download_sp500, load_data
from src.data_cleaning import clean_raw_data, save_cleaned_data
from src.feature_engineering import create_volatility_features
from src.models import split_data, run_random_forest
import logging
from src.logger import setup_logging

setup_logging()

logger = logging.getLogger(__name__)
logger.info("Logging works!")

data = download_sp500()
df = load_data()

clean_df = clean_raw_data(df)
save_cleaned_data(clean_df)

final_df = create_volatility_features(clean_df)


X_train, X_test, y_train, y_test = split_data(final_df)

model, y_pred = run_random_forest(X_train, X_test, y_train)

evaluate_regression(
    model,
    y_test,
    y_pred,
    model_name="Random Forest",
    X_train=X_train
)