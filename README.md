# S&P 500 Volatility Forecasting

Predicting S&P 500 realized volatility 30 days ahead using 
Random Forest, XGBoost, and ARIMA with rolling forecast evaluation.

## Project Structure
|--data / 
|       raw / sp500.csv
|       processed / sp500_cleaned.csv
|                   feature_engineering.csv 
|--src / 
|       config_loader.py
|       data_loader.py
|       data_cleaning.py
|       feature_engineering.py
|       models.py
|       evaluation.py
|       logger.py 
|
|--reports / 
|       model_comparison.png 
|
|config / 
|       config.yaml
|       logging.yaml
|
|--testing.ipynb
|--requirements.txt
|--.gitignore
|--README.md
|___


## Setup
pip install -r requirements.txt
python main.py

## Problem Definition
- Volatility shows how much the data fluctuates and is very important for time series data 
- We're predicting volatility of the next 30 days 
- It is hard because 30 days is not a little time to predict so we need rolling features because they contain more stabilized information and less noise 

## Data
- Source: yfinance, S&P 500 from 2000-01-01
- Train: 2001-02-14 to 2021-01-28 (5020 rows)
- Test: 2021-01-29 to 2026-01-29 (1256 rows)

## Features
- Lag features (vol_lag_1, vol_lag_5...) — captures volatility memory, 
  yesterday's volatility is the strongest predictor of tomorrow's
- Rolling means (mean_5, mean_10, mean_30) — smooths noise, tells the model 
  whether price is high or low relative to recent history
- Range (high-low)/close — measures intraday volatility independent of close price
- Distance from MA (dist_ma10, dist_ma30) — how far price has deviated from its average
- Volume lags — captures whether unusual trading activity preceded volatility


## Models
**Random Forest** — trained on lag and rolling features, 
single 80/20 temporal split. Struggles with distribution shift 
between training (2001-2021) and test (2021-2026) periods.

**XGBoost** — same feature set and split as Random Forest.
Sequential boosting slightly improves MAE but same regime bias remains.

**ARIMA(1,0,1)** — univariate rolling forecast with 252-day fixed window.
Refits on each step using only the past trading year, 
avoiding bias from outdated regimes like 2008 or COVID.



## Results
| Model         | MAE      |
|---------------|----------|
| Baseline      | 0.003450 |
| Random Forest | 0.007315 |
| XGBoost       | 0.006988 |
| ARIMA         | 0.000284 |

![Model Comparison](reports/model_comparison.png)

## Key Finding
Random Forest and XGboost train on 20 years including extreme events like 2008 and COVID, so its averaged predictions are permanently biased toward high volatility. ARIMA with a 252-day window only looks at the past year, so it reflects the current regime — the same way prices from last year tell you more about today's cost of living than prices from 2001.

