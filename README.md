# Machine learning for financial time series forecating

This project leverages various machine learning models to forecast stock price movements and lays the groundwork for building an optimized trading portfolio based on these predictions. The core idea is to apply a systematic, data-driven approach to generate trading signals from historical price data.

## Project Overview

The investment process is structured as a machine learning workflow:

1.  **Data Preparation**: Sourcing, cleaning, and engineering features from historical daily stock prices. This involves creating technical indicators that may serve as predictive signals (alpha factors).
2.  **Predictive Modeling**: Training and evaluating several machine learning models to forecast future stock returns. We explore a range of models from simple linear regressions to more complex ensemble methods.
3.  **Signal Generation**: Using model predictions to generate trading signals (e.g., long, short, or neutral).
4.  **Portfolio Strategy (Next Steps)**: The ultimate goal is to use these signals to construct and optimize a portfolio. This involves translating signals into positions and managing portfolio risk and return, for example, using techniques like Mean-Variance Optimization or Hierarchical Risk Parity.

This project draws inspiration from the principles outlined in the Machine Learning for Trading book.

## Important note and current results
The project is under active development and the main task was to build a baseline models with further development of deep learning models.   
Current results showed that XGBoost model showed the best performance with Sharpe ratio (2.12) and overall cumulative returns over 200%. It is the only model that outperforms 'buy&hold' strategy in term of Sharpe ratio which although the difference is not huge it is the most promising strategy at the moment. The random forest model showed very poor performance which is even worse then baseline linear regression model. From the differen variants of linear regression Ridge linear regression showed the best results in terms of Sharpe ration (1.25) which is still worse then the Shape ratio of 'buy&hold' strategy, at the same time it outperformed the strategy by the cumulative outcome.    
The future task for the project is to improve feature engineering part, build deep learning models and understand why some models fails to capture the complex patterns of financial time series. 

## Project Structure

```
.
├── data/                   # Contains historical stock data in CSV format
├── data_preparation.ipynb  # Jupyter notebook for data loading and feature engineering
├── linear_regression.ipynb # Stock return forecasting using Linear Regression
├── random_forest.ipynb     # Stock return forecasting using Random Forest
├── xgboost.ipynb           # Stock return forecasting using XGBoost
├── README.md               # This file
└── resourses.txt           # Additional resources and references
```

## Methodology

### 1. Data and Feature Engineering

The process starts with historical daily price data for a universe of stocks, located in the [`data/`](time_series_forecasting/data/) directory. The [`data_preparation.ipynb`](time_series_forecasting/data_preparation.ipynb) notebook handles:
- Loading the raw price data.
- Engineering features (alpha factors) such as moving averages, RSI, and momentum indicators. The goal is to create informative features that capture patterns relevant for return prediction.
- Defining the prediction target, typically the forward returns over a specific horizon.

### 2. Predictive Modeling

We experiment with several supervised learning models to predict stock returns. Each model is implemented in its own notebook:

- **[Linear Regression](time_series_forecasting/linear_regression.ipynb)**: Serves as a baseline model to understand linear relationships between the features and returns.
- **[Random Forest](time_series_forecasting/random_forest.ipynb)**: An ensemble model that can capture non-linear interactions and is generally robust to overfitting.
- **[XGBoost](time_series_forecasting/xgboost.ipynb)**: A powerful gradient boosting implementation known for its performance and speed, often used in trading strategies.

The models are trained and evaluated using time-series cross-validation to ensure the temporal nature of the data is respected and to avoid lookahead bias.

### 3. Portfolio Construction and Backtesting

While the current notebooks focus on generating predictions, the logical next step is to use these signals to build a trading strategy. This involves:

1.  **Defining Trading Rules**: Translating model predictions into buy/sell decisions. For instance, going long on stocks with the highest predicted returns and short on those with the lowest.
2.  **Portfolio Optimization**: Sizing positions to manage risk. Techniques like mean-variance optimization..

## How to Use

1.  **Setup**: Ensure you have a Python environment with standard data science libraries (`pandas`, `scikit-learn`, `xgboost`, `matplotlib`).
2.  **Data Preparation**: Run the [`data_preparation.ipynb`](time_series_forecasting/data_preparation.ipynb) notebook to generate the features for modeling.
3.  **Modeling**: Run the individual model notebooks ([`linear_regression.ipynb`](time_series_forecasting/linear_regression.ipynb), [`random_forest.ipynb`](time_series_forecasting/random_forest.ipynb), [`xgboost.ipynb`](time_series_forecasting/xgboost.ipynb)) to train the models and analyze their predictive power.
4.  **Extend**: Use the generated predictions as a starting point to build your own portfolio strategies.