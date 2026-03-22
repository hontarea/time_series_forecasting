# Deep Learning and Reinforcement Learning for Time-Series.

> **This project is under active development.** The core pipeline (data, forecasting, rule-based and RL executions) is functional. Interfaces may change.

## Overview

The system combines supervised forecasting models with reinforcement learning-based trade execution, evaluated on time-series data. 

Realistic constraints are one of the main concerns throughout the design. The walk-forward engine enforces an embargo period between training and test data, fits scalers per fold to avoid preprocessing leakage, and separates in-sample from out-of-sample data when training the RL agent. The goal is not just predictive accuracy, but a backtest that honestly represents what a deployed system would experience.

## Project Structure

```
├── tsf/                          # Core library
│   ├── data/
│   │   ├── dataset.py            # Unified data container (features, labels, OHLCV)
│   │   ├── loader.py             # CSV ingestion with named presets
│   │   ├── feature_engineer.py   # Registry-based technical indicators and labels
│   │   ├── window.py             # Sliding / expanding window generator
│   │   └── scaler.py             # Per-fold feature and label scaling
│   ├── models/
│   │   ├── base.py               # Abstract model interface (tabular vs sequence)
│   │   ├── sklearn_adapter.py    # Wrapper for scikit-learn estimators
│   │   └── torch_adapter.py      # Wrapper for PyTorch modules
│   ├── execution/
│   │   ├── walk_forward.py       # Walk-forward engine with embargo and fold dispatch
│   │   ├── backtester.py         # Rule-based strategy evaluation
│   │   ├── orchestrator.py       # RL execution interleaved with forecasting
│   │   └── strategy.py           # Signal generation 
│   ├── agents/
│   │   ├── base.py               # Abstract RL agent interface
│   │   ├── dqn.py                # DQN with experience replay and target network
│   │   └── replay_buffer.py      # Fixed-capacity transition buffer
│   ├── environments/
│   │   └── trading.py            # Gymnasium trading environment
│   ├── optimization/
│   │   └── optimizer.py          # Optuna hyperparameter search
│   └── utils/
│       └── evaluation.py         # Performance metrics 
│
├── models/                       # Concrete model architectures
│   ├── linear.py                 # LTSF-Linear (Zeng et al., 2023)
│   └── lstm.py                   # LSTM encoder with MLP head
│
├── scripts/                      # Runnable pipelines
│   ├── linear_pipeline.py        # Linear model: train -> backtest -> optimize
│   ├── lstm_pipeline.py          # LSTM model: train -> backtest -> optimize
│   └── lstm_dqn.py               # LSTM + DQN: forecasting with RL execution
│
└── notebooks/
    └── data_module_test.ipynb    # Data pipeline walkthrough
```

## Pipeline

Each pipeline script runs the same three-stage flow:

**Data** — Load OHLCV-style data, create/compute technical indicators and/or other types of features, and generate the forecasting target. The window generator then slices the timeline into overlapping train/test folds.

**Forecasting** — The walk-forward engine iterates over folds. For each fold it trims the training set by the embargo period, fits the scaler on the trimmed data, trains the model, and generates predictions over the test window. Tabular models (scikit-learn) receive auto-injected lagged features and a direct multi-step target matrix. Sequence models (torch) receive sliding-window batches via a DataLoader. The engine exposes both a batch `run()` method and a fold-by-fold generator for RL interleaving.

**Execution** — Predictions are turned into trading positions by either a rule-based strategy evaluated through the backtester, or an RL agent that interacts with a Gymnasium trading environment through the orchestrator. Transaction costs are applied on position changes. Performance is measured by Sharpe ratio, total return, maximum drawdown, volatility, and win rate.

## Quick Start

```bash
git clone https://github.com/hontarea/time_series_forecasting.git
cd time_series_forecasting
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running a Pipeline

An example of the pipelines are located is `scripts/`. Each script under `scripts/` is a self-contained pipeline with CLI flags:

```bash
# LSTM with default hyperparameters (no Optuna)
python scripts/lstm_pipeline.py --no-optimize

# LSTM with Optuna search, save plots and results
python scripts/lstm_pipeline.py --save-plots plots/ --save-results results/lstm.txt

# LSTM forecasting + DQN execution
python scripts/lstm_dqn.py --no-optimize --save-results results/dqn.txt

# Linear model, full pipeline
python scripts/linear_pipeline.py --save-plots plots/ --save-model models/linear.pt
```
