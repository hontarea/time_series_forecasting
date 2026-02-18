# Fully automated trading pipeline based on application of machine learning and deep learning models for financial time series forecasting

The goal of this project is to test the performance of different machine learning and deep learning models for the forecasting of financial time series. 

For this purpose we build the full pipeline that contains the following three main stages:
- Data loading/preprocessing;
- Model building/training;
- Execution of the strategies based on the available data and predictions of the model. 

### Data loading/preprocessing
List of modules/classes responsible for data loading and data preprocessing:
- **data**:
    - **dataset.py** - responsible for storing and retrieving the data
    - **loader.py** - initializes and returns an instanse of *Dataset* class implemented in dataset.py
    - **feature_engineering.py** - contains a library of methods to add various technical indicators and label columns
    - **window.py** - responsible for slicing the dataset into windows
    - **scaler.py** - wrapper around standard scikit-learn scaler methods for scalling the data from the provided window and avoid look-ahead bias

### Model building/training
List of modules/classes responsible for data loading and data preprocessing:
- **model**:
    - **base.py** - common abstract class for scikit-learn and torch models that unifies the model workflow: fit() -> predict() -> reset().
    - **sklear_adapter.py** - wrapper for scikit-learn models.
    - **torch_adapter.py** - wrapper for torch models.