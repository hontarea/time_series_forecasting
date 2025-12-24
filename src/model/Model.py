import inspect
import pandas as pd
from src.datawrapper.DataWrapper import DataWrapper
from src.utils.MLFlowTracker import MLFlowTracker


class Model:
    """
    Base class for machine learning models. Defines common attributes such as input columns
    (`in_cols`) and the underlying estimator or neural model
    """
    input_columns = []
    model = None

    def __init__(self, **kwargs):
        """
        Initialize the Model with given parameters.

        Args:
        **kwargs : dict
            Additional keyword arguments for model configuration.
        """
        self._init_params = self._get_init_params()

    def _get_init_params(self) -> dict:
        """
        Retrieve the initialization parameters of the model.

        Returns:
        dict
            A dictionary of parameter names and their corresponding values.
        """
        signature = inspect.signature(self.__init__)
        params = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty and k != 'self'
        }
        return params
    def fit(self, features: pd.DataFrame, labels: pd.Series):
        """
        Fit the model to the provided features and labels.

        Args:
        features : pd.DataFrame
            The input features for training.
        labels : pd.Series
            The target labels for training.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict labels for the provided features.
        Args:
        features : pd.DataFrame
            The input features for prediction. 
        Returns:
        pd.Series
            The predicted labels.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    def set_params_from_wrapper(self, wrapper: DataWrapper):
        """
        Set model parameters from a given wrapper object.

        Args:
        wrapper : object
            An object that contains model parameters as attributes.
        """
        for param in self._init_params.keys():
            if hasattr(wrapper, param):
                setattr(self, param, getattr(wrapper, param))
    def reset_state(self):
        """
        Reset the model's internal state.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def set_params(self, params: dict):
        """
        Set model parameters from a given dictionary.

        Args:
        params : dict
            A dictionary of parameter names and their corresponding values.
        """
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
    def _get_init_params(self) -> dict:
        """
        Retrieve the initialization parameters of the model.

        Returns:
        dict
            A dictionary of parameter names and their corresponding values.
        """
        signature = inspect.signature(self.__init__)
        params = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty and k != 'self'
        }
        return params
    def log_params(self) -> None:
        """
        Log the model's parameters to MLFlowTracker.
        """
        MLFlowTracker.log_params(self._init_params)