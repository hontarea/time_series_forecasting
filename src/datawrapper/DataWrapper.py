import pandas as pd
import copy
from src.datawrapper.DataHandler import DataHandler

class DataWrapper:
    """
    Higher-level interface for interacting with a `DataHandler`. Designed to be extended by 
    specialized wrappers that add domain-specific logic.
    """
    
    #  Default name of time column is 'open_time_iso'
    time_col = 'open_time_iso'

    # Variables will be overridden in specialized wrappers
    open_column = None
    high_column = None
    low_column = None
    close_column = None
    volume_column = None

    def __init__(self, data_handler: DataHandler):
        """
        Initializes the DataWrapper with a DataHandler.

        Args:
            data_handler (DataHandler): The underlying data handler.
        """
        self.data_handler = data_handler
        self._ensure_datetime()

    def _ensure_datetime(self):
        """
        Validates, converts, and sorts the time column to support TimeWindowScope.
        """

        if self.time_col not in self.get_dataframe().columns:
            raise ValueError(f"Time column '{self.time_col}' not found in dataset.")

        if not pd.api.types.is_datetime64_any_dtype(self.get_dataframe()[self.time_col]):
            try:
                self.set_dataframe(self.get_dataframe().copy())
                self.get_dataframe()[self.time_col] = pd.to_datetime(self.get_dataframe()[self.time_col])
                
            except Exception as e:
                raise ValueError(f"Could not parse time column '{self.time_col}': {e}")
        self.get_dataframe().sort_values(by=self.time_col, ascending=True, inplace=True, ignore_index=True)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the underlying DataFrame from the DataHandler.
        """
        return self.data_handler.get_dataframe()

    def set_dataframe(self, dataframe: pd.DataFrame):
        """
        Sets a new DataFrame in the DataHandler.

        Args:
            dataframe (pd.DataFrame): The new DataFrame to set.
        """
        self.data_handler.set_dataframe(dataframe)

    def get_features(self) -> pd.DataFrame:
        """
        Returns the feature columns DataFrame from the DataHandler.
        """
        return self.data_handler.get_features()

    def add_features(self, features, on = None):
        """
        Adds new feature columns to the DataHandler.

        Args:
            features (pd.DataFrame): DataFrame containing new feature columns.
            on (str or list, optional): Column(s) to join on. If None, indexes are used.
        """
        self.data_handler.add_features(features, on = on)

    def add_features_from_csv(self, path_to_csv: str, index = None, on = None):
        """
        Loads feature columns from a CSV file and adds them to the DataHandler.

        Args:
            path_to_csv (str): Path to the CSV file containing feature columns.
            on (str or list, optional): Column(s) to join on. If None, indexes are used.
        """
        features = pd.read_csv(path_to_csv, index_col = index)
        self.add_features(features, on = on)

    def get_feature_columns(self):
        """Returns a list of feature column names."""
        return list(self.data_handler.feature_cols)

    def get_labels(self) -> pd.DataFrame:
        """
        Returns the label columns DataFrame from the DataHandler.
        """
        return self.data_handler.get_labels()
    
    def add_labels(self, labels):
        """
        Adds new label columns to the DataHandler.

        Args:
            labels (pd.DataFrame): DataFrame containing new label columns.
        """
        self.data_handler.add_labels(labels)

    def get_label_columns(self):
        """Returns a list of label column names."""
        return list(self.data_handler.label_cols)
    
    def add_predictions(self, predictions):
        """Delegate the call to the internal data handler"""
        if hasattr(self.data_handler, 'add_predictions'):
            self.data_handler.add_predictions(predictions)
        else:
            raise AttributeError(f"Underlying {type(self.data_handler).__name__} has no 'add_predictions' method.")
    
    def get_predictions(self) -> pd.DataFrame:
        """
        Returns the prediction columns DataFrame from the DataHandler.
        """
        return self.data_handler.get_predictions()
    
    def add_columns(self, new_data: pd.DataFrame):
        """
        Adds new columns to the underlying DataFrame in the DataHandler.

        Args:
            new_data (pd.DataFrame): DataFrame containing new columns to add.
        """
        self.data_handler.add_columns(new_data)

    def get_columns(self, column_names) -> pd.DataFrame:
        """
        Returns specified columns from the underlying DataFrame in the DataHandler.
        """
        return self.data_handler.get_columns(column_names)
    
    def empty(self):
        """
        Returns whether the underlying DataFrame in the DataHandler is empty.
        """
        return self.get_dataframe().empty
    
    def deepcopy(
        self,
        dataframe: pd.DataFrame = None,
        feature_cols=None,
        label_cols=None,
    ):
        """
        Returns an independent copy of this wrapper and its underlying DataHandler.
        """
        new_handler = self.data_handler.copy(
            dataframe=dataframe,
            feature_cols=feature_cols,
            label_cols=label_cols,
        )
        new = self.__class__(new_handler)

        for attribute_key, value in self.__dict__.items():
            if attribute_key == "data_handler":
                continue
            setattr(new, attribute_key, copy.deepcopy(value))

        return new
