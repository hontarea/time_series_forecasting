import pandas as pd
from src.datawrapper.DataHandler import DataHandler

class DataWrapper:
    """
    Higher-level interface for interacting with a `DataHandler`. Designed to be extended by 
    specialized wrappers that add domain-specific logic.
    """
    
    date_column = 'Date'

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
