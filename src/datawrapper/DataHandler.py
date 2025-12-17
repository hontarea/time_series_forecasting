import pandas as pd

class DataHandler:
    """
    Utility class for managing features, labels, and predictions within a pandas DataFrame. 
    Provides methods for accessing, modifying, slicing, and copying the underlying data.
    """
    
    def __init__(self, dataframe: pd.DataFrame = None, feature_cols = None, label_cols = None, prediction_cols = None):
        """
        Initializes the DataHandler with a DataFrame and optional feature, label, and prediction columns.

        Args:
            dataframe (pd.DataFrame): The underlying data.
            feature_cols (list): List of column names representing features.
            label_cols (list): List of column names representing labels.
            prediction_cols (list): List of column names representing predictions.
        """
        self.dataframe = dataframe
        self.prediction_cols = prediction_cols

        self.feature_cols = set(feature_cols) if feature_cols is not None else set()
        self.label_cols = set(label_cols) if label_cols is not None else set()

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the underlying DataFrame.
        """
        return self.dataframe
    
    def set_dataframe(self, dataframe: pd.DataFrame):
        """
        Sets the new underlying DataFrame.

        Args:
            dataframe (pd.DataFrame): The new DataFrame to set.
        """
        self.dataframe = dataframe

    def get_features(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing only the feature columns.
        """
        return self.dataframe[list(self.feature_cols)]

    def add_features(self, features, on=None):
        """
        Adds new feature columns to the DataFrame and updates the feature columns set.

        Args:
            features (pd.DataFrame): DataFrame containing new feature columns.
            on (str or list, optional): Column(s) to join on. If None, indexes are used.
        """
        self.dataframe = self.dataframe.join(features, on = on) if on else self.dataframe.join(features)
        self.feature_cols.update(features.columns.tolist())

    def get_labels(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing only the label columns.
        """
        return self.dataframe[list(self.label_cols)]
    
    def add_labels(self, labels):
        """
        Adds new label columns to the DataFrame and updates the label columns set.

        Args:
            labels (pd.DataFrame): DataFrame containing new label columns.
        """
        self.dataframe = self.dataframe.join(labels)
        self.label_cols.update(labels.columns.tolist() if isinstance(labels, pd.DataFrame) else [labels.name])

    def get_predictions(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing only the prediction columns.
        """
        return self.dataframe[self.prediction_cols] if self.prediction_cols else pd.DataFrame()
    
    def add_predictions(self, predictions):
        """
        Adds new prediction columns to the DataFrame and updates the prediction columns list.

        Args:
            predictions (pd.DataFrame): DataFrame containing new prediction columns.
        """
        self.dataframe = self.dataframe.join(predictions)
        if self.prediction_cols is None:
            self.prediction_cols = []
        self.prediction_cols.extend(predictions.columns.tolist())

    def get_columns(self, cols) -> pd.DataFrame:
        """
        Returns a DataFrame containing only the specified columns.

        Args:
            cols (list): List of column names to retrieve.

        Returns:
            pd.DataFrame: DataFrame with the specified columns.
        """
        return self.dataframe[cols]
    
    def add_columns(self, new_data):
        """
        Adds new columns to the DataFrame.

        Args:
            new_data (pd.DataFrame): DataFrame containing new columns to add.
        """
        self.dataframe = self.dataframe.join(new_data)

    def copy(self, dataframe: pd.DataFrame = None, feature_cols = None, label_cols = None, prediction_cols = None):
        """
        Creates a copy of the DataHandler.
        """
        if dataframe is None:
            dataframe = self.dataframe.copy() 
        if feature_cols is None:
            feature_cols = list(self.feature_cols)
        if label_cols is None:
            label_cols = list(self.label_cols)
        if self.prediction_cols is not None:
            prediction_cols = list(self.prediction_cols)
        
        return DataHandler(dataframe, feature_cols = feature_cols, label_cols = label_cols, prediction_cols = prediction_cols)