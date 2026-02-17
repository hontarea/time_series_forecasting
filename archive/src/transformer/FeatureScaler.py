import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src.datawrapper.DataWrapper import DataWrapper

class FeatureScaler:
    """
    Scales features within a DataWrapper using Scikit-Learn scalers.
    Ensures that scaling parameters (mean, std, etc.) are learned from training data
    and applied consistently to test data.
    """
    def __init__(self, method: str = 'standard', columns: list = None):
        """
        Args:
            method (str): 'standard' (Z-score), 'minmax' (0-1), or 'robust' (outlier resilient).
            columns (list): Specific columns to scale. If None, scales all feature columns.
        """
        self.method = method
        self.columns = columns
        self.fitted = False
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    def fit(self, wrapper: DataWrapper):
        """
        Fits the scaler to the features in the DataWrapper.
        """
        df = wrapper.get_features()
        
        # Determine which columns to scale
        cols_to_scale = self.columns if self.columns else df.columns
        
        # Fit internal sklearn scaler
        self.scaler.fit(df[cols_to_scale])
        self.fitted = True

    def transform(self, wrapper: DataWrapper):
        """
        Applies the fitted scaling to the DataWrapper in-place.
        """
        if not self.fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform.")

        df = wrapper.get_dataframe()
        features = wrapper.get_features()
        cols_to_scale = self.columns if self.columns else features.columns
        
        scaled_values = self.scaler.transform(df[cols_to_scale])
        
        df[cols_to_scale] = scaled_values
        
        wrapper.set_dataframe(df)
        
    def fit_transform(self, wrapper: DataWrapper):
        """
        Fits and transforms in one step.
        """
        self.fit(wrapper)
        self.transform(wrapper)