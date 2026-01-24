from src.datawrapper.DataWrapper import *
from src.model.Model import Model
from src.transformer.FeatureScaler import FeatureScaler
class Trainer:
    """
    Handles training loop logic. Initializes model parameters from the `DataWrapper` and fits the
    model using features and labels.
    """
    def __init__(self, model: Model = None, feature_scaler: FeatureScaler = None):
        """
        Initialize the Trainer with a model.

        Args:
        model : Model
            The model to be trained.
        """
        self.model = model
        self.feature_scaler = feature_scaler

    def train(self, data_wrapper: DataWrapper):
        """
        Train the model using features and labels from the DataWrapper.

        Args:
        data_wrapper : DataWrapper
            The DataWrapper containing training data.
        """
        if self.feature_scaler:
            self.feature_scaler.fit(data_wrapper)
            self.feature_scaler.transform(data_wrapper)

        features = data_wrapper.get_features()
        
        # Filter only input columns defined in the model
        if self.model.input_columns:
            features = features[self.model.input_columns]
        labels = data_wrapper.get_labels()
        self.model.set_params_from_wrapper(data_wrapper)
        self.model.fit(features, labels)
    def reset_state(self):
        """
        Reset the state of the model.
        """
        self.model.reset_state()
    def set_model_params(self, params):
        """Pass hyperparameter updates to the underlying model."""
        if self.model:
            self.model.set_params(params)