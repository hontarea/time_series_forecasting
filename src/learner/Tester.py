import pandas as pd
from src.model.Model import Model
from src.model.ScikitModel import ScikitModel
class Tester:
    """
    Handles inference and prediction generation for a trained model on datasets wrapped by 
    `DataWrapper`.
    """
    def __init__(self, model: Model = None):
        """
        Initialize the Tester with a model.

        Args:
        model : Model
            The model to be used for testing.
        """
        self.model = model

    def test(self, data_wrapper):
        """
        Generate predictions using the model on features from the DataWrapper.

        Args:
        data_wrapper : DataWrapper
            The DataWrapper containing test data.

        Returns:
        pd.DataFrame:
            The predicted labels.
        """
        features = data_wrapper.get_features()
        # Filter only input columns defined in the model
        if self.model.input_columns:
            features = features[self.model.input_columns]
        
        if isinstance(self.model, ScikitModel):
            if features.isnull().values.any():
                raise ValueError("Input features contain NaN values. Please handle missing data before testing.")
            predictions = self.model.predict(features)

            #  Check dimensions before accessing shape[1]
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                columns = self.model.scikit_model.classes_
            else:
                columns = ['prediction'] 
                if isinstance(predictions, pd.Series):
                    predictions = predictions.to_frame(name=columns[0])
            return pd.DataFrame(predictions.values, index=features.index, columns=columns)
        else:
            predictions = self.model.predict(features)
            return pd.DataFrame(index=data_wrapper.get_dataframe().index, data=predictions)

