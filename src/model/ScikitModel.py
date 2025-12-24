import sklearn
import pandas as pd
from src.model.Model import Model
from sklearn.base import BaseEstimator, is_classifier

class ScikitModel(Model):
    """
    Wrapper for scikit-learn estimators. Integrates sklearn models into the unified modeling 
    interface.
    """
    def __init__(self, sklearn_estimator: BaseEstimator, **kwargs):
        """
        Initialize the ScikitModel with an EXISTING scikit-learn estimator instance.

        Args:
            sklearn_estimator (BaseEstimator): An instance of a scikit-learn model 
                                               (e.g., RandomForestClassifier(n_estimators=100))
        """
        super().__init__(**kwargs)  
        # Store the model directly
        self.scikit_model = sklearn_estimator
        
        # Detect model type automatically
        self.is_classifier = is_classifier(self.scikit_model)

    def fit(self, features: pd.DataFrame, labels: pd.Series):
        # Squeeze 2D labels to 1D if necessary (common sklearn requirement)
        if isinstance(labels, pd.DataFrame):
            labels = labels.squeeze()
        
        if labels.ndim > 1 and labels.shape[1] == 1:
            labels = labels.squeeze()
            
        self.scikit_model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.is_classifier:
            # Check if the model supports probability prediction
            if hasattr(self.scikit_model, "predict_proba"):
                predictions = self.scikit_model.predict_proba(features)
                # Return probabilities for the positive class (class 1)
                return pd.Series(predictions[:, 1], index=features.index)
            else:
                # Fallback for classifiers without proba (e.g., SVM without probability=True)
                return pd.Series(self.scikit_model.predict(features), index=features.index)
        else:
            predictions = self.scikit_model.predict(features)
            return pd.Series(predictions, index=features.index)
        
    def reset_state(self):
        """
        Reset the state of the model to unfitted, preserving hyperparameters.
        """
        # sklearn.base.clone creates a new unfitted instance with the same params
        self.scikit_model = sklearn.base.clone(self.scikit_model)

    def set_params(self, params: dict):
        """Allows dynamic hyperparameter updates."""
        self.scikit_model.set_params(**params)