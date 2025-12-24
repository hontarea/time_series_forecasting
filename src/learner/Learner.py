import pandas as pd
from src.learner.Trainer import Trainer
from src.learner.Tester import Tester
from src.transformer.DataSelector import DataSelector
from src.datawrapper.DataWrapper import DataWrapper

class Learner:
    """
    Coordinates training and testing of a model using a `Trainer` and `Tester`.
    Serves as the base class for single-fold or multi-fold (walk-forward) execution.
    """
    def __init__(self, 
                 trainer: Trainer = None, 
                 tester: Tester = None, 
                 data_wrapper: DataWrapper = None, 
                 data_selector: DataSelector = None,
                 is_last: bool = True,
                 **kwargs):
        self.trainer = trainer
        self.tester = tester
        self.data_wrapper = data_wrapper
        self.data_selector = data_selector
        self.last = is_last

    def set_model_params(self, model_params):
        """Set parameters for the model used in training and testing."""
        if self.trainer:
            self.trainer.set_model_params(model_params)
        if self.tester:
            self.tester.set_model_params(model_params)

    def train(self, dataset: DataWrapper):
        """Train the model using the provided dataset wrapper."""
        if self.trainer and not dataset.get_dataframe().empty:
            self.trainer.train(dataset)

    def test(self, dataset: DataWrapper) -> pd.DataFrame:
        """Test the model and return predictions."""
        if self.tester and not dataset.get_dataframe().empty:
            return self.tester.test(dataset)
        return pd.DataFrame()
    
    def reset_state(self):
        """Reset the internal state of the trainer (e.g., clear model weights)."""
        if self.trainer:
            self.trainer.reset_state()
    
    def update(self):
        """Advance the data selector to the next window."""
        if self.data_selector:
            self.data_selector.update()

    def run(self) -> pd.DataFrame:
        """
        Executes a single train-test cycle based on the CURRENT state of the selector.
        """
        train_data = self.data_selector.get_train_data()
        test_data = self.data_selector.get_test_data()
        
        if train_data.get_dataframe().empty or test_data.get_dataframe().empty:
            return pd.DataFrame()
            
        self.train(train_data)
        return self.test(test_data)
    
    def compute(self, wrapper: DataWrapper) -> DataWrapper:
        """
        High-level entry point. Executes the 'run' logic and merges results 
        back into a DataWrapper.
        """
        # Execute the training/testing logic (single fold or walk-forward)
        results_df = self.run()
        
        if results_df.empty:
            return wrapper
            
        # Remove any potential overlaps in the prediction index
        results_df = results_df[~results_df.index.duplicated(keep='first')]
        
        # Create a new wrapper to hold the results
        # We use deepcopy to ensure the original input wrapper remains pristine
        pwrapper = wrapper.deepcopy()
        
        if self.last:
            # Final output: These are the actual Buy/Sell/Price predictions
            pwrapper.add_predictions(results_df)
        else:
            # Intermediate step: These predictions will be used as features 
            # for a meta-model
            pwrapper.add_features(results_df)
            
        return pwrapper
    
class UpdatingLearner(Learner):
    """
    Implements Walk-Forward Validation. Iteratively trains and tests 
    as the window moves across the dataset.
    """
    def __init__(self, 
                 trainer: Trainer = None, 
                 tester: Tester = None, 
                 data_wrapper: DataWrapper = None, 
                 data_selector: DataSelector = None,
                 **kwargs):
        # By default, we assume UpdatingLearner is the last step of the backtest
        super().__init__(trainer, tester, data_wrapper, data_selector, is_last=True, **kwargs)

    def run(self) -> pd.DataFrame:
        """
        The Walk-Forward engine.
        """
        all_predictions = []
        
        # Reset the selector to ensure we start from the initial window
        self.data_selector.reset()

        # Iterate while there are valid train/test windows left in the dataset
        while self.data_selector.is_last_window():
            train_data = self.data_selector.get_train_data()
            test_data = self.data_selector.get_test_data()
            
            if train_data.get_dataframe().empty or test_data.get_dataframe().empty:
                break
                
            # Retrain model on the new training window
            self.train(train_data)
            
            # Predict on the new testing window
            fold_predictions = self.test(test_data)
            all_predictions.append(fold_predictions)
            
            # Slide/Expand the windows forward
            self.data_selector.update()
            
        if all_predictions:
            # Combine all windows into a single continuous time series of predictions
            return pd.concat(all_predictions).sort_index()
            
        return pd.DataFrame()