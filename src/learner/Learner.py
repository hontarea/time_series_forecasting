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
        """
        Test the model and return predictions.
        """
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
        results_df = self.run()
        
        if results_df.empty:
            return wrapper
            
        # Remove any potential overlaps in the prediction index
        results_df = results_df[~results_df.index.duplicated(keep='first')]

        pwrapper = wrapper.deepcopy()
        
        if self.last:
            # These are the actual predictions
            pwrapper.add_predictions(results_df)
        else:
            # These predictions will be used as features for a meta-model
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
        all_predictions = pd.DataFrame()
        
        self.data_selector.reset()

        while True:
            train_data = self.data_selector.get_train_data()
            test_data = self.data_selector.get_test_data()

            # print(self.data_selector.current_state_info())
            
            if train_data.get_dataframe().empty or test_data.get_dataframe().empty:
                break
                
            self.train(train_data)
            fold_predictions = self.test(test_data)

            # print(f"{fold_predictions.shape} predictions generated for current window.")
            # print(f"{test_data.get_dataframe().shape} test samples processed for current window.")

            predictions = pd.concat([test_data.get_dataframe()[['open_time_iso', 'log_return']], fold_predictions], axis=1, ignore_index=True)
            predictions.columns = ['open_time_iso', 'log_return', 'prediction']
            # print(f"{predictions.shape} total items (timestamps + predictions) for current window.")

            all_predictions = pd.concat([all_predictions, predictions])

            # print(test_data.get_dataframe())
            
            if self.data_selector.is_last_window():
                break
            
            self.data_selector.update()
        all_predictions.reset_index(drop=True, inplace=True)
        return all_predictions