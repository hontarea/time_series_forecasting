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
            
    def reset_state(self):
        """Reset the internal state of the trainer."""
        if self.trainer:
            self.trainer.reset_state()
    
    def update(self):
        """Advance the data selector to the next window."""
        if self.data_selector:
            self.data_selector.update()

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
    
    def compute(self) -> DataWrapper:
        """
        High-level entry point used by Optimizer. 
        Executes the 'run' logic and merges results back into a DataWrapper.
        """
        wrapper = self.data_wrapper
        # 1. Generate predictions (Single fold or Walk-Forward)
        results_df = self.run()
        
        if results_df.empty:
            return wrapper
            
        # 2. Clean up duplicates if the sliding windows overlapped
        results_df = results_df[~results_df.index.duplicated(keep='last')]

        # 3. Create a deep copy to avoid modifying the original data during optimization trials
        pwrapper = wrapper.deepcopy()
        

        pwrapper.add_predictions(results_df)
        pwrapper.get_dataframe().dropna(inplace=True)
            
        return pwrapper

class UpdatingLearner(Learner):
    """
    Implements Walk-Forward Validation. 
    Iteratively trains and tests as the window moves across the dataset.
    """
    def __init__(self, 
                 trainer: Trainer = None, 
                 tester: Tester = None, 
                 data_wrapper: DataWrapper = None, 
                 data_selector: DataSelector = None,
                 **kwargs):
        super().__init__(trainer, tester, data_wrapper, data_selector, is_last=True, **kwargs)

    def run(self) -> pd.DataFrame:
        """
        The Walk-Forward engine.
        Returns a DataFrame of predictions aligned with the original dataset index.
        """
        all_predictions = []
        
        # Ensure we start from the beginning of the timeline
        self.data_selector.reset()

        while True:
            train_data = self.data_selector.get_train_data()
            test_data = self.data_selector.get_test_data()
            
            # Stop if we run out of data
            if train_data.get_dataframe().empty or test_data.get_dataframe().empty:
                break
                
            # 1. Train on the rolling window
            self.train(train_data)
            
            # 2. Predict on the "future" (testing window)
            # IMPORTANT: fold_predictions must have the same index as test_data
            fold_predictions = self.test(test_data)
            
            if not fold_predictions.empty:
                all_predictions.append(fold_predictions)

            # # 3. Check termination condition
            # if self.data_selector.is_last_window():
            #     break
            
            # 4. Move the window forward
            self.data_selector.update()

        # Concatenate all time steps
        if not all_predictions:
            return pd.DataFrame()
            
        # We simply concatenate. Since we used the original indices in `test()`, 
        # this DataFrame is perfectly aligned with the original DataWrapper.
        return pd.concat(all_predictions)