import pandas as pd
import copy
from src.datawrapper.DataWrapper import DataWrapper
from src.transformer.Scope import Scope, TestingWindowScope

class DataSelector:
    """
    Orchestrator for Walk-Forward Validation.
    """
    def __init__(self, wrapper: DataWrapper, training_scope: Scope, test_parameters: dict = None):
        self.wrapper = wrapper
        
        self.training_scope = copy.deepcopy(training_scope)
        
        params = test_parameters if test_parameters else {'testing_window_size': "24h"}
        
        self.testing_scope = TestingWindowScope(
            training_window_scope=self.training_scope, 
            parameters=params
        )

    def update(self):
        """Move the window forward by one step"""
        self.training_scope.shift()
        self.testing_scope.shift() 

    def is_last_window(self):
        return not(self.training_scope.is_in_scope() and self.testing_scope.is_in_scope())

    def reset(self):
        self.training_scope.reset_state()
        self.testing_scope.reset_state()

    def get_train_data(self):
            return self._slice_data(self.training_scope)

    def get_test_data(self):
        return self._slice_data(self.testing_scope)

    def _slice_data(self, scope):
        """
        Internal helper: Slices data based on Time logic.
        """
        start, end = scope.current_state()
        time_col = scope.time_column
        
        df = self.wrapper.get_dataframe()
        
        # FIX: Robust Timezone Handling
        # If the column is already datetime, use it. If not, convert.
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
             time_series = df[time_col]
        else:
             time_series = pd.to_datetime(df[time_col])

        # Ensure Naive (No Timezone) for comparison with Scope
        if time_series.dt.tz is not None:
             time_series = time_series.dt.tz_localize(None)

        mask = (time_series >= start) & (time_series < end)
        
        sliced_df = df.loc[mask].copy()
        
        return DataWrapper(
            self.wrapper.data_handler.__class__(
                sliced_df, 
                feature_cols=self.wrapper.get_feature_columns(),
                label_cols=self.wrapper.get_label_columns()
            )
        )
    
    def current_state_info(self):
        t_start, t_end = self.training_scope.current_state()
        v_start, v_end = self.testing_scope.current_state()
        return {
            "train_indices": (t_start, t_end),
            "test_indices": (v_start, v_end)
        }