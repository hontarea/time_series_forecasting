from datawrapper.DataWrapper import DataWrapper
from transformer.Scope import Scope, TestingWindowScope

class DataSelector:
    """
    Orchestrator for Walk-Forward Validation.
    Manages the Training Scope and automatically updates the dependent Testing Scope.
    """
    def __init__(self, wrapper: DataWrapper, training_scope: Scope, test_parameters: dict = None):
        self.wrapper = wrapper
        self.training_scope = training_scope
        
        # Initialize Testing Scope dependent on the Training Scope
        params = test_parameters if test_parameters else {'testing_window_size': 1}
        self.testing_scope = TestingWindowScope(
            training_window_scope=self.training_scope, 
            parameters=params
        )

    def update(self):
        """Move the window forward by one step"""
        self.training_scope.shift()
        self.testing_scope.shift() 

    def is_last_window(self):
        """Check if we can perform another iteration"""
        return self.training_scope.is_in_scope() and self.testing_scope.is_in_scope()

    def reset(self):
        """Reset scopes to initial configuration"""
        self.training_scope.reset_state()
        self.testing_scope.reset_state()

    # Data Extraction Methods 

    def get_train_data(self):
        """Return the features for current training window"""
        start, end = self.training_scope.current_state()
        return self.wrapper.get_features().iloc[start:end]

    def get_test_data(self):
        """Return the labels for current testing window"""
        start, end = self.testing_scope.current_state()
        return self.wrapper.get_labels().iloc[start:end]

    def current_state_info(self):
        """Debugging helper: returns indices of current windows"""
        t_start, t_end = self.training_scope.current_state()
        v_start, v_end = self.testing_scope.current_state()
        return {
            "train_indices": (t_start, t_end),
            "test_indices": (v_start, v_end)
        }