from abc import ABC, abstractmethod
from src.datawrapper.DataWrapper import DataWrapper
from src.utils.AttributeSetter import AttributeSetter
import numpy as np

class Scope(ABC):
    """
    Abstract base class for data segmentation.
    """
    def __init__(self, wrapper: DataWrapper = None, parameters=None):
        self.wrapper = wrapper 
        self.parameters = parameters if parameters is not None else {}
        AttributeSetter.set_attributes(self, parameters)

    @abstractmethod
    def shift(self):
        pass

    @abstractmethod
    def is_in_scope(self):
        pass

    @abstractmethod
    def reset_state(self):
        pass

    def current_state(self):
        """Get the current state (start_index, end_index)"""
        # Ensure we return integers for iloc slicing
        if hasattr(self, 'start_value') and hasattr(self, 'window_size'):
            return int(self.start_value), int(self.start_value + self.window_size)
        return None

class WindowScope(Scope):
    """
    Implements a sliding window scope based on DATAFRAME INDICES.
    """
    default_parameters = { 
        "start_value": 0,    # Default to first row
        "step_size": 1,      # Move 1 row at a time
        "window_size": 10    # 10 rows per window
        # Removed "column" and "end_value" defaults as they depend on data
    }

    def __init__(self, wrapper: DataWrapper = None, parameters=None):
        params = self.default_parameters.copy()
        if parameters:
            params.update(parameters)
            
        super().__init__(wrapper, params)

        self.step_size = self.parameters.get("step_size", 1)
        self.window_size = self.parameters.get("window_size", 10)

        # FIX 1: Start Value is an Index (0), not a timestamp
        if "start_value" in self.parameters:
            self.start_value = self.parameters["start_value"]
        else:
            self.start_value = 0

        # FIX 2: End Value is the Length of the Dataframe (Max Index)
        if "end_value" in self.parameters:
            self.end_value = self.parameters["end_value"]
        elif self.wrapper is not None:
            # This ensures the loop stops exactly at the last row
            self.end_value = len(self.wrapper.get_dataframe())
        else:
            self.end_value = np.inf # Only if no data is provided (dangerous)

        self.start_value_initial = self.start_value
        self.window_size_initial = self.window_size

    def reset_state(self):
        self.start_value = self.start_value_initial
        self.window_size = self.window_size_initial

    def shift(self):
        self.start_value += self.step_size

    def is_in_scope(self):
        # Stop if the END of the window exceeds the dataframe length
        return (self.start_value + self.window_size) <= self.end_value

class ScopeExpander(WindowScope):
    """
    Expands the current scope (Anchored Walk Forward).
    """
    def shift(self):
        self.window_size += self.step_size

    def is_in_scope(self):
        return (self.start_value + self.window_size) <= self.end_value

class ScopeShifter(WindowScope):
    """
    Shifts the current scope forward (Rolling Window).
    """
    def shift(self):
        self.start_value += self.step_size

    def is_in_scope(self):
        return (self.start_value + self.window_size) <= self.end_value
    
class TestingWindowScope(WindowScope):
    """
    Implementation of testing windows that depends on training windows.
    """
    name = 'testing_window_scope'
    init_parameters = {'testing_window_size': 1}

    def __init__(self, training_window_scope: WindowScope, parameters=None, **kwargs):
        # Inherit parameters but override with testing specific ones
        params = self.init_parameters.copy()
        if parameters:
             # Check if parameters are nested under the class name or flat
             params.update(parameters.get(self.name, parameters))

        super().__init__(wrapper=training_window_scope.wrapper, parameters=params, **kwargs)
        
        self.training_window_scope = training_window_scope
        self.testing_window_size = params.get('testing_window_size', 1)
        
        # Ensure we share the same boundary as the training scope
        self.end_value = self.training_window_scope.end_value
        
        self.sync_with_training()

    def sync_with_training(self):
        """
        Aligns the test window to start immediately after the training window.
        """
        # Test Start = Training Start + Training Length
        self.start_value = self.training_window_scope.start_value + self.training_window_scope.window_size 
        self.window_size = self.testing_window_size
        self.step_size = self.training_window_scope.step_size

    def shift(self):
        self.sync_with_training()

    def is_in_scope(self):
        # The loop is valid only if:
        # 1. The training window is valid
        # 2. AND the testing window fits inside the data
        train_valid = self.training_window_scope.is_in_scope()
        test_fits = (self.start_value + self.window_size) <= self.end_value
        return train_valid and test_fits