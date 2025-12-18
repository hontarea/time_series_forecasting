from abc import ABC, abstractmethod
from datawrapper.DataWrapper import DataWrapper
from utils.AttributeSetter import AttributeSetter

import numpy as np
class Scope(ABC):
    """
    Abstract base class for data segmentation. Defines the interface and shared initialization logic
    for all scope implementations.
    """
    def __init__(self, wrapper: DataWrapper = None, parameters=None):
        self.wrapper = wrapper 
        self.parameters = parameters if parameters is not None else {}
        AttributeSetter.set_attributes(self, parameters)

    @abstractmethod
    def shift(self):
        """Move the scope forward"""
        pass

    @abstractmethod
    def isInScope(self):
        """Check if the current is within the valid bounds"""
        pass

    @abstractmethod
    def reset_state(self):
        """Reset the scope to its initial state"""
        pass

    def current_state(self):
        """Get the current state (start, end)"""
        if hasattr(self, 'start_value') and hasattr(self, 'window_size'):
            return self.start_value, self.start_value + self.window_size
        return None

class WindowScope(Scope):
    """
    Implements a sliding window scope for segmenting time series data.
    """
    default_parameters = { 
        "column": "open_time",
        "start_value": 0,
        "end_value": np.inf(), 
        "step_size": 1,
        "window_size": 10
    }

    def __init__(self, wrapper: DataWrapper = None, parameters=None):
        # Handle default parameters merge
        params = self.default_parameters.copy()
        if parameters:
            params.update(parameters)
            
        super().__init__(wrapper, params)

        self.column = self.parameters.get("column", "open_time")
        self.step_size = self.parameters.get("step_size", 1)
        self.window_size = self.parameters.get("window_size", 10)

        if "start_value" in self.parameters:
            self.start_value = self.parameters["start_value"]
        elif self.wrapper is not None and self.column in self.wrapper.get_dataframe().columns:
            self.start_value = self.wrapper.get_dataframe()[self.column].min()
        else:
            self.start_value = 0

        if "end_value" in self.parameters:
            self.end_value = self.parameters["end_value"]
        elif self.wrapper is not None and self.column in self.wrapper.get_dataframe().columns:
            self.end_value = self.wrapper.get_dataframe()[self.column].max()
        else:
            self.end_value = 100

        self.start_value_initial = self.start_value
        self.window_size_initial = self.window_size

    def reset_state(self):
        """
        Reset the scope to its initial state
        """
        self.start_value = self.start_value_initial
        self.window_size = self.window_size_initial

    def shift(self):
        pass
    def isInScope(self):
        return True

class ScopeExpander(WindowScope):
    """
    Expands the current scope (Anchored Walk Forward).
    Start remains fixed, window size grows.
    """
    def shift(self):
        self.window_size += self.step_size

    def isInScope(self):
        return (self.start_value + self.window_size) <= self.end_value

class ScopeShifter(WindowScope):
    """
    Shifts the current scope forward (Rolling Window).
    Start moves forward, window size stays fixed.
    """
    def shift(self):
        self.start_value += self.step_size

    def isInScope(self):
        return (self.start_value + self.window_size) <= self.end_value
    
class TestingWindowScope(WindowScope):
    """
    Implementation of testing windows that depends on training windows.
    """
    name = 'testing_window_scope'
    init_parameters = {'testing_window_size': 1}

    def __init__(self, training_window_scope: WindowScope, parameters=None, **kwargs):
        params = self.init_parameters.copy()
        if parameters and self.name in parameters:
             params.update(parameters[self.name])
        elif parameters:
             params.update(parameters)

        super().__init__(wrapper=training_window_scope.wrapper, parameters=params, **kwargs)
        
        self.training_window_scope = training_window_scope
        self.testing_window_size = params.get('testing_window_size', 1)
        
        self.sync_with_training()

    def sync_with_training(self):
        """
        Set testing window parameters relative to the training window.
        """
        self.start_value = self.training_window_scope.start_value + self.training_window_scope.window_size 
        
        self.window_size = self.testing_window_size
        
        self.step_size = self.training_window_scope.step_size
        self.end_value = self.training_window_scope.end_value

    def shift(self):
        self.sync_with_training()

    def isInScope(self):
        train_valid = self.training_window_scope.isInScope()
        test_fits = (self.start_value + self.window_size) <= self.end_value
        return train_valid and test_fits