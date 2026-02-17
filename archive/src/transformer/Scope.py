from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from src.datawrapper.DataWrapper import DataWrapper
from src.utils.AttributeSetter import AttributeSetter

class Scope(ABC):
    """Abstract base class for data segmentation."""
    def __init__(self, wrapper: DataWrapper = None, parameters=None):
        self.wrapper = wrapper 
        self.parameters = parameters if parameters is not None else {}
        AttributeSetter.set_attributes(self, parameters)

    @abstractmethod
    def shift(self): pass

    @abstractmethod
    def is_in_scope(self): pass

    @abstractmethod
    def reset_state(self): pass

    def current_state(self):
        if hasattr(self, 'start_value') and hasattr(self, 'window_size'):
            return self.start_value, self.start_value + self.window_size
        return None

class WindowScope(Scope):
    """
    Implements a sliding window scope based on TIME (Timestamps).
    """
    default_parameters = { 
        "time_column": "open_time_iso", 
        "start_value": None,
        "end_value": None,
        "step_size": "24h",
        "window_size": "30d"
    }

    def __init__(self, wrapper: DataWrapper = None, parameters=None):
        params = self.default_parameters.copy()
        if parameters:
            params.update(parameters)
        super().__init__(wrapper, params)

        self.time_column = self.parameters.get("time_column", "open_time")
        
        def parse_param(key, default_val, default_unit='h'):
            val = self.parameters.get(key, default_val)
            if isinstance(val, (int, float)):
                return pd.to_timedelta(val, unit=default_unit)
            return pd.to_timedelta(val)

        self.step_size = parse_param("step_size", "24h", default_unit='h')
        self.window_size = parse_param("window_size", "30d", default_unit='d')

        min_date = pd.Timestamp.now().normalize()
        max_date = pd.Timestamp.now().normalize()

        if wrapper is not None:
            df = wrapper.get_dataframe()
            if self.time_column in df.columns:
                series = pd.to_datetime(df[self.time_column])
                if series.dt.tz is not None:
                     series = series.dt.tz_localize(None)
                
                if not series.empty:
                    min_date = series.min()
                    max_date = series.max()

        if self.parameters.get("start_value"):
            self.start_value = pd.to_datetime(self.parameters["start_value"])
        else:
            self.start_value = min_date

        if self.parameters.get("end_value"):
             self.end_value = pd.to_datetime(self.parameters["end_value"])
        else:
             self.end_value = max_date

        self.start_value_initial = self.start_value
        self.window_size_initial = self.window_size

        print(f"[WindowScope] Initialized with start: {self.start_value}, "
              f"end: {self.end_value}, step_size: {self.step_size}, window_size: {self.window_size}")

    def reset_state(self):
        self.start_value = self.start_value_initial
        self.window_size = self.window_size_initial

    def shift(self):
        self.start_value += self.step_size

    def is_in_scope(self):
        return (self.start_value + self.window_size) <= self.end_value

class ScopeExpander(WindowScope):
    def shift(self):
        self.window_size += self.step_size

    def is_in_scope(self):
        return (self.start_value + self.window_size) <= self.end_value

class ScopeShifter(WindowScope):
    def shift(self):
        self.start_value += self.step_size

    def is_in_scope(self):
        return (self.start_value + self.window_size) <= self.end_value
    
class TestingWindowScope(WindowScope):
    name = 'testing_window_scope'
    init_parameters = {'testing_window_size': "24h"}

    def __init__(self, training_window_scope: WindowScope, parameters=None, **kwargs):
        params = self.init_parameters.copy()
        if parameters:
             params.update(parameters.get(self.name, parameters))

        super().__init__(wrapper=training_window_scope.wrapper, parameters=params, **kwargs)
        
        self.training_window_scope = training_window_scope
        
        # Parse testing window size safely
        raw_size = params.get('testing_window_size', "24h")
        if isinstance(raw_size, (int, float)):
            self.testing_window_size = pd.to_timedelta(raw_size, unit='h')
        else:
            self.testing_window_size = pd.to_timedelta(raw_size)
        
        self.end_value = self.training_window_scope.end_value
        self.sync_with_training()

    def sync_with_training(self):
        train_end = self.training_window_scope.start_value + self.training_window_scope.window_size
        self.start_value = train_end
        self.window_size = self.testing_window_size
        self.step_size = self.training_window_scope.step_size

    def shift(self):
        self.sync_with_training()

    def is_in_scope(self):
        train_valid = self.training_window_scope.is_in_scope()
        test_fits = (self.start_value + self.window_size) <= self.end_value
        return train_valid and test_fits

    def reset_state(self):
        self.sync_with_training()