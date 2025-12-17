import pandas as pd
from src.datawrapper.DataWrapper import DataWrapper
from src.datawrapper.DataHandler import DataHandler
from src.datawrapper.TickWrapper import TickWrapper
class DataLoader:
    """A utility class for loading and wrapping data from a CSV file."""
    
    @classmethod
    def load_wrap(cls, path_to_csv: str, data_type: str) -> DataWrapper:
        df = pd.read_csv(path_to_csv)

        if data_type == "tick":
            feature_cols = ["open", "high", "low", "close", "volume"]
            label_cols = []
            prediction_cols = None

            # Filter columns to only include relevant ones
            df = df[["open_time"] + feature_cols]

            data_handler = DataHandler(
                dataframe=df,
                feature_cols=feature_cols,
                label_cols=label_cols,
                prediction_cols=prediction_cols,
            )
            return TickWrapper(data_handler)

        raise ValueError(f"Unsupported data type: {data_type}")