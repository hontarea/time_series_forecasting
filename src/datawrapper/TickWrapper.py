import numpy as np
from src.datawrapper.DataHandler import DataHandler
from src.datawrapper.DataWrapper import DataWrapper

class TickWrapper(DataWrapper):
    """
    Specialized DataWrapper for handling tick data in financial time series.
    """
    open_column = 'open'
    high_column = 'high'
    low_column = 'low'
    close_column = 'close'
    volume_column = 'volume'

    def __init__(self, data_handler: DataHandler):
        """
        Initializes the TickWrapper with a DataHandler.

        Args:
            data_handler (DataHandler): The underlying data handler.
        """
        super().__init__(data_handler)

    def compute_log_return(self):
        """
        Computes the logarithmic returns of the open prices.
        """
        open_price_col = self.get_dataframe()[self.open_column]
        log_return = np.log1p(open_price_col / open_price_col.shift(1)).shift(-1).rename('log_return')
        self.data_handler.add_labels(log_return)
        self.get_dataframe().dropna(inplace=True)
        self.get_dataframe().reset_index(drop=True, inplace=True)