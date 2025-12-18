import pandas as pd
import talib as ta
from src.datawrapper.DataWrapper import DataWrapper

class TechIndicGenerator:
    def __init__(self):
        pass

    # Momentum Indicators
    def compute_RSI(self, data_wrapper: DataWrapper, period=14):
        rsi_feature = ta.RSI(data_wrapper.get_dataframe()['close'], timeperiod=period).rename('rsi')
        data_wrapper.add_features(rsi_feature)
        data_wrapper.get_dataframe().dropna(inplace=True)
        data_wrapper.get_dataframe().reset_index(drop=True, inplace=True)

    def compute_KAMA(self, data_wrapper: DataWrapper, period=30):
        kama_feature = ta.KAMA(data_wrapper.get_dataframe()['close'], timeperiod=period).rename('kama')
        data_wrapper.add_features(kama_feature)
        data_wrapper.get_dataframe().dropna(inplace=True)
        data_wrapper.get_dataframe().reset_index(drop=True, inplace=True)

    def compute_SWMA(self, data_wrapper: DataWrapper):
        close_prices = data_wrapper.get_dataframe()['close']
        swma_feature = (close_prices + 2 * close_prices.shift(1) + 2 * close_prices.shift(2) + close_prices.shift(3)) / 6
        data_wrapper.add_features(swma_feature.rename('swma'))
        data_wrapper.get_dataframe().dropna(inplace=True)
        data_wrapper.get_dataframe().reset_index(drop=True, inplace=True)

    def compute_HLC3(self, data_wrapper: DataWrapper):
        hlc3_feature = ((data_wrapper.get_dataframe()['high'] + data_wrapper.get_dataframe()['low'] + data_wrapper.get_dataframe()['close']) / 3).rename('hlc3')
        data_wrapper.add_features(hlc3_feature)
        data_wrapper.get_dataframe().dropna(inplace=True)
        data_wrapper.get_dataframe().reset_index(drop=True, inplace=True)

    # Trend Indicators
    def compute_MI():
        pass  # Placeholder for future implementation

    def compute_EMA(self, data_wrapper: DataWrapper, period=12):
        ema_feature = ta.EMA(data_wrapper.get_dataframe()['close'], timeperiod=period).rename('ema')
        data_wrapper.add_features(ema_feature)
        data_wrapper.get_dataframe().dropna(inplace=True)
        data_wrapper.get_dataframe().reset_index(drop=True, inplace=True)

    def compute_TEMA(self, data_wrapper: DataWrapper, period=12):
        tema_feature = ta.TEMA(data_wrapper.get_dataframe()['close'], timeperiod=period).rename('tema')
        data_wrapper.add_features(tema_feature)
        data_wrapper.get_dataframe().dropna(inplace=True)
        data_wrapper.get_dataframe().reset_index(drop=True, inplace=True)   

    # Volatility Indicators
    def compute_ATR(self, data_wrapper: DataWrapper, period=14):
        atr_feature = ta.ATR(data_wrapper.get_dataframe()['high'], data_wrapper.get_dataframe()['low'], data_wrapper.get_dataframe()['close'], timeperiod=period).rename('atr')
        data_wrapper.add_features(atr_feature)
        data_wrapper.get_dataframe().dropna(inplace=True)
        data_wrapper.get_dataframe().reset_index(drop=True, inplace=True)

    def compute_BBANDS(self, data_wrapper: DataWrapper, period=20, nbdevup=2, nbdevdn=2, matype=0):
        upperband, middleband, lowerband = ta.BBANDS(data_wrapper.get_dataframe()['close'], timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
        data_wrapper.add_features(upperband.rename('bb_upper'))
        data_wrapper.add_features(middleband.rename('bb_middle'))
        data_wrapper.add_features(lowerband.rename('bb_lower'))
        data_wrapper.get_dataframe().dropna(inplace=True)
        data_wrapper.get_dataframe().reset_index(drop=True, inplace=True)

    # Volume Indicators
    def compute_OBV(self, data_wrapper: DataWrapper):
        obv_feature = ta.OBV(data_wrapper.get_dataframe()['close'], data_wrapper.get_dataframe()['volume']).rename('obv')
        data_wrapper.add_features(obv_feature)
        data_wrapper.get_dataframe().dropna(inplace=True)
        data_wrapper.get_dataframe().reset_index(drop=True, inplace=True)

    def compute_MFI(self, data_wrapper: DataWrapper, period=14):
        mfi_feature = ta.MFI(data_wrapper.get_dataframe()['high'], data_wrapper.get_dataframe()['low'], data_wrapper.get_dataframe()['close'], data_wrapper.get_dataframe()['volume'], timeperiod=period).rename('mfi')
        data_wrapper.add_features(mfi_feature)
        data_wrapper.get_dataframe().dropna(inplace=True)
        data_wrapper.get_dataframe().reset_index(drop=True, inplace=True)
    

    