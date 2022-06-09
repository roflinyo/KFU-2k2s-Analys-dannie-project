import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class CustomNormalizer:

    def normalize(data: pd.DataFrame):
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(data.values)
        return x_scaled

    def standardize(data: pd.DataFrame):
        min_max_scaler = StandardScaler()
        x_scaled = min_max_scaler.fit_transform(data.values)
        return x_scaled
    