import pandas as pd
import numpy as np

COLUMNS = ["distance_from_home","distance_from_last_transaction","ratio_to_median_purchase_price"]

class ThreeQRule:

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data 
        val_dict = self.standard_deviation()
        self.q3_clear(val_dict)

    def standard_deviation(self):
        val_dict = {}
        for column in COLUMNS:
            std = np.std(self.data[column].values, axis=0)
            mean = np.mean(self.data[column].values, axis=0)
            val_dict[column] = (mean - std * 3, mean + std * 3)
        return val_dict

    def q3_clear(self, val_dict):
        for column in COLUMNS:
            self.data = self.data.drop(self.data[self.data[column] < val_dict[column][0]].index)
            self.data = self.data.drop(self.data[self.data[column] > val_dict[column][1]].index)
        
    def get_data(self):
        return self.data