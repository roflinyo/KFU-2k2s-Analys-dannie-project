import pandas as pd
import numpy as np


class ThreeQRule:

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data 
        val_dict = self.standard_deviation()
        self.q3_clear(val_dict)

    def standard_deviation(self):
        val_dict = {}
        for column in self.data.columns:
            std = np.std(self.data[column].values, axis=0)
            mean = np.mean(self.data[column].values, axis=0)
            val_dict[column] = (mean - std * 3, mean + std * 3)
        return val_dict

    def q3_clear(self, val_dict):
        for column in self.data.columns:
            self.data = self.data.drop([val_dict[column][0] > self.data[column].values and \
                                         self.data[column].values < val_dict[column][1]])

    def get_data(self):
        return self.data