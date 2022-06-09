import pandas as pd
import numpy as np
from data_treat.treat_func import for_age_column, clear_quotes, clear_quotes_and_point, clear_y_colum
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


class DataCleaner:

    def __init__(self, file) -> None:
        self.path = f"./data/{file}.csv"
        self.data = pd.read_csv(self.path)

    def head(self):
        return self.data.head()

    def get_data(self):
        return self.data



class Equlizer:

    def __init__(self, data, flag_full, num) -> None:
        self.data = data
        self.num = num
        if not flag_full:
            self.equl()

    def equl(self):
        data_0 = self.data[self.data["fraud"] == 0][:self.num]
        data_1 = self.data[self.data["fraud"] == 1]
        self.data = pd.concat([data_0,data_1]).sample(frac=1)
        

    def get_data(self):
        return self.data

class ThreeQRule:

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data 
        val_dict = self.standard_deviation()
        self.q3_clear(val_dict)

    def standard_deviation(self):
        val_dict = {}
        for column in self.data.columns:
            std = self.data[column].values
            val_dict[column] = (std - std * 3, std + std * 3)
        return val_dict

    def q3_clear(self, val_dict):
        for column in self.data.columns:
            self.data[column] = self.data[val_dict[column][0] < self.data["fraud"].values < val_dict[column][1]]

    def get_data(self):
        return self.data
##b = DataEncoder(a.get_data())
#c = Equlizer(b.get_data())
#print(c)
#print(pd.value_counts(b.get_data()["y"]))

'''
    index: The index of the row.
    age: The age of the person.
    job: The job of the person.
    marital: The marital status of the person.
    education: The education level of the person.
    default: Whether or not the person has credit in default.
    balance: The balance of the person.
    housing: Whether or not the person has a housing loan.
    loan: Whether or not the person has a personal loan.
    contact: The contact communication type of the person.
    day: The day of the week of the last contact.
    month: The month of the year of the last contact.
    duration: The duration of the last contact, in seconds.
    campaign: The number of contacts performed during this campaign and for this client.
    pdays: The number of days that passed by after the client was last contacted from a previous campaign.
    previous: The number of contacts performed before this campaign and for this client.
    poutcome: The outcome of the previous marketing campaign.
    y: Whether or not the client has subscribed a term deposit
'''


