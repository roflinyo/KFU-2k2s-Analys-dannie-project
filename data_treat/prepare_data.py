import pandas as pd
import numpy as np
from treat_func import for_age_column, clear_quotes, clear_quotes_and_point, clear_y_colum
from sklearn.preprocessing import LabelEncoder

TEXT_FEATURES = ["job", "marital", "education", "default", "housing", 
                "loan", "contact", "month", "day_of_week", "poutcome", "y"]

class DataCleaner:

    def __init__(self, file) -> None:
        self.path = f"./data/{file}.csv"
        self.data = pd.read_csv(self.path)
        self.data_treat()

    def data_treat(self):
        self.remane_colums()
        self.treat_columns()

    def remane_colums(self):
        for colum in self.data:
            if "age" in colum:
                self.data = self.data.rename(columns={colum:"age"})
            else:
                self.data = self.data.rename(columns={colum:colum.replace("\"","")})

    def treat_columns(self):
        for colum in self.data:
            if colum == "age":
                self.data["age"] = self.data["age"].apply(for_age_column)
            elif colum == "job":
                self.data["job"] = self.data["job"].apply(clear_quotes_and_point)
            elif colum == "y":
                self.data["y"] = self.data["y"].apply(clear_y_colum)
            else:
                self.data[colum] = self.data[colum].apply(clear_quotes)
        self.data = self.data.astype({'age':'int64'})

    def head(self):
        return self.data.head()

    def get_data(self):
        return self.data

class DataEncoder:

    def __init__(self, data) -> None:
        self.data = data
        self.label_encoding()

    def label_encoding(self):
        label_encoder = LabelEncoder()
        for col in TEXT_FEATURES:
            self.data[col] = label_encoder.fit_transform(self.data[col]) + 1

    def get_data(self):
        return self.data

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


