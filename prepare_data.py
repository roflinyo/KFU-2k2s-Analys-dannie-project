import pandas as pd
import numpy as np

class DataPrepare:

    def __init__(self, file) -> None:
        self.path = f"./data/{file}.csv"
        self.data = pd.read_csv(self.path)

    def data_treat(self):
        pass

    def remane_colums(self):
        for colum in self.data:
            if "age" in colum:
                self.data = self.data.rename(columns={colum:"age"})
            else:
                self.data = self.data.rename(columns={colum:colum.replace("\"","")})

    def treat_rows(self):
        pass

a = DataPrepare("bank-additional")
#b = DataPrepare("bank-additional")
print(a.data.columns)
a.remane_colums()
print(a.data.columns)


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


