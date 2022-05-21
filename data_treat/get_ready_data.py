import pandas as pd
from data_treat.prepare_data import DataCleaner, DataEncoder, Equlizer

def get_data():
    a = DataCleaner("bank-additional")
    b = DataEncoder(a.get_data())
    c = Equlizer(b.get_data())



    return c.get_data()



