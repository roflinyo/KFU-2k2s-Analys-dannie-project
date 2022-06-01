import pandas as pd
from sqlalchemy import true
from data_treat.prepare_data import DataCleaner, DataEncoder, Equlizer

def get_data(full=True):
    a = DataCleaner("bank-additional")
    b = DataEncoder(a.get_data())
    c = Equlizer(b.get_data(), full)



    return c.get_data()

def get_data_for_analiz():
    a = DataCleaner("bank-additional")
    return a.get_data()



