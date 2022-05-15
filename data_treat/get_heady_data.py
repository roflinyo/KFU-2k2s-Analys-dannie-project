from prepare_data import DataCleaner, DataEncoder

def get_data():
    a = DataCleaner("bank-additional")
    b = DataEncoder(a.get_data())
    return b.get_data()