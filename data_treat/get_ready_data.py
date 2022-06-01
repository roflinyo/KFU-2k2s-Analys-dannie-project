from data_treat.prepare_data import DataCleaner, Equlizer

NUM = 87403

def get_data(full=False):
    a = DataCleaner("card_transdata")
    c = Equlizer(a.get_data(), full, NUM)

    return c.get_data()

def get_data_for_analiz():
    a = DataCleaner("card_transdata")
    return a.get_data()



