

def for_age_column(item):
    return item.split("\"")[1]

def clear_quotes(item):
    return item.replace("\"", "") if type(item) == str else item

def clear_quotes_and_point(item):
    return item.replace("\"", "").replace(".","") if type(item) == str else item

def clear_y_colum(item):
    return item.split("\"")[0]