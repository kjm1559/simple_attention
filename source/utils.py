import numpy as np
from datetime import timedelta, datetime

# string index data
string_index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',\
                'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', \
                'w', 'x', 'y', 'z', '-', ' ', ','] 

TIME_STEPS = 30 # length of max string

def encoding_string(string, string_index):
    '''
    string -> one hot
    Arg:
        string (str): target string
        string_index (list): string list for indexing
    Returns:
        encoded_data (numpy.array): one hot encoded data
    '''
    if len(string) < TIME_STEPS:
        for i in range(TIME_STEPS - len(string)):
            string += ' '
    encoded_data = np.zeros((len(string), len(string_index)))
    for i in range(len(string)):
        encoded_data[i][string_index.index(string[i])] = 1
    return encoded_data

def decoding_string(encoded_data, string_index):
    '''
    one hot -> string
    Arg: 
        encoded_data (numpy.array): one hot vectors
        string_index (list): string list for indexing
    Returns:
        string (str): translated data
    '''
    string = ''
    for idata in encoded_data:
        string += string_index[np.argmax(idata)]
    return string

def make_train_data():
    '''
    make data for tringing
    Returns:
        train_X (list): input train data 
        train_y (list): target train data
    '''
    train_X = []
    train_y = []
    td = timedelta(days=1)
    date = datetime(1970, 1, 1)
    now = datetime(2020, 3, 10)
    while True:
        train_X.append(encoding_string(date.strftime("%Y-%m-%d") + ', ' + str(date.weekday()), string_index))
        train_y.append(encoding_string(date.strftime("%A, %d %B %Y").lower(), string_index))
        if date == now:
            break
        date = date + td
    return train_X, train_y