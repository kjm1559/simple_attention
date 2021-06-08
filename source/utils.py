import numpy as np
from datetime import timedelta, datetime

# string index data
string_index = ['!', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',\
                'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', \
                'w', 'x', 'y', 'z', '-', ' ', ',', '@'] 

TIME_STEPS = 30 # length of max string

def encoding_embeding(string, string_index):
    '''
    string -> string index
    Arg:
        string (str): target string
        string_index (list): string list for indexing
    '''
    string += '@'
    string += '!' * (TIME_STEPS - len(string))
    encoded_data = [string_index.index(s) for s in string]
    return encoded_data

def decoding_embeding(encoded_data, string_index):
    '''
    string index -> string
    Arg: 
        encoded_data (numpy.array): one hot vectors
        string_index (list): string list for indexing
    Returns:
        string (str): translated data
    '''
    string = ''
    for i in range(len(encoded_data)):
        if encoded_data[i] == string_index.index('@'):
            break
        string += string_index[encoded_data[i]]
    return string

def encoding_string(string, string_index, flag=True):
    '''
    string -> one hot
    Arg:
        string (str): target string
        string_index (list): string list for indexing
    Returns:
        encoded_data (numpy.array): one hot encoded data
    '''
    string += '@'
    string += '!' * (TIME_STEPS - len(string))
    encoded_data = np.zeros((len(string), len(string_index)))
    for i in range(len(string)):
        if flag | (string[i] != '!'):
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
        if np.argmax(idata) == string_index.index('@'):
            break
        string += string_index[np.argmax(idata)]
    return string

def make_data(s, e):
    train_X = []
    train_y = []
    td = timedelta(days=1)
    date = datetime(s[0], s[1], s[2])
    now = datetime(e[0], e[1], e[2])
    while True:
        train_X.append(encoding_embeding(date.strftime("%Y-%m-%d") + ', ' + str(date.weekday()), string_index))
        train_y.append(encoding_string(date.strftime("%A, %d %B %Y").lower(), string_index))
        if date == now:
            break
        date = date + td
    return train_X, train_y

def make_train_data():
    '''
    make data for tringing
    Returns:
        train_X (list): input train data 
        train_y (list): target train data
    '''
    return make_data((1970, 1, 1), (2020, 3, 10))

def make_test_data():
    return make_data((2020, 3, 11), (2021, 6, 4))

def levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(current_row[1:])

        previous_row = current_row

    return previous_row[-1]