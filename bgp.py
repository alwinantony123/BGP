from tensorboard import summary
from tensorflow import keras


from keras.models import load_model
custom_objects = {"f1_m": 0.4498 , "precision_m": 0.6627, "recall_m": 0.3429}
model = load_model('book_model_one_lstm.h5', custom_objects=custom_objects)


import json


token_dict=json.load(open('token_dict.json'))


vocab_dict = {v:k for (k, v) in token_dict.items()}
#def reviewBook(model,text):




import re
import numpy as np
min_desc_length=6
max_desc_length=200
def tokenizer(desc, vocab_dict, max_desc_length):
    '''
    Function to tokenize descriptions
    Inputs:
    - desc, description
    - vocab_dict, dictionary mapping words to their corresponding tokens
    - max_desc_length, used for pre-padding the descriptions where the no. of words is less than this number
    Returns:

    List of length max_desc_length, pre-padded with zeroes if the desc length was less than max_desc_length
    '''
    a=[vocab_dict[i] if i in vocab_dict else 0 for i in desc.split()]
    b=[0] * max_desc_length
    if len(a)<max_desc_length:
        return np.asarray(b[:max_desc_length-len(a)]+a).squeeze()
    else:
        return np.asarray(a[:max_desc_length]).squeeze()


def _removeNonAscii(s):
    return "".join(i for i in s if ord(i)<128)
    
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = text.replace('(ap)', '')
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    text = re.sub('[^a-zA-Z ?!]+', '', text)
    text = _removeNonAscii(text)
    text = text.strip()
    return text

def cleaner(df):
    # removing others reduces the dataset to only fiction
    # df = df[df['label'] != 'others']

    # df = df[df['language'] != 'nil']

    df['clean_desc'] = df['book_desc'].apply(clean_text)

    return df


from collections import defaultdict
import cufflinks as cf
import pandas as pd

import matplotlib.pyplot as plt
import string

valid_chars = string.ascii_letters+string.digits+' '

import plotly.offline as pyoff
import plotly.graph_objs as go

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

pyoff.init_notebook_mode()
cf.go_offline()

def reviewBook(text,model):
    threshold = 0.5 # possibly change it back to 0.2
    a = clean_text(text)
    a = tokenizer(a, vocab_dict, max_desc_length)
    a = np.reshape(a, (1,max_desc_length))
    a=a.astype(np.float)
    #predict_x=model.predict(text) 
    #classes_x=np.argmax(predict_x,batch_size=1)
    output = model.predict(a, batch_size=1)
    #print(output)
    return(output) 
    return(classes_x)


