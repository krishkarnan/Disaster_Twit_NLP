import string

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import re
from abbreviations import abbreviations


dft=pd.read_csv("train.csv")
#print(df)
dft["length"]=dft["text"].apply(len)

#df["length"].describe()
def toklean_text(text):
    clean_text=[char for char in text if char not in string.punctuation]
    clean_text=''.join(clean_text)
    return clean_text

dft['clean_text']=dft['text'].apply(toklean_text)
#print(dft)

def remove_URL(text):
    url=re.compile(r'https?://\s+/www\.\s+')
    return url.sub(r'URL',text)

def remove_HTML(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_not_ASCII(text):
    text=''.join([word for word in text if word in string.printable])
    return text

def word_abbrev(word):
    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word

def replace_abbrev(text):
    string =" "
    for word in text.split():
        string += word_abbrev(word) + " "
    return string

def remove_mention(text):
    at = re.compile(r'@\s+')
    return at.sub(r'USER',text)

def remove_number(text):
    num = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
    return num.sub(r'NUMBER',text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'EMOJI',text)


