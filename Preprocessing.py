import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import re
import nltk
nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS
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
def transcription_sad(text):
    eyes =  "[8:=;]"
    nose = "['\-]"
    smiley = re.compile(r'[8:=;][\'\-]?[(\\/]')
    return smiley.sub(r'SADFACE',text)

def transcription_smile(text):
    eyes = "[8:=;]"
    nose = "['\-]"
    smiley = re.compile(r'[8:=;][\'\-]?[)dDp]')
    return smiley.sub(r'SMILE',text)

def transcription_heart(text):
    heart=re.compile(r'<3')
    return heart.sub(r'HEART',text)


# For clean the text

def clean_tweet(text):
    text = remove_URL(text)
    text= remove_HTML(text)
    text=remove_not_ASCII(text)
    text=remove_mention(text)
    text=remove_number(text)
    text=replace_abbrev(text)
    text=remove_emoji(text)
    text=transcription_smile(text)
    text=transcription_sad(text)
    text=transcription_heart(text)
    return text



dft["clean_text"]=dft["clean_text"].apply(clean_tweet)
#ak1

#print(stopwords.words('english'))

def toremove_stopword(text):
    remove_stopword=[word for word in text.split() if word.lower() not in stopwords.words('english')]
    return remove_stopword

dft["clean_text"]=dft["clean_text"].apply(toremove_stopword)

#print(dft)

#Tokenization

max_features=3000
tokenizer=Tokenizer(num_words=max_features,split=' ')
tokenizer.fit_on_texts(dft['clean_text'].values)
X=tokenizer.texts_to_sequences(dft['clean_text'].values)
X=pad_sequences(X)

#print(X[0])

#tokenizer.sequences_to_texts([[ 713,  154,   56, 1434,   14]])