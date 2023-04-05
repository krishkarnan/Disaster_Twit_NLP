import pandas as pd
import keras

from Preprocessing import toklean_text
from Model import y
from Preprocessing import clean_tweet,dft
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,Embedding, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
import seaborn as sns
from tensorflow.keras.utils import plot_model
#from Preprocessing import Tokenizer

dfts= pd.read_csv("test.csv")
dfts.head().style.background_gradient(cmap='coolwarm')
dfts['clean_text']=dfts['text'].apply(toklean_text)
dfts["clean_text"]=dfts["clean_text"].apply(clean_tweet)


l=50
max_features=5000
tokenizer=Tokenizer(num_words=max_features,split=' ')
tokenizer.fit_on_texts(dft['clean_text'].values)
X=tokenizer.texts_to_sequences(dft['clean_text'].values)
X=pad_sequences(X,maxlen=l)

tokenizer.fit_on_texts(dfts['clean_text'].values)
test_token = tokenizer.texts_to_sequences(dfts['clean_text'].values)
test_token = pad_sequences(test_token,maxlen=l)

embed_dim=100
lstm_out=100
model = Sequential()
model.add(Embedding(max_features,embed_dim,input_length=X.shape[1]))
model.add(Dropout(0.2))
model.add(LSTM(lstm_out,dropout=0.2,recurrent_dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
adam=optimizers.Adam(learning_rate=2e-3)
model.compile(optimizer= keras.optimizers.Adam(), loss= keras.losses.BinaryCrossentropy(from_logits= False),metrics=[keras.metrics.Accuracy()])
#print(model.summary())

#kp=plot_model(model, to_file='model.png')
#print(kp)

es_callback=keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)
model.fit(X,y,epochs=10,validation_split=0.2,callbacks=[es_callback],batch_size=32)

y_hat= model.predict(test_token).round()
print(y_hat)
sns.barplot(y_hat)