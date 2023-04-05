from Preprocessing import dft
import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding, LSTM, Dropout
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from Preprocessing import X
from Preprocessing import max_features

#Library for Testing
from sklearn import metrics
from functools import reduce
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


y=dft['target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =41)

embed_dim=32
lstm_out = 32
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length = X.shape[1]))
model.add(Dropout(0.2))
model.add(LSTM(lstm_out,dropout=0.2,recurrent_dropout=0.4))
model.add(Dense(1,activation='sigmoid'))
adam = optimizers.Adam(learning_rate=0.002)
#model.compile(loss='binary_crossentropy',optimizer=adam,metrices=['accuracy'])
#model.compile(loss="binary_crossentropy", optimizer='adam',metrices="accuracy")
model.compile(optimizer= keras.optimizers.Adam(), loss= keras.losses.BinaryCrossentropy(from_logits= False),metrics=[keras.metrics.Accuracy()])
#print(model.summary())


model.fit(X_train,y_train,epochs=10,batch_size=32,validation_data=(X_test,y_test))

y_pred = model.predict(X_test).round()
#print(y_pred)