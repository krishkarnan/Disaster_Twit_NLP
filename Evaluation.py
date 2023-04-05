#Library for Testing
from sklearn import metrics
from Model import model
from Model import y_train, X_train, y_test,y_pred
from functools import reduce
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
train_accuracy=round(metrics.accuracy_score(y_train,model.predict(X_train).round())*100)

print('Accuracy is:',(metrics.accuracy_score(y_test,y_pred)))
print('Recall is :',(metrics.recall_score(y_test,y_pred)))
print('precision is:',(metrics.precision_score(y_test,y_pred)))

confu_matrix=confusion_matrix(y_test,y_pred)
print(confu_matrix)

print(classification_report(y_test,y_pred))