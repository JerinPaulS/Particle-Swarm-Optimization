import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

data = pd.read_csv("pima-indians-diabetes.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

d = preprocessing.normalize(X)
X = pd.DataFrame(d)

X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=4)

model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(64,activation='tanh',input_shape=(8,)),
    tf.keras.layers.Dense(64,activation='tanh'),
    tf.keras.layers.Dense(2,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#fit model on train dataset
model.fit(X_train,y_train,epochs=1500)
predicted = model.predict(X_test)
predicted = np.argmax(predicted, axis = 1)
print(model.evaluate(X_test, y_test))

confusion_matrix = metrics.confusion_matrix(y_test, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()

actual_values, predicted_values = y_test, predicted

print("Accuracy score",accuracy_score(actual_values, predicted_values))
print("Precision score",precision_score(actual_values, predicted_values, pos_label=1))
print("Recall score",recall_score(actual_values, predicted_values))
print("F1 score",f1_score(actual_values, predicted_values))
