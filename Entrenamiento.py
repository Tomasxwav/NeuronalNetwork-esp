from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import time

data = np.loadtxt('mnist_train.csv', delimiter=',')
print("Lectura de la base de datos completa")
tic = time.process_time()
ncol = data.shape[1]

X = data[:,1:ncol]
y = data[:,0]
print(len(X[0]))
print(len(X))
print(X)

#MODELO DE REGRESION LOGISTICA

clf = LogisticRegression()
clf.fit(X, y)
print("Entrenamiento de regresión logistica completo")

filename_RL = 'finalized_model_RL.sav'
pickle.dump(clf, open(filename_RL, 'wb'))


#MODELO REDES NEURONALES
model = MLPClassifier(hidden_layer_sizes=(15, 10, 5), activation='relu', solver='adam', max_iter=1000, random_state=1)
model.fit(X, y)
print("Entrenamiento de RNA completo")

filename_RNA = 'finalized_model_RNA.sav'
pickle.dump(model, open(filename_RNA, 'wb'))

#PRUEBAS
X_test = data[:,1:ncol]
y_test = data[:,0]

loaded_model_RL = pickle.load(open('finalized_model_RL.sav', 'rb'))

loaded_model_RNA = pickle.load(open('finalized_model_RNA.sav', 'rb'))

predicted_LR = loaded_model_RL.predict(X_test)
print(f"Predicciones RL: {predicted_LR}")
error_RL = 1 - accuracy_score(y_test, predicted_LR)
print(f"el error del modelo de regresion logistica es del: {round(error_RL*100, 2)} %")

predicted_RNA = loaded_model_RNA.predict(X_test)
print(f"Predicciones RNA: {predicted_RNA}")
error_RNA = 1 - accuracy_score(y_test, predicted_RNA)
print(f"El error del modelo RNA es del: {(round(error_RNA*100, 2))} %")

toc = time.process_time()
print(toc - tic)