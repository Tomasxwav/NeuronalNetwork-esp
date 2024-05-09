import numpy as np
import pickle
import matplotlib.pyplot as plt

data = np.loadtxt('mnist_test.csv', delimiter=',')
ncol = data.shape[1]

X_test = data[:, 1:ncol]
y_test = data[:, 0]

select_image = np.random.random([28, 28])

ex = np.random.randint(0, 10000)
ex = 5000 - 1
print(X_test)
print(type(X_test))
for ex in range(1000):
    print(f"Para el dato {ex}")
    loaded_model_LR = pickle.load(open('finalized_model_RL.sav', 'rb'))
    loaded_model_RNA = pickle.load(open('finalized_model_RNA.sav', 'rb'))


    xtest = X_test[ex,].reshape(1, -1)
    predicted_LR = loaded_model_LR.predict(xtest)
    print(f"La regresion logistica predice un: {predicted_LR}")


    xtest = X_test[ex,].reshape(1, -1)
    predicted_RNA = loaded_model_RNA.predict(xtest)
    print(f"La red neuronal predice un: {predicted_RNA}")


    select_image1 = 1 - X_test[ex,] / 255
    k = 0
    for i in range(0, 28):
        for j in range(0, 28):
            select_image[i, j] = select_image1[k]
            k = k + 1
    plt.imshow(select_image, cmap='gray', interpolation='nearest')
    plt.show()
    ex += 23