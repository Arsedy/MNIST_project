import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

# Veri yükleme ve normalize etme
data = pd.read_csv("mnist_train.csv").to_numpy()
np.random.shuffle(data)
X = data[:, 1:] / 255.
Y = data[:, 0]
X_train = X[1000:].T
Y_train = Y[1000:]
X_dev = X[:1000].T
Y_dev = Y[:1000]

# Parametreleri başlat
def init_params():
    W1 = np.random.randn(64, 784) * 0.01
    b1 = np.zeros((64, 1))
    W2 = np.random.randn(32, 64) * 0.01
    b2 = np.zeros((32, 1))
    W3 = np.random.randn(10, 32) * 0.01
    b3 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3

# Aktivasyonlar
def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def one_hot(Y, num_classes=10):
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

# Forward propagation
def forward_prop(X, W1, b1, W2, b2, W3, b3, keep_prob=0.9):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    D1 = (np.random.rand(*A1.shape) < keep_prob) / keep_prob
    A1 *= D1

    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    D2 = (np.random.rand(*A2.shape) < keep_prob) / keep_prob
    A2 *= D2

    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, D1, Z2, A2, D2, Z3, A3

# Backward propagation
def backward_prop(X, Y, Z1, A1, D1, Z2, A2, D2, Z3, A3, W1, W2, W3):
    m = X.shape[1]
    one_hot_Y = one_hot(Y, num_classes=10)
    dZ3 = A3 - one_hot_Y
    dW3 = 1/m * dZ3.dot(A2.T)
    db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = W3.T.dot(dZ3) * D2
    dZ2 = dA2 * ReLU_deriv(Z2)
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = W2.T.dot(dZ2) * D1
    dZ1 = dA1 * ReLU_deriv(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

# Parametre güncelleme
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha=0.05):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    return W1, b1, W2, b2, W3, b3

# Tahmin ve doğruluk
def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Mini-batch gradient descent
def gradient_descent(X, Y, alpha=0.05, iterations=500, batch_size=128, keep_prob=0.9):
    W1, b1, W2, b2, W3, b3 = init_params()

    for i in range(iterations):
        permutation = np.random.permutation(X.shape[1])
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[permutation]

        for j in range(0, X.shape[1], batch_size):
            X_batch = X_shuffled[:, j:j+batch_size]
            Y_batch = Y_shuffled[j:j+batch_size]

            Z1, A1, D1, Z2, A2, D2, Z3, A3 = forward_prop(X_batch, W1, b1, W2, b2, W3, b3, keep_prob)
            dW1, db1, dW2, db2, dW3, db3 = backward_prop(X_batch, Y_batch, Z1, A1, D1, Z2, A2, D2, Z3, A3, W1, W2, W3)
            W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        if i % 50 == 0:
            _, _, _, _, _, _, _, A3_full = forward_prop(X, W1, b1, W2, b2, W3, b3, keep_prob=1.0)
            acc = get_accuracy(get_predictions(A3_full), Y)
            print(f"Iteration {i}, Accuracy: {acc:.4f}")

    return W1, b1, W2, b2, W3, b3

# Eğitimi başlat
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train)

# Modeli kaydet
with open("mnist_model.pkl", "wb") as f:
    pickle.dump((W1, b1, W2, b2, W3, b3), f)

# Test fonksiyonu
def test_prediction(index, X, Y, W1, b1, W2, b2, W3, b3):
    current_image = X[:, index, None]
    _, _, _, _, _, _, _, A3 = forward_prop(current_image, W1, b1, W2, b2, W3, b3, keep_prob=1.0)
    prediction = get_predictions(A3)[0]
    label = Y[index]
    print(f"Prediction: {prediction}, Label: {label}")
    plt.imshow(current_image.reshape(28, 28) * 255, cmap='gray')
    plt.show()

# Örnek test
test_prediction(0, X_train, Y_train, W1, b1, W2, b2, W3, b3)
