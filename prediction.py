import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Modeli yükle
# -----------------------------
with open("mnist_model.pkl", "rb") as f:
    W1, b1, W2, b2, W3, b3 = pickle.load(f)

print("✅ Model yüklendi.")

# -----------------------------
# Aktivasyon Fonksiyonları
# -----------------------------
def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    # Daha stabil softmax
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# -----------------------------
# Forward Propagation
# -----------------------------
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# -----------------------------
# Tahmin Fonksiyonu
# -----------------------------
def predict_custom_image(W1, b1, W2, b2, W3, b3, image_path):
    # Resmi aç ve grayscale yap
    img = Image.open(image_path).convert("L")  # grayscale
    img = img.resize((28, 28))  # 28x28 boyuta getir
    img_array = np.array(img)

    # Normal
    normal = img_array.reshape(28*28, 1) / 255.0
    _, _, _, _, _, A3_normal = forward_prop(W1, b1, W2, b2, W3, b3, normal)
    pred_normal = np.argmax(A3_normal)


    # ---------------- Sonuçları göster
    print(f"Normal resim tahmini: {pred_normal}")


    plt.subplot(1,2,1)
    plt.title(f"Normal: {pred_normal}")
    plt.imshow(normal.reshape(28,28), cmap="gray")

    plt.show()

# -----------------------------
# Kullanım Örneği
# -----------------------------
predict_custom_image(W1, b1, W2, b2, W3, b3, "img.png")
