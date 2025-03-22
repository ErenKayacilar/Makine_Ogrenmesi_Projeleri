import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(X @ weights)
    cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    return cost


def gradient_descent(X, y, weights, learning_rate, epochs):
    m = len(y)
    cost_history = []
    for _ in range(epochs):
        h = sigmoid(X @ weights)
        gradient = (1/m) * (X.T @ (h - y))
        weights -= learning_rate * gradient
        cost_history.append(compute_cost(X, y, weights))
    return weights, np.array(cost_history).flatten()


file_path = r"C:\Users\S.EREN\Downloads\BankNoteAuthentication_cleaned.csv"
df = pd.read_csv(file_path)
X = df.drop(columns=["class"]).values
y = df["class"].values.reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]


weights = np.zeros((X_train.shape[1], 1))


learning_rate = 0.1
epochs = 1000
start_time = time.time()
weights, cost_history = gradient_descent(X_train, y_train, weights, learning_rate, epochs)
manual_train_time = time.time() - start_time


start_time = time.time()
y_pred_prob = sigmoid(X_test @ weights)
y_pred = (y_pred_prob >= 0.5).astype(int)
manual_test_time = time.time() - start_time


conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)


print("Manuel Logistic Regression Model Sonuçları:")
print("Doğruluk:", accuracy)
print("Karmaşıklık Matrisi:\n", conf_matrix)
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print(f"Eğitim Süresi: {manual_train_time:.6f} saniye")
print(f"Test Süresi: {manual_test_time:.6f} saniye")


plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Gerçek", "Sahte"], yticklabels=["Gerçek", "Sahte"])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Karmaşıklık Matrisi (Manuel Model)")
plt.show()


plt.plot(cost_history)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost Function'un Azalışı")
plt.show()
