import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns


file_path = r"C:\Users\S.EREN\Downloads\BankNoteAuthentication_cleaned.csv"
df = pd.read_csv(file_path)


X = df.drop(columns=["class"])
y = df["class"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression()
start_time = time.time()
model.fit(X_train, y_train)
sklearn_train_time = time.time() - start_time


start_time = time.time()
y_pred = model.predict(X_test)
sklearn_test_time = time.time() - start_time


conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)


print("Scikit-learn Logistic Regression Model Sonuçları:")
print("Doğruluk:", accuracy)
print("Karmaşıklık Matrisi:\n", conf_matrix)
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print(f"Eğitim Süresi: {sklearn_train_time:.6f} saniye")
print(f"Test Süresi: {sklearn_test_time:.6f} saniye")


plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Gerçek", "Sahte"], yticklabels=["Gerçek", "Sahte"])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Karmaşıklık Matrisi")
plt.show()
