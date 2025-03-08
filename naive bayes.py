import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Veri setini yükleme
df = pd.read_csv(r"C:\Users\S.EREN\Downloads\Cleaned_Task.csv", encoding="latin1")

# Gerekli sütunları seçme
df = df[['News_Headline', 'Source', 'Stated_On', 'Label']]

# Label sütununu sayısal hale getirme
label_mapping = {
    'pants-fire': 0, 'false': 0, 'barely-true': 0,
    'half-true': 1, 'mostly-true': 1, 'true': 1
}  # 0: Yanlış, 1: Doğru

df['Label'] = df['Label'].str.lower().map(label_mapping)

# Boş değerleri temizleme
df = df.dropna()

# TF-IDF vektörizasyonu (News_Headline)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_text = vectorizer.fit_transform(df['News_Headline'])

# One-Hot Encoding (Source ve Stated_On)
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_categorical = ohe.fit_transform(df[['Source', 'Stated_On']])

# Özellikleri birleştirme
X = hstack([X_text, X_categorical])
y = df['Label']

# SMOTE ile dengesiz veriyi dengeleme
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Eğitim ve test veri kümelerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Logistic Regression modelini oluşturma ve parametre ayarlama
model = LogisticRegression(class_weight='balanced', C=0.5, max_iter=300, solver='liblinear')
start_time = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_time

# Modeli test etme
start_time = time.time()
y_pred = model.predict(X_test)
test_time = time.time() - start_time

# Performans değerlendirme
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Sonuçları ekrana yazdırma
print("-------------Sonuçlar--------------")
print(f"Model Doğruluk Oranı: {accuracy:.4f}")
print(f"Eğitim Süresi: {train_time:.4f} saniye")
print(f"Test Süresi: {test_time:.4f} saniye")
print("Karmaşıklık Matrisi:")
print(conf_matrix)

# Karmaşıklık matrisini görselleştirme
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Logistic Regression - Confusion Matrix')
plt.show()















