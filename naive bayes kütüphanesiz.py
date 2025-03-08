import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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

df = df.dropna()

# TF-IDF vektörizasyonunu elle yapma
def compute_tf_idf(corpus):
    term_counts = [Counter(doc.split()) for doc in corpus]
    df_counts = Counter(word for doc in term_counts for word in doc)
    N = len(corpus)

    tf_idf_matrix = []
    for term_count in term_counts:
        tf_idf_vector = {word: (term_count[word] / sum(term_count.values())) * np.log(N / (df_counts[word] + 1))
                         for word in term_count}
        tf_idf_matrix.append(tf_idf_vector)

    return tf_idf_matrix, list(df_counts.keys())

X_text, vocabulary = compute_tf_idf(df['News_Headline'])

# Naive Bayes Modeli
class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}

    def fit(self, X, y):
        # Class probabilities
        class_counts = Counter(y)
        total_samples = len(y)
        self.class_probs = {c: class_counts[c] / total_samples for c in class_counts}

        # Word probabilities for each class
        word_counts = {c: Counter() for c in class_counts}
        for doc, label in zip(X, y):
            word_counts[label].update(doc)

        # Calculate word probabilities with Laplace smoothing
        self.word_probs = {
            c: {word: (word_counts[c][word] + 1) / (sum(word_counts[c].values()) + len(vocabulary))
                for word in vocabulary}
            for c in class_counts
        }

    def predict(self, X):
        predictions = []
        for doc in X:
            class_scores = {c: np.log(self.class_probs[c]) for c in self.class_probs}
            for word in doc:
                for c in self.class_probs:
                    class_scores[c] += np.log(
                        self.word_probs[c].get(word, 1 / (sum(self.word_probs[c].values()) + len(vocabulary))))
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)  # NumPy dizisine çevirme

# Modeli eğit ve test et
X = X_text
y = np.array(df['Label'])

# Eğitim ve test veri kümelerini ayırma
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = NaiveBayes()

# Modeli eğitme
start_time = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_time

# Model ile tahmin yapma
start_time = time.time()
y_pred = model.predict(X_test)
test_time = time.time() - start_time

# Performans değerlendirme
accuracy = np.mean(y_pred == y_test)

# Karmaşıklık matrisini manuel hesaplama
def compute_confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)  # 2x2 karmaşıklık matrisi
    for true, pred in zip(y_true, y_pred):
        true = int(true)  # Tam sayıya dönüştür
        pred = int(pred)  # Tam sayıya dönüştür
        cm[true, pred] += 1
    return cm

conf_matrix = compute_confusion_matrix(y_test, y_pred)

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
plt.title('Naive Bayes - Confusion Matrix')
plt.show()

print("Tahmin edilen sınıf dağılımı:", np.unique(y_pred, return_counts=True))
print("Gerçek sınıf dağılımı:", np.unique(y_test, return_counts=True))





