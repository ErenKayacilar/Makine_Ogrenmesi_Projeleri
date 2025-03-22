 **Logistic Regression Model Karşılaştırması**

Bu projede Logistic Regression algoritmasını hem Scikit-learn kullanarak hem de kullanmayarak eğitip performanslarını karşılaştırdık. 
Elde edilen sonuçlar doğruluk, eğitim süresi, test süresi ve karmaşıklık matrisi gibi metrikler açısından inceledik.

 1. Scikit-learn Kullanarak Logistic Regression Sonuçları
- Doğruluk: 0.965
- Karmaşıklık Matrisi:
  
  [[113   6]
   [  1  80]]
  
- Sınıflandırma Raporu:
  
                 precision    recall  f1-score   support

           0       0.99      0.95      0.97       119
           1       0.93      0.99      0.96        81

    accuracy                           0.96       200
   macro avg       0.96      0.97      0.96       200
weighted avg       0.97      0.96      0.97       200
  
- Eğitim Süresi: 0.004562 saniye
- Test Süresi: 0.00000 saniye

 2. Kütüphanesiz Logistic Regression Sonuçları
- Doğruluk: 0.96
- Karmaşıklık Matrisi:
  
  [[112   7]
   [  1  80]]
  
- Sınıflandırma Raporu:
  
                 precision    recall  f1-score   support

           0       0.99      0.94      0.97       119
           1       0.92      0.99      0.95        81

    accuracy                           0.96       200
   macro avg       0.96      0.96      0.96       200
weighted avg       0.96      0.96      0.96       200
  
- Eğitim Süresi: 0.106348 saniye
- Test Süresi: 0.000000 saniye

 3. Sonuç ve Karşılaştırma

Modeller karşılaştırıldığında:
- Scikit-learn modeli daha yüksek doğruluk oranına sahiptir ancak fark çok küçük (%96.5 vs. %96.0).
- Eğitim süresi açısından büyük fark bulunmaktadır. Scikit-learn modeli 0.0045 saniyede eğitilirken, kütüphanesiz model 0.1063 saniyede eğitildi.
- Scikit-learn modeli optimize edilmiş ve hızlıdır, ancak kütüphanesiz model, algoritmanın temel bileşenlerini anlamak için faydalıdır.

 4. Logistic Regression Uygulamalarında Dikkat Edilmesi Gerekenler

 Dengesiz Veri Kümeleri:
- Eğer sınıflar arasında ciddi bir dengesizlik varsa, doğruluk oranı yanıltıcı olabilir. Örneğin, veri kümesinin %90'ı "0" etiketi içeriyorsa, model sürekli "0" tahmin ederek %90 doğruluk sağlayabilir.
   Ancak bu gerçek performansı yansıtmaz.
- Bu tür durumlarda F1 skoru, kesinlik (precision) ve duyarlılık (recall) gibi metriklere odaklanmak gerekir.

 Problem Türüne Göre Metrik Seçimi:
- Tıbbi teşhis (Hastalık tahmini): Yanlış negatifler (FN) kritik olduğu için hassasiyet (recall) öncelikli olmalıdır.
- Spam filtreleme: Yanlış pozitiflerin (FP) önlenmesi gerektiğinden kesinlik (precision) daha önemli olabilir.
- Genel sınıflandırma problemleri: Veri dengeliyse doğruluk (accuracy) veya F1 skoru uygun bir değerlendirme metriği olabilir.

Bu çalışma hem makine öğrenmesi kütüphanelerinin avantajlarını hem de manuel uygulamaların temel mantığını anlamak açısından önemli bir karşılaştırma sunmaktadır. Kütüphanesiz yazmak algoritmaların mantığı için
faydalıdır ama iş hayatında kütüphaneler kullanılmalıdır.

