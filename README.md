# Naive bayes uygulamaları
 yzm202 Makine öğrenmesi dersi projelerinden naive bayes projesinde scikit learn kullanarak ve kullanmayarak naive bayes uygulamaları yaptım ve aralarındaki farkları karşılaştırdım

Scikit-learn Kullanarak Elde Edilen Sonuçlar:
Model Doğruluk Oranı: 0.7101
Eğitim Süresi: 0.0156 saniye
Test Süresi: 0.0010 saniye
Karmaşıklık Matrisi:
   [[33 25]
    [15 65]]

Scikit-learn Kullanılmadan Yazılan Naive Bayes Modeli:
Model Doğruluk Oranı: 0.6639
Eğitim Süresi: 1.5096 saniye
Test Süresi: 1.4656 saniye
Karmaşıklık Matrisi:
  [[54 17]
    [23 25]]

Karşılaştırma sonucu :
Modeller incelendiğinde Scikit-learn kullanılarak yapılan modelin doğruluk oranının daha yüksek olduğu görülmektedir ama bu tamamen kodlama biçimine göre değişebilir. Ancak eğitim ve test süreleri açısından büyük bir fark bulunmaktadır. Scikit-learn kullanmayan modelin eğitimi ve test süresi çok daha uzundur. Bunun sebebi, optimize edilmiş kütüphanelerin eksikliği ve temel Python uygulamalarının daha fazla hesaplama maliyeti gerektirmesidir.Özelleştirilmiş bir model gerekliyse, Naive Bayes gibi algoritmalar sıfırdan uygulanarak detaylı analiz yapılabilir ancak işlem süresi uzun olabilir.

Şimdi veri setine - amaca göre naive bayes uygulamalarını nasıl farklı değerlendirmemiz gerektiğine bakalım

Dengesiz Veri Kümeleri: 
Eğer sınıflar arasında büyük bir dengesizlik varsa, doğruluk metriği yanıltıcı olabilir. Örneğin, %90 oranında "0" içeren bir veri kümesinde modelin her zaman "0" tahmin etmesi %90 doğruluk verebilir, ancak bu model işlevsel değildir. Bu tür durumlarda F1 skoru, hassasiyet (recall) ve kesinlik (precision) gibi metrikler daha anlamlıdır.

Problem Türü:
Tıbbi Teşhis: Yanlış negatiflerin (FN) kritik olduğu durumlarda hassasiyet (recall) ön planda olmalıdır.
Spam Filtreleme: Yanlış pozitiflerin (FP) önemli olduğu durumlarda kesinlik (precision) daha önemli olabilir.
Genel Sınıflandırma Problemleri: Dengeli veri kümelerinde F1 skoru veya doğruluk (accuracy) iyi bir gösterge olabilir.




