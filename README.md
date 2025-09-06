#MNIST Rakam Tanıma

Bu proje, MNIST veri setini kullanarak rakam tanıma için bir sinir ağı uygular. İki ana betik içerir:

1.train.py: Modeli eğitmek için.

2.prediction.py: Eğitilmiş modeli kullanarak tahmin yapmak için.

---


##Özellikler

-**Eğitim**: MNIST veri seti üzerinde mini-batch gradyan inişi kullanarak bir sinir ağı eğitin.

-**Tahmin**: Eğitilmiş modeli kullanarak özel resimlerden rakam tahmin edin.

-**Model Kalıcılığı**: Eğitilmiş modeli pickle ile kaydedin ve yükleyin.

-**3 Alt Katmanlı Model**: Sinir ağı, 3 gizli katmana (hidden layer) sahiptir.

---

##Gereksinimler

Gerekli Python kütüphanelerini yükleyin:
1.MNIST datasetleri dosyaya eklenmeli:
-https://www.kaggle.com/datasets/oddrationale/mnist-in-csv (csv olması lazım!)
-prediction.py denemek için img.png dosya örneklerini https://www.kaggle.com/datasets/alexanderyyy/mnist-png/data bu linkten kullandım
2.pip install numpy matplotlib pillow

---

##ÖNEMLİ NOT:
-Proje içinde zaten 3 katmanında kaydedilmiş modeli vardır (mnist_model.pkl) o yüzden prediction.py direkt çalışır durumdadır.
-Öğrenim amaçlı yazılmıştır
