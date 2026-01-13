# 🇹🇷 Türkçe Duygu Analizi Sistemi

Makine öğrenmesi algoritmaları kullanarak Türkçe metinlerdeki duyguları (pozitif/negatif/nötr) analiz eden web tabanlı uygulama.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## 📋 Özellikler

- ✅ **5 Farklı ML Algoritması:** Naive Bayes, Logistic Regression, SVM, Random Forest, Voting Ensemble
- ✅ **Grid Search Optimizasyonu:** Otomatik hyperparameter tuning
- ✅ **İki Model Seti:** Ham veri ve dengeli veri modelleri
- ✅ **Modern Web Arayüzü:** Streamlit ile kullanıcı dostu interface
- ✅ **Batch İşleme:** CSV dosyası yükleyerek toplu analiz
- ✅ **Detaylı Raporlama:** Model performans karşılaştırmaları ve görselleştirmeler

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Python 3.8 veya üzeri
- pip paket yöneticisi

### Kurulum

1. Repository'yi klonlayın:
```bash
git clone https://github.com/KULLANICI_ADINIZ/turkish-sentiment-analysis.git
cd turkish-sentiment-analysis
```

2. Virtual environment oluşturun (önerilir):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

4. Veri setini indirin ve modelleri eğitin:
```bash
python run_all.py
```

5. Streamlit uygulamasını başlatın:
```bash
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` adresini açın.

## 📁 Proje Yapısı

```
turkish-sentiment-analysis/
├── src/                          # Kaynak kodlar
│   ├── data_preprocessing.py     # Veri ön işleme
│   ├── model_trainer.py          # Model eğitimi
│   ├── model_evaluator.py        # Model değerlendirme
│   └── download_data.py          # Veri indirme
├── data/                         # Veri setleri (gitignore'da)
├── models_raw/                   # Ham veri modelleri (gitignore'da)
├── models_balanced/              # Dengeli veri modelleri (gitignore'da)
├── app.py                        # Streamlit web arayüzü
├── run_all.py                    # Ana eğitim pipeline'ı
├── requirements.txt              # Python bağımlılıkları
└── README.md                     # Bu dosya
```

## 🎯 Kullanım

### 1. Tekil Metin Analizi
- "Tahmin" sayfasından tek bir metni analiz edin
- Model seçimi yapın
- Anlık sonuç alın

### 2. Toplu Analiz
- "Toplu Analiz" sayfasından CSV dosyası yükleyin
- Metin sütununu seçin
- Tüm metinleri aynı anda analiz edin

### 3. Model Karşılaştırma
- "Model Karşılaştırma" sayfasından tüm modellerin performansını görün
- Accuracy, F1 Score, hız metriklerini inceleyin
- Confusion matrix'leri görüntüleyin

### 4. Veri İstatistikleri
- Ham ve dengeli veri setlerini karşılaştırın
- Sınıf dağılımlarını görüntüleyin

## 🧠 Kullanılan Algoritmalar

1. **Naive Bayes:** Hızlı ve klasik text classification
2. **Logistic Regression:** Yüksek accuracy
3. **Linear SVM:** Margin maximization
4. **Random Forest:** Ensemble method
5. **Voting Ensemble:** 4 modelin kombinasyonu (en iyi performans)

## 📊 Performans

### Dengeli Veri Modelleri

| Model | Accuracy | F1 Score | Hız (ms/örnek) |
|-------|----------|----------|----------------|
| Naive Bayes | 87.56% | 0.8754 | 0.0012 |
| Logistic Regression | 89.88% | 0.8987 | 0.0034 |
| SVM | 89.76% | 0.8975 | 0.0089 |
| Random Forest | 88.34% | 0.8835 | 0.0156 |
| **Voting Ensemble** | **89.88%** | **0.8984** | 0.0245 |

## 🔧 Teknik Detaylar

- **Veri Seti:** 440,679 Türkçe yorum ([HuggingFace](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset))
- **Vektörleştirme:** TF-IDF with N-gram (1,2)
- **Optimizasyon:** Grid Search with Stratified 3-Fold CV
- **Dengeleme:** Undersampling yöntemi
- **Framework:** scikit-learn, Streamlit, NLTK

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 👤 Yazar

**[Adınız]**
- GitHub: [@kullaniciadi](https://github.com/kullaniciadi)
- Email: email@example.com

## 🙏 Teşekkürler

- [winvoker](https://huggingface.co/winvoker) - Türkçe duygu analizi veri seti
- [Streamlit](https://streamlit.io/) - Web framework
- [scikit-learn](https://scikit-learn.org/) - ML kütüphanesi

## 📞 İletişim

Sorularınız için GitHub Issues kullanabilirsiniz.

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
