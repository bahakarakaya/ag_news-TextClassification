> 📄 **Projenin tam değerlendirme ve analiz raporu için: [metrik_raporu.md](./metrik_raporu.md)**

# 📄 LLM Metin Sınıflandırma Raporu

Bu proje, AG News veri seti üzerinde farklı prompt stratejileriyle haber sınıflandırması yaparak bu stratejilerin doğruluk ve verimlilik açısından karşılaştırılmasını amaçlamaktadır.  

## 🎯 Amaç

- Farklı istem yapılarının (zero-shot, few-shot, CoT vb.) sınıflandırma performansını karşılaştırmak  
- LLM tarafından üretilen çıktılarda doğruluk ve token verimliliğini ölçmek  
- Stratejilere göre performans / maliyet dengesi önerileri sunmak  

## 🧠 Kullanılan LLM

- **Model:**`gemma-3-12b-it`
- **API:** `google-generativeai` Python SDK
- **Çıktılar:** LLM tarafından doğrudan üretilmiş olup manuel müdahale yoktur.

## 🧪 Test Detayları

- **Veri seti:** AG News (HuggingFace üzerinden)
- **Test örneği:** 100 adet rastgele haber (seed=42)
- **Metot:** Macro-average ile accuracy, precision, recall, F1 hesaplandı

## 📊 Örnek Çıktılar

Tüm test sonuçları, confusion matrix görselleri ve performans metrikleri `stats/` klasöründe yer almaktadır.

| Strateji                  | Doğruluk | F1 Score | Toplam Token | Doğruluk/Token |
|---------------------------|----------|----------|--------------|----------------|
| Rol + Few-Shot + CoT      | %93      | 0.932    | 486.56       | 0.00191        |
| Zero-Shot + Rol           | %89      | 0.889    | 114.98       | 0.00774        |
| ...                       | ...      | ...      | ...          | ...            |


## 🛠 Kurulum

```bash
# Ortam oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

# Gereksinimleri yükle
pip install -r requirements.txt

## 🔐 Ortam Değişkeni
Gemini API anahtarınızı .env dosyası olarak tanımlayın:

```
# .env
GEMINI_API_KEY=your_api_key_here
```

## 🚀 Çalıştırma
```
python main.py
```

Kod, 100 haber üzerinde inference çalıştırır, tahminleri alır, token sayılarını hesaplar ve metrikleri stats/ klasörüne kayıt eder.


### 📌 Notlar
* metrik_raporu.md dosyası, tüm istemler, tablolar, değerlendirme ve sonuç önerilerini içermektedir.
* İstem örnekleri ve LLM çıktıları da raporda yer almaktadır.