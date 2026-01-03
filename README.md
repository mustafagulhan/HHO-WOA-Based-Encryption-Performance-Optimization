# HHO-WOA Based Encryption Performance Optimization

Bu proje, **Harris Hawks Optimization (HHO)** ve **Whale Optimization Algorithm (WOA)** algoritmalarının hibrit kullanımıyla, kriptografik algoritmaların **performans–güvenlik dengesini** optimize eden yapay zeka tabanlı bir sistemdir.

Sistem; **AES, ChaCha20, 3DES, Blowfish, CAST5** gibi simetrik şifreleme algoritmalarını ve **CBC, GCM, CTR** gibi çalışma modlarını analiz eder. Çalıştığı donanımın **anlık sistem durumuna** göre en uygun konfigürasyonu belirler:

- Şifreleme Algoritması  
- Çalışma Modu  
- Anahtar Uzunluğu  
- Buffer Boyutu  

---

## Özellikler

- **Hibrit Meta-Sezgisel Yaklaşım**  
  HHO algoritmasının güçlü **keşif (exploration)** yeteneği ile WOA algoritmasının etkili **sömürü (exploitation)** mekanizması birleştirilmiştir.

- **Karşılaştırmalı Optimizasyon**  
  Aynı maliyet fonksiyonunda **HHO-WOA**, **Differential Evolution (DE)** ve **Particle Swarm Optimization (PSO)** paralel koşturulur; yakınsama grafiği tek çıktıda karşılaştırılır.

- **Robust (Dayanıklı) Optimizasyon**  
  Min-Max normalizasyonu yerine **Median / IQR** tabanlı *Robust Scaler* kullanılmıştır. Bu sayede işletim sistemi kaynaklı anlık takılmalar ve ölçüm sapmaları (outlier) analiz sonucunu bozmaz.

- **Çok Amaçlı (Multi-Objective) Optimizasyon**  
  Aşağıdaki kriterleri içeren ağırlıklı bir maliyet fonksiyonu minimize edilir:
  - Performans: Süre, CPU kullanımı, RAM tüketimi  
  - Güvenlik: NIST kriptografik standartlarına uygunluk  

---

## Dinamik Adaptasyon ve Sonuç Değişkenliği

Bu sistem **deterministik değil**, **stokastik ve dinamik** bir yapıdadır. Bu nedenle farklı çalıştırmalarda farklı sonuçlar elde edilebilir (ör. bir çalıştırmada `AES-CTR`, diğerinde `ChaCha20`).

Bu durum bir hata değil, sistemin **adaptasyon kabiliyetinin doğal bir sonucudur**.

### Bunun Temel Nedenleri

1. **Dinamik Sistem Kaynakları**  
   CPU yükü, önbellek (cache) durumu ve RAM kullanımı milisaniyeler içinde değişir. Optimizasyon algoritması, **o anki donanım koşullarına en uygun** çözümü seçer.

2. **Meta-Sezgisel Algoritmaların Doğası**  
   Çözüm uzayı rastgelelik içeren akıllı bir arama süreciyle keşfedilir. Performansları birbirine çok yakın olan güçlü adaylar (global optima) arasında geçişler yaşanması matematiksel olarak beklenen bir durumdur.

---

## Kurulum (Installation)

Projenin çalışabilmesi için **Python 3.x** ve gerekli bağımlılıkların yüklü olması gerekmektedir.

1. Projeyi bilgisayarınıza indirin.
2. Terminal üzerinden proje dizinine gidin.
3. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

---

## Kullanım (Usage)

Optimizasyon süreci, sistem kaynaklarını gerçek zamanlı ölçerek başlatılır ve her çalıştırmada farklı sonuçlar üretebilir.

### Optimizasyonu Başlatma

Aşağıdaki komut, HHO-WOA, DE ve PSO’yu aynı parametre uzayı ve maliyet fonksiyonuyla karşılaştırmalı olarak çalıştırır:

```bash
python optimizer.py
```

Bu işlem sırasında sistem:

- CPU ve RAM kullanımını anlık olarak ölçer  
- Seçili kriptografik algoritmalar ve modlar üzerinde testler yapar  
- Çok amaçlı maliyet fonksiyonunu minimize eden en uygun konfigürasyonu belirler  
- `robust_result.png` içinde üç algoritmanın yakınsama eğrilerini üretir  
- Konsolda her algoritma için en iyi konfigürasyon (algo/mod/key/buffer) ile yeni sürekli parametreler (`data_size_mb`, `repeats`) raporlanır  

---

## Tekrar Çalıştırma ve Değerlendirme

Sistem stokastik olduğu için:

- Aynı komut tekrar çalıştırıldığında farklı ama **yakın-optimal** sonuçlar elde edilebilir.
- Bu durum sistemin kararsızlığı değil, **adaptif optimizasyon yeteneğinin** doğrudan sonucudur.

Gerçekçi değerlendirme için:
- Birden fazla çalıştırma yapılması  
- Sonuçların ortalama veya median değerler üzerinden karşılaştırılması önerilir.

---

## Parametre Özelleştirme

Optimizasyon parametreleri `optimizer.py` dosyası içerisinden değiştirilebilir:

- Popülasyon büyüklüğü  
- Maksimum iterasyon sayısı  
- Performans / güvenlik ağırlıkları  
- Sürekli parametreler:  
  - `data_size_mb` (1–8 MB arası dilimlenmiş veri)  
  - `repeats` (1–10 arası benchmark tekrar sayısı)  

Bu sayede sistem:
- Performans odaklı  
- Güvenlik öncelikli  
- Dengeli (balanced) senaryolara göre yapılandırılabilir.

---

## Not

Bu proje, kriptografik algoritmalar arasında **tek başına en hızlı veya en güvenli olanı** seçmeyi değil;  
**çalıştığı sistem, anlık yük ve güvenlik gereksinimleri bağlamında en dengeli çözümü** bulmayı hedefler.
