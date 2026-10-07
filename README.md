# 🔥 Hibrit Yangın Tespit ve Doğrulama Sistemi

**YOLO + Görsel Dil Modeli (VLM) + Telegram Alarm**

Bu proje, kamera veya video görüntüsünden **gerçek zamanlı yangın / duman tespiti** yapar. Tespiti tek bir modele bırakmaz; **hibrit** bir yapı kullanır:

1. **YOLO** her kareyi çok hızlı tarar ve "burada yangın olabilir" der (hızlı ama bazen yanılabilir).
2. Şüpheli bölge kırpılıp **yerel bir VLM'e** (Ollama üzerinde çalışan Qwen2.5-VL) gönderilir. VLM görüntüye bakıp "gerçekten yangın mı, yoksa sigara / araba farı / buhar mı?" diye **teyit eder** (yavaş ama akıllı).
3. Yangın teyit edilirse **Telegram'a fotoğraf + durum raporu + 112 butonu** gönderilir, ardından **10 saniyelik olay videosu** (5 sn öncesi + 5 sn sonrası) iletilir.

> ✅ VLM tamamen **yerel (offline)** çalışır. Görüntüler internete çıkmaz, API ücreti yoktur. Sadece Telegram bildirimi için internet gerekir.

---

## 📑 İçindekiler

1. [Sistem Nasıl Çalışır? (Hibrit Akış)](#-sistem-nasıl-çalışır-hibrit-akış)
2. [Gereksinimler](#-gereksinimler)
3. [Kurulum Akışı (Genel Bakış)](#-kurulum-akışı-genel-bakış)
4. [Adım Adım Kurulum](#-adım-adım-kurulum)
5. [.env Dosyası Ayarları](#-env-dosyası-ayarları)
6. [Telegram Botu Oluşturma](#-telegram-botu-oluşturma)
7. [Çalıştırma](#-çalıştırma)
8. [Ayarlanabilir Parametreler](#-ayarlanabilir-parametreler)
9. [Sorun Giderme](#-sorun-giderme)
10. [Proje Yapısı](#-proje-yapısı)

---

## 🧠 Sistem Nasıl Çalışır? (Hibrit Akış)

Sistemin mantığı basit: **hızlı model şüphelenir, akıllı model karar verir.** Böylece hem gerçek zamanlı akıcılık hem de düşük yanlış alarm oranı elde edilir.

```mermaid
flowchart TD
    A["🎥 Video / Kamera Karesi"] --> B["⚡ YOLO Tespiti<br/>(best1.pt - her karede çalışır)"]
    B --> C{"Ateş veya Duman<br/>tespit edildi mi?"}
    C -- "Hayır" --> A
    C -- "Evet" --> D{"VLM meşgul değil<br/>ve yangın kilidi yok<br/>ve 2 sn bekleme geçti mi?"}
    D -- "Hayır" --> A
    D -- "Evet" --> E["✂️ Akıllı Kırpma<br/>(çevre bağlamı + padding)"]
    E --> F["🤖 Yerel VLM - Ollama<br/>Qwen2.5-VL (arka plan thread)"]
    F --> G{"VLM Kararı"}
    G -- "ONAY: HAYIR<br/>(yanlış alarm)" --> A
    G -- "ONAY: EVET<br/>(gerçek yangın)" --> H["🚨 ALARM"]
    H --> I["📸 Telegram: Fotoğraf + Durum +<br/>Müdahale Önerisi + 112 Butonu"]
    H --> J["🎞️ 10 sn Olay Klibi Kaydı<br/>(5 sn öncesi + 5 sn sonrası)"]
    J --> K["📤 Telegram: Video Klip"]
    I --> A
    K --> A

    style B fill:#fff3cd,stroke:#ffc107,color:#000
    style F fill:#d1ecf1,stroke:#17a2b8,color:#000
    style H fill:#f8d7da,stroke:#dc3545,color:#000
```

### Adım adım ne oluyor?

| Adım | Ne yapılıyor? | Kim yapıyor? |
|------|---------------|--------------|
| 1 | Video karesi okunur, sağ üstte ve altta HUD paneli çizilir | OpenCV |
| 2 | Karede `fire` (ateş) veya `smoke` (duman) aranır. `other` sınıfı yok sayılır | YOLO (`best1.pt`) |
| 3 | Tespit varsa, tespit edilen alan **çevresiyle birlikte** kırpılır (VLM ortamı anlasın diye) | Python / NumPy |
| 4 | Kırpılan görsel yerel VLM'e gönderilir. Bu işlem **arka planda** yapılır, video donmaz | Ollama + Qwen2.5-VL |
| 5 | VLM, Türkçe olarak `ONAY: EVET/HAYIR` formatında cevap verir | Qwen2.5-VL |
| 6 | Cevap EVET ise alarm başlar: yangın "kilitlenir", klip kaydı başlar | Ana pipeline |
| 7 | Telegram'a fotoğraf, VLM raporu, müdahale önerisi ve butonlar gönderilir | `telegram_notifier.py` |
| 8 | 5 saniye sonra video klip tamamlanır ve Telegram'a yüklenir | `telegram_notifier.py` |
| 9 | Ortamda 3 saniye boyunca tehdit görünmezse sistem sıfırlanır | Ana pipeline |


### Neden hibrit?

- **Sadece YOLO:** Çok hızlı ama araba farı, sigara dumanı, buhar gibi şeylere yanlış alarm verebilir.
- **Sadece VLM:** Çok akıllı ama her kareyi işleyemeyecek kadar yavaş.
- **Hibrit:** YOLO "filtre", VLM "hakem" olur. Sadece şüpheli anlarda VLM çağrılır.

---

## 📋 Gereksinimler

| Gereksinim | Önerilen |
|------------|----------|
| İşletim Sistemi | Windows 10/11, Linux veya macOS |
| Python | **3.10 – 3.12** |
| RAM | En az 8 GB (16 GB önerilir) |
| Disk | ~10 GB boş alan (modeller + paketler) |
| GPU | Zorunlu değil. NVIDIA GPU varsa VLM yanıtı çok daha hızlı olur |
| İnternet | Kurulum sırasında + Telegram bildirimleri için |
| Telegram hesabı | Alarm bildirimlerini almak için |

> 💡 GPU yoksa sistem yine çalışır; ancak VLM yanıtı CPU'da 10–60 saniye sürebilir. Bu durumda `LOCAL_VLM_TIMEOUT` değerini artırmanız gerekebilir.

---

## 🗺️ Kurulum Akışı (Genel Bakış)

```mermaid
flowchart LR
    S1["1️⃣ Python<br/>Kur"] --> S2["2️⃣ Proje<br/>dosyalarını al"]
    S2 --> S3["3️⃣ Sanal ortam<br/>oluştur"]
    S3 --> S4["4️⃣ Python<br/>paketlerini kur"]
    S4 --> S5["5️⃣ Ollama'yı<br/>kur"]
    S5 --> S6["6️⃣ Qwen2.5-VL<br/>modelini indir"]
    S6 --> S7["7️⃣ Telegram<br/>botu oluştur"]
    S7 --> S8["8️⃣ .env<br/>dosyasını doldur"]
    S8 --> S9["9️⃣ Test et<br/>ve çalıştır"]

    style S1 fill:#e8f4fd,stroke:#0d6efd,color:#000
    style S5 fill:#e8f4fd,stroke:#0d6efd,color:#000
    style S7 fill:#e8f4fd,stroke:#0d6efd,color:#000
    style S9 fill:#d4edda,stroke:#28a745,color:#000
```

---

## 🛠️ Adım Adım Kurulum

### Adım 1 — Python'u Kurun

1. [python.org/downloads](https://www.python.org/downloads/) adresinden **Python 3.10, 3.11 veya 3.12** sürümünü indirin.
2. **Windows'ta kurulum ekranında mutlaka `Add Python to PATH` kutucuğunu işaretleyin.**
3. Kurulumu doğrulayın (Terminal / PowerShell / CMD):

```bash
python --version
pip --version
```

> Linux/macOS'ta komut `python3` ve `pip3` olabilir.

### Adım 2 — Proje Dosyalarını Hazırlayın

Proje klasörünüzde şu dosyalar bulunmalıdır:

```
yangin-tespit/
├── pipline_local.py        # Ana program
├── telegram_notifier.py    # Telegram bildirim modülü
├── test_telegram.py        # Telegram test aracı
├── best1.pt                # YOLO model ağırlıkları (yangın/duman)
├── video2.mp4              # Test videosu (veya web kamerası kullanılacak)
└── .env                    # Ayar dosyası (Adım 8'de oluşturulacak)
```

Terminalde proje klasörüne girin:

```bash
cd yangin-tespit
```

### Adım 3 — Sanal Ortam (Virtual Environment) Oluşturun

Sanal ortam, projenin paketlerini bilgisayarınızdaki diğer Python işlerinden ayrı tutar. **Önerilir.**

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Aktifleşince terminalin başında `(venv)` yazısı görünür.

### Adım 4 — Python Paketlerini Kurun

Önce pip'i güncelleyin, ardından gerekli paketleri kurun:

```bash
python -m pip install --upgrade pip
pip install ultralytics opencv-python numpy requests python-dotenv
```

| Paket | Ne işe yarıyor? |
|-------|-----------------|
| `ultralytics` | YOLO modelini çalıştırır (PyTorch'u da otomatik kurar, bu yüzden indirme biraz uzun sürebilir) |
| `opencv-python` | Video okuma, görüntü işleme, ekrana çizim, klip kaydı |
| `numpy` | Görüntü ve kutu koordinatı hesapları |
| `requests` | Ollama'ya ve Telegram API'ye HTTP istekleri |
| `python-dotenv` | `.env` dosyasındaki ayarları okur |

> 💡 **İpucu:** Aşağıdaki içerikle bir `requirements.txt` dosyası oluşturursanız tek komutla kurabilirsiniz: `pip install -r requirements.txt`
>
> ```text
> ultralytics
> opencv-python
> numpy
> requests
> python-dotenv
> ```

Kurulumu doğrulayın:

```bash
python -c "import ultralytics, cv2, numpy, requests, dotenv; print('Tum paketler hazir!')"
```

### Adım 5 — Ollama'yı Kurun

Ollama, yapay zekâ modellerini bilgisayarınızda yerel olarak çalıştıran bir programdır.

1. [ollama.com/download](https://ollama.com/download) adresinden işletim sisteminize uygun sürümü indirip kurun.
2. **Linux** için tek satırlık kurulum:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
3. Kurulumu doğrulayın:
   ```bash
   ollama --version
   ```

> Windows ve macOS'ta Ollama kurulumdan sonra arka planda otomatik çalışır (sistem tepsisinde simgesi görünür). Varsayılan adresi: `http://localhost:11434`

### Adım 6 — Qwen2.5-VL Modelini İndirin

Aşağıdaki komut modeli indirir (yaklaşık 3–4 GB, bir kez yapılır):

```bash
ollama pull qwen2.5vl:3b
```

Modelin indiğini kontrol edin:

```bash
ollama list
```

Listede `qwen2.5vl:3b` görünmelidir.

**İsteğe bağlı hızlı test:**
```bash
ollama run qwen2.5vl:3b "Merhaba"
```
Cevap geliyorsa model hazırdır. Çıkmak için `/bye` yazın.

> 💡 Daha güçlü bir bilgisayarınız varsa `qwen2.5vl:7b` modelini de deneyebilirsiniz (daha doğru ama daha yavaş/ağır). Bu durumda `.env` içindeki `LOCAL_MODEL_NAME` değerini de güncelleyin.

### Adım 7 — Telegram Botunu Oluşturun

Detaylar aşağıdaki [Telegram Botu Oluşturma](#-telegram-botu-oluşturma) bölümündedir. Sonuç olarak elinizde iki bilgi olmalı: **Bot Token** ve **Chat ID**.

### Adım 8 — `.env` Dosyasını Oluşturun

Proje klasöründe `.env` adında bir dosya oluşturun (başında nokta olduğuna dikkat edin, uzantısı yok) ve içine aşağıdakini yazın:

```dotenv
# Telegram Bot Entegrasyon Ayarları
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
TELEGRAM_CHAT_ID=123456789
CAMERA_LOCATION=Kamera 01 - Fabrika / Depo Alanı
CAMERA_MAPS_URL=https://maps.google.com/?q=41.0082,28.9784
LOCAL_MODEL_NAME=qwen2.5vl:3b
LOCAL_VLM_TIMEOUT=90
```

Her satırın ne anlama geldiği bir sonraki bölümde anlatılıyor.

### Adım 9 — Test Edin ve Çalıştırın

Önce Telegram bağlantısını test edin, sonra ana programı çalıştırın. Detaylar: [Çalıştırma](#-çalıştırma).

---

## ⚙️ .env Dosyası Ayarları

`.env` dosyası, programın **kişiye ve ortama özel** ayarlarını tutar. Bu dosyayı kendi bilgisayarınızda **siz** oluşturursunuz; Git'e / başkalarına göndermeyin (içinde bot şifreniz var).

| Değişken | Zorunlu mu? | Açıklama | Örnek |
|----------|:-----------:|----------|-------|
| `TELEGRAM_BOT_TOKEN` | ✅ Evet | BotFather'dan aldığınız bot anahtarı. Botunuzun "şifresi" gibidir, kimseyle paylaşmayın. | `123456789:ABCdef...` |
| `TELEGRAM_CHAT_ID` | ✅ Evet | Alarmların gönderileceği sohbetin numarası (kişi veya grup). | `123456789` veya grup için `-1001234567890` |
| `CAMERA_LOCATION` | ❌ Hayır | Alarm mesajında "Mekan / Kamera" olarak görünen yazı. Kameranın nerede olduğunu anlatın. | `Kamera 01 - Fabrika / Depo Alanı` |
| `CAMERA_MAPS_URL` | ❌ Hayır | Telegram'daki "📍 Olay Yeri (Harita)" butonunun açacağı adres. `q=` sonrasına enlem,boylam yazın. | `https://maps.google.com/?q=41.0082,28.9784` |
| `LOCAL_MODEL_NAME` | ❌ Hayır | Ollama'da indirdiğiniz VLM modelinin adı. `ollama list` çıktısıyla **birebir aynı** olmalı. | `qwen2.5vl:3b` |
| `LOCAL_VLM_TIMEOUT` | ❌ Hayır | VLM'in yanıt vermesi için beklenecek maksimum süre (saniye). Yavaş bilgisayarda artırın. | `90` |

### Opsiyonel (ileri düzey) değişken

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `LOCAL_VLM_URL` | `http://localhost:11434/v1/chat/completions` | Ollama farklı bir bilgisayarda/portta çalışıyorsa buradan değiştirin. Normalde dokunmanız gerekmez. |

### 📌 Dikkat edilmesi gerekenler

- Değerlerin etrafına **tırnak koymayın** ve `=` işaretinin etrafına **boşluk bırakmamaya** özen gösterin: `LOCAL_MODEL_NAME=qwen2.5vl:3b` ✅
- `TELEGRAM_BOT_TOKEN` ve `TELEGRAM_CHAT_ID` boşsa sistem çalışır ama **Telegram bildirimi gönderilmez** (konsolda uyarı görürsünüz).
- Konum koordinatını bulmak için: Google Haritalar'da yeri sağ tıklayın, çıkan koordinatı (`41.0082, 28.9784`) kopyalayıp URL'ye yapıştırın.
- `.env` dosyasını başkalarına göndereceğiniz ZIP'e koymayın; token'ınız çalınırsa botunuz kötüye kullanılabilir.

---

## 🤖 Telegram Botu Oluşturma

### 1) Bot Token'ı Alma

1. Telegram'da **[@BotFather](https://t.me/BotFather)** hesabını arayın ve açın.
2. `/newbot` yazın.
3. Bota bir **isim** verin (örn. `Yangin Alarm Botu`).
4. Bota benzersiz bir **kullanıcı adı** verin; sonu `bot` ile bitmeli (örn. `fabrika_yangin_alarm_bot`).
5. BotFather size şuna benzer bir token verecek:
   ```
   123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
   ```
   Bunu `.env` içindeki `TELEGRAM_BOT_TOKEN=` satırına yapıştırın.

### 2) Chat ID'yi Bulma

**Yöntem A — Kendi hesabınıza bildirim almak için:**

1. Telegram'da oluşturduğunuz **botu bulun** ve **`/start`** yazın (bu adımı atlamayın, yoksa bot size mesaj gönderemez).
2. Tarayıcıda şu adresi açın (kendi token'ınızı yazın):
   ```
   https://api.telegram.org/bot<TOKEN_BURAYA>/getUpdates
   ```
3. Çıkan yanıtta `"chat":{"id":123456789,...}` kısmındaki sayı **Chat ID**'nizdir.
4. Bunu `.env` içindeki `TELEGRAM_CHAT_ID=` satırına yazın.

**Yöntem B — Bir gruba bildirim göndermek için:**

1. Botu grubunuza ekleyin.
2. Grupta herhangi bir mesaj yazın.
3. Yukarıdaki `getUpdates` adresini açın.
4. Grup ID'si genelde **eksi (-) ile başlar**, örn. `-1001234567890`. Eksi işaretiyle birlikte yazın.

> Yanıt boş (`"result":[]`) geliyorsa bota veya gruba yeni bir mesaj yazıp sayfayı yenileyin.

---

## ▶️ Çalıştırma

Her seferinde şu sırayı izleyin:

```mermaid
flowchart TD
    R1["1. Ollama çalışıyor mu?<br/>(ollama list)"] --> R2["2. Sanal ortamı aç<br/>(venv)"]
    R2 --> R3["3. Telegram testi<br/>python test_telegram.py"]
    R3 --> R4{"Telegram'a mesaj<br/>geldi mi?"}
    R4 -- "Hayır" --> R5["Token / Chat ID'yi<br/>kontrol et"]
    R5 --> R3
    R4 -- "Evet" --> R6["4. Ana programı başlat<br/>python pipline_local.py"]
    R6 --> R7["✅ Sistem çalışıyor<br/>Çıkmak için 'q' tuşu"]

    style R6 fill:#d4edda,stroke:#28a745,color:#000
    style R7 fill:#d4edda,stroke:#28a745,color:#000
```

### 1) Ollama'nın çalıştığından emin olun

```bash
ollama list
```
Hata alırsanız Ollama'yı başlatın (Windows/macOS'ta uygulamayı açın, Linux'ta `ollama serve` komutunu ayrı bir terminalde çalıştırın).

### 2) Sanal ortamı etkinleştirin

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3) Telegram bağlantısını test edin

```bash
python test_telegram.py
```

Telegram'ınıza örnek bir alarm fotoğrafı, rapor ve butonlar gelmelidir. Gelmediyse [Sorun Giderme](#-sorun-giderme) bölümüne bakın.

### 4) Ana sistemi başlatın

```bash
python pipline_local.py
```

- Bir pencere açılır ve video üzerinde YOLO tespitleri, sağ üstte VLM önizleme paneli, altta VLM analizi görünür.
- **Çıkmak için pencere seçiliyken `q` tuşuna basın.**
- Test videosu bittiğinde otomatik olarak başa sarar.

### Video kaynağını değiştirme

`pipline_local.py` dosyasında şu satırı düzenleyin:

```python
VIDEO_SOURCE = "video2.mp4"    # Video dosyası için dosya adı
VIDEO_SOURCE = 0               # Bilgisayardaki web kamerası için
VIDEO_SOURCE = "rtsp://kullanici:sifre@192.168.1.10:554/stream"   # IP kamera için
```

### Ekranda görecekleriniz

| Durum yazısı | Anlamı |
|--------------|--------|
| 🟢 `DURUM: Taraniyor` | Sistem normal, tehdit yok |
| 🟡 `DURUM: Yerel VLM Dogruluyor...` | YOLO şüphelendi, VLM inceliyor |
| 🔴 `DURUM: YANGIN TEYITLI (Kilitli)` | Yangın doğrulandı, alarm gönderildi |

---

## 🎛️ Ayarlanabilir Parametreler

`pipline_local.py` dosyasının başında bulunan değerler, sistemin hassasiyetini ayarlar:

| Parametre | Varsayılan | Ne yapar? |
|-----------|-----------|-----------|
| `CONF_THRESHOLD` | `0.40` | YOLO'nun tespit güven eşiği. Düşürürseniz daha hassas olur (daha çok yanlış alarm adayı); yükseltirseniz daha seçici olur. |
| `PADDING_RATIO` | `0.35` | Kırpılan bölgenin çevresine eklenen bağlam payı (%35). |
| `MIN_CONTEXT_SIZE` | `350` | Küçük alevlerde VLM'in odayı anlaması için kırpmanın en az kaç piksel olacağı. |
| `VLM_COOLDOWN_SECONDS` | `2.0` | İki VLM sorgusu arasındaki minimum bekleme süresi. |

---

## 🧯 Sorun Giderme

| Sorun | Olası neden / Çözüm |
|-------|---------------------|
| `Yerel VLM sunucusuna bağlanılamadı` | Ollama çalışmıyor. Uygulamayı açın veya `ollama serve` çalıştırın. |
| `model not found` (HTTP 404) | Model indirilmemiş veya `.env`'deki isim yanlış. `ollama list` ile kontrol edip `ollama pull qwen2.5vl:3b` yapın. |
| `VLM Zaman Aşımı` | Bilgisayar yavaş. `.env`'de `LOCAL_VLM_TIMEOUT=180` gibi artırın veya GPU kullanın. |
| `Telegram UYARI: ... eksik` | `.env` dosyası yok, yanlış klasörde ya da token/chat id boş. |
| Telegram'a mesaj gelmiyor (HTTP 400/401/403) | Token hatalı olabilir (401), Chat ID hatalı olabilir (400) veya bota `/start` yazılmamış / bot gruptan atılmış olabilir (403). |
| `YOLO model dosyası bulunamadı` | `best1.pt` (veya `best.pt`) dosyası `pipline_local.py` ile **aynı klasörde** olmalı. |
| `Video kaynağı açılamadı` | `VIDEO_SOURCE` dosya adı/yolu yanlış ya da kamera başka program tarafından kullanılıyor. |
| `ModuleNotFoundError` | Sanal ortam aktif değil veya paketler kurulmamış. Adım 3 ve 4'ü tekrarlayın. |
| Yangın yokken sürekli alarm | `CONF_THRESHOLD` değerini yükseltin (örn. `0.55`). |
| Gerçek yangını kaçırıyor | `CONF_THRESHOLD` değerini düşürün (örn. `0.30`). |
| Pencere/HUD çok yavaş | Daha küçük çözünürlüklü video kullanın, GPU kullanın veya daha küçük VLM seçin. |

---

## 📁 Proje Yapısı

```
yangin-tespit/
├── pipline_local.py        # Ana pipeline: YOLO + VLM doğrulama + HUD + klip kaydı
├── telegram_notifier.py    # Telegram'a fotoğraf, rapor, buton ve video gönderimi
├── test_telegram.py        # Telegram ayarlarını test eden yardımcı araç
├── best1.pt                # Eğitilmiş YOLO modeli (fire / smoke / other)
├── video2.mp4              # Örnek test videosu
├── .env                    # Kişisel ayarlar (GİZLİ - paylaşmayın)
├── requirements.txt        # (İsteğe bağlı) Python paket listesi
└── yangin_olay_*.mp4       # Alarm anında otomatik kaydedilen olay klipleri
```

---

## 🔒 Güvenlik Notları

- `.env` dosyasını **asla** herkese açık bir depoya (GitHub vb.) yüklemeyin. Git kullanıyorsanız `.gitignore` dosyasına `.env` ekleyin.
- Token yanlışlıkla paylaşılırsa BotFather'da `/revoke` ile yenisini oluşturun.
- Bu sistem yardımcı bir **erken uyarı** aracıdır; resmi yangın algılama ve söndürme sistemlerinin yerine geçmez.

---

## 🙌 Kullanılan Teknolojiler

- [Ultralytics YOLO](https://docs.ultralytics.com/) — gerçek zamanlı nesne tespiti
- [Ollama](https://ollama.com/) + [Qwen2.5-VL](https://ollama.com/library/qwen2.5vl) — yerel görsel dil modeli
- [OpenCV](https://opencv.org/) — görüntü işleme
- [Telegram Bot API](https://core.telegram.org/bots/api) — bildirim altyapısı
