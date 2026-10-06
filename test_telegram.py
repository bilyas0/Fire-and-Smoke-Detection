import os
import cv2
import numpy as np
from dotenv import load_dotenv
from telegram_notifier import send_fire_alert_async, is_telegram_configured

load_dotenv()

print("=" * 50)
print("       TELEGRAM BOT TEST ARACI")
print("=" * 50)

if not is_telegram_configured():
    print("\n[!] DİKKAT: .env dosyasında TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID henüz ayarlanmamış!")
    print("    Lütfen .env dosyanızı açıp bilgilerinizi girin.\n")
    print("    TELEGRAM_BOT_TOKEN=123456789:ABC...XYZ")
    print("    TELEGRAM_CHAT_ID=123456789")
    print("=" * 50)
    exit(1)

print("[*] Test görseli oluşturuluyor...")
test_img = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.putText(test_img, "YANGIN TEST ALARMI", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
_, buffer = cv2.imencode(".jpg", test_img)

test_verdict = (
    "ONAY: EVET - Odak alanında ahşap zemin üzerinde aktif alev ve yoğun duman yayılımı görülüyor.\n"
    "MÜDAHALE: Ortam ivedilikle tahliye edilmeli, 112 Acil İtfaiye aranmalı ve KKT yangın söndürme tüpüyle güvenli mesafeden ilk müdahale yapılmalıdır."
)

print("[*] Telegram'a test alarmı (VLM kırpması + 2 satırlı durum ve müdahale raporu + butonlar) gönderiliyor...")
send_fire_alert_async(
    image_bytes=buffer.tobytes(),
    vlm_verdict=test_verdict,
    model_name="Gemini 3.5 Flash (Test)",
    confidence=0.92
)

# Klasörde mevcut olan örnek bir olay klibini de test amacıyla gönder
from telegram_notifier import send_fire_clip_async
import glob

clips = sorted(glob.glob("yangin_olay_*.mp4"))
if clips:
    sample_clip = clips[-1]
    print(f"[*] Test videosu gönderiliyor: {sample_clip}...")
    send_fire_clip_async(sample_clip)
else:
    print("[i] Test videosu için henüz kaydedilmiş klip bulunamadı (ana sistem çalışınca otomatik kaydedilir).")

print("[+] İstekler arka planda Telegram sunucusuna iletildi!")
print("[i] Telegram sohbetinizi kontrol edin. Çıkmak için 5 saniye bekleniyor...")
import time
time.sleep(5)

