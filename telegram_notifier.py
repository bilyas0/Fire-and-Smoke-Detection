import os
import time
import threading
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CAMERA_LOCATION = os.getenv("CAMERA_LOCATION", "Kamera 01 - Fabrika / Depo Alanı")
MAPS_URL = os.getenv("CAMERA_MAPS_URL", "https://maps.google.com/?q=41.0082,28.9784")


def is_telegram_configured() -> bool:
    """Telegram yapılandırmasının tam olup olmadığını kontrol eder."""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and "BURAYA" not in TELEGRAM_BOT_TOKEN)


def _send_telegram_photo(image_bytes: bytes, caption: str, inline_keyboard: list = None):
    """Telegram üzerinden fotoğraf ve bilgi mesajı gönderir."""
    if not is_telegram_configured():
        print("[Telegram] UYARI: TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID eksik. Bildirim gönderilmedi.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "caption": caption,
        "parse_mode": "Markdown",
    }
    if inline_keyboard:
        import json
        data["reply_markup"] = json.dumps({"inline_keyboard": inline_keyboard})

    files = {
        "photo": ("fire_detection.jpg", image_bytes, "image/jpeg")
    }

    try:
        resp = requests.post(url, data=data, files=files, timeout=20)
        if resp.status_code == 200:
            print("[Telegram] [+] Yangin alarm gorseli ve acil durum raporu basariyla iletildi!")
        else:
            print(f"[Telegram] Hata (Fotograf): {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"[Telegram] Baglanti Hatasi (Fotograf): {e}")



def _send_telegram_video(video_path: str, caption: str):
    """Telegram üzerinden 10 saniyelik olay klibini gönderir."""
    if not is_telegram_configured():
        return

    if not os.path.exists(video_path):
        print(f"[Telegram] Hata: Video dosyasi bulunamadi: {video_path}")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "caption": caption,
        "parse_mode": "Markdown",
        "supports_streaming": True,
    }

    try:
        print(f"[Telegram] [*] 10 saniyelik olay klibi yukleniyor ({os.path.basename(video_path)})...")
        with open(video_path, "rb") as vf:
            files = {"video": (os.path.basename(video_path), vf, "video/mp4")}
            resp = requests.post(url, data=data, files=files, timeout=60)
            if resp.status_code == 200:
                print("[Telegram] [+] 10 saniyelik olay klibi Telegram'a basariyla iletildi!")
            else:
                print(f"[Telegram] Hata (Video): {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"[Telegram] Baglanti Hatasi (Video): {e}")



def send_fire_alert_async(image_bytes: bytes, vlm_verdict: str, model_name: str, confidence: float = 0.0):
    """
    Yangın teyit edildiği anda VLM'de doğrulanan kırpılmış hedef görselini,
    zamanı, mekanı, VLM durum analizini, bir alt satırda müdahale önerisini ve
    aksiyon butonlarını arka planda asenkron olarak gönderir.
    """
    now_str = datetime.now().strftime("%d.%m.%Y - %H:%M:%S")

    # Telegram butonları (Fotoğrafın hemen altında doğrudan arama yapan buton)
    inline_keyboard = [
        [
            {"text": "🚨 112 İtfaiyeyi Ara", "url": "https://tinyurl.com/3fx8dda"},
            {"text": "📍 Olay Yeri (Harita)", "url": MAPS_URL}
        ]
    ]

    # Müdahale önerisini bir alt satıra ve belirgin bölüme ayır
    clean_verdict = vlm_verdict.strip()
    status_text = clean_verdict
    advice_text = "Derhal alan tahliyesini başlatın ve 112 Acil Çağrı Merkezine haber verin."

    if "MÜDAHALE:" in clean_verdict:
        parts = clean_verdict.split("MÜDAHALE:")
        status_text = parts[0].strip()
        advice_text = parts[1].strip()
    elif "Müdahale Önerisi:" in clean_verdict:
        parts = clean_verdict.split("Müdahale Önerisi:")
        status_text = parts[0].strip()
        advice_text = parts[1].strip()
    elif "\n" in clean_verdict:
        lines = [line.strip() for line in clean_verdict.split("\n") if line.strip()]
        if len(lines) >= 2:
            status_text = lines[0]
            advice_text = " ".join(lines[1:])

    caption = (
        f"🚨 *ACİL YANGIN TESPİT ALARMI* 🚨\n\n"
        f"📍 *Mekan / Kamera:* `{CAMERA_LOCATION}`\n"
        f"🕒 *Zaman:* `{now_str}`\n"
        f"🎯 *YOLO Skoru:* `%{int(confidence * 100)}` | 🤖 *VLM:* `{model_name}`\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🔍 *Durum Tespiti:*\n{status_text}\n\n"
        f"⚠️ *Müdahale Önerisi:*\n{advice_text}\n"
        f"━━━━━━━━━━━━━━━━━━━\n\n"
        f"📞 *Acil Arama:* 112\n"
        f"👉 *Aşağıdaki [🚨 112 İtfaiyeyi Ara] butonuna basarak doğrudan arama yapabilirsiniz.*\n"
        f"⏳ *Ekranın 10 saniyelik tam video kaydı yükleniyor...*"
    )

    thread = threading.Thread(
        target=_send_telegram_photo,
        args=(image_bytes, caption, inline_keyboard),
        daemon=True
    )
    thread.start()




def send_fire_clip_async(video_path: str):
    """
    Ekranın 10 saniyelik tam video kaydı bittiğinde arka planda Telegram'a gönderir.
    """
    now_str = datetime.now().strftime("%H:%M:%S")
    caption = (
        f"🎥 *Ekranın 10 Saniyelik Tam Olay Video Kaydı*\n"
        f"📍 *Mekan / Bölge:* `{CAMERA_LOCATION}`\n"
        f"⏱ *Kayıt Sonu:* `{now_str}`\n"
        f"ℹ️ *Detay:* 5 sn olay öncesi + 5 sn tespit anı (Tüm ekran HUD & kutular dahil)."
    )

    thread = threading.Thread(
        target=_send_telegram_video,
        args=(video_path, caption),
        daemon=True
    )
    thread.start()

