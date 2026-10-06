import os
import sys
import time
import math
import base64
import threading
from collections import deque
import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from ultralytics import YOLO
from telegram_notifier import send_fire_alert_async, send_fire_clip_async

load_dotenv()

# Yerel VLM Yapılandırması 
# Ollama varsayılan olarak http://localhost:11434/v1/chat/completions adresinde OpenAI uyumlu çalışır
LOCAL_VLM_URL = os.getenv("LOCAL_VLM_URL", "http://localhost:11434/v1/chat/completions").strip().strip('"').strip("'")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen2.5vl:3b").strip().strip('"').strip("'")
LOCAL_VLM_TIMEOUT = int(os.getenv("LOCAL_VLM_TIMEOUT", "90"))

print("=" * 65)
print(f"[*] YEREL (LOCAL) YANGIN DOĞRULAMA PIPELINE BAŞLATILIYOR")
print(f"[*] Yerel VLM Modeli: {LOCAL_MODEL_NAME}")
print(f"[*] VLM Sunucu Adresi: {LOCAL_VLM_URL}")
print(f"[*] VLM Zaman Aşımı: {LOCAL_VLM_TIMEOUT} saniye")
print("=" * 65)

# Model ve Dosya Yapılandırması
YOLO_MODEL_PATH = "best1.pt" if os.path.exists("best1.pt") else "best.pt"
VIDEO_SOURCE = "video2.mp4"    # Video dosyanız veya web kamerası için 0
CONF_THRESHOLD = 0.40          # YOLO tespit eşiği
PADDING_RATIO = 0.35           # Kutuyu çevre bağlamı için %35 genişletme payı
MIN_CONTEXT_SIZE = 350         # Küçük hedeflerde VLM'in odayı/çevreyi anlaması için minimum piksel bağlamı (px)

# VLM İstek Limitleme 
VLM_COOLDOWN_SECONDS = 2.0     # İki yerel sorgu arasındaki minimum bekleme süresi 2 sn

# Video ve Model Başlatma
if not os.path.exists(YOLO_MODEL_PATH):
    print(f"[HATA] YOLO model dosyası bulunamadı: {YOLO_MODEL_PATH}")
    sys.exit(1)

model = YOLO(YOLO_MODEL_PATH)
# 'other' sınıfını tespit ve bounding box çiziminden hariç tut
TARGET_CLASSES = [cls_id for cls_id, name in model.names.items() if name.lower() != "other"]

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"[HATA] Video kaynağı açılamadı: '{VIDEO_SOURCE}'")
    print(f"[i] Lütfen dosya adını ve yolunu kontrol edin.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
fps = 30 if (fps <= 0 or fps != fps) else int(fps)
raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Çözünürlük Standardizasyonu:
# Düşük çözünürlüklü videolarda (örn. 144p, 360p, 480p) HUD ve panellerin devleşip
# ekranı kaplamaması için çizim tuvalini en az 1280px genişliğe ölçeklendiriyoruz.
if raw_width > 0 and raw_height > 0:
    if raw_width < 1280:
        width = 1280
        height = int(1280 * (raw_height / raw_width))
    else:
        width = raw_width
        height = raw_height
else:
    width = 1280
    height = 720

# 5 saniyelik geçmiş tamponu (Ring Buffer - Deque)
BUFFER_LEN = fps * 5
frame_buffer = deque(maxlen=BUFFER_LEN)

# Pencere Yapılandırması (Kullanıcı tarafından serbestçe yeniden boyutlandırılabilir)
WINDOW_NAME = f"YOLO + Local VLM ({LOCAL_MODEL_NAME}) Smart Verification"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, width, height)

# Durum Değişkenleri (State Machine)
is_fire_active = False        # Yangın sürüyor mu? (True ise yeni VLM isteği engellenir)
is_vlm_busy = False           # VLM arka planda şu an çalışıyor mu?
recording_post_event = False  # Doğrulama sonrası 5 sn kayıt sayacı devrede mi?
post_event_counter = 0        # Kalan kayıt karesi
clip_writer = None            # Kaydedici nesnesi
current_clip_filename = None  # Aktif kaydedilen 10 sn'lik video dosya yolu
last_vlm_verdict = "Henüz doğrulama yapılmadı"
last_vlm_time = 0.0           # VLM son yanıt süresi (saniye)
vlm_request_start_time = 0.0  # VLM istek başlama anı
last_vlm_request_time = 0.0   # Son VLM isteğinin atıldığı zaman

# Sabit VLM Önizleme Verileri
active_vlm_crop = None        # VLM'e gönderilmiş olan sabit kırpma karesi
active_vlm_conf = 0.0         # VLM'e gönderilen hedefin güven skoru
last_active_box = None        # Video üzerinde çizilecek alarm odak kutusu


# ================= YARDIMCI FONKSİYONLAR =================

def tr_to_ascii(text: str) -> str:
    """OpenCV putText Türkçe karakter bozulmalarını engeller."""
    tr_map = {
        'ç': 'c', 'Ç': 'C',
        'ğ': 'g', 'Ğ': 'G',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S',
        'ü': 'u', 'Ü': 'U'
    }
    for tr_char, en_char in tr_map.items():
        text = text.replace(tr_char, en_char)
    return text


def enhance_crop_for_vlm(crop_img):
    """
    Küçük yangın kırpmalarını yerel VLM'in rahat görebilmesi için ölçeklendirir.
    Aşırı büyük kırpmaları da CPU/GPU işlem süresini şişirmemek için sınırlar.
    """
    if crop_img is None or crop_img.size == 0:
        return crop_img

    h, w = crop_img.shape[:2]
    # Küçük hedefleri büyüt (en az 280px)
    if min(h, w) < 280:
        scale = 280.0 / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        crop_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = crop_img.shape[:2]

    # Aşırı büyük hedefleri küçült (VLM görsel token sayısını ve CPU yükünü dengeler)
    if max(h, w) > 448:
        scale = 448.0 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        crop_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return crop_img


def get_padded_crop(frame, box, pad_ratio=0.35, min_context_size=350):
    """
    YOLO kutusunu çevre bağlamını alacak şekilde genişletir.
    Eğer tespit edilen nesne küçükse (örn. uzaktaki alev/kıvılcım), VLM'in
    çevredeki odayı/nesneleri anlayabilmesi için en az 'min_context_size' (örn. 350px)
    genişliğinde bağlam alanı kırpar.
    """
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(int)

    bw = x2 - x1
    bh = y2 - y1

    # Oransal genişletme payı
    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    crop_x1 = x1 - pad_w
    crop_y1 = y1 - pad_h
    crop_x2 = x2 + pad_w
    crop_y2 = y2 + pad_h

    # Küçük hedefler için minimum piksel bağlam garantisi
    target_crop_w = crop_x2 - crop_x1
    target_crop_h = crop_y2 - crop_y1

    if target_crop_w < min_context_size:
        center_x = (x1 + x2) // 2
        half_w = min_context_size // 2
        crop_x1 = center_x - half_w
        crop_x2 = center_x + half_w

    if target_crop_h < min_context_size:
        center_y = (y1 + y2) // 2
        half_h = min_context_size // 2
        crop_y1 = center_y - half_h
        crop_y2 = center_y + half_h

    # Görüntü sınırlarını aşmaması için sınırla
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(w_img, crop_x2)
    crop_y2 = min(h_img, crop_y2)

    return frame[crop_y1:crop_y2, crop_x1:crop_x2]


def get_smart_target_crop(frame, target_boxes, pad_ratio=0.35, min_context_size=350):
    """
    Birden fazla alev/duman parçası varsa hepsini kapsayan birleşik kutuyu kırpar.
    """
    if len(target_boxes) == 1:
        box = target_boxes[0].xyxy[0].cpu().numpy()
        conf = target_boxes[0].conf.item()
        crop = get_padded_crop(frame, box, pad_ratio=pad_ratio, min_context_size=min_context_size)
        return crop, conf, box

    all_boxes = np.array([b.xyxy[0].cpu().numpy() for b in target_boxes])
    u_x1 = np.min(all_boxes[:, 0])
    u_y1 = np.min(all_boxes[:, 1])
    u_x2 = np.max(all_boxes[:, 2])
    u_y2 = np.max(all_boxes[:, 3])

    best_conf = max(b.conf.item() for b in target_boxes)
    union_box = np.array([u_x1, u_y1, u_x2, u_y2])
    crop = get_padded_crop(frame, union_box, pad_ratio=pad_ratio, min_context_size=min_context_size)
    return crop, best_conf, union_box


def local_vlm_verify_worker(crop_img):
    """
    Yerel VLM'e (Ollama / vLLM / Local Server) sorgu atar.
    İnternet bağlantısı gerektirmez, tamamen offline çalışır.
    Arka planda (Thread) çalışır, ana video akışını dondurmaz.
    """
    global is_fire_active, is_vlm_busy, recording_post_event, post_event_counter
    global clip_writer, last_vlm_verdict, last_vlm_time, vlm_request_start_time
    global current_clip_filename

    req_start = time.time()
    vlm_request_start_time = req_start

    processed_crop = enhance_crop_for_vlm(crop_img)
    _, buffer = cv2.imencode(".jpg", processed_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64_img = base64.b64encode(buffer).decode("utf-8")

    prompt = (
        "Görsel güvenlik kamerası / yangın tespit sisteminden kırpılmış bir alandır (çevre bağlamı dahildir).\n\n"
        "GÖREV:\n"
        "1. Görselde alev, yangın, köz veya yangın kaynaklı duman emaresi varsa KESİNLİKLE şu formatta TAM OLARAK 2 SATIR yanıt ver:\n"
        "ONAY: EVET - [Alevin boyutu, duman ve tehlike durumunun kısa açıklaması]\n"
        "MÜDAHALE: [Tahliye durumu, 112 aranmalı mı ve hangi yangın tüpüyle müdahale edilmeli]\n"
        "2. Yalnızca yangınla ilgisi olmayan alakasız sıradan nesneler, araba farı/lamba veya sigara/çakmak ise:\n"
        "ONAY: HAYIR - [Kısa açıklama]\n"
        "3. KESİNLİKLE SADECE TÜRKÇE YANIT VER."
    )

    payload = {
        "model": LOCAL_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    },
                ],
            }
        ],
        "max_tokens": 150,
        "temperature": 0.1,
    }

    try:
        response = requests.post(
            LOCAL_VLM_URL,
            json=payload,
            timeout=LOCAL_VLM_TIMEOUT,
        )

        elapsed_time = time.time() - req_start
        last_vlm_time = elapsed_time

        if response.status_code == 200:
            res_json = response.json()
            choices = res_json.get("choices", [])
            if choices:
                message_obj = choices[0].get("message", {})
                content = (message_obj.get("content") or "").strip()
                reasoning = (message_obj.get("reasoning") or "").strip()
                raw_answer = content if content else reasoning
            else:
                raw_answer = ""

            upper_ans = raw_answer.upper()

            # Karar ayrıştırma
            is_false_alarm = False
            is_confirmed_fire = False

            if "ONAY: EVET" in upper_ans:
                is_confirmed_fire = True
            elif "ONAY: HAYIR" in upper_ans:
                is_false_alarm = True
            elif any(neg in upper_ans for neg in [
                "NOT A FIRE", "NOT VISIBLE", "NO FIRE", "ORDINARY SMOKE",
                "CIGARETTE", "SIGARA", "STEAM", "BUHAR", "FALSE ALARM", "YANLIS ALARM"
            ]):
                is_false_alarm = True
            elif "EVET" in upper_ans[:25] and "HAYIR" not in upper_ans[:25]:
                is_confirmed_fire = True
            else:
                is_false_alarm = True

            if is_confirmed_fire and not is_false_alarm:
                is_fire_active = True
                if not raw_answer.startswith("ONAY: EVET"):
                    formatted_verdict = f"ONAY: EVET - {raw_answer}"
                else:
                    formatted_verdict = raw_answer

                last_vlm_verdict = formatted_verdict
                print(f"\n[!] [YEREL VLM YANGIN TEYİDİ - {LOCAL_MODEL_NAME} - {elapsed_time:.2f} sn]:\n{formatted_verdict}\n")

                # Video klip kaydı (5 sn geçmiş + 5 sn gelecek)
                filename = f"yangin_olay_{int(time.time())}.mp4"
                current_clip_filename = filename
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                clip_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

                for past_frame in list(frame_buffer):
                    clip_writer.write(past_frame)

                recording_post_event = True
                post_event_counter = BUFFER_LEN
                print(f"[!] [ALARM] Gerçek Yangın Teyit Edildi! Klip kaydediliyor: {filename}\n")

                # Telegram Bildirimi
                send_fire_alert_async(buffer.tobytes(), formatted_verdict, f"Local-{LOCAL_MODEL_NAME}", active_vlm_conf)
            else:
                is_fire_active = False
                if raw_answer.startswith("ONAY: HAYIR"):
                    formatted_verdict = raw_answer
                elif "CIGARETTE" in upper_ans or "SIGARA" in upper_ans:
                    formatted_verdict = "ONAY: HAYIR - Sigara dumani / tehlikesiz duman (Yangin degil)."
                elif "STEAM" in upper_ans or "BUHAR" in upper_ans:
                    formatted_verdict = "ONAY: HAYIR - Su buhari / sis tespit edildi (Yangin degil)."
                else:
                    formatted_verdict = f"ONAY: HAYIR - {raw_answer}"

                last_vlm_verdict = formatted_verdict
                print(f"\n[+] [YEREL VLM Analizi - {LOCAL_MODEL_NAME} - {elapsed_time:.2f} sn]:\n{formatted_verdict}\n")
        else:
            print(f"[Yerel VLM HTTP Hatası]: {response.status_code} - {response.text}")
            last_vlm_verdict = f"VLM Baglanti Hatasi (HTTP {response.status_code})"
            is_fire_active = False

    except requests.exceptions.Timeout:
        print(f"\n[UYARI] Yerel VLM yanıt süresi zaman aşımına uğradı ({LOCAL_VLM_TIMEOUT} sn)!")
        print(f"[i] Model CPU/GPU üzerinde yanıt üretirken {LOCAL_VLM_TIMEOUT} saniyeyi aştı.")
        last_vlm_verdict = "VLM Zaman Asimi (Cevap Gecikti)"
        is_fire_active = False
    except requests.exceptions.ConnectionError:
        print(f"\n[UYARI] Yerel VLM sunucusuna bağlanılamadı ({LOCAL_VLM_URL})!")
        print(f"[i] Lütfen Ollama'nın çalıştığından emin olun: terminalde 'ollama run {LOCAL_MODEL_NAME}' çalıştırın.\n")
        last_vlm_verdict = f"VLM Sunucusuna Baglanilamadi! ({LOCAL_MODEL_NAME})"
        is_fire_active = False
    except Exception as e:
        print(f"[Yerel VLM Hatası]: {e}")
        last_vlm_verdict = f"VLM Hatasi: {tr_to_ascii(str(e))[:40]}"
        is_fire_active = False
    finally:
        is_vlm_busy = False


def draw_tactical_corners(frame, x1, y1, x2, y2, color, length=14, thickness=2):
    """Köşe çerçeve çizgileri çizer."""
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)


def draw_single_alarm_panel(frame, primary_crop, primary_conf, is_busy, is_fire):
    """Sağ üst alarm hedef paneli çizer."""
    h, w = frame.shape[:2]
    panel_w = max(190, min(250, int(w * 0.22)))
    panel_h = max(160, min(210, int(panel_w * 0.95)))
    pad = 12

    x2 = w - pad
    x1 = x2 - panel_w
    y1 = pad
    y2 = y1 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    now = time.time()
    if is_busy:
        pulse = (math.sin(now * 8) + 1) / 2
        card_color = (0, int(160 + 95 * pulse), 255)
        dots = "." * (int(now * 3) % 4)
        header_text = f"LOCAL VLM ISLIYOR{dots}"
    elif is_fire:
        pulse = (math.sin(now * 6) + 1) / 2
        card_color = (0, 0, int(170 + 85 * pulse))
        header_text = "ALARM: [YANGIN TEYITLI]"
    else:
        card_color = (0, 220, 120) if primary_crop is not None else (70, 70, 70)
        header_text = "LOCAL VLM [BEKLEMEDE]" if primary_crop is None else "SON VLM GIRDISI"

    cv2.putText(frame, header_text, (x1 + 6, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, card_color, 1)
    if primary_conf > 0:
        cv2.putText(frame, f"YOLO: %{int(primary_conf * 100)}", (x2 - 75, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    img_x1, img_y1 = x1 + 4, y1 + 24
    img_x2, img_y2 = x2 - 4, y2 - 6
    iw = img_x2 - img_x1
    ih = img_y2 - img_y1

    if primary_crop is not None and primary_crop.size > 0:
        ch, cw = primary_crop.shape[:2]
        scale = min(iw / cw, ih / ch)
        nw, nh = int(cw * scale), int(ch * scale)
        resized = cv2.resize(primary_crop, (nw, nh))

        off_x = img_x1 + (iw - nw) // 2
        off_y = img_y1 + (ih - nh) // 2
        frame[off_y:off_y + nh, off_x:off_x + nw] = resized
        cv2.rectangle(frame, (off_x, off_y), (off_x + nw, off_y + nh), (50, 50, 50), 1)
    else:
        cx, cy = (img_x1 + img_x2) // 2, (img_y1 + img_y2) // 2
        cv2.putText(frame, "BEKLEMEDE", (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    draw_tactical_corners(frame, x1, y1, x2, y2, card_color, length=14, thickness=2)


def draw_hud(frame, status_text, status_color, vlm_text, fps_val):
    """Video üzerine yarı saydam modern HUD bilgi paneli çizer."""
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    box_h = 135
    cv2.rectangle(overlay, (10, h - box_h - 10), (w - 10, h - 10), (20, 20, 20), -1)
    
    top_bar_w = min(440, max(260, w - 280))
    cv2.rectangle(overlay, (10, 10), (top_bar_w, 75), (20, 20, 20), -1)

    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Üst Durum Metinleri & FPS
    cv2.putText(frame, status_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
    cv2.putText(frame, f"FPS: {fps_val:.1f} | Model: {LOCAL_MODEL_NAME}", (20, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Alt VLM Açıklama Metinleri
    cv2.putText(frame, "LOCAL VLM ANALIZI & ACIKLAMASI:", (20, h - box_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)

    clean_vlm = tr_to_ascii(vlm_text)
    
    max_chars_per_line = max(40, int((w - 40) / 10))
    words = clean_vlm.split()
    lines = []
    current_line = []

    for word in words:
        if len(" ".join(current_line + [word])) <= max_chars_per_line:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    y_offset = h - box_h + 45
    for line in lines[:3]:
        cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25


# ================= ANA VİDEO DÖNGÜSÜ =================

consecutive_no_threat_frames = 0
RESET_THRESHOLD_FRAMES = fps * 3

print(f"[*] Video taranıyor... Çıkmak için 'q' tuşuna basın.")

prev_loop_time = time.time()
current_fps = float(fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # Video bittiğinde başa sar (sürekli test edebilmek için)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            break

    # Anlık FPS Hesaplama
    now_loop_time = time.time()
    dt = now_loop_time - prev_loop_time
    prev_loop_time = now_loop_time
    if dt > 0:
        inst_fps = 1.0 / dt
        current_fps = 0.85 * current_fps + 0.15 * inst_fps

    # YOLO Çıkarımı
    results = model(frame, conf=CONF_THRESHOLD, classes=TARGET_CLASSES, verbose=False)
    boxes = results[0].boxes
    annotated_frame = results[0].plot()

    fire_boxes = [box for box in boxes if model.names.get(int(box.cls.item()), "").lower() == "fire"]
    smoke_boxes = [box for box in boxes if model.names.get(int(box.cls.item()), "").lower() == "smoke"]

    if len(fire_boxes) > 0:
        target_boxes = fire_boxes
        target_type = "Ateş (Fire)"
    elif len(smoke_boxes) > 0:
        target_boxes = smoke_boxes
        target_type = "Duman (Smoke)"
    else:
        target_boxes = []
        target_type = None

    detected_target_count = len(target_boxes)

    # Tetikleme Mekanizması
    if detected_target_count > 0:
        consecutive_no_threat_frames = 0
        now_time = time.time()
        if not is_fire_active and not is_vlm_busy and (now_time - last_vlm_request_time >= VLM_COOLDOWN_SECONDS):
            smart_crop, conf_val, target_box = get_smart_target_crop(
                frame, target_boxes, pad_ratio=PADDING_RATIO, min_context_size=MIN_CONTEXT_SIZE
            )
            
            active_vlm_crop = smart_crop.copy()
            active_vlm_conf = conf_val
            last_active_box = target_box.copy()

            is_vlm_busy = True
            last_vlm_request_time = now_time
            threading.Thread(target=local_vlm_verify_worker, args=(active_vlm_crop,), daemon=True).start()
            print(f"[*] [Yerel Trigger] {detected_target_count} adet {target_type} tespit edildi. Yerel VLM'e gönderildi...")

    else:
        if is_fire_active:
            consecutive_no_threat_frames += 1
            if consecutive_no_threat_frames > RESET_THRESHOLD_FRAMES:
                is_fire_active = False
                last_active_box = None
                print("[+] Yangın tehdidi sona erdi. Sistem sıfırlandı.")

    # Durum Bilgileri
    if is_fire_active:
        status_text = "DURUM: YANGIN TEYITLI (Kilitli)"
        status_color = (0, 0, 255)
    elif is_vlm_busy:
        status_text = "DURUM: Yerel VLM Dogruluyor..."
        status_color = (0, 255, 255)
    else:
        status_text = "DURUM: Taraniyor"
        status_color = (0, 255, 0)

    # Çizim Tuvalini Standartlaştır (Küçük videolarda HUD oranını korur)
    if (annotated_frame.shape[1], annotated_frame.shape[0]) != (width, height):
        annotated_frame = cv2.resize(annotated_frame, (width, height), interpolation=cv2.INTER_LINEAR)

    # Sağ Üst: Sabit Önizleme Çerçevesi
    draw_single_alarm_panel(
        annotated_frame,
        active_vlm_crop,
        active_vlm_conf,
        is_vlm_busy,
        is_fire_active
    )

    # Alt Taraf: HUD Bilgi Paneli
    draw_hud(annotated_frame, status_text, status_color, last_vlm_verdict, current_fps)

    # 10 Saniyelik Olay Tamponu
    frame_buffer.append(annotated_frame.copy())

    if recording_post_event and clip_writer is not None:
        clip_writer.write(annotated_frame)
        post_event_counter -= 1
        if post_event_counter <= 0:
            recording_post_event = False
            clip_writer.release()
            clip_writer = None
            print("[+] 10 sn'lik olay klibi başarıyla kaydedildi.")
            if current_clip_filename:
                send_fire_clip_async(current_clip_filename)

    cv2.imshow(WINDOW_NAME, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Temizlik
if clip_writer is not None:
    clip_writer.release()
cap.release()
cv2.destroyAllWindows()
