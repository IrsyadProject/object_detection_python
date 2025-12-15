import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
import sys
from collections import Counter
import os

# Windows sound playback (fallback to no-op on other platforms)
try:
    import winsound
    WINSOUND_AVAILABLE = True
except Exception:
    WINSOUND_AVAILABLE = False

# Suara default dan cooldown (detik)
SOUND_FILE = "hidup-jokowi.wav"
SOUND_COOLDOWN = 1.0
_last_sound_time = 0.0

# Face cascade untuk mengecualikan wajah dari mask kulit (mengurangi false-positive)
try:
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    FACE_CASCADE_AVAILABLE = not FACE_CASCADE.empty()
except Exception:
    FACE_CASCADE = None
    FACE_CASCADE_AVAILABLE = False

# Thresholds for open-hand detection
OPEN_DEFECTS_THRESHOLD = 3

# HSV calibration (default). Use `c` key to calibrate from center ROI.
HSV_LOWER = np.array([0, 30, 60], dtype=np.uint8)
HSV_UPPER = np.array([20, 150, 255], dtype=np.uint8)

# status message to show on-screen after actions (reset/calibrate)
status_msg = ""
status_msg_frames = 0
STATUS_MSG_DURATION = 90  # number of frames to display message (~3s at 30fps)

def play_alert():
    """Putar alert.wav secara non-blocking (Windows winsound)."""
    if not WINSOUND_AVAILABLE:
        return
    if not os.path.isfile(SOUND_FILE):
        return
    try:
        winsound.PlaySound(SOUND_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as e:
        print(f"Warning: gagal memutar suara: {e}", file=sys.stderr)

def is_fist_cv(frame):
    # Delegate to get_hand_debug and evaluate returned geometry
    mask, contours, c, hull_pts, defects = get_hand_debug(frame)
    if c is None:
        return False

    area = cv2.contourArea(c)
    if area < 2000:
        return False

    if hull_pts is None or len(hull_pts) < 3:
        return False

    if defects is None:
        # use solidity if no defects
        hull_area = cv2.contourArea(hull_pts) if hull_pts is not None else 0
        if hull_area > 0:
            solidity = area / hull_area
            return solidity > 0.85
        return False

    cnt_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(c[s][0])
        end = tuple(c[e][0])
        far = tuple(c[f][0])
        a = np.linalg.norm(np.array(start) - np.array(end))
        b = np.linalg.norm(np.array(start) - np.array(far))
        cdist = np.linalg.norm(np.array(end) - np.array(far))
        if a == 0 or b == 0 or cdist == 0:
            continue
        angle = np.arccos(max(-1.0, min(1.0, (b*b + cdist*cdist - a*a) / (2*b*cdist)))) * 180.0 / np.pi
        if angle < 90 and d > 1000:
            cnt_defects += 1

    return cnt_defects <= 1


def get_hand_debug(frame):
    """Return processed hand mask and geometry for debugging.
    Returns (mask, contours, largest_contour, hull_points, defects) where mask is the resized
    binary mask used for detection (uint8), contours is list of contours on the small image,
    largest_contour is the chosen contour (in small-image coords), hull_points is convex hull points
    (in small-image coords) and defects is the convexity defects array (or None).
    """
    h, w = frame.shape[:2]
    small_w = max(160, w // 3)
    small_h = max(120, h // 3)
    small = cv2.resize(frame, (small_w, small_h))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # use calibrated HSV ranges
    lower = HSV_LOWER
    upper = HSV_UPPER
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

    # Optionally remove faces from mask so face skin doesn't become a large contour
    if FACE_CASCADE_AVAILABLE:
        try:
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (fx, fy, fw, fh) in faces:
                # remove face region from mask
                x1 = max(0, fx)
                y1 = max(0, fy)
                x2 = min(mask.shape[1], fx + fw)
                y2 = min(mask.shape[0], fy + fh)
                mask[y1:y2, x1:x2] = 0
        except Exception:
            pass

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, contours, None, None, None

    c = max(contours, key=cv2.contourArea)
    hull_pts = cv2.convexHull(c)
    hull_idx = cv2.convexHull(c, returnPoints=False)
    defects = None
    try:
        if hull_idx is not None and len(hull_idx) >= 3:
            defects = cv2.convexityDefects(c, hull_idx)
    except Exception:
        defects = None

    return mask, contours, c, hull_pts, defects


def is_hand_open_cv(frame):
    """Detect an open hand (visible fingers) using convexity defects heuristic.
    Returns True when the number of finger-like defects meets the OPEN_DEFECTS_THRESHOLD.
    """
    mask, contours, c, hull_pts, defects = get_hand_debug(frame)
    if c is None:
        return False
    area = cv2.contourArea(c)
    if area < 2000:
        return False
    if defects is None:
        return False

    cnt_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(c[s][0])
        end = tuple(c[e][0])
        far = tuple(c[f][0])
        a = np.linalg.norm(np.array(start) - np.array(end))
        b = np.linalg.norm(np.array(start) - np.array(far))
        cdist = np.linalg.norm(np.array(end) - np.array(far))
        if a == 0 or b == 0 or cdist == 0:
            continue
        angle = np.arccos(max(-1.0, min(1.0, (b*b + cdist*cdist - a*a) / (2*b*cdist)))) * 180.0 / np.pi
        # treat defects as fingers when angle small and defect depth sufficient
        if angle < 90 and d > 1000:
            cnt_defects += 1

    return cnt_defects >= OPEN_DEFECTS_THRESHOLD

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 realtime inference")
    parser.add_argument("--model", "-m", default="yolo11n.pt", help="Path to model")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index")
    parser.add_argument("--device", "-d", default=None, help="Device (e.g. cpu or cuda:0). Leave None for auto")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional inference image size")
    parser.add_argument("--debug-hand", action="store_true", help="Show hand mask and contour overlay for tuning the fist detector")
    return parser.parse_args()

args = parse_args()

model = YOLO(args.model)

camera = cv2.VideoCapture(args.camera)
if not camera.isOpened():
    print(f"Error: tidak dapat membuka kamera index {args.camera}", file=sys.stderr)
    sys.exit(1)

NUM_CLASSES = len(model.names)
COLORS = np.random.randint(0, 256, size=(NUM_CLASSES, 3), dtype=np.uint8)

# Membuat jendela yang bisa diubah ukurannya
WINDOW_NAME = "YOLOv11 Inference"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Kamus untuk menerjemahkan label kelas ke Bahasa Indonesia (lengkap untuk COCO)
translation_dict = {
    'person': 'Manusia',
    'bicycle': 'Sepeda',
    'car': 'Mobil',
    'motorcycle': 'Motor',
    'airplane': 'Pesawat',
    'bus': 'Bus',
    'train': 'Kereta',
    'truck': 'Truk',
    'boat': 'Kapal',
    'traffic light': 'Lampu Lalu Lintas',
    'fire hydrant': 'Hidran',
    'stop sign': 'Rambu Stop',
    'parking meter': 'Meter Parkir',
    'bench': 'Bangku',
    'bird': 'Burung',
    'cat': 'Kucing',
    'dog': 'Anjing',
    'horse': 'Kuda',
    'sheep': 'Domba',
    'cow': 'Sapi',
    'elephant': 'Gajah',
    'bear': 'Beruang',
    'zebra': 'Zebra',
    'giraffe': 'Jerapah',
    'backpack': 'Ransel',
    'umbrella': 'Payung',
    'handbag': 'Tas Tangan',
    'tie': 'Dasi',
    'suitcase': 'Koper',
    'frisbee': 'Frisbee',
    'skis': 'Ski',
    'snowboard': 'Snowboard',
    'sports ball': 'Bola Olahraga',
    'kite': 'Layang-layang',
    'baseball bat': 'Pemukul Baseball',
    'baseball glove': 'Sarung Tangan Baseball',
    'skateboard': 'Skateboard',
    'surfboard': 'Papan Selancar',
    'tennis racket': 'Raket Tenis',
    'bottle': 'Botol',
    'wine glass': 'Gelas Anggur',
    'cup': 'Cangkir',
    'fork': 'Garpu',
    'knife': 'Pisau',
    'spoon': 'Sendok',
    'bowl': 'Mangkuk',
    'banana': 'Pisang',
    'apple': 'Apel',
    'sandwich': 'Sandwich',
    'orange': 'Jeruk',
    'broccoli': 'Brokoli',
    'carrot': 'Wortel',
    'hot dog': 'Hotdog',
    'pizza': 'Pizza',
    'donut': 'Donat',
    'cake': 'Kue',
    'chair': 'Kursi',
    'couch': 'Sofa',
    'potted plant': 'Tanaman Pot',
    'bed': 'Tempat Tidur',
    'dining table': 'Meja Makan',
    'toilet': 'Toilet',
    'tv': 'TV',
    'laptop': 'Laptop',
    'mouse': 'Mouse',
    'remote': 'Remote',
    'keyboard': 'Keyboard',
    'cell phone': 'Handphone',
    'microwave': 'Microwave',
    'oven': 'Oven',
    'toaster': 'Pemanggang Roti',
    'sink': 'Wastafel',
    'refrigerator': 'Kulkas',
    'book': 'Buku',
    'clock': 'Jam',
    'vase': 'Vas',
    'scissors': 'Gunting',
    'teddy bear': 'Boneka Beruang',
    'hair drier': 'Pengering Rambut',
    'toothbrush': 'Sikat Gigi',
}

# Untuk menghitung FPS
prev_time = time.perf_counter()
fps = 0.0
# status open-hand sebelumnya (untuk mendeteksi transisi)
prev_open = False
# untuk mengurangi false-positive: harus terdeteksi beberapa frame berturut-turut
open_counter = 0
OPEN_CONSECUTIVE = 3
# lewati beberapa frame pertama sebagai warmup (kamera/eksposure stabilisasi)
WARMUP_FRAMES = 10
frame_idx = 0

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Warning: frame tidak tersedia (camera.read() returned False). Mengakhiri.", file=sys.stderr)
            break

        # Opsional: berikan imgsz ke model jika disediakan
        infer_kwargs = {"verbose": False}
        if args.imgsz:
            infer_kwargs["imgsz"] = args.imgsz
        if args.device:
            infer_kwargs["device"] = args.device

        results = model(frame, **infer_kwargs)

        annotated_frame = frame.copy() # Salin frame asli untuk digambari
        # Kumpulkan nama-nama deteksi pada frame ini untuk perhitungan
        detected_names = []

        # Cek apakah ada tangan terbuka menggunakan fallback OpenCV
        frame_idx += 1
        detected_now = False
        if frame_idx > WARMUP_FRAMES:
            try:
                detected_now = is_hand_open_cv(frame)
            except Exception:
                detected_now = False

        # hitung konfirmasi berturut-turut
        if detected_now:
            open_counter += 1
        else:
            open_counter = 0

        # hanya anggap open-hand valid bila terdeteksi beberapa frame berturut-turut
        open_hand = open_counter >= OPEN_CONSECUTIVE

        # Hanya trigger suara pada transisi False -> True untuk menghindari loop
        if open_hand and not prev_open:
            now = time.perf_counter()
            if (now - _last_sound_time) > SOUND_COOLDOWN:
                play_alert()
                _last_sound_time = now

        prev_open = open_hand

        # Iterasi melalui setiap objek yang terdeteksi
        for box in results[0].boxes:
            # Dapatkan koordinat kotak
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
            except Exception:
                # fallback jika struktur berbeda
                coords = np.array(box.xyxy).reshape(-1)
                x1, y1, x2, y2 = map(int, coords[:4])

            # Dapatkan skor kepercayaan diri
            confidence = float(box.conf[0]) if hasattr(box.conf, "__getitem__") else float(box.conf)
            # Dapatkan ID kelas
            class_id = int(box.cls[0]) if hasattr(box.cls, "__getitem__") else int(box.cls)

            # Lindungi indexing warna jika ada mismatch
            class_id_safe = class_id % len(COLORS)
            color_bgr = (int(COLORS[class_id_safe][0]), int(COLORS[class_id_safe][1]), int(COLORS[class_id_safe][2]))

            # Dapatkan nama kelas asli (Inggris)
            class_name_en = model.names.get(class_id, str(class_id)) if isinstance(model.names, dict) else model.names[class_id]
            # Terjemahkan ke Bahasa Indonesia, jika tidak ada di kamus, gunakan nama asli
            class_name_id = translation_dict.get(class_name_en, class_name_en)

            # Tambah ke daftar deteksi untuk perhitungan
            detected_names.append(class_name_id)

            # Format label: "NamaLabel Persentase%"
            label = f"{class_name_id} {int(confidence * 100)}%"

            # Gambar kotak
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 2)

            # Gambar background untuk teks supaya terbaca
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            # koordinat background (jangan keluar frame)
            bx1 = x1
            by1 = max(0, y1 - th - 8)
            bx2 = x1 + tw + 6
            by2 = y1
            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color_bgr, -1)
            # teks putih di atas background
            cv2.putText(annotated_frame, label, (x1 + 3, y1 - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Hitung FPS
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time)) if prev_time else 0.0
        prev_time = now

        # Dapatkan ukuran jendela saat ini
        try:
            window_width = cv2.getWindowImageRect(WINDOW_NAME)[2]
            window_height = cv2.getWindowImageRect(WINDOW_NAME)[3]
        except cv2.error:
            window_width = annotated_frame.shape[1]
            window_height = annotated_frame.shape[0]

        # Hitung rasio aspek
        frame_height, frame_width = annotated_frame.shape[:2]
        aspect_ratio = frame_width / frame_height

        # Ubah ukuran frame agar sesuai dengan jendela sambil mempertahankan rasio aspek
        new_width = window_width
        new_height = int(new_width / aspect_ratio)

        if new_height > window_height:
            new_height = window_height
            new_width = int(new_height * aspect_ratio)

        resized_frame = cv2.resize(annotated_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Buat kanvas hitam dan letakkan frame di tengah
        canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        y_offset = (window_height - new_height) // 2
        x_offset = (window_width - new_width) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

        # Jika debug hand aktif, tampilkan mask kecil dan kontur di pojok kanan atas
        if args.debug_hand:
            try:
                mask, contours, c, hull_pts, defects = get_hand_debug(frame)
                # gambarkan kontur dan hull pada annotated_frame (skala kecil -> perlu rescale coords)
                if c is not None:
                    # draw contour on resized_frame for visibility (scale contour to resized_frame)
                    # compute scale between small mask and resized_frame area
                    small_w = mask.shape[1]
                    small_h = mask.shape[0]
                    scale_x = new_width / small_w
                    scale_y = new_height / small_h
                    # draw contour (rescaled) on the region inside canvas
                    contour_scaled = (c.astype(np.float32) * [scale_x, scale_y]).astype(np.int32)
                    # shift to region origin (x_offset, y_offset)
                    contour_shifted = contour_scaled + np.array([x_offset, y_offset])
                    cv2.drawContours(canvas, [contour_shifted], -1, (0, 255, 0), 2)
                    if hull_pts is not None:
                        hull_scaled = (hull_pts.astype(np.float32) * [scale_x, scale_y]).astype(np.int32)
                        hull_shifted = hull_scaled + np.array([x_offset, y_offset])
                        cv2.drawContours(canvas, [hull_shifted], -1, (255, 0, 0), 1)

                # buat kecilkan mask menjadi panel kecil
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                panel_w = min(240, mask_bgr.shape[1])
                panel_h = int(panel_w * mask_bgr.shape[0] / mask_bgr.shape[1])
                panel = cv2.resize(mask_bgr, (panel_w, panel_h))
                px = window_width - panel_w - 10
                py = 10
                # border
                cv2.rectangle(canvas, (px - 2, py - 2), (px + panel_w + 2, py + panel_h + 2), (0, 0, 0), -1)
                canvas[py:py + panel_h, px:px + panel_w] = panel
            except Exception:
                pass

        # --- Menampilkan jumlah objek di kiri atas ---
        counts = Counter(detected_names)
        total = sum(counts.values())
        # Siapkan baris teks: baris pertama total, lalu per-kelas paling banyak
        lines = [f"Total Objek: {total}"]
        for name, cnt in counts.most_common(10):
            lines.append(f"{name}: {cnt}")

        # Gambar kotak latar di kiri atas
        left_margin = 10
        top_margin = 10
        line_height = 20
        box_width = 220
        box_height = 8 + line_height * len(lines)
        cv2.rectangle(canvas, (left_margin, top_margin), (left_margin + box_width, top_margin + box_height), (0, 0, 0), -1)
        # Tulis setiap baris
        text_x = left_margin + 8
        text_y = top_margin + 16
        small_font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(lines):
            y = text_y + i * line_height
            cv2.putText(canvas, line, (text_x, y), small_font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Menambahkan Judul pada Kanvas (tengah) dan FPS (pojok kanan atas) ---
        # Judul di tengah
        title_text = "Realtime Object Detection by Irsyad"
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 1
        title_thickness = 2
        title_color = (0, 255, 255)  # Warna kuning (BGR)

        # Dapatkan ukuran teks untuk menempatkannya di tengah
        title_size = cv2.getTextSize(title_text, font, title_scale, title_thickness)[0]
        title_x = (window_width - title_size[0]) // 2
        title_y = y_offset - 20 if y_offset > 40 else 40
        # Gambar shadow (bayangan) hitam sedikit bergeser, lalu teks utama putih di atasnya
        shadow_offset = 2
        cv2.putText(canvas, title_text, (title_x + shadow_offset, title_y + shadow_offset), font, title_scale, (0, 0, 0), title_thickness + 1, cv2.LINE_AA)
        cv2.putText(canvas, title_text, (title_x, title_y), font, title_scale, title_color, title_thickness, cv2.LINE_AA)

        # FPS di pojok kanan atas dengan background agar terbaca
        fps_text = f"FPS: {fps:.1f}"
        fps_scale = 0.7
        fps_thickness = 2
        (fw, fh), _ = cv2.getTextSize(fps_text, font, fps_scale, fps_thickness)
        fx = max(10, window_width - fw - 10)
        fy = 30
        # Gambar background kotak hitam di belakang teks FPS
        cv2.rectangle(canvas, (fx - 6, fy - fh - 6), (fx + fw + 6, fy + 6), (0, 0, 0), -1)
        cv2.putText(canvas, fps_text, (fx, fy), font, fps_scale, (255, 255, 255), fps_thickness, cv2.LINE_AA)

        # Tampilkan pesan status singkat jika ada
        if status_msg_frames > 0 and status_msg:
            msg_font = cv2.FONT_HERSHEY_SIMPLEX
            msg_scale = 0.7
            msg_th = 2
            (mw, mh), _ = cv2.getTextSize(status_msg, msg_font, msg_scale, msg_th)
            mx = 10
            my = window_height - 20
            cv2.rectangle(canvas, (mx - 6, my - mh - 6), (mx + mw + 6, my + 6), (0, 0, 0), -1)
            cv2.putText(canvas, status_msg, (mx, my), msg_font, msg_scale, (255, 255, 255), msg_th, cv2.LINE_AA)
            status_msg_frames -= 1

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # reset counters/warmup and status
            frame_idx = 0
            open_counter = 0
            prev_open = False
            _last_sound_time = 0.0
            status_msg = "Reset: warmup and counters cleared"
            status_msg_frames = STATUS_MSG_DURATION
        elif key == ord('c'):
            # calibrate HSV from central ROI of the current frame
            try:
                fh, fw = frame.shape[:2]
                rw = max(40, fw // 6)
                rh = max(40, fh // 6)
                cx = fw // 2
                cy = fh // 2
                roi = frame[cy - rh:cy + rh, cx - rw:cx + rw]
                if roi.size == 0:
                    raise ValueError("ROI kosong")
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                vals = hsv_roi.reshape(-1, 3)
                low = np.percentile(vals, 5, axis=0).astype(int)
                high = np.percentile(vals, 95, axis=0).astype(int)
                # expand a bit for tolerance
                pad = np.array([10, 30, 30])
                new_lower = np.maximum(low - pad, [0, 0, 0]).astype(np.uint8)
                new_upper = np.minimum(high + pad, [179, 255, 255]).astype(np.uint8)
                HSV_LOWER[:] = new_lower
                HSV_UPPER[:] = new_upper
                status_msg = f"Calibrated HSV: {HSV_LOWER.tolist()} - {HSV_UPPER.tolist()}"
                status_msg_frames = STATUS_MSG_DURATION
            except Exception as e:
                status_msg = f"Calibrate failed: {e}"
                status_msg_frames = STATUS_MSG_DURATION
finally:
    camera.release()
    cv2.destroyAllWindows()