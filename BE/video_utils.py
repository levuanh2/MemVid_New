import os
import cv2
import qrcode
import numpy as np
from typing import List
from datetime import datetime

VIDEOS_DIR = 'videos'
QR_FRAME_RATE = 1

# Tạo thư mục videos nếu chưa tồn tại
os.makedirs(VIDEOS_DIR, exist_ok=True)

def generate_qr_frames(chunks: List[str]) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for txt in chunks:
        qr = qrcode.QRCode(
            version=None,  # None + fit=True = auto chọn version hợp lệ (1–40)
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(txt)
        qr.make(fit=True)

        qr_img = qr.make_image(fill_color="black", back_color="white")
        resized = qr_img.resize((512, 512))
        frame = cv2.cvtColor(np.array(resized.convert("RGB")), cv2.COLOR_RGB2BGR)

        frames.append(frame)
    return frames

# Lưu các frame thành video MP4 với tên động
def save_qr_frames_to_video(frames: List[np.ndarray], prefix: str = 'memory') -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f"{VIDEOS_DIR}/{prefix}_{ts}.mp4"
    print(f"Saving video to: {out_path}")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        QR_FRAME_RATE,
        (width, height)
    )
    for frame in frames:
        writer.write(frame)
    writer.release()
    return out_path

def decode_video_qr(path: str) -> List[str]:
    cap = cv2.VideoCapture(path)
    detector = cv2.QRCodeDetector()
    decoded_texts: set = set()  # Use set to avoid duplicates
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        text, points, _ = detector.detectAndDecode(frame)
        if text:
            decoded_texts.add(text)
        idx += 1

    cap.release()
    return list(decoded_texts)