from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def _auto_crop_to_content(gray: np.ndarray) -> tuple[np.ndarray, dict[str,int]]:
    h, w = gray.shape[:2]
    inv = 255 - gray
    _, th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(th)
    if coords is None:
        return gray, {"x": 0, "y": 0, "w": w, "h": h}
    x, y, bw, bh = cv2.boundingRect(coords)
    pad_x = max(8, int(bw * 0.02))
    pad_y = max(8, int(bh * 0.02))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)
    return gray[y1:y2, x1:x2], {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def normalize_document(image_bgr: np.ndarray) -> dict[str, Any]:
    original = image_bgr.copy()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    cropped_gray, crop_box = _auto_crop_to_content(gray)
    # fondo suave para compensar sombras sin empastar líneas finas
    blur = cv2.GaussianBlur(cropped_gray, (0, 0), 15)
    shadow_corrected = cv2.divide(cropped_gray, blur, scale=255)
    shadow_corrected = cv2.normalize(shadow_corrected, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    normalized_gray = clahe.apply(shadow_corrected)
    # unsharp muy leve, documental y no agresivo
    soft = cv2.GaussianBlur(normalized_gray, (0, 0), 0.8)
    sharpened = cv2.addWeighted(normalized_gray, 1.08, soft, -0.08, 0)
    preserved_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return {
        "gray": cropped_gray,
        "normalized_gray": normalized_gray,
        "preserved_bgr": preserved_bgr,
        "crop_box": crop_box,
    }
