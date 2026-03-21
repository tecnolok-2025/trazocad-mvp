from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def _rect_mask(shape: tuple[int, int], regions: list[dict[str, int]]) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in regions:
        x = max(0, int(box.get("x", 0)))
        y = max(0, int(box.get("y", 0)))
        bw = max(0, int(box.get("w", 0)))
        bh = max(0, int(box.get("h", 0)))
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        if x2 > x and y2 > y:
            cv2.rectangle(mask, (x, y), (x2, y2), 255, -1)
    return mask


def _find_title_block_candidate(binary: np.ndarray) -> dict[str, int] | None:
    h, w = binary.shape[:2]
    x0, y0 = int(w * 0.52), int(h * 0.62)
    roi = binary[y0:h, x0:w]
    if roi.size == 0:
        return None
    inv = 255 - roi
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < (w * h) * 0.01:
            continue
        if bw < w * 0.16 or bh < h * 0.05:
            continue
        aspect = bw / float(max(bh, 1))
        if aspect < 1.2 or aspect > 8.5:
            continue
        # premiar posición abajo-derecha y tamaño razonable
        cx = x0 + x + bw / 2.0
        cy = y0 + y + bh / 2.0
        pos_score = (cx / max(w, 1)) * 0.55 + (cy / max(h, 1)) * 0.45
        score = area * pos_score
        if score > best_score:
            best_score = score
            best = {"x": x0 + x, "y": y0 + y, "w": bw, "h": bh}
    if best is not None:
        pad_x = max(8, int(best['w'] * 0.04))
        pad_y = max(8, int(best['h'] * 0.08))
        best = {
            'x': max(0, best['x'] - pad_x),
            'y': max(0, best['y'] - pad_y),
            'w': min(w - max(0, best['x'] - pad_x), best['w'] + pad_x * 2),
            'h': min(h - max(0, best['y'] - pad_y), best['h'] + pad_y * 2),
        }
    return best


def segment_document_regions(normalized_gray: np.ndarray, binary: np.ndarray) -> dict[str, Any]:
    h, w = normalized_gray.shape[:2]
    title_block = _find_title_block_candidate(binary)
    if not title_block:
        title_block = {'x': int(w * 0.68), 'y': int(h * 0.72), 'w': int(w * 0.28), 'h': int(h * 0.20)}

    notes_box = {
        'x': max(0, int(title_block['x'] - w * 0.04)),
        'y': max(0, int(title_block['y'] - h * 0.16)),
        'w': min(w - max(0, int(title_block['x'] - w * 0.04)), int(title_block['w'] + w * 0.08)),
        'h': min(h - max(0, int(title_block['y'] - h * 0.16)), int(max(title_block['h'] * 0.70, h * 0.12))),
    }
    title_band = {
        'x': int(w * 0.25),
        'y': int(h * 0.88),
        'w': int(w * 0.50),
        'h': int(h * 0.08),
    }
    documental_regions = [title_block, notes_box, title_band]
    documental_mask = _rect_mask((h, w), documental_regions)
    documental_mask = cv2.dilate(documental_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)

    # Geometría base = binario reparado menos documento
    geometry_mask = cv2.bitwise_and(binary, cv2.bitwise_not(documental_mask))

    # Texto regional aproximado dentro de zonas documentales.
    inv_doc = cv2.bitwise_and(255 - binary, documental_mask)
    text_seed = cv2.morphologyEx(inv_doc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    text_seed = cv2.dilate(text_seed, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    debug = cv2.cvtColor(normalized_gray, cv2.COLOR_GRAY2BGR)
    colors = [(180, 30, 170), (30, 140, 220), (40, 170, 70)]
    labels = ['ROTULO', 'NOTAS', 'TITULO']
    for box, color, label in zip(documental_regions, colors, labels):
        x, y, bw, bh = box['x'], box['y'], box['w'], box['h']
        cv2.rectangle(debug, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(debug, label, (x + 6, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return {
        'title_block': title_block,
        'notes_region': notes_box,
        'title_region': title_band,
        'documental_regions': documental_regions,
        'documental_mask': documental_mask,
        'geometry_mask': geometry_mask,
        'text_seed_mask': text_seed,
        'debug_regions': debug,
    }
