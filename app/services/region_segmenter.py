from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def _clamp_box(box: dict[str, int], w: int, h: int) -> dict[str, int]:
    x = max(0, min(w - 1, int(box.get("x", 0))))
    y = max(0, min(h - 1, int(box.get("y", 0))))
    bw = max(1, int(box.get("w", 0)))
    bh = max(1, int(box.get("h", 0)))
    x2 = max(x + 1, min(w, x + bw))
    y2 = max(y + 1, min(h, y + bh))
    return {"x": x, "y": y, "w": x2 - x, "h": y2 - y}


def _rect_mask(shape: tuple[int, int], regions: list[dict[str, int]]) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in regions:
        box = _clamp_box(box, w, h)
        x, y, bw, bh = box['x'], box['y'], box['w'], box['h']
        cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
    return mask


def _largest_box_from_roi(inv_roi: np.ndarray, x0: int, y0: int, min_area: int, close_kernel: tuple[int, int]) -> dict[str, int] | None:
    merged = cv2.morphologyEx(inv_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel), iterations=2)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area:
            continue
        if area > best_area:
            best_area = area
            best = {"x": x0 + x, "y": y0 + y, "w": bw, "h": bh}
    return best


def _find_title_block_candidate(binary: np.ndarray) -> dict[str, int] | None:
    h, w = binary.shape[:2]
    # Buscar estrictamente en el cuadrante inferior derecho, no invadir la geometría principal.
    x0, y0 = int(w * 0.73), int(h * 0.74)
    roi = binary[y0:h, x0:w]
    if roi.size == 0:
        return None
    inv = roi.copy()  # binary ya viene invertido: trazos blancos sobre fondo negro.
    candidate = _largest_box_from_roi(inv, x0, y0, min_area=max(3000, int(w * h * 0.006)), close_kernel=(17, 9))
    if candidate is None:
        return None
    pad_x = max(6, int(candidate['w'] * 0.03))
    pad_y = max(6, int(candidate['h'] * 0.05))
    candidate = {
        'x': candidate['x'] - pad_x,
        'y': candidate['y'] - pad_y,
        'w': candidate['w'] + pad_x * 2,
        'h': candidate['h'] + pad_y * 2,
    }
    return _clamp_box(candidate, w, h)


def _find_notes_region(binary: np.ndarray, title_block: dict[str, int]) -> dict[str, int] | None:
    h, w = binary.shape[:2]
    x0 = max(0, int(title_block['x'] - title_block['w'] * 0.05))
    x1 = min(w, int(title_block['x'] + title_block['w'] * 0.98))
    y1 = max(0, int(title_block['y'] - h * 0.17))
    y2 = max(0, int(title_block['y'] - h * 0.015))
    if y2 <= y1 or x1 <= x0:
        return None
    roi = binary[y1:y2, x0:x1]
    if roi.size == 0:
        return None
    merged = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5)), iterations=1)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < max(300, int(w * h * 0.00018)):
            continue
        if bw < int(title_block['w'] * 0.16) or bh < 8:
            continue
        boxes.append((x, y, bw, bh))
    if not boxes:
        fallback = {
            'x': x0,
            'y': y1,
            'w': max(10, x1 - x0),
            'h': max(10, y2 - y1),
        }
        return _clamp_box(fallback, w, h)
    minx = min(x for x, _, _, _ in boxes)
    miny = min(y for _, y, _, _ in boxes)
    maxx = max(x + bw for x, _, bw, _ in boxes)
    maxy = max(y + bh for _, y, _, bh in boxes)
    return _clamp_box({'x': x0 + minx - 6, 'y': y1 + miny - 4, 'w': (maxx - minx) + 12, 'h': (maxy - miny) + 8}, w, h)


def _find_title_band(binary: np.ndarray) -> dict[str, int] | None:
    h, w = binary.shape[:2]
    x0, x1 = int(w * 0.28), int(w * 0.72)
    y0, y1 = int(h * 0.86), int(h * 0.96)
    roi = binary[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    merged = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)), iterations=1)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < max(120, int(w * h * 0.00008)):
            continue
        aspect = bw / float(max(bh, 1))
        if aspect < 2.2:
            continue
        score = area
        if score > best_score:
            best_score = score
            best = {'x': x0 + x - 6, 'y': y0 + y - 4, 'w': bw + 12, 'h': bh + 8}
    if best is None:
        best = {'x': int(w * 0.36), 'y': int(h * 0.89), 'w': int(w * 0.20), 'h': int(h * 0.045)}
    return _clamp_box(best, w, h)


def segment_document_regions(normalized_gray: np.ndarray, binary: np.ndarray) -> dict[str, Any]:
    h, w = normalized_gray.shape[:2]
    title_block = _find_title_block_candidate(binary)
    if not title_block:
        title_block = _clamp_box({'x': int(w * 0.79), 'y': int(h * 0.77), 'w': int(w * 0.17), 'h': int(h * 0.18)}, w, h)

    notes_box = _find_notes_region(binary, title_block)
    if not notes_box:
        notes_box = _clamp_box({
            'x': int(title_block['x'] - title_block['w'] * 0.02),
            'y': int(title_block['y'] - h * 0.11),
            'w': int(title_block['w'] * 1.02),
            'h': int(h * 0.10),
        }, w, h)

    title_band = _find_title_band(binary)

    documental_regions = [title_block, notes_box, title_band]
    # Dilatación mucho más conservadora para no invadir media lámina.
    documental_mask = _rect_mask((h, w), documental_regions)
    documental_mask = cv2.dilate(documental_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    geometry_mask = cv2.bitwise_and(binary, cv2.bitwise_not(documental_mask))

    inv_doc = cv2.bitwise_and(binary, documental_mask)
    text_seed = cv2.morphologyEx(inv_doc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    text_seed = cv2.dilate(text_seed, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

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
