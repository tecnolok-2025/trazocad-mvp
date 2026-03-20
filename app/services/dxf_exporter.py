from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import ezdxf
import numpy as np
from ezdxf.enums import TextEntityAlignment
from PIL import Image


def _documental_regions_from_geometry(geometry: dict[str, Any], image_width: int, image_height: int) -> list[dict[str, float]]:
    regions = [dict(box) for box in geometry.get('title_blocks', [])]
    if not regions:
        regions.append({'x': image_width * 0.72, 'y': image_height * 0.68, 'w': image_width * 0.24, 'h': image_height * 0.24})
    regions.append({'x': image_width * 0.68, 'y': image_height * 0.56, 'w': image_width * 0.30, 'h': image_height * 0.24})
    regions.append({'x': image_width * 0.30, 'y': image_height * 0.86, 'w': image_width * 0.40, 'h': image_height * 0.10})
    return regions


def _expanded_region(region: dict[str, float], pad: float) -> dict[str, float]:
    return {
        'x': float(region['x']) - pad,
        'y': float(region['y']) - pad,
        'w': float(region['w']) + 2 * pad,
        'h': float(region['h']) + 2 * pad,
    }


def _line_overlap_ratio(line: dict[str, Any], region: dict[str, float]) -> float:
    x1, y1, x2, y2 = float(line.get('x1', 0)), float(line.get('y1', 0)), float(line.get('x2', 0)), float(line.get('y2', 0))
    minx, maxx = min(x1, x2), max(x1, x2)
    miny, maxy = min(y1, y2), max(y1, y2)
    rx1, ry1 = float(region['x']), float(region['y'])
    rx2, ry2 = rx1 + float(region['w']), ry1 + float(region['h'])
    ix1, iy1 = max(minx, rx1), max(miny, ry1)
    ix2, iy2 = min(maxx, rx2), min(maxy, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max((maxx - minx) * (maxy - miny), 1.0)
    return inter / area


def _normalized_line_key(line: dict[str, Any], snap: int = 6) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(line.get(k, 0)) / snap) * snap) for k in ('x1', 'y1', 'x2', 'y2')]
    pts = sorted([(x1, y1), (x2, y2)])
    return pts[0][0], pts[0][1], pts[1][0], pts[1][1]


def _is_axis_like(line: dict[str, Any]) -> bool:
    x1, y1, x2, y2 = float(line.get('x1', 0)), float(line.get('y1', 0)), float(line.get('x2', 0)), float(line.get('y2', 0))
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    return dx < 1.5 or dy < 1.5 or (min(dx, dy) / max(dx, dy, 1.0) < 0.08)


def _line_length(line: dict[str, Any]) -> float:
    return math.hypot(float(line.get('x2', 0)) - float(line.get('x1', 0)), float(line.get('y2', 0)) - float(line.get('y1', 0)))


def _endpoint_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build_connectivity(lines: list[dict[str, Any]], tolerance: float = 16.0) -> list[int]:
    endpoints: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for line in lines:
        endpoints.append(((float(line['x1']), float(line['y1'])), (float(line['x2']), float(line['y2']))))
    scores = [0] * len(lines)
    for i, (a1, a2) in enumerate(endpoints):
        for j in range(i + 1, len(endpoints)):
            b1, b2 = endpoints[j]
            if min(_endpoint_distance(a1, b1), _endpoint_distance(a1, b2), _endpoint_distance(a2, b1), _endpoint_distance(a2, b2)) <= tolerance:
                scores[i] += 1
                scores[j] += 1
    return scores


def _near_text(line: dict[str, Any], text_items: list[dict[str, Any]], pad: float = 18.0) -> bool:
    minx = min(float(line.get('x1', 0)), float(line.get('x2', 0)))
    maxx = max(float(line.get('x1', 0)), float(line.get('x2', 0)))
    miny = min(float(line.get('y1', 0)), float(line.get('y2', 0)))
    maxy = max(float(line.get('y1', 0)), float(line.get('y2', 0)))
    for item in text_items:
        x1 = float(item.get('x', 0)) - pad
        y1 = float(item.get('y', 0)) - pad
        x2 = float(item.get('x', 0) + item.get('w', 0)) + pad
        y2 = float(item.get('y', 0) + item.get('h', 0)) + pad
        if not (maxx < x1 or minx > x2 or maxy < y1 or miny > y2):
            return True
    return False


def _load_support_map(raster_path: Path | None) -> np.ndarray | None:
    if not raster_path or not raster_path.exists():
        return None
    img = cv2.imread(str(raster_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 7)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return binary


def _line_support_ratio(line: dict[str, Any], support_map: np.ndarray | None) -> float:
    if support_map is None:
        return 1.0
    h, w = support_map.shape[:2]
    x1 = int(round(float(line.get('x1', 0))))
    y1 = int(round(float(line.get('y1', 0))))
    x2 = int(round(float(line.get('x2', 0))))
    y2 = int(round(float(line.get('y2', 0))))
    dist = max(int(round(math.hypot(x2 - x1, y2 - y1))), 1)
    samples = max(min(dist // 4, 180), 12)
    hit = 0
    for i in range(samples + 1):
        t = i / max(samples, 1)
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        x0, x3 = max(0, x - 1), min(w, x + 2)
        y0, y3 = max(0, y - 1), min(h, y + 2)
        if np.any(support_map[y0:y3, x0:x3] > 0):
            hit += 1
    return hit / max(samples + 1, 1)


def _detect_document_boxes(support_map: np.ndarray | None, image_width: int, image_height: int) -> list[dict[str, float]]:
    if support_map is None:
        return []
    h, w = support_map.shape[:2]
    roi = support_map[int(h * 0.52):, int(w * 0.54):]
    if roi.size == 0:
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    merged = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[dict[str, float]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < (w * h) * 0.01:
            continue
        if bw < w * 0.14 or bh < h * 0.08:
            continue
        abs_x = x + int(w * 0.54)
        abs_y = y + int(h * 0.52)
        abs_w = min(bw, max(1, image_width - abs_x))
        abs_h = min(bh, max(1, image_height - abs_y))
        boxes.append({
            'x': abs_x,
            'y': abs_y,
            'w': abs_w,
            'h': abs_h,
        })
    return boxes


def _extract_supported_hv_lines(support_map: np.ndarray | None, image_width: int, image_height: int) -> list[dict[str, float]]:
    if support_map is None:
        return []
    lines_p = cv2.HoughLinesP(support_map, 1, np.pi / 180, threshold=80, minLineLength=max(40, int(max(image_width, image_height) * 0.08)), maxLineGap=8)
    if lines_p is None:
        return []
    extra: list[dict[str, float]] = []
    for item in lines_p[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in item]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if min(dx, dy) > max(4.0, max(dx, dy) * 0.08):
            continue
        extra.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    return extra


def _sanitize_lines_for_dxf(lines: list[dict[str, Any]], documental: list[dict[str, float]], text_items: list[dict[str, Any]], image_width: int, image_height: int, support_map: np.ndarray | None = None) -> list[dict[str, Any]]:
    img_long = max(image_width, image_height)
    expanded_doc = [_expanded_region(r, 18.0) for r in documental]
    connectivity = _build_connectivity(lines, tolerance=max(14.0, img_long * 0.006))
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for idx, line in enumerate(lines):
        length = _line_length(line)
        if length < 14:
            continue
        overlap = max((_line_overlap_ratio(line, r) for r in expanded_doc), default=0.0)
        axis_like = _is_axis_like(line)
        connected = connectivity[idx]
        near_text = _near_text(line, text_items)
        support = _line_support_ratio(line, support_map)

        if support < 0.28 and length < img_long * 0.22:
            continue
        if support < 0.16:
            continue
        if overlap > 0.55 and length < img_long * 0.35 and support < 0.72:
            continue
        if overlap > 0.18 and near_text and length < img_long * 0.18 and support < 0.68:
            continue
        if connected == 0 and length < img_long * 0.1 and not axis_like:
            continue
        if connected <= 1 and length < img_long * 0.06 and support < 0.7:
            continue
        if near_text and length < img_long * 0.05 and not axis_like:
            continue

        key = _normalized_line_key(line)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
    return cleaned


def _classify_line_layer(line: dict[str, Any], dimension_lines: list[dict[str, Any]], cota_texts: list[dict[str, Any]], documental: list[dict[str, float]]) -> str:
    key = _normalized_line_key(line)
    if any(_normalized_line_key(item) == key for item in dimension_lines):
        return 'COTAS'
    if _near_text(line, cota_texts, pad=16.0):
        return 'COTAS'
    overlap = max((_line_overlap_ratio(line, r) for r in documental), default=0.0)
    if overlap > 0.25:
        return 'ROTULO'
    return 'GEOMETRIA'


def _poly_bbox(poly: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    xs = [float(pt['x']) for pt in poly]
    ys = [float(pt['y']) for pt in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _poly_is_curve_like(poly: list[dict[str, Any]]) -> bool:
    if len(poly) < 3:
        return False
    turns = 0
    for a, b, c in zip(poly[:-2], poly[1:-1], poly[2:]):
        abx, aby = float(b['x']) - float(a['x']), float(b['y']) - float(a['y'])
        bcx, bcy = float(c['x']) - float(b['x']), float(c['y']) - float(b['y'])
        cross = abx * bcy - aby * bcx
        if abs(cross) > 2:
            turns += 1
    return turns >= 2


def _add_text_item(msp, item: dict[str, Any], layer: str, height_mm: float, mm_per_px: float) -> None:
    text = str(item.get('text', '')).strip()
    if not text:
        return
    x_mm = float(item.get('x', 0)) * mm_per_px
    y_mm = height_mm - float(item.get('y', 0)) * mm_per_px
    box_w_mm = max(float(item.get('w', 0)) * mm_per_px, 2.0)
    box_h_mm = max(float(item.get('h', 0)) * mm_per_px, 1.2)
    text_height = max(min(box_h_mm * 0.78, 6.0), 1.8)
    if len(text) > 24 or box_w_mm > 24:
        mtext = msp.add_mtext(text, dxfattribs={'layer': layer, 'char_height': text_height})
        mtext.dxf.insert = (x_mm, y_mm)
        mtext.dxf.width = max(box_w_mm, text_height * 6)
    else:
        entity = msp.add_text(text, dxfattribs={'layer': layer, 'height': text_height})
        entity.set_placement((x_mm, y_mm), align=TextEntityAlignment.LEFT)


def _add_title_block_fallback(msp, boxes: list[dict[str, Any]], height_mm: float, mm_per_px: float) -> None:
    for box in boxes:
        x = float(box['x']) * mm_per_px
        y_top = height_mm - float(box['y']) * mm_per_px
        w = float(box['w']) * mm_per_px
        h = float(box['h']) * mm_per_px
        pts = [(x, y_top), (x + w, y_top), (x + w, y_top - h), (x, y_top - h), (x, y_top)]
        msp.add_lwpolyline(pts, dxfattribs={'layer': 'ROTULO', 'closed': True})
        row1 = y_top - h * 0.42
        row2 = y_top - h * 0.72
        msp.add_line((x, row1), (x + w, row1), dxfattribs={'layer': 'ROTULO'})
        msp.add_line((x, row2), (x + w, row2), dxfattribs={'layer': 'ROTULO'})
        for frac in (0.22, 0.44, 0.66, 0.82):
            xx = x + w * frac
            msp.add_line((xx, y_top), (xx, row1), dxfattribs={'layer': 'ROTULO'})
        for frac in (0.35, 0.7):
            xx = x + w * frac
            msp.add_line((xx, row1), (xx, row2), dxfattribs={'layer': 'ROTULO'})


def export_to_dxf(
    output_path: Path,
    geometry: dict[str, Any],
    image_width: int,
    image_height: int,
    mm_per_px: float,
    raster_path: Path | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = ezdxf.new(dxfversion='R2010')
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    for layer, color in [('GEOMETRIA', 7), ('CURVAS', 4), ('COTAS', 3), ('TEXTOS', 2), ('ROTULO', 5)]:
        if layer not in doc.layers:
            doc.layers.add(layer, color=color)

    height_mm = image_height * mm_per_px
    support_map = _load_support_map(raster_path)
    documental = _documental_regions_from_geometry(geometry, image_width, image_height)
    inferred_doc_boxes = _detect_document_boxes(support_map, image_width, image_height)
    if inferred_doc_boxes:
        documental.extend(inferred_doc_boxes)
    text_items = geometry.get('texts', []) or []
    source_lines = list(geometry.get('lines', []))
    source_lines.extend(_extract_supported_hv_lines(support_map, image_width, image_height))
    lines = _sanitize_lines_for_dxf(source_lines, documental, text_items, image_width, image_height, support_map=support_map)
    dimension_lines = geometry.get('dimension_lines', []) or []
    cota_texts = geometry.get('cota_texts', []) or []

    for line in lines:
        layer = _classify_line_layer(line, dimension_lines, cota_texts, documental)
        msp.add_line(
            (line['x1'] * mm_per_px, height_mm - line['y1'] * mm_per_px),
            (line['x2'] * mm_per_px, height_mm - line['y2'] * mm_per_px),
            dxfattribs={'layer': layer},
        )

    for poly in geometry.get('polylines', []):
        if len(poly) < 2:
            continue
        minx, miny, maxx, maxy = _poly_bbox(poly)
        region_hit = any(not (maxx < r['x'] or minx > r['x'] + r['w'] or maxy < r['y'] or miny > r['y'] + r['h']) for r in documental)
        width = maxx - minx
        height = maxy - miny
        if region_hit and width < image_width * 0.12 and height < image_height * 0.12:
            continue
        pts = [(pt['x'] * mm_per_px, height_mm - pt['y'] * mm_per_px) for pt in poly]
        closed = len(poly) >= 3 and (poly[0]['x'], poly[0]['y']) == (poly[-1]['x'], poly[-1]['y'])
        layer = 'CURVAS' if _poly_is_curve_like(poly) else ('ROTULO' if region_hit else 'GEOMETRIA')
        msp.add_lwpolyline(pts, dxfattribs={'layer': layer, 'closed': closed})

    for line in dimension_lines:
        if _line_length(line) < 10:
            continue
        if _line_support_ratio(line, support_map) < 0.22:
            continue
        msp.add_line(
            (line['x1'] * mm_per_px, height_mm - line['y1'] * mm_per_px),
            (line['x2'] * mm_per_px, height_mm - line['y2'] * mm_per_px),
            dxfattribs={'layer': 'COTAS'},
        )

    for item in geometry.get('cota_texts', []):
        _add_text_item(msp, item, 'COTAS', height_mm, mm_per_px)
    for item in geometry.get('general_texts', []):
        _add_text_item(msp, item, 'TEXTOS', height_mm, mm_per_px)
    for item in geometry.get('rotulo_texts', []):
        _add_text_item(msp, item, 'ROTULO', height_mm, mm_per_px)

    title_boxes = geometry.get('title_blocks', []) or inferred_doc_boxes
    _add_title_block_fallback(msp, title_boxes, height_mm, mm_per_px)

    doc.saveas(output_path)
    return output_path


def _add_point(msp, x_px: float, y_px: float, height_mm: float, mm_per_px: float) -> None:
    msp.add_point((x_px * mm_per_px, height_mm - y_px * mm_per_px), dxfattribs={'layer': 'PUNTOS'})


def _sample_geometry(msp, geometry: dict[str, Any], height_mm: float, mm_per_px: float, step_px: float) -> int:
    count = 0

    def sample_line(x1: float, y1: float, x2: float, y2: float, local_step: float | None = None):
        nonlocal count
        step = max(local_step or step_px, 1.4)
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(int(dist / step), 1)
        for i in range(steps + 1):
            t = i / steps
            _add_point(msp, x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, height_mm, mm_per_px)
            count += 1

    for line in geometry.get('lines', []):
        sample_line(float(line.get('x1', 0)), float(line.get('y1', 0)), float(line.get('x2', 0)), float(line.get('y2', 0)))
    for line in geometry.get('dimension_lines', []):
        sample_line(float(line.get('x1', 0)), float(line.get('y1', 0)), float(line.get('x2', 0)), float(line.get('y2', 0)), local_step=max(1.4, step_px * 0.4))
    for poly in geometry.get('polylines', []):
        if len(poly) < 2:
            continue
        local_step = max(1.4, step_px * 0.4) if _poly_is_curve_like(poly) else step_px
        for p1, p2 in zip(poly[:-1], poly[1:]):
            sample_line(float(p1.get('x', 0)), float(p1.get('y', 0)), float(p2.get('x', 0)), float(p2.get('y', 0)), local_step=local_step)

    return count


def _sample_raster(msp, raster_path: Path, image_width: int, image_height: int, height_mm: float, mm_per_px: float, geometry: dict[str, Any] | None = None, step_px: float = 4.0) -> int:
    count = 0
    geometry = geometry or {}
    support_map = _load_support_map(raster_path)
    if support_map is None:
        return 0
    documental = _documental_regions_from_geometry(geometry, image_width, image_height)
    inferred_doc_boxes = _detect_document_boxes(support_map, image_width, image_height)
    if inferred_doc_boxes:
        documental.extend(inferred_doc_boxes)
    height, width = support_map.shape[:2]
    base_step = max(int(round(step_px)), 2)
    scale_x = image_width / float(width)
    scale_y = image_height / float(height)
    for y in range(0, height, base_step):
        for x in range(0, width, base_step):
            mapped_x = x * scale_x
            mapped_y = y * scale_y
            local_step = base_step
            documental_hit = False
            for region in documental:
                if region['x'] <= mapped_x <= region['x'] + region['w'] and region['y'] <= mapped_y <= region['y'] + region['h']:
                    documental_hit = True
                    local_step = max(2, base_step // 2)
                    break
            thresh_hit = support_map[y, x] > 0
            if thresh_hit:
                _add_point(msp, mapped_x, mapped_y, height_mm, mm_per_px)
                count += 1
                neigh = [(local_step, 0), (0, local_step), (local_step, local_step), (-local_step, local_step)]
                if documental_hit:
                    neigh += [(2 * local_step, 0), (0, 2 * local_step), (2 * local_step, local_step)]
                for dx, dy in neigh:
                    xn = max(0, min(width - 1, x + dx))
                    yn = max(0, min(height - 1, y + dy))
                    if support_map[yn, xn] > 0:
                        _add_point(msp, xn * scale_x, yn * scale_y, height_mm, mm_per_px)
                        count += 1
    return count


def export_to_point_cloud_dxf(
    output_path: Path,
    geometry: dict[str, Any],
    image_width: int,
    image_height: int,
    mm_per_px: float,
    raster_path: Path | None = None,
    step_px: float = 4.0,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = ezdxf.new(dxfversion='R2010')
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    if 'PUNTOS' not in doc.layers:
        doc.layers.add('PUNTOS', color=7)

    height_mm = image_height * mm_per_px
    point_count = 0
    if raster_path and raster_path.exists():
        point_count = _sample_raster(msp, raster_path, image_width, image_height, height_mm, mm_per_px, geometry=geometry, step_px=step_px)
    if point_count == 0:
        _sample_geometry(msp, geometry, height_mm, mm_per_px, step_px)

    doc.saveas(output_path)
    return output_path
