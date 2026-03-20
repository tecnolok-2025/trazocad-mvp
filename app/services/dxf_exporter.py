from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import ezdxf
from ezdxf.enums import TextEntityAlignment
from PIL import Image


def _documental_regions_from_geometry(geometry: dict[str, Any], image_width: int, image_height: int) -> list[dict[str, float]]:
    regions = [dict(box) for box in geometry.get('title_blocks', [])]
    if not regions:
        regions.append({'x': image_width * 0.72, 'y': image_height * 0.68, 'w': image_width * 0.24, 'h': image_height * 0.24})
    # franja de notas/referencias y título inferior
    regions.append({'x': image_width * 0.68, 'y': image_height * 0.58, 'w': image_width * 0.28, 'h': image_height * 0.22})
    regions.append({'x': image_width * 0.34, 'y': image_height * 0.88, 'w': image_width * 0.34, 'h': image_height * 0.08})
    return regions


def _line_intersects_region(line: dict[str, Any], region: dict[str, float]) -> bool:
    x1, y1, x2, y2 = float(line.get('x1', 0)), float(line.get('y1', 0)), float(line.get('x2', 0)), float(line.get('y2', 0))
    minx, maxx = min(x1, x2), max(x1, x2)
    miny, maxy = min(y1, y2), max(y1, y2)
    rx1, ry1 = float(region['x']), float(region['y'])
    rx2, ry2 = rx1 + float(region['w']), ry1 + float(region['h'])
    return not (maxx < rx1 or minx > rx2 or maxy < ry1 or miny > ry2)


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


def _sanitize_lines_for_dxf(lines: list[dict[str, Any]], documental: list[dict[str, float]], image_width: int, image_height: int) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    img_long = max(image_width, image_height)
    for line in lines:
        x1, y1, x2, y2 = float(line.get('x1', 0)), float(line.get('y1', 0)), float(line.get('x2', 0)), float(line.get('y2', 0))
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 10:
            continue
        overlap = max((_line_overlap_ratio(line, r) for r in documental), default=0.0)
        if overlap > 0.35 and length < img_long * 0.22:
            continue
        if overlap > 0.05 and not _is_axis_like(line) and length < img_long * 0.12:
            continue
        key = _normalized_line_key(line)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
    return cleaned


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


def _add_title_block_fallback(msp, geometry: dict[str, Any], height_mm: float, mm_per_px: float) -> None:
    for box in geometry.get('title_blocks', []) or []:
        x1 = float(box['x']) * mm_per_px
        y1 = height_mm - float(box['y']) * mm_per_px
        x2 = float(box['x'] + box['w']) * mm_per_px
        y2 = height_mm - float(box['y'] + box['h']) * mm_per_px
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        msp.add_lwpolyline(pts, dxfattribs={'layer': 'ROTULO', 'closed': True})


def export_to_dxf(
    output_path: Path,
    geometry: dict[str, Any],
    image_width: int,
    image_height: int,
    mm_per_px: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = ezdxf.new(dxfversion='R2010')
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    for layer, color in [('GEOMETRIA', 7), ('COTAS', 3), ('TEXTOS', 2), ('ROTULO', 5)]:
        if layer not in doc.layers:
            doc.layers.add(layer, color=color)

    height_mm = image_height * mm_per_px

    documental = _documental_regions_from_geometry(geometry, image_width, image_height)
    lines = _sanitize_lines_for_dxf(geometry.get('lines', []), documental, image_width, image_height)
    for line in lines:
        msp.add_line(
            (line['x1'] * mm_per_px, height_mm - line['y1'] * mm_per_px),
            (line['x2'] * mm_per_px, height_mm - line['y2'] * mm_per_px),
            dxfattribs={'layer': 'GEOMETRIA'},
        )

    for poly in geometry.get('polylines', []):
        if len(poly) < 2:
            continue
        xs = [float(pt['x']) for pt in poly]
        ys = [float(pt['y']) for pt in poly]
        region_hit = any(not (max(xs) < r['x'] or min(xs) > r['x'] + r['w'] or max(ys) < r['y'] or min(ys) > r['y'] + r['h']) for r in documental)
        if region_hit and (max(xs) - min(xs)) < image_width * 0.08 and (max(ys) - min(ys)) < image_height * 0.08:
            continue
        pts = [(pt['x'] * mm_per_px, height_mm - pt['y'] * mm_per_px) for pt in poly]
        closed = len(poly) >= 3 and (poly[0]['x'], poly[0]['y']) == (poly[-1]['x'], poly[-1]['y'])
        msp.add_lwpolyline(pts, dxfattribs={'layer': 'GEOMETRIA', 'closed': closed})

    for item in geometry.get('cota_texts', []):
        _add_text_item(msp, item, 'COTAS', height_mm, mm_per_px)
    for item in geometry.get('general_texts', []):
        _add_text_item(msp, item, 'TEXTOS', height_mm, mm_per_px)
    for item in geometry.get('rotulo_texts', []):
        _add_text_item(msp, item, 'ROTULO', height_mm, mm_per_px)

    _add_title_block_fallback(msp, geometry, height_mm, mm_per_px)

    doc.saveas(output_path)
    return output_path


def _add_point(msp, x_px: float, y_px: float, height_mm: float, mm_per_px: float) -> None:
    msp.add_point((x_px * mm_per_px, height_mm - y_px * mm_per_px), dxfattribs={'layer': 'PUNTOS'})


def _sample_geometry(msp, geometry: dict[str, Any], height_mm: float, mm_per_px: float, step_px: float) -> int:
    count = 0

    def sample_line(x1: float, y1: float, x2: float, y2: float):
        nonlocal count
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(int(dist / max(step_px, 1.0)), 1)
        for i in range(steps + 1):
            t = i / steps
            _add_point(msp, x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, height_mm, mm_per_px)
            count += 1

    for line in geometry.get('lines', []):
        sample_line(float(line.get('x1', 0)), float(line.get('y1', 0)), float(line.get('x2', 0)), float(line.get('y2', 0)))

    for poly in geometry.get('polylines', []):
        if len(poly) < 2:
            continue
        for p1, p2 in zip(poly[:-1], poly[1:]):
            sample_line(float(p1.get('x', 0)), float(p1.get('y', 0)), float(p2.get('x', 0)), float(p2.get('y', 0)))

    return count


def _sample_raster(msp, raster_path: Path, image_width: int, image_height: int, height_mm: float, mm_per_px: float, geometry: dict[str, Any] | None = None, step_px: float = 8.0) -> int:
    count = 0
    geometry = geometry or {}
    documental = _documental_regions_from_geometry(geometry, image_width, image_height)
    with Image.open(raster_path) as image:
        gray = image.convert('L')
        width, height = gray.size
        if width <= 0 or height <= 0:
            return 0
        base_step = max(int(round(step_px)), 2)
        black_threshold = 225
        px = gray.load()
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
                        local_step = max(2, base_step // 3)
                        break
                if px[x, y] < black_threshold:
                    _add_point(msp, mapped_x, mapped_y, height_mm, mm_per_px)
                    count += 1
                    neigh = [(local_step, 0), (0, local_step)]
                    if documental_hit:
                        neigh += [(local_step, local_step), (-local_step, local_step)]
                    for dx, dy in neigh:
                        xn = max(0, min(width - 1, x + dx))
                        yn = max(0, min(height - 1, y + dy))
                        if px[xn, yn] < black_threshold:
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
    step_px: float = 9.0,
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
