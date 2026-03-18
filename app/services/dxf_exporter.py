from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import ezdxf
from ezdxf.enums import TextEntityAlignment
from PIL import Image


def _add_text_item(msp, item: dict[str, Any], layer: str, height_mm: float, mm_per_px: float) -> None:
    text = str(item.get('text', '')).strip()
    if not text:
        return
    x_mm = float(item.get('x', 0)) * mm_per_px
    y_mm = height_mm - float(item.get('y', 0)) * mm_per_px
    box_h_mm = max(float(item.get('h', 0)) * mm_per_px, 1.0)
    text_height = max(min(box_h_mm * 0.72, 6.0), 1.8)
    entity = msp.add_text(text, dxfattribs={'layer': layer, 'height': text_height})
    entity.set_placement((x_mm, y_mm), align=TextEntityAlignment.LEFT)


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

    for line in geometry.get('lines', []):
        msp.add_line(
            (line['x1'] * mm_per_px, height_mm - line['y1'] * mm_per_px),
            (line['x2'] * mm_per_px, height_mm - line['y2'] * mm_per_px),
            dxfattribs={'layer': 'GEOMETRIA'},
        )

    for poly in geometry.get('polylines', []):
        if len(poly) < 2:
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


def _sample_raster(msp, raster_path: Path, image_width: int, image_height: int, height_mm: float, mm_per_px: float) -> int:
    count = 0
    with Image.open(raster_path) as image:
        gray = image.convert('L')
        width, height = gray.size
        if width <= 0 or height <= 0:
            return 0
        step = max(int(round(max(width, height) / 450)), 2)
        black_threshold = 200
        px = gray.load()
        scale_x = image_width / float(width)
        scale_y = image_height / float(height)
        for y in range(0, height, step):
            for x in range(0, width, step):
                if px[x, y] < black_threshold:
                    _add_point(msp, x * scale_x, y * scale_y, height_mm, mm_per_px)
                    count += 1
    return count


def export_to_point_cloud_dxf(
    output_path: Path,
    geometry: dict[str, Any],
    image_width: int,
    image_height: int,
    mm_per_px: float,
    raster_path: Path | None = None,
    step_px: float = 14.0,
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
        point_count = _sample_raster(msp, raster_path, image_width, image_height, height_mm, mm_per_px)
    if point_count == 0:
        _sample_geometry(msp, geometry, height_mm, mm_per_px, step_px)

    doc.saveas(output_path)
    return output_path
