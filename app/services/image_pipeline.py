from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from collections import defaultdict
import math
import os
import time
import resource

import cv2
import numpy as np

SHEET_SIZES_MM = {
    "A4": (210, 297),
    "A3": (297, 420),
    "A2": (420, 594),
    "A1": (594, 841),
    "CUSTOM": (0, 0),
}
ORIENTATION_LABELS = {"AUTO": "automatica", "VERTICAL": "vertical", "HORIZONTAL": "apaisada"}


@dataclass
class PipelineResult:
    image_width_px: int
    image_height_px: int
    processed_width_px: int
    processed_height_px: int
    threshold_ratio: float
    detected_line_count: int
    detected_contour_count: int
    closed_shapes_count: int
    detected_text_regions: int
    recognized_text_count: int
    estimated_scale_mm_per_px: float
    sheet_width_mm: int
    sheet_height_mm: int
    confidence_score: float
    reference_mode: str
    inferred_dpi: int
    quality_band: str
    page_detected: bool
    auto_upscaled: bool
    deskew_angle_deg: float
    assumptions: list[str]
    warnings: list[str]
    insights: list[str]
    quality_metrics: dict[str, Any]
    geometry: dict[str, Any]
    output_files: dict[str, str]
    sheet_orientation: str
    preserved_aspect_ratio: bool
    document_orientation: str
    calibration_summary: str
    calibration_distance_mm: float | None
    calibration_pixel_span: float | None
    ocr_engine: str
    detected_dimension_count: int
    detected_title_block_count: int
    detected_label_count: int
    detected_arrow_count: int
    review_text_count: int
    line_segments_raw_count: int
    line_segments_after_cleanup_count: int
    duplicate_lines_removed_count: int
    detected_symbol_count: int
    electrical_symbol_count: int
    sanitary_symbol_count: int
    mechanical_symbol_count: int
    discipline_guess: str
    suggested_block_count: int
    discipline_rule_count: int
    precision_index: float
    calibration_reliability: str
    geometry_stability: str
    text_separation_quality: str
    recommended_precision_action: str
    precision_observations: list[str]
    precision_class: str
    expected_positional_error_mm: float
    suggested_linear_tolerance_mm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_image(input_path: Path) -> np.ndarray:
    raw = np.fromfile(str(input_path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo leer la imagen de entrada.")
    return image


def _save_image(path: Path, image: np.ndarray) -> None:
    _ensure_parent(path)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"No se pudo guardar la imagen {path.name}.")
    path.write_bytes(encoded.tobytes())


def _resize_for_report(image: np.ndarray, max_width: int | None = None) -> np.ndarray:
    max_width = max_width or int(os.getenv('TRAZOCAD_REPORT_MAX_WIDTH', '1400'))
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    ratio = max_width / float(width)
    return cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_AREA)


def _apply_memory_budget(image: np.ndarray) -> np.ndarray:
    max_dimension = int(os.getenv('TRAZOCAD_MAX_DIMENSION_PX', '2200'))
    max_megapixels = float(os.getenv('TRAZOCAD_MAX_MEGAPIXELS', '4.2'))
    h, w = image.shape[:2]
    scale = 1.0
    longest = max(h, w)
    if longest > max_dimension:
        scale = min(scale, max_dimension / float(longest))
    megapixels = (h * w) / 1_000_000.0
    if megapixels > max_megapixels:
        scale = min(scale, (max_megapixels / megapixels) ** 0.5)
    if scale >= 0.999:
        return image
    new_w = max(640, int(round(w * scale)))
    new_h = max(640, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _parse_user_directives(notes: str) -> dict[str, bool]:
    text = (notes or '').lower()
    return {
        'prioritize_fidelity': any(token in text for token in ['[fidelity]', 'fidelidad', 'preservar plano completo', 'preservar el plano', 'priorizar fidelidad visual']),
        'preserve_title_block': any(token in text for token in ['[title]', 'rotulo', 'rótulo', 'referencias', 'cuadro de plano']),
        'prioritize_dimensions': any(token in text for token in ['[dimensions]', 'cota', 'cotas', 'ejes', 'texto técnico', 'textos técnicos']),
        'reconstruct_perimeters': any(token in text for token in ['[reconstruct]', '[geometry]', 'reconstruir perímetros', 'reconstruir perimetros', 'arcos', 'contornos', 'cerrar cortes']),
        'preserve_dashed': any(token in text for token in ['[dashed]', 'puntead', 'trazo fino', 'trazos finos', 'líneas de cota', 'lineas de cota']),
        'force_ocr': '[ocr]' in text,
        'safe_mode': '[safe]' in text or '[ocr]' not in text,
    }


def _should_run_ocr(directives: dict[str, bool] | None = None) -> bool:
    directives = directives or {}
    env_default = os.getenv('TRAZOCAD_ENABLE_OCR_BY_DEFAULT', '0').strip().lower() in {'1', 'true', 'yes', 'si'}
    if directives.get('force_ocr'):
        return True
    return env_default


def _blend_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.min(a, b)


def _memory_usage_mb() -> float:
    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux ru_maxrss is in KiB.
        return float(rss_kb) / 1024.0
    except Exception:
        return 0.0


def _adaptive_runtime_profile(image_shape: tuple[int, int, int], directives: dict[str, bool] | None = None) -> dict[str, Any]:
    directives = directives or {}
    h, w = image_shape[:2]
    megapixels = (h * w) / 1_000_000.0
    memory_mb = _memory_usage_mb()
    safety_reserve_mb = float(os.getenv('TRAZOCAD_MEMORY_SAFETY_RESERVE_MB', '60'))
    soft_limit_mb = float(os.getenv('TRAZOCAD_MEMORY_SOFT_LIMIT_MB', '410'))
    hard_limit_mb = float(os.getenv('TRAZOCAD_MEMORY_HARD_LIMIT_MB', '470'))
    pressure = 'normal'
    if memory_mb >= hard_limit_mb - safety_reserve_mb:
        pressure = 'critical'
    elif memory_mb >= soft_limit_mb - safety_reserve_mb:
        pressure = 'high'
    elif megapixels > 3.5:
        pressure = 'medium'
    # Instancias chicas: jamás forzar OCR global.
    ocr_allowed = directives.get('force_ocr', False) and pressure != 'critical'
    if directives.get('safe_mode') and pressure in {'high', 'critical'}:
        ocr_allowed = False
    point_cloud_step = 6.0 if pressure == 'normal' else 8.0 if pressure == 'medium' else 10.0 if pressure == 'high' else 12.0
    return {
        'memory_mb': round(memory_mb, 1),
        'memory_pressure': pressure,
        'ocr_allowed': ocr_allowed,
        'ocr_regions_multiplier': 1.0 if pressure == 'normal' else 0.8 if pressure == 'medium' else 0.55 if pressure == 'high' else 0.35,
        'ocr_pixel_fraction_multiplier': 1.0 if pressure == 'normal' else 0.8 if pressure == 'medium' else 0.55 if pressure == 'high' else 0.4,
        'point_cloud_step_px': point_cloud_step,
        'safety_reserve_mb': safety_reserve_mb,
    }


def _region_intersection_ratio(line_box: tuple[int,int,int,int], region: dict[str,int]) -> float:
    lx1, ly1, lx2, ly2 = line_box
    rx1, ry1 = int(region.get('x', 0)), int(region.get('y', 0))
    rx2, ry2 = rx1 + int(region.get('w', 0)), ry1 + int(region.get('h', 0))
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max((lx2 - lx1) * (ly2 - ly1), 1)
    return inter / float(area)


def _sanitize_geometry(geometry: dict[str, Any], documental_regions: list[dict[str, int]], image_shape: tuple[int, int, int], directives: dict[str, bool] | None = None) -> dict[str, Any]:
    directives = directives or {}
    h, w = image_shape[:2]
    clean_lines: list[dict[str, int]] = []
    removed = 0
    for line in geometry.get('lines', []):
        x1, y1 = int(line['x1']), int(line['y1'])
        x2, y2 = int(line['x2']), int(line['y2'])
        length = math.hypot(x2 - x1, y2 - y1)
        box = (min(x1, x2), min(y1, y2), max(x1, x2) + 1, max(y1, y2) + 1)
        overlap = max((_region_intersection_ratio(box, region) for region in documental_regions), default=0.0)
        if overlap > 0.55 and length < max(w, h) * 0.18:
            removed += 1
            continue
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        axis_like = dx < 2 or dy < 2 or (min(dx, dy) / max(dx, dy, 1.0) < 0.08)
        if overlap > 0.08 and not axis_like and length < max(w, h) * 0.10:
            removed += 1
            continue
        if length < 10:
            removed += 1
            continue
        clean_lines.append(line)
    clean_polys: list[list[dict[str, int]]] = []
    for poly in geometry.get('polylines', []):
        if len(poly) < 2:
            continue
        xs = [int(pt['x']) for pt in poly]
        ys = [int(pt['y']) for pt in poly]
        box = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
        overlap = max((_region_intersection_ratio(box, region) for region in documental_regions), default=0.0)
        if overlap > 0.75 and (max(xs) - min(xs)) * (max(ys) - min(ys)) < (w * h) * 0.03:
            removed += 1
            continue
        if len(poly) < 3 and (max(xs) - min(xs) + max(ys) - min(ys)) < max(w, h) * 0.03:
            removed += 1
            continue
        clean_polys.append(poly)
    geometry['lines'] = clean_lines
    geometry['polylines'] = clean_polys
    geometry['removed_documental_vectors'] = removed
    return geometry


def _ocr_budgets(directives: dict[str, bool] | None = None, runtime_profile: dict[str, Any] | None = None) -> dict[str, float | int]:
    directives = directives or {}
    runtime_profile = runtime_profile or {}
    wants_documental = directives.get('preserve_title_block') or directives.get('prioritize_dimensions')
    wants_heavy = wants_documental and directives.get('prioritize_fidelity')
    mult = float(runtime_profile.get('ocr_regions_multiplier', 1.0))
    pix_mult = float(runtime_profile.get('ocr_pixel_fraction_multiplier', 1.0))
    base_boxes = 28 if wants_heavy else 18 if wants_documental else 10
    base_fraction = 0.24 if wants_heavy else 0.16 if wants_documental else 0.10
    pressure = str(runtime_profile.get('memory_pressure', 'normal'))
    base_time = 20 if wants_heavy else 12 if wants_documental else 7
    if pressure == 'high':
        base_time = min(base_time, 9)
    elif pressure == 'critical':
        base_time = min(base_time, 5)
    return {
        'max_boxes': max(4, int(round(base_boxes * mult))),
        'pixel_fraction': max(0.05, base_fraction * pix_mult),
        'time_budget_s': float(os.getenv('TRAZOCAD_OCR_TIME_BUDGET_S', str(base_time))),
        'max_roi_long_side': 520 if wants_heavy else 420 if wants_documental else 320,
    }


def _reconstruct_structural_gaps(binary: np.ndarray, directives: dict[str, bool] | None = None) -> np.ndarray:
    directives = directives or {}
    work = binary.copy()
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (17 if directives.get('reconstruct_perimeters') else 11, 1))
    vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 17 if directives.get('reconstruct_perimeters') else 11))
    diag = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    h_closed = cv2.morphologyEx(work, cv2.MORPH_CLOSE, horiz, iterations=1)
    v_closed = cv2.morphologyEx(work, cv2.MORPH_CLOSE, vert, iterations=1)
    d_closed = cv2.morphologyEx(work, cv2.MORPH_CLOSE, diag, iterations=1)
    merged = cv2.bitwise_or(work, h_closed)
    merged = cv2.bitwise_or(merged, v_closed)
    merged = cv2.bitwise_or(merged, d_closed)
    return merged


def _restore_region_from_gray(base_gray: np.ndarray, original_gray: np.ndarray, box: dict[str, int], pad: int = 10) -> np.ndarray:
    restored = base_gray.copy()
    h, w = restored.shape[:2]
    x1 = max(0, int(box['x']) - pad)
    y1 = max(0, int(box['y']) - pad)
    x2 = min(w, int(box['x'] + box['w']) + pad)
    y2 = min(h, int(box['y'] + box['h']) + pad)
    if x2 <= x1 or y2 <= y1:
        return restored
    restored[y1:y2, x1:x2] = _blend_min(restored[y1:y2, x1:x2], original_gray[y1:y2, x1:x2])
    return restored


def _documental_regions(image_shape: tuple[int, int], title_blocks: list[dict[str, int]] | None = None) -> list[dict[str, int]]:
    h, w = image_shape[:2]
    title_blocks = title_blocks or []
    regions: list[dict[str, int]] = []
    # Zona documental: rótulo, referencias/notas sobre rótulo y título inferior del plano.
    if title_blocks:
        for box in title_blocks:
            regions.append(dict(box))
            regions.append({
                'x': max(0, int(box['x'] - w * 0.02)),
                'y': max(0, int(box['y'] - h * 0.14)),
                'w': min(w - max(0, int(box['x'] - w * 0.02)), int(box['w'] + w * 0.04)),
                'h': min(h - max(0, int(box['y'] - h * 0.14)), int(box['h'] + h * 0.16)),
            })
    else:
        regions.append({'x': int(w * 0.72), 'y': int(h * 0.68), 'w': int(w * 0.24), 'h': int(h * 0.24)})
        regions.append({'x': int(w * 0.70), 'y': int(h * 0.58), 'w': int(w * 0.26), 'h': int(h * 0.16)})
    regions.append({'x': int(w * 0.34), 'y': int(h * 0.88), 'w': int(w * 0.34), 'h': int(h * 0.08)})
    return regions


def _enhance_documental_roi(roi: np.ndarray) -> np.ndarray:
    if roi.size == 0:
        return roi
    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(6, 6))
    enhanced = clahe.apply(roi)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 0.35)
    return cv2.addWeighted(enhanced, 1.06, blur, -0.06, 0)


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _detect_document(image: np.ndarray) -> tuple[np.ndarray | None, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    area_image = image.shape[0] * image.shape[1]
    best = None
    best_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_image * 0.18:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx.reshape(4, 2)
            best_area = area
    if best is None:
        return None, 0.0

    rect = _order_points(best.astype("float32"))
    angle = math.degrees(math.atan2(rect[1][1] - rect[0][1], rect[1][0] - rect[0][0]))
    return rect, angle


def _warp_document(image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rect = _order_points(corners)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = max(int(round(max(width_a, width_b))), 1)
    max_height = max(int(round(max(height_a, height_b))), 1)
    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped, matrix


def _auto_upscale(image: np.ndarray, target_min_side: int | None = None) -> tuple[np.ndarray, bool, float]:
    if target_min_side is None:
        target_min_side = int(os.getenv('TRAZOCAD_TARGET_MIN_SIDE', '1600'))
    h, w = image.shape[:2]
    min_side = min(h, w)
    if min_side >= target_min_side:
        return image, False, 1.0
    scale = target_min_side / float(max(min_side, 1))
    if scale < 1.25:
        return image, False, 1.0
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), True, scale


def _normalize_background(gray: np.ndarray) -> np.ndarray:
    sigma = max(gray.shape[:2]) / 32.0
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = cv2.divide(gray, background, scale=255)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)


def _enhance_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = _normalize_background(gray)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    contrast = clahe.apply(normalized)
    # En v71 priorizamos preservar texto fino y rótulo sobre "limpiar" de más.
    # Cambiamos NLM por un filtrado mucho más liviano y menos destructivo.
    denoised = cv2.GaussianBlur(contrast, (0, 0), 0.45)
    sharpened = cv2.addWeighted(contrast, 1.10, denoised, -0.10, 0)
    whitened = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return sharpened, normalized, whitened


def _build_binary(gray: np.ndarray) -> np.ndarray:
    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 9)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.bitwise_or(adapt, otsu)
    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_small, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=1)
    return binary


def _repair_broken_traces(binary: np.ndarray, directives: dict[str, bool] | None = None) -> np.ndarray:
    directives = directives or {}
    # En v70 la reparación es conservadora por defecto para evitar empastado.
    if not directives.get('reconstruct_perimeters') and not directives.get('preserve_dashed'):
        return binary.copy()
    repaired = binary.copy()
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
    ]
    if directives.get('reconstruct_perimeters'):
        kernels.extend([
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        ])
    if directives.get('preserve_dashed'):
        kernels.extend([
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)),
        ])
    before_fill = float(cv2.countNonZero(repaired)) / float(max(repaired.size, 1))
    for kernel in kernels:
        candidate = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, kernel, iterations=1)
        candidate_fill = float(cv2.countNonZero(candidate)) / float(max(candidate.size, 1))
        if candidate_fill <= min(0.18, before_fill + 0.012):
            repaired = candidate
            before_fill = candidate_fill
    repaired = cv2.bitwise_or(repaired, binary)
    return repaired


def _reinforce_title_block(binary: np.ndarray, directives: dict[str, bool] | None = None) -> np.ndarray:
    directives = directives or {}
    # En v70 evitamos reforzar el rótulo con cierres agresivos: solo una limpieza mínima si se pidió explícitamente.
    if not directives.get('preserve_title_block'):
        return binary
    h, w = binary.shape[:2]
    y1, x1 = int(h * 0.70), int(w * 0.68)
    roi = binary[y1:h, x1:w].copy()
    if roi.size == 0:
        return binary
    candidate = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    out = binary.copy()
    out[y1:h, x1:w] = cv2.bitwise_or(out[y1:h, x1:w], candidate)
    return out


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    img = binary.copy()
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _detect_text_regions(binary: np.ndarray) -> tuple[list[dict[str, int]], np.ndarray]:
    h, w = binary.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidate_mask = np.zeros_like(binary)
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        if area < 8 or bh < 5 or bw < 2:
            continue
        if bh > h * 0.085 or bw > w * 0.30:
            continue
        aspect = bw / float(max(bh, 1))
        fill = area / float(max(bw * bh, 1))
        if 0.15 <= fill <= 0.90 and 0.15 <= aspect <= 12.0:
            candidate_mask[labels == idx] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(14, w // 90), max(3, h // 220)))
    grouped = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    grouped = cv2.dilate(grouped, kernel, iterations=1)

    boxes: list[dict[str, int]] = []
    text_mask = np.zeros_like(binary)
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < 60:
            continue
        if bh < 8 or bh > h * 0.10:
            continue
        if bw < 8 or bw > w * 0.55:
            continue
        aspect = bw / float(max(bh, 1))
        if aspect < 0.35 or aspect > 40:
            continue
        cv2.rectangle(text_mask, (x, y), (x + bw, y + bh), 255, -1)
        boxes.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)})

    boxes.sort(key=lambda item: (item["y"], item["x"]))
    return boxes, text_mask


def _dedupe_lines(lines: list[dict[str, int]], tolerance: int = 5) -> list[dict[str, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[dict[str, int]] = []
    for item in lines:
        x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
        if (x1, y1) > (x2, y2):
            x1, y1, x2, y2 = x2, y2, x1, y1
        key = (round(x1 / tolerance), round(y1 / tolerance), round(x2 / tolerance), round(y2 / tolerance))
        if key in seen:
            continue
        seen.add(key)
        unique.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return unique


def _line_length(item: dict[str, int]) -> float:
    return float(math.hypot(item["x2"] - item["x1"], item["y2"] - item["y1"]))


def _line_orientation(item: dict[str, int], diagonal_tolerance_deg: float = 8.0) -> str:
    dx = item["x2"] - item["x1"]
    dy = item["y2"] - item["y1"]
    angle = abs(math.degrees(math.atan2(dy, dx)))
    angle = min(angle, abs(180 - angle))
    if angle <= diagonal_tolerance_deg:
        return "horizontal"
    if abs(angle - 90) <= diagonal_tolerance_deg:
        return "vertical"
    return "diagonal"


def _line_overlaps_text(item: dict[str, int], text_boxes: list[dict[str, int]], padding: int = 4) -> bool:
    x1, y1 = item["x1"], item["y1"]
    x2, y2 = item["x2"], item["y2"]
    lx1, lx2 = min(x1, x2), max(x1, x2)
    ly1, ly2 = min(y1, y2), max(y1, y2)
    for box in text_boxes:
        bx1 = box["x"] - padding
        by1 = box["y"] - padding
        bx2 = box["x"] + box["w"] + padding
        by2 = box["y"] + box["h"] + padding
        if lx2 < bx1 or lx1 > bx2 or ly2 < by1 or ly1 > by2:
            continue
        return True
    return False


def _extract_axis_lines(graphics_binary: np.ndarray) -> list[dict[str, int]]:
    h, w = graphics_binary.shape[:2]
    min_len = max(28, int(min(h, w) * 0.035))
    out: list[dict[str, int]] = []
    kernels = {
        "horizontal": cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 40), 1)),
        "vertical": cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 40))),
    }
    for kind, kernel in kernels.items():
        isolated = cv2.morphologyEx(graphics_binary, cv2.MORPH_OPEN, kernel, iterations=1)
        lines = cv2.HoughLinesP(isolated, 1, np.pi / 180, threshold=max(30, min_len // 2), minLineLength=min_len, maxLineGap=max(8, min_len // 5))
        if lines is None:
            continue
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, line)
            if kind == "horizontal":
                y = int(round((y1 + y2) / 2))
                out.append({"x1": min(x1, x2), "y1": y, "x2": max(x1, x2), "y2": y})
            else:
                x = int(round((x1 + x2) / 2))
                out.append({"x1": x, "y1": min(y1, y2), "x2": x, "y2": max(y1, y2)})
    return out


def _merge_collinear_lines(lines: list[dict[str, int]], tolerance: int = 6, gap_tolerance: int = 24) -> list[dict[str, int]]:
    buckets: dict[tuple[str, int], list[dict[str, int]]] = defaultdict(list)
    diagonals: list[dict[str, int]] = []
    for line in lines:
        orient = _line_orientation(line)
        if orient == "horizontal":
            buckets[(orient, round(((line["y1"] + line["y2"]) / 2) / tolerance))].append(line)
        elif orient == "vertical":
            buckets[(orient, round(((line["x1"] + line["x2"]) / 2) / tolerance))].append(line)
        else:
            diagonals.append(line)

    merged: list[dict[str, int]] = []
    for (orient, anchor), items in buckets.items():
        if orient == "horizontal":
            spans = sorted((min(it["x1"], it["x2"]), max(it["x1"], it["x2"]), int(round((it["y1"] + it["y2"]) / 2))) for it in items)
            start, end, y = spans[0]
            for xs, xe, yy in spans[1:]:
                if xs <= end + gap_tolerance:
                    end = max(end, xe)
                    y = int(round((y + yy) / 2))
                else:
                    merged.append({"x1": start, "y1": y, "x2": end, "y2": y})
                    start, end, y = xs, xe, yy
            merged.append({"x1": start, "y1": y, "x2": end, "y2": y})
        else:
            spans = sorted((min(it["y1"], it["y2"]), max(it["y1"], it["y2"]), int(round((it["x1"] + it["x2"]) / 2))) for it in items)
            start, end, x = spans[0]
            for ys, ye, xx in spans[1:]:
                if ys <= end + gap_tolerance:
                    end = max(end, ye)
                    x = int(round((x + xx) / 2))
                else:
                    merged.append({"x1": x, "y1": start, "x2": x, "y2": end})
                    start, end, x = ys, ye, xx
            merged.append({"x1": x, "y1": start, "x2": x, "y2": end})

    diagonals = _dedupe_lines(diagonals, tolerance=max(3, tolerance))
    for line in diagonals:
        if _line_length(line) >= max(24, gap_tolerance * 1.5):
            merged.append(line)
    return _dedupe_lines(merged, tolerance=max(3, tolerance))


def _detect_geometry(graphics_binary: np.ndarray, text_boxes: list[dict[str, int]]) -> dict[str, Any]:
    h, w = graphics_binary.shape[:2]
    skeleton = _skeletonize(graphics_binary)
    line_segments: list[dict[str, int]] = []
    line_segments_raw = 0

    try:
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        detected = detector.detect(skeleton)[0]
    except Exception:
        detected = None

    if detected is not None:
        for item in detected[:, 0, :]:
            x1, y1, x2, y2 = map(int, map(round, item))
            length = math.hypot(x2 - x1, y2 - y1)
            if length < max(18, w * 0.012):
                continue
            line_segments.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            line_segments_raw += 1

    hough = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 180,
        threshold=max(40, w // 14),
        minLineLength=max(24, w // 28),
        maxLineGap=max(10, w // 120),
    )
    if hough is not None:
        for line in hough[:, 0, :]:
            x1, y1, x2, y2 = map(int, line)
            line_segments.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            line_segments_raw += 1

    line_segments.extend(_extract_axis_lines(graphics_binary))
    deduped_segments = _dedupe_lines(line_segments)
    non_text_segments = [item for item in deduped_segments if not _line_overlaps_text(item, text_boxes)]
    line_segments = _merge_collinear_lines(non_text_segments)

    contours, _ = cv2.findContours(graphics_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polylines: list[list[dict[str, int]]] = []
    closed_shapes = 0
    useful_contours = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 3 or bh < 3:
            continue
        if bw <= w * 0.30 and bh <= h * 0.10:
            overlap = 0
            for box in text_boxes:
                ix1 = max(x, box["x"])
                iy1 = max(y, box["y"])
                ix2 = min(x + bw, box["x"] + box["w"])
                iy2 = min(y + bh, box["y"] + box["h"])
                if ix2 > ix1 and iy2 > iy1:
                    overlap = max(overlap, (ix2 - ix1) * (iy2 - iy1))
            if overlap / float(max(bw * bh, 1)) > 0.35:
                continue
        useful_contours += 1
        epsilon = 0.009 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = [{"x": int(p[0][0]), "y": int(p[0][1])} for p in approx]
        if len(pts) >= 3 and cv2.arcLength(approx, True) >= 24:
            polylines.append(pts)
            closed_shapes += 1 if cv2.isContourConvex(approx) else 0

    long_lines = sum(1 for item in line_segments if _line_length(item) >= max(graphics_binary.shape[:2]) * 0.12)
    return {
        "lines": line_segments,
        "polylines": polylines,
        "closed_shapes_count": closed_shapes,
        "raw_contour_count": useful_contours,
        "long_line_count": long_lines,
        "text_boxes": text_boxes,
        "line_segments_raw_count": line_segments_raw,
        "line_segments_after_cleanup_count": len(line_segments),
        "duplicate_lines_removed_count": max(line_segments_raw - len(line_segments), 0),
    }


def _fit_inside_sheet_dimensions(sheet_size: str, sheet_orientation: str, image_width: int, image_height: int) -> tuple[int, int]:
    base_w, base_h = SHEET_SIZES_MM.get(sheet_size, SHEET_SIZES_MM["A3"])
    if base_w == 0 or base_h == 0:
        base_w, base_h = SHEET_SIZES_MM["A3"]
    orientation = sheet_orientation.upper()
    if orientation == "AUTO":
        orientation = "HORIZONTAL" if image_width >= image_height else "VERTICAL"
    return (max(base_w, base_h), min(base_w, base_h)) if orientation == "HORIZONTAL" else (min(base_w, base_h), max(base_w, base_h))


def _compute_quality_metrics(
    binary: np.ndarray,
    graphics_binary: np.ndarray,
    geometry: dict[str, Any],
    original_shape: tuple[int, int, int],
    processed_shape: tuple[int, int, int],
    sheet_size: str,
    sheet_orientation: str,
    recognized_text_count: int,
) -> dict[str, Any]:
    h0, w0 = original_shape[:2]
    hp, wp = processed_shape[:2]
    black_ratio = float(np.count_nonzero(binary)) / float(binary.size)
    graphics_ratio = float(np.count_nonzero(graphics_binary)) / float(max(graphics_binary.size, 1))
    line_density = geometry["long_line_count"] / max((wp * hp) / 1_000_000.0, 0.1)
    sheet_w, sheet_h = _fit_inside_sheet_dimensions(sheet_size, sheet_orientation, wp, hp)
    dpi_x = int(round(wp / (sheet_w / 25.4)))
    dpi_y = int(round(hp / (sheet_h / 25.4)))
    inferred_dpi = int(round((dpi_x + dpi_y) / 2))
    return {
        "original_pixels": int(w0 * h0),
        "processed_pixels": int(wp * hp),
        "black_ratio": round(black_ratio, 4),
        "graphics_ratio": round(graphics_ratio, 4),
        "line_density": round(line_density, 2),
        "inferred_dpi": inferred_dpi,
        "useful_contours": geometry["raw_contour_count"],
        "long_lines": geometry["long_line_count"],
        "text_regions": len(geometry.get("text_boxes", [])),
        "recognized_text_count": recognized_text_count,
        "line_segments_raw_count": geometry.get("line_segments_raw_count", 0),
        "line_segments_after_cleanup_count": geometry.get("line_segments_after_cleanup_count", 0),
        "duplicate_lines_removed_count": geometry.get("duplicate_lines_removed_count", 0),
    }


def _transform_calibration_points(
    points: list[tuple[float, float]] | None,
    matrix: np.ndarray | None,
    upscale_factor: float,
) -> list[tuple[float, float]]:
    if not points or len(points) != 2:
        return []
    arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    if matrix is not None:
        arr = cv2.perspectiveTransform(arr, matrix)
    arr = arr * float(upscale_factor)
    return [(float(item[0][0]), float(item[0][1])) for item in arr]


def _estimate_scale_mm_per_px(
    width_px: int,
    height_px: int,
    sheet_size: str,
    reference_mm: float | None,
    sheet_orientation: str,
    calibration: dict[str, Any] | None,
    transformed_points: list[tuple[float, float]],
) -> tuple[float, str, int, int, str, float | None, float | None]:
    sheet_w, sheet_h = _fit_inside_sheet_dimensions(sheet_size, sheet_orientation, width_px, height_px)

    if calibration and calibration.get("mode") == "TWO_POINT" and len(transformed_points) == 2:
        dx = transformed_points[1][0] - transformed_points[0][0]
        dy = transformed_points[1][1] - transformed_points[0][1]
        pixel_span = float(math.hypot(dx, dy))
        distance_mm = float(calibration.get("distance_mm", 0) or 0)
        if pixel_span > 0 and distance_mm > 0:
            mm_per_px = round(max(distance_mm / pixel_span, 0.0001), 6)
            summary = f"{distance_mm:.2f} mm entre dos puntos ({pixel_span:.1f} px medidos)"
            return mm_per_px, "dos_puntos", sheet_w, sheet_h, summary, distance_mm, pixel_span

    if reference_mm and reference_mm > 0:
        mm_per_px = round(max(reference_mm / max(width_px, height_px), 0.001), 6)
        summary = f"medida global: {reference_mm:.2f} mm sobre dimensión principal"
        return mm_per_px, "cota_referencia_global", sheet_w, sheet_h, summary, reference_mm, None

    mm_per_px = round(max(sheet_w / max(width_px, 1), sheet_h / max(height_px, 1)), 6)
    summary = f"ajuste automático por hoja {sheet_size} {ORIENTATION_LABELS.get(sheet_orientation, 'automatica')}"
    return mm_per_px, "formato_hoja", sheet_w, sheet_h, summary, None, None


def _quality_band(confidence: float) -> str:
    return "alta" if confidence >= 86 else "media" if confidence >= 72 else "preliminar"


def _confidence(quality_metrics: dict[str, Any], page_detected: bool, auto_upscaled: bool, reference_mode: str) -> float:
    raw = (
        42.0
        + min(quality_metrics["long_lines"] * 1.1, 20)
        + min(quality_metrics["useful_contours"] * 0.25, 14)
        + min(quality_metrics["recognized_text_count"] * 1.4, 10)
        + (8 if page_detected else 0)
        + (6 if 0.012 <= quality_metrics["graphics_ratio"] <= 0.32 else -5)
        + (10 if quality_metrics["inferred_dpi"] >= 170 else 4 if quality_metrics["inferred_dpi"] >= 120 else -8)
        + (4 if auto_upscaled else 0)
        + (6 if reference_mode == "dos_puntos" else 2 if reference_mode == "cota_referencia_global" else 0)
    )
    return round(max(1.0, min(raw, 98.0)), 2)


def _infer_document_orientation(width_px: int, height_px: int) -> str:
    return "apaisada" if width_px >= height_px else "vertical"


def _preparar_region_ocr(roi_bgr: np.ndarray) -> np.ndarray:
    gris = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.fastNlMeansDenoising(gris, None, 8, 7, 21)
    gris = cv2.equalizeHist(gris)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binaria) < 127:
        binaria = cv2.bitwise_not(binaria)
    return cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)


def _collect_priority_ocr_regions(text_boxes: list[dict[str, int]], graphics_binary: np.ndarray, image_shape: tuple[int, int, int], directives: dict[str, bool] | None = None) -> list[dict[str, int]]:
    directives = directives or {}
    h, w = image_shape[:2]
    regions = [dict(box) for box in text_boxes]
    title_blocks = _detect_title_block(graphics_binary)
    regions.extend(title_blocks)
    regions.append({'x': int(w * 0.62), 'y': int(h * 0.67), 'w': int(w * 0.34), 'h': int(h * 0.26)})
    regions.append({'x': int(w * 0.30), 'y': int(h * 0.84), 'w': int(w * 0.32), 'h': int(h * 0.10)})
    if directives.get('preserve_title_block'):
        regions.append({'x': int(w * 0.58), 'y': int(h * 0.62), 'w': int(w * 0.40), 'h': int(h * 0.34)})
    if directives.get('prioritize_dimensions'):
        regions.append({'x': int(w * 0.08), 'y': int(h * 0.02), 'w': int(w * 0.84), 'h': int(h * 0.18)})
        regions.append({'x': int(w * 0.05), 'y': int(h * 0.78), 'w': int(w * 0.75), 'h': int(h * 0.16)})
    deduped: list[dict[str, int]] = []
    for item in regions:
        if item['w'] <= 0 or item['h'] <= 0:
            continue
        duplicate = False
        for keep in deduped:
            ix1 = max(item['x'], keep['x'])
            iy1 = max(item['y'], keep['y'])
            ix2 = min(item['x'] + item['w'], keep['x'] + keep['w'])
            iy2 = min(item['y'] + item['h'], keep['y'] + keep['h'])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area = max(item['w'] * item['h'], 1)
            if inter / area > 0.70:
                duplicate = True
                break
        if not duplicate:
            deduped.append(item)
    deduped.sort(key=lambda b: ((1 if b['y'] > h * 0.70 else 0) + (1 if b['x'] > w * 0.55 else 0), b['w'] * b['h']), reverse=True)
    return deduped


def _run_ocr(image_bgr: np.ndarray, text_boxes: list[dict[str, int]], graphics_binary: np.ndarray, directives: dict[str, bool] | None = None, runtime_profile: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], str, str | None]:
    directives = directives or {}
    runtime_profile = runtime_profile or {}
    if not runtime_profile.get('ocr_allowed', directives.get('force_ocr', False)):
        return [], "desactivado por presupuesto", "El OCR se omitió en esta corrida para preservar estabilidad y memoria del servicio."
    candidate_regions = _collect_priority_ocr_regions(text_boxes, graphics_binary, image_bgr.shape, directives)
    if not candidate_regions:
        return [], "sin zonas de texto", None
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception:
        return [], "no disponible", "No se encontró un motor OCR instalado. Para activar OCR real en Render instalá rapidocr-onnxruntime."

    try:
        engine = RapidOCR()
    except Exception as exc:
        return [], "falló", f"El motor OCR no pudo inicializarse correctamente: {exc}"

    def _normalize_text(value: str) -> str:
        return " ".join(value.replace("\n", " ").split())

    def _add_text(target: list[dict[str, Any]], text: str, score: float, x: int, y: int, w: int, h: int) -> None:
        text = _normalize_text(text)
        if not text or score < 0.35 or w < 3 or h < 3:
            return
        target.append({
            "text": text,
            "score": round(score, 3),
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
        })

    # Presupuesto OCR seguro: limitamos regiones, píxeles y tiempo para evitar
    # que el motor textual tumbe el proceso en servicios chicos.
    warnings: list[str] = []
    texts: list[dict[str, Any]] = []
    boxes = list(candidate_regions)
    budgets = _ocr_budgets(directives, runtime_profile)
    max_boxes = int(budgets['max_boxes'])
    if len(boxes) > max_boxes:
        warnings.append(f"OCR reducido: se analizaron {max_boxes} regiones prioritarias de {len(boxes)} detectadas para evitar sobrecarga del servidor.")
        boxes = boxes[:max_boxes]

    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    analyzed_pixels = 0
    pixel_budget = int(total_pixels * float(budgets['pixel_fraction']))
    started_at = time.monotonic()

    for box in boxes:
        pad = 4
        x1 = max(0, box["x"] - pad)
        y1 = max(0, box["y"] - pad)
        x2 = min(image_bgr.shape[1], box["x"] + box["w"] + pad)
        y2 = min(image_bgr.shape[0], box["y"] + box["h"] + pad)
        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        roi_pixels = int(roi.shape[0] * roi.shape[1])
        if runtime_profile.get('memory_pressure') == 'critical':
            warnings.append('OCR detenido por presión crítica de memoria.')
            break
        if time.monotonic() - started_at > float(budgets['time_budget_s']):
            warnings.append('OCR reducido por presupuesto de tiempo para estabilizar el proceso.')
            break
        if analyzed_pixels + roi_pixels > pixel_budget and analyzed_pixels > 0:
            warnings.append('OCR reducido por presupuesto de cálculo: se priorizaron las regiones más grandes y legibles.')
            break
        analyzed_pixels += roi_pixels
        long_side = max(roi.shape[0], roi.shape[1])
        max_roi_long_side = int(budgets['max_roi_long_side'])
        if long_side > max_roi_long_side:
            scale_down = max_roi_long_side / float(long_side)
            roi = cv2.resize(roi, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_AREA)
        scale = max(1.0, 56.0 / max(roi.shape[0], 1))
        if scale > 1.05:
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        roi_preparada = _preparar_region_ocr(roi)
        try:
            region_result = engine(roi_preparada)
            region_items = region_result[0] if isinstance(region_result, tuple) else region_result
        except Exception as exc:
            warnings.append(f"OCR parcial no disponible en una región: {exc}")
            region_items = []
        if not region_items:
            continue
        best_text = ""
        best_score = 0.0
        for item in region_items:
            try:
                txt = str(item[1]).strip()
                scr = float(item[2])
            except Exception:
                continue
            if scr > best_score and txt:
                best_text = txt
                best_score = scr
        if best_text:
            _add_text(texts, best_text, best_score, box["x"], box["y"], box["w"], box["h"])

    deduped: list[dict[str, Any]] = []
    for item in sorted(texts, key=lambda d: (-d["score"], d["y"], d["x"])):
        duplicate = False
        for keep in deduped:
            ix1 = max(item["x"], keep["x"])
            iy1 = max(item["y"], keep["y"])
            ix2 = min(item["x"] + item["w"], keep["x"] + keep["w"])
            iy2 = min(item["y"] + item["h"], keep["y"] + keep["h"])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area = max(item["w"] * item["h"], 1)
            if inter / area > 0.45:
                duplicate = True
                break
        if not duplicate:
            deduped.append(item)

    deduped.sort(key=lambda item: (item["y"], item["x"]))
    warning = "; ".join(dict.fromkeys(warnings)) if warnings else None
    return deduped, "RapidOCR por regiones en modo seguro", warning

def _classify_text_items(ocr_items: list[dict[str, Any]], image_shape: tuple[int, int, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    h, w = image_shape[:2]
    cotas: list[dict[str, Any]] = []
    rotulos: list[dict[str, Any]] = []
    textos: list[dict[str, Any]] = []
    for item in ocr_items:
        text = str(item.get("text", "")).strip()
        box = dict(item)
        compacto = text.replace(" ", "")
        tiene_digitos = any(ch.isdigit() for ch in compacto)
        es_cota = tiene_digitos and len(compacto) <= 12
        region_baja = item.get("y", 0) > h * 0.72
        ancho_rel = item.get("w", 0) / float(max(w, 1))
        if es_cota:
            box["tipo_texto"] = "cota"
            cotas.append(box)
        elif region_baja and (ancho_rel > 0.08 or len(text) >= 4):
            box["tipo_texto"] = "rotulo"
            rotulos.append(box)
        else:
            box["tipo_texto"] = "texto_general"
            textos.append(box)
    return cotas, rotulos, textos


def _detect_title_block(graphics_binary: np.ndarray) -> list[dict[str, int]]:
    h, w = graphics_binary.shape[:2]
    contours, _ = cv2.findContours(graphics_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < (w * h) * 0.015:
            continue
        if x < w * 0.35 or y < h * 0.55:
            continue
        if bw < w * 0.16 or bh < h * 0.08:
            continue
        if bw > w * 0.80 or bh > h * 0.35:
            continue
        aspect = bw / float(max(bh, 1))
        if aspect < 1.2 or aspect > 8.5:
            continue
        candidates.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "area": int(area)})
    candidates.sort(key=lambda item: item["area"], reverse=True)
    if not candidates:
        return []
    best = candidates[0]
    return [{"x": best["x"], "y": best["y"], "w": best["w"], "h": best["h"]}]


def _associate_dimension_lines(lines: list[dict[str, int]], cota_texts: list[dict[str, Any]]) -> list[dict[str, int]]:
    if not lines or not cota_texts:
        return []
    out: list[dict[str, int]] = []
    for line in lines:
        length = _line_length(line)
        orient = _line_orientation(line)
        if orient not in {"horizontal", "vertical"}:
            continue
        if length < 16 or length > 3200:
            continue
        lx1, lx2 = min(line["x1"], line["x2"]), max(line["x1"], line["x2"])
        ly1, ly2 = min(line["y1"], line["y2"]), max(line["y1"], line["y2"])
        for txt in cota_texts:
            tx1 = txt["x"] - max(10, txt["w"] // 2)
            ty1 = txt["y"] - max(10, txt["h"] * 2)
            tx2 = txt["x"] + txt["w"] + max(10, txt["w"] // 2)
            ty2 = txt["y"] + txt["h"] + max(10, txt["h"] * 2)
            if orient == "horizontal":
                cerca = (ly2 >= ty1 and ly1 <= ty2 and lx2 >= tx1 and lx1 <= tx2)
            else:
                cerca = (lx2 >= tx1 and lx1 <= tx2 and ly2 >= ty1 and ly1 <= ty2)
            if cerca:
                out.append({"x1": int(line["x1"]), "y1": int(line["y1"]), "x2": int(line["x2"]), "y2": int(line["y2"])})
                break
    return _dedupe_lines(out, tolerance=4)



def _detect_dimension_arrows(graphics_binary: np.ndarray, dimension_lines: list[dict[str, int]]) -> list[dict[str, int]]:
    if not dimension_lines:
        return []
    contours, _ = cv2.findContours(graphics_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 6 or area > 220:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 3 or h < 3 or w > 36 or h > 36:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
        if len(approx) < 3 or len(approx) > 6:
            continue
        candidates.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    out: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for line in dimension_lines:
        orient = _line_orientation(line)
        endpoints = [(line["x1"], line["y1"]), (line["x2"], line["y2"])]
        for ex, ey in endpoints:
            for cand in candidates:
                cx = cand["x"] + cand["w"] / 2.0
                cy = cand["y"] + cand["h"] / 2.0
                if orient == "horizontal":
                    close = abs(cx - ex) <= 18 and abs(cy - ey) <= 14
                elif orient == "vertical":
                    close = abs(cx - ex) <= 14 and abs(cy - ey) <= 18
                else:
                    close = abs(cx - ex) <= 16 and abs(cy - ey) <= 16
                if close:
                    key = (round(cx / 6), round(cy / 6))
                    if key not in seen:
                        seen.add(key)
                        out.append(cand)
                    break
    return out


def _estimate_text_review_items(ocr_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review: list[dict[str, Any]] = []
    for item in ocr_items:
        txt = str(item.get("text", "")).strip()
        score = float(item.get("score", 0.0) or 0.0)
        weird = sum(1 for ch in txt if ch in "|/_~")
        if score < 0.58 or len(txt) <= 1 or weird >= 3:
            review.append(item)
    return review


def _detect_title_block_refined(graphics_binary: np.ndarray, rotulo_texts: list[dict[str, Any]]) -> list[dict[str, int]]:
    base = _detect_title_block(graphics_binary)
    if base:
        return base
    if not rotulo_texts:
        return []
    xs = [t["x"] for t in rotulo_texts]
    ys = [t["y"] for t in rotulo_texts]
    x2s = [t["x"] + t["w"] for t in rotulo_texts]
    y2s = [t["y"] + t["h"] for t in rotulo_texts]
    x1, y1, x2, y2 = min(xs), min(ys), max(x2s), max(y2s)
    pad_x = max(18, int((x2 - x1) * 0.10))
    pad_y = max(12, int((y2 - y1) * 0.20))
    return [{"x": max(0, x1 - pad_x), "y": max(0, y1 - pad_y), "w": x2 - x1 + pad_x * 2, "h": y2 - y1 + pad_y * 2}]

def _guess_discipline(ocr_items: list[dict[str, Any]], geometry: dict[str, Any]) -> str:
    text_blob = " ".join(str(item.get("text", "")).lower() for item in ocr_items)
    score = {"arquitectura": 0, "electricidad": 0, "sanitaria": 0, "mecanica": 0}
    for token in ["tablero", "fase", "llave", "circuito", "tomacorriente", "iluminacion", "electric"]:
        if token in text_blob:
            score["electricidad"] += 2
    for token in ["agua", "desague", "cloaca", "bomba", "sanitario", "tanque", "pluvial"]:
        if token in text_blob:
            score["sanitaria"] += 2
    for token in ["corte", "planta", "muro", "puerta", "ventana", "nivel"]:
        if token in text_blob:
            score["arquitectura"] += 2
    for token in ["motor", "rodamiento", "eje", "chapa", "soporte", "tornillo", "diam", "ø"]:
        if token in text_blob:
            score["mecanica"] += 2
    score["arquitectura"] += min(3, len(geometry.get("closed_polys", [])))
    score["electricidad"] += min(3, len(geometry.get("electrical_symbols", [])))
    score["sanitaria"] += min(3, len(geometry.get("sanitary_symbols", [])))
    score["mecanica"] += min(3, len(geometry.get("mechanical_symbols", [])))
    best = max(score, key=score.get)
    return best if score[best] > 0 else "general"


def _detect_symbols(graphics_binary: np.ndarray, lines: list[dict[str, int]], text_boxes: list[dict[str, int]]) -> dict[str, list[dict[str, int]]]:
    h, w = graphics_binary.shape[:2]
    line_mask = np.zeros_like(graphics_binary)
    for line in lines:
        cv2.line(line_mask, (line["x1"], line["y1"]), (line["x2"], line["y2"]), 255, 3)
    residual = cv2.bitwise_and(graphics_binary, cv2.bitwise_not(line_mask))
    for box in text_boxes:
        cv2.rectangle(residual, (box["x"], box["y"]), (box["x"] + box["w"], box["y"] + box["h"]), 0, -1)
    contours, _ = cv2.findContours(residual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    symbols = {"electrical_symbols": [], "sanitary_symbols": [], "mechanical_symbols": [], "generic_symbols": []}
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < 20 or area > (w * h) * 0.01:
            continue
        if bw < 4 or bh < 4 or bw > 90 or bh > 90:
            continue
        aspect = bw / float(max(bh, 1))
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.06 * peri, True)
        circularity = 0.0
        cnt_area = cv2.contourArea(cnt)
        if peri > 0:
            circularity = 4.0 * math.pi * cnt_area / (peri * peri)
        box = {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)}
        if circularity > 0.55 and 0.7 <= aspect <= 1.3:
            symbols["electrical_symbols"].append(box)
        elif len(approx) >= 6 and bh > bw * 1.2:
            symbols["sanitary_symbols"].append(box)
        elif len(approx) in {3, 4, 5} and 0.4 <= aspect <= 2.5:
            symbols["mechanical_symbols"].append(box)
        else:
            symbols["generic_symbols"].append(box)
    return symbols



def _infer_symbol_block_name(symbol_type: str, box: dict[str, int], discipline_guess: str) -> str:
    aspect = float(box.get("w", 1)) / float(max(box.get("h", 1), 1))
    area = int(box.get("w", 0)) * int(box.get("h", 0))
    if symbol_type == "electrical_symbols":
        if 0.85 <= aspect <= 1.15:
            return "TOMA_ELECTRICA"
        if aspect > 1.6:
            return "LUMINARIA_LINEAL"
        return "INTERRUPTOR"
    if symbol_type == "sanitary_symbols":
        if aspect < 0.75:
            return "MONTANTE_SANITARIA"
        if area > 900:
            return "ARTEFACTO_SANITARIO"
        return "VALVULA_SANITARIA"
    if symbol_type == "mechanical_symbols":
        if area > 1200:
            return "EQUIPO_MECANICO"
        if 0.85 <= aspect <= 1.15:
            return "RODAMIENTO"
        return "VALVULA_MECANICA"
    if discipline_guess == "arquitectura":
        return "REFERENCIA_ARQUITECTURA"
    return "BLOQUE_GENERAL"


def _build_symbol_blocks(symbols: dict[str, list[dict[str, int]]], discipline_guess: str) -> list[dict[str, Any]]:
    bloques: list[dict[str, Any]] = []
    for key, items in symbols.items():
        for idx, box in enumerate(items, start=1):
            cx = box["x"] + box["w"] / 2.0
            cy = box["y"] + box["h"] / 2.0
            nombre = _infer_symbol_block_name(key, box, discipline_guess)
            bloques.append({
                "nombre": nombre,
                "tipo": key,
                "indice": idx,
                "x": float(cx),
                "y": float(cy),
                "w": int(box["w"]),
                "h": int(box["h"]),
                "disciplina": discipline_guess,
            })
    return bloques


def _build_discipline_rules(discipline_guess: str, cota_texts: list[dict[str, Any]], rotulo_texts: list[dict[str, Any]], symbols: dict[str, list[dict[str, int]]]) -> list[str]:
    reglas_comunes = [
        "separar textos del trazado antes de vectorizar",
        "preservar la proporcion original del dibujo",
        "mantener capas independientes para textos, cotas y simbologia",
    ]
    reglas_por_disciplina = {
        "arquitectura": [
            "priorizar cerramientos, aberturas y cuadros de plano",
            "agrupar simbolos residuales como referencia de arquitectura",
            "mantener referencias de ambientes y puertas fuera del barrido geometrico",
        ],
        "electricidad": [
            "priorizar tablero, circuitos y simbologia electrica",
            "mantener textos de circuito y referencias fuera de la geometria",
            "separar luminarias, tomas e interruptores en bloques preliminares distintos",
        ],
        "sanitaria": [
            "priorizar montantes, artefactos y simbologia sanitaria",
            "preservar ejes de cañerias y etiquetas de servicio",
            "distinguir artefactos, valvulas y montantes en capas de revision",
        ],
        "mecanica": [
            "priorizar ejes, diametros y notas tecnicas de piezas",
            "mantener cotas y llamados de mecanizado en capa separada",
            "distinguir equipos, valvulas y apoyos mecanicos en bloques preliminares",
        ],
        "general": [
            "mantener lectura neutra y dejar revision asistida activada",
        ],
    }
    reglas = list(reglas_comunes)
    reglas.extend(reglas_por_disciplina.get(discipline_guess, reglas_por_disciplina["general"]))
    if cota_texts:
        reglas.append("proteger las cotas detectadas de la limpieza geometrica")
    if rotulo_texts:
        reglas.append("preservar el rotulo y el cuadro del plano para trazabilidad")
    if sum(len(v) for v in symbols.values()) > 0:
        reglas.append("sugerir bloques CAD preliminares para la simbologia detectada")
    return reglas

def _build_analysis_preview(base_image: np.ndarray, text_boxes: list[dict[str, int]]) -> np.ndarray:
    preview = base_image.copy()
    for box in text_boxes:
        cv2.rectangle(preview, (box["x"], box["y"]), (box["x"] + box["w"], box["y"] + box["h"]), (205, 120, 20), 1)
    return preview


def _build_vector_base(graphics_binary: np.ndarray, ocr_items: list[dict[str, Any]]) -> np.ndarray:
    clean = cv2.cvtColor(cv2.bitwise_not(graphics_binary), cv2.COLOR_GRAY2BGR)
    for item in ocr_items:
        cv2.rectangle(clean, (item["x"], item["y"]), (item["x"] + item["w"], item["y"] + item["h"]), (230, 240, 255), -1)
        cv2.rectangle(clean, (item["x"], item["y"]), (item["x"] + item["w"], item["y"] + item["h"]), (170, 170, 220), 1)
    return clean

def _build_presentation_image(source_bgr: np.ndarray, enhanced_bgr: np.ndarray, normalized_gray: np.ndarray, binary: np.ndarray, repaired_binary: np.ndarray, title_blocks: list[dict[str, int]] | None = None, directives: dict[str, bool] | None = None) -> np.ndarray:
    directives = directives or {}
    title_blocks = title_blocks or []
    source_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    original_gray = cv2.normalize(normalized_gray, None, 0, 255, cv2.NORM_MINMAX)
    base_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

    # En v71 la presentación sale desde el documento original suavemente normalizado.
    presentation_gray = cv2.addWeighted(source_gray, 0.97, original_gray, 0.03, 0)

    # Refuerzo muy leve solo para líneas ya presentes; evita empastar texto y rótulo.
    line_source = binary if not directives.get('reconstruct_perimeters') else cv2.bitwise_or(binary, cv2.bitwise_and(repaired_binary, cv2.bitwise_not(binary)))
    line_mask = cv2.GaussianBlur(line_source, (0, 0), 0.40)
    line_mask_f = line_mask.astype(np.float32) / 255.0
    darken = 3.0 if directives.get('prioritize_fidelity') or directives.get('preserve_dashed') else 1.5
    presentation_gray = np.clip(presentation_gray.astype(np.float32) - line_mask_f * darken, 0, 255).astype(np.uint8)

    # Zonas documentales: restauración y micro-realce del original para mantener textos/rótulo nítidos.
    for box in _documental_regions(presentation_gray.shape, title_blocks):
        x1 = max(0, int(box['x']))
        y1 = max(0, int(box['y']))
        x2 = min(presentation_gray.shape[1], x1 + int(box['w']))
        y2 = min(presentation_gray.shape[0], y1 + int(box['h']))
        if x2 <= x1 or y2 <= y1:
            continue
        roi = source_gray[y1:y2, x1:x2]
        enhanced_roi = _enhance_documental_roi(roi)
        presentation_gray[y1:y2, x1:x2] = cv2.min(presentation_gray[y1:y2, x1:x2], enhanced_roi)

    # Si se pidió fidelidad, se conserva más detalle fino del original sin endurecerlo.
    if directives.get('prioritize_fidelity') or directives.get('preserve_title_block'):
        detail = cv2.GaussianBlur(source_gray, (0, 0), 0.15)
        presentation_gray = cv2.addWeighted(presentation_gray, 0.98, detail, 0.02, 0)

    return cv2.cvtColor(presentation_gray, cv2.COLOR_GRAY2BGR)


def _insights(
    reference_mode: str,
    page_detected: bool,
    quality_metrics: dict[str, Any],
    sheet_size: str,
    auto_upscaled: bool,
    sheet_orientation: str,
    document_orientation: str,
    calibration_summary: str,
    ocr_engine: str,
) -> list[str]:
    out: list[str] = []
    if reference_mode == "dos_puntos":
        out.append(f"La escala se calibró con dos puntos de referencia cargados por el usuario: {calibration_summary}.")
    elif reference_mode == "cota_referencia_global":
        out.append(f"Se aplicó una calibración global usando la medida real cargada por el usuario: {calibration_summary}.")
    else:
        out.append(
            f"La escala de salida se calibró con el formato {sheet_size} y orientación {ORIENTATION_LABELS.get(sheet_orientation, 'automatica')} seleccionados; para metrología fina conviene una referencia real adicional."
        )
    out.append(
        "Se detectó el borde principal del documento y se corrigió la perspectiva manteniendo la proporción original del dibujo."
        if page_detected
        else "No se detectó un borde de hoja confiable; se procesó la imagen completa, manteniendo la orientación y proporción originales."
    )
    out.append(f"El documento fue interpretado como {document_orientation}. La hoja de salida se ajusta para contenerlo sin deformación.")
    if auto_upscaled:
        out.append("Se aplicó un reescalado automático para reforzar líneas finas y mejorar la lectura de contornos.")
    out.append(
        f"Se separaron {quality_metrics['text_regions']} regiones candidatas de texto para que no entren en la misma rutina de vectorización geométrica."
    )
    out.append(
        f"El motor OCR activo fue: {ocr_engine}. Se reconocieron {quality_metrics['recognized_text_count']} bloques de texto aprovechables, con {quality_metrics.get('cota_texts', 0)} posibles cotas y {quality_metrics.get('rotulo_texts', 0)} textos de rótulo o cuadro."
        if ocr_engine not in {"no disponible", "sin zonas de texto", "falló"}
        else "El proceso separa zonas candidatas de texto del dibujo, pero en esta ejecución no pudo consolidar OCR suficiente para convertir más textos del plano en entidades útiles."
    )
    if quality_metrics.get("title_blocks", 0) > 0:
        out.append(f"Se detectó {quality_metrics.get('title_blocks', 0)} cuadro de plano o rótulo principal en la zona inferior derecha para ayudar al orden del DXF y la lectura del documento.")
    if quality_metrics.get("dimension_lines", 0) > 0:
        out.append(f"Se asociaron {quality_metrics.get('dimension_lines', 0)} líneas de cota con sus textos cercanos para que no se mezclen con la geometría principal.")
    if quality_metrics.get("dimension_arrows", 0) > 0:
        out.append(f"Además se detectaron {quality_metrics.get('dimension_arrows', 0)} flechas o remates de acotación, lo que ayuda a distinguir cotas de líneas estructurales.")
    if quality_metrics.get("duplicate_lines_removed_count", 0) > 0:
        out.append(f"Se limpiaron {quality_metrics.get('duplicate_lines_removed_count', 0)} segmentos repetidos o superpuestos para entregar una vectorización más ordenada.")
    if quality_metrics.get("review_texts", 0) > 0:
        out.append(f"Se marcaron {quality_metrics.get('review_texts', 0)} textos para revisión asistida porque su lectura OCR quedó dudosa o incompleta.")
    if quality_metrics.get("electrical_symbols", 0) + quality_metrics.get("sanitary_symbols", 0) + quality_metrics.get("mechanical_symbols", 0) > 0:
        out.append(
            f"Se detectaron símbolos preliminares para ayudar al orden por disciplina: {quality_metrics.get('electrical_symbols', 0)} eléctricos, {quality_metrics.get('sanitary_symbols', 0)} sanitarios y {quality_metrics.get('mechanical_symbols', 0)} mecánicos. Disciplina sugerida: {quality_metrics.get('discipline_guess', 'general')}."
        )
        out.append(
            f"Además se propusieron {quality_metrics.get('symbol_blocks', 0)} bloques CAD preliminares con nombres más específicos para acelerar la revisión técnica."
        )
    if quality_metrics.get("symbol_blocks", 0) > 0:
        out.append(f"Se sugirieron {quality_metrics.get('symbol_blocks', 0)} bloques CAD preliminares para acelerar la limpieza por disciplina.")
    if quality_metrics.get("discipline_rules", 0) > 0:
        out.append(f"El ojo inteligente aplicó {quality_metrics.get('discipline_rules', 0)} reglas de disciplina para ordenar la vectorización durante el proceso.")
    return out


def _warnings(
    reference_mode: str,
    quality_metrics: dict[str, Any],
    geometry: dict[str, Any],
    quality_band: str,
    calibration: dict[str, Any] | None,
    ocr_warning: str | None,
) -> list[str]:
    warnings: list[str] = []
    if calibration and calibration.get("mode") == "TWO_POINT" and reference_mode != "dos_puntos":
        warnings.append("La calibración por dos puntos no pudo validarse correctamente. Se usó una calibración de respaldo para no frenar el proceso.")
    if reference_mode == "formato_hoja":
        warnings.append("No se definió una referencia geométrica real. La salida queda útil para reconstrucción y trazado, pero no como verificación dimensional fina.")
    if quality_metrics["inferred_dpi"] < 110:
        warnings.append("La definición efectiva estimada es menor a 110 dpi. Para detalle fino de cotas, textos o símbolos conviene reescanear o cargar una imagen de mayor resolución.")
    if geometry["long_line_count"] < 4:
        warnings.append("Se detectaron pocas líneas largas estructurales. Puede haber trazos débiles, fondo complejo o un plano con predominio de curvas que requiera revisión asistida.")
    if quality_metrics["graphics_ratio"] > 0.38:
        warnings.append("La base gráfica quedó muy cargada. Puede haber sombreado, sellos o ruido que afecten parte de la vectorización.")
    elif quality_metrics["graphics_ratio"] < 0.008:
        warnings.append("La imagen tiene poco contraste útil. Conviene revisar iluminación, compresión o exposición del archivo cargado.")
    if quality_metrics["recognized_text_count"] == 0:
        warnings.append("No se incorporaron textos OCR al DXF en esta corrida. La geometría puede salir bien, pero los rótulos seguirán necesitando una etapa de reconstrucción textual o una imagen con mejor nitidez.")
    if quality_metrics.get("cota_texts", 0) == 0:
        warnings.append("No se pudieron identificar cotas confiables en esta corrida. Conviene una imagen con mejor definición o calibración manual por dos puntos.")
    if quality_metrics.get("title_blocks", 0) == 0:
        warnings.append("No se detectó un cuadro de plano claro en la zona inferior del documento. El rótulo podría requerir revisión asistida.")
    if quality_metrics.get("review_texts", 0) > max(3, quality_metrics.get("recognized_text_count", 0) * 0.35):
        warnings.append("Una parte importante de los textos quedó marcada para revisión asistida. Conviene validar rótulos y cotas antes de usar la salida como base documental.")
    if ocr_warning:
        warnings.append(ocr_warning)
    if quality_band == "preliminar":
        warnings.append("El resultado es preliminar. Se recomienda corrección interactiva antes de usarlo como base de proyecto.")
    return warnings


def _calibration_reliability(reference_mode: str, calibration_distance_mm: float | None, calibration_pixel_span: float | None, inferred_dpi: int) -> str:
    if reference_mode == "dos_puntos" and calibration_distance_mm and calibration_pixel_span and calibration_pixel_span >= 80:
        return "alta" if inferred_dpi >= 160 else "media"
    if reference_mode == "cota_referencia_global":
        return "media" if inferred_dpi >= 140 else "preliminar"
    return "preliminar"


def _geometry_stability(quality_metrics: dict[str, Any], line_count: int, duplicate_removed: int) -> str:
    raw_segments = max(int(quality_metrics.get("long_lines", 0)) + duplicate_removed + 1, 1)
    ratio = line_count / float(raw_segments)
    if ratio >= 0.85 and duplicate_removed <= max(8, line_count * 0.08):
        return "alta"
    if ratio >= 0.60:
        return "media"
    return "preliminar"


def _text_separation_quality(text_regions: int, recognized_texts: int) -> str:
    if text_regions <= 0:
        return "sin texto relevante"
    ratio = recognized_texts / float(max(text_regions, 1))
    if ratio >= 0.70:
        return "alta"
    if ratio >= 0.40:
        return "media"
    return "preliminar"


def _precision_index(confidence: float, reference_mode: str, inferred_dpi: int, geometry_stability: str, text_quality: str) -> float:
    score = confidence * 0.45
    score += 20 if reference_mode == "dos_puntos" else 12 if reference_mode == "cota_referencia_global" else 6
    score += 18 if inferred_dpi >= 180 else 12 if inferred_dpi >= 140 else 6
    score += 10 if geometry_stability == "alta" else 6 if geometry_stability == "media" else 2
    score += 7 if text_quality == "alta" else 4 if text_quality == "media" else 1
    return round(min(score, 100.0), 1)



def _precision_class(precision_index: float, calibration_reliability: str, geometry_stability: str) -> str:
    if precision_index >= 85 and calibration_reliability == "alta" and geometry_stability == "alta":
        return "alta"
    if precision_index >= 70:
        return "media"
    return "preliminar"


def _expected_positional_error_mm(mm_per_px: float, inferred_dpi: int, calibration_reliability: str) -> float:
    dpi_factor = 0.8 if inferred_dpi >= 180 else 1.0 if inferred_dpi >= 140 else 1.35
    reliability_factor = 0.85 if calibration_reliability == "alta" else 1.0 if calibration_reliability == "media" else 1.25
    base_error = max(mm_per_px * 2.0, 0.18)
    return round(base_error * dpi_factor * reliability_factor, 2)


def _suggested_linear_tolerance_mm(expected_error_mm: float, precision_class: str) -> float:
    factor = 2.0 if precision_class == "alta" else 2.5 if precision_class == "media" else 3.2
    return round(max(expected_error_mm * factor, 0.5), 2)

def _recommended_precision_action(precision_index: float, calibration_reliability: str) -> str:
    if precision_index >= 82 and calibration_reliability == "alta":
        return "Apta para revisión técnica fina antes de emitir documentación final."
    if precision_index >= 68:
        return "Apta para documentación preliminar; conviene revisar cotas, textos y símbolos antes de liberar."
    return "Conviene recalibrar, revisar OCR y hacer corrección asistida antes de usarla como base técnica."


def _precision_observations(reference_mode: str, calibration_reliability: str, geometry_stability: str, text_quality: str, inferred_dpi: int) -> list[str]:
    items = [
        f"Confiabilidad de calibración: {calibration_reliability}.",
        f"Estabilidad geométrica estimada: {geometry_stability}.",
        f"Calidad de separación texto/dibujo: {text_quality}.",
        f"Resolución efectiva estimada: {inferred_dpi} dpi.",
    ]
    if reference_mode == "formato_hoja":
        items.append("La calibración está apoyada en el formato de hoja; para mayor precisión conviene usar dos puntos reales.")
    elif reference_mode == "cota_referencia_global":
        items.append("La calibración usa una única medida global; mejora si se corrige con dos puntos reales sobre la imagen.")
    else:
        items.append("La calibración por dos puntos aporta la mejor base para precisión dentro del flujo actual.")
    return items



def process_drawing(
    input_path: Path,
    output_dir: Path,
    sheet_size: str,
    drawing_type: str,
    reference_mm: float | None,
    notes: str = "",
    sheet_orientation: str = "AUTO",
    calibration: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> PipelineResult:
    directives = _parse_user_directives(notes)
    if progress_callback:
        progress_callback("preparando_archivo")
    image = _read_image(input_path)
    original_h, original_w = image.shape[:2]
    corners, angle = _detect_document(image)
    page_detected = corners is not None
    transform_matrix = None
    if page_detected:
        image, transform_matrix = _warp_document(image, corners)

    image = _apply_memory_budget(image)
    runtime_profile = _adaptive_runtime_profile(image.shape, directives)

    if progress_callback:
        progress_callback("reconstruyendo_base")
    image, auto_upscaled, upscale_factor = _auto_upscale(image)
    reconstructed_gray, normalized_gray, enhanced_bgr = _enhance_image(image)
    binary = _build_binary(reconstructed_gray)
    repaired_binary = _repair_broken_traces(binary, directives)
    repaired_binary = _reinforce_title_block(repaired_binary, directives)
    if progress_callback:
        progress_callback("separando_texto_y_dibujo")
    text_boxes, text_mask = _detect_text_regions(repaired_binary)
    graphics_binary = cv2.bitwise_and(repaired_binary, cv2.bitwise_not(text_mask))
    graphics_binary = cv2.morphologyEx(graphics_binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    if progress_callback:
        progress_callback("reconociendo_texto")
    if _should_run_ocr(directives):
        runtime_profile = _adaptive_runtime_profile(enhanced_bgr.shape, directives)
        ocr_items, ocr_engine, ocr_warning = _run_ocr(enhanced_bgr, text_boxes, graphics_binary, directives, runtime_profile)
    else:
        ocr_items, ocr_engine, ocr_warning = [], "desactivado por estabilidad", "El OCR quedó desactivado en esta corrida para priorizar estabilidad en Render. Activá una acción OCR solo cuando realmente necesites recuperar textos."
    cota_texts, rotulo_texts, general_texts = _classify_text_items(ocr_items, image.shape)
    if progress_callback:
        progress_callback("vectorizando_geometria")
    geometry = _detect_geometry(graphics_binary, text_boxes)
    documental_regions = _documental_regions(image.shape, geometry.get("title_blocks", []))
    geometry = _sanitize_geometry(geometry, documental_regions, image.shape, directives)
    geometry["texts"] = cota_texts + rotulo_texts + general_texts
    geometry["cota_texts"] = cota_texts
    geometry["rotulo_texts"] = rotulo_texts
    geometry["general_texts"] = general_texts
    geometry["dimension_lines"] = _associate_dimension_lines(geometry["lines"], cota_texts)
    geometry["dimension_arrows"] = _detect_dimension_arrows(graphics_binary, geometry["dimension_lines"])
    geometry["review_texts"] = _estimate_text_review_items(ocr_items)
    geometry["title_blocks"] = _detect_title_block_refined(graphics_binary, rotulo_texts)
    documental_regions = _documental_regions(image.shape, geometry.get("title_blocks", []))
    geometry = _sanitize_geometry(geometry, documental_regions, image.shape, directives)
    _symbols = _detect_symbols(graphics_binary, geometry["lines"], text_boxes)
    geometry.update(_symbols)
    geometry["discipline_guess"] = _guess_discipline(ocr_items, geometry)
    geometry["symbol_blocks"] = _build_symbol_blocks(_symbols, geometry["discipline_guess"])
    geometry["discipline_rules"] = _build_discipline_rules(geometry["discipline_guess"], cota_texts, rotulo_texts, _symbols)
    geometry["runtime_profile"] = runtime_profile

    quality_metrics = _compute_quality_metrics(binary, graphics_binary, geometry, (original_h, original_w, 3), image.shape, sheet_size.upper(), sheet_orientation.upper(), len(ocr_items))
    quality_metrics["cota_texts"] = len(cota_texts)
    quality_metrics["rotulo_texts"] = len(rotulo_texts)
    quality_metrics["title_blocks"] = len(geometry["title_blocks"])
    quality_metrics["dimension_lines"] = len(geometry["dimension_lines"])
    quality_metrics["dimension_arrows"] = len(geometry["dimension_arrows"])
    quality_metrics["review_texts"] = len(geometry["review_texts"])
    quality_metrics["electrical_symbols"] = len(geometry.get("electrical_symbols", []))
    quality_metrics["sanitary_symbols"] = len(geometry.get("sanitary_symbols", []))
    quality_metrics["mechanical_symbols"] = len(geometry.get("mechanical_symbols", []))
    quality_metrics["generic_symbols"] = len(geometry.get("generic_symbols", []))
    quality_metrics["symbol_blocks"] = len(geometry.get("symbol_blocks", []))
    quality_metrics["discipline_rules"] = len(geometry.get("discipline_rules", []))
    quality_metrics["discipline_guess"] = geometry.get("discipline_guess", "general")
    threshold_ratio = float(np.count_nonzero(graphics_binary)) / float(max(graphics_binary.size, 1))

    width_px, height_px = int(image.shape[1]), int(image.shape[0])
    transformed_points = _transform_calibration_points(
        calibration.get("points") if calibration else None,
        transform_matrix,
        upscale_factor,
    )
    mm_per_px, reference_mode, sheet_w, sheet_h, calibration_summary, calibration_distance_mm, calibration_pixel_span = _estimate_scale_mm_per_px(
        width_px,
        height_px,
        sheet_size.upper(),
        reference_mm,
        sheet_orientation.upper(),
        calibration,
        transformed_points,
    )
    confidence = _confidence(quality_metrics, page_detected, auto_upscaled, reference_mode)
    quality_band = _quality_band(confidence)
    document_orientation = _infer_document_orientation(width_px, height_px)

    overlay = enhanced_bgr.copy()
    for item in geometry["lines"]:
        cv2.line(overlay, (item["x1"], item["y1"]), (item["x2"], item["y2"]), (0, 0, 255), 1)
    for poly in geometry["polylines"]:
        pts = np.array([[pt["x"], pt["y"]] for pt in poly], dtype=np.int32)
        cv2.polylines(overlay, [pts], True, (0, 150, 0), 1)
    for item in ocr_items:
        color = (180, 120, 10)
        if item in cota_texts:
            color = (20, 120, 210)
        elif item in rotulo_texts:
            color = (130, 30, 170)
        cv2.rectangle(overlay, (item["x"], item["y"]), (item["x"] + item["w"], item["y"] + item["h"]), color, 1)
        cv2.putText(overlay, item["text"][:24], (item["x"], max(18, item["y"] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    for line in geometry.get("dimension_lines", []):
        cv2.line(overlay, (line["x1"], line["y1"]), (line["x2"], line["y2"]), (20, 120, 210), 2)
    for box in geometry.get("title_blocks", []):
        cv2.rectangle(overlay, (box["x"], box["y"]), (box["x"] + box["w"], box["y"] + box["h"]), (130, 30, 170), 2)
        cv2.putText(overlay, "Cuadro detectado", (box["x"], max(24, box["y"] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 30, 170), 2, cv2.LINE_AA)
    for key, color, label in [("electrical_symbols", (0, 180, 255), "Simbolo el."), ("sanitary_symbols", (0, 180, 120), "Simbolo san."), ("mechanical_symbols", (255, 120, 0), "Simbolo mec.")]:
        for box in geometry.get(key, []):
            cv2.rectangle(overlay, (box["x"], box["y"]), (box["x"] + box["w"], box["y"] + box["h"]), color, 1)
            cv2.putText(overlay, label, (box["x"], max(18, box["y"] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    if page_detected:
        cv2.putText(overlay, "Perspectiva corregida", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 120, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, "Proporcion preservada", (24, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (80, 160, 255), 2, cv2.LINE_AA)
    if len(transformed_points) == 2:
        p1 = tuple(int(round(v)) for v in transformed_points[0])
        p2 = tuple(int(round(v)) for v in transformed_points[1])
        cv2.line(overlay, p1, p2, (0, 165, 255), 2)
        cv2.circle(overlay, p1, 7, (30, 60, 220), -1)
        cv2.circle(overlay, p2, 7, (220, 50, 50), -1)
        cv2.putText(overlay, "A", (p1[0] + 10, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 60, 220), 2, cv2.LINE_AA)
        cv2.putText(overlay, "B", (p2[0] + 10, p2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 50, 50), 2, cv2.LINE_AA)

    reconstruction_preview = enhanced_bgr
    vector_base = _build_vector_base(graphics_binary, ocr_items)
    presentation_base = _build_presentation_image(image, enhanced_bgr, normalized_gray, binary, repaired_binary, geometry.get("title_blocks", []), directives)

    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_path = output_dir / "reconstruccion_previa.png"
    cleaned_path = output_dir / "limpio.png"
    presentation_path = output_dir / "presentacion.png"
    overlay_path = output_dir / "overlay.png"
    original_path = output_dir / "original_preview.png"
    _save_image(reconstruction_path, _resize_for_report(reconstruction_preview))
    _save_image(cleaned_path, _resize_for_report(vector_base))
    _save_image(presentation_path, _resize_for_report(presentation_base))
    if os.getenv('TRAZOCAD_DISABLE_OVERLAY_OUTPUT', '1').strip().lower() not in {'1', 'true', 'yes', 'si'}:
        _save_image(overlay_path, _resize_for_report(overlay))
    if os.getenv('TRAZOCAD_DISABLE_ANALYSIS_PREVIEW', '1').strip().lower() not in {'1', 'true', 'yes', 'si'}:
        analysis_preview = _build_analysis_preview(enhanced_bgr, text_boxes)
        _save_image(original_path, _resize_for_report(analysis_preview))

    assumptions = [
        f"Tipo de plano interpretado: {drawing_type}.",
        f"Formato de salida solicitado: {sheet_size.upper()}.",
        f"Orientación de hoja solicitada: {ORIENTATION_LABELS.get(sheet_orientation.upper(), 'automatica')}.",
        f"Modo de calibración resuelto: {reference_mode.replace('_', ' ')}.",
        f"Detalle de calibración: {calibration_summary}.",
        f"Cotas OCR detectadas: {len(cota_texts)}.",
        f"Rótulos OCR detectados: {len(rotulo_texts)}.",
        f"Cuadros de plano detectados: {len(geometry.get('title_blocks', []))}.",
        f"Segmentos vectoriales depurados: {geometry.get('line_segments_after_cleanup_count', len(geometry['lines']))}.",
        f"Disciplina sugerida por el ojo inteligente: {geometry.get('discipline_guess', 'general')}.",
        f"Símbolos preliminares detectados: {len(geometry.get('electrical_symbols', [])) + len(geometry.get('sanitary_symbols', [])) + len(geometry.get('mechanical_symbols', []))}.",
        f"Bloques CAD sugeridos: {len(geometry.get('symbol_blocks', []))}. Tipos preliminares detectados: {len({item.get('nombre', '') for item in geometry.get('symbol_blocks', []) if item.get('nombre')})}.",
        f"Reglas de disciplina aplicadas: {len(geometry.get('discipline_rules', []))}.",
        f"Perfil adaptativo: memoria {runtime_profile.get('memory_mb', 0)} MB, presión {runtime_profile.get('memory_pressure', 'normal')}, OCR {'habilitado' if runtime_profile.get('ocr_allowed') else 'limitado'}.",
        f"Observaciones del usuario: {notes.strip() or 'sin notas adicionales'}.",
        f"Directivas activas: {', '.join([k for k, v in directives.items() if v]) or 'ninguna'}.",
        "La salida sigue siendo una reconstrucción asistida y debe validarse antes de utilizarse como documentación definitiva.",
    ]
    warnings = _warnings(reference_mode, quality_metrics, geometry, quality_band, calibration, ocr_warning)
    insights = _insights(
        reference_mode,
        page_detected,
        quality_metrics,
        sheet_size.upper(),
        auto_upscaled,
        sheet_orientation.upper(),
        document_orientation,
        calibration_summary,
        ocr_engine,
    )

    calibration_reliability = _calibration_reliability(reference_mode, calibration_distance_mm, calibration_pixel_span, quality_metrics["inferred_dpi"])
    geometry_stability = _geometry_stability(quality_metrics, len(geometry["lines"]), geometry.get("duplicate_lines_removed_count", 0))
    text_separation_quality = _text_separation_quality(len(text_boxes), len(ocr_items))
    precision_index = _precision_index(confidence, reference_mode, quality_metrics["inferred_dpi"], geometry_stability, text_separation_quality)
    precision_class = _precision_class(precision_index, calibration_reliability, geometry_stability)
    expected_positional_error_mm = _expected_positional_error_mm(mm_per_px, quality_metrics["inferred_dpi"], calibration_reliability)
    suggested_linear_tolerance_mm = _suggested_linear_tolerance_mm(expected_positional_error_mm, precision_class)
    precision_observations = _precision_observations(reference_mode, calibration_reliability, geometry_stability, text_separation_quality, quality_metrics["inferred_dpi"])
    precision_observations.append(f"Clase de precisión estimada: {precision_class}.")
    precision_observations.append(f"Desvío posicional estimado: ±{expected_positional_error_mm} mm.")
    precision_observations.append(f"Tolerancia lineal sugerida para revisión: ±{suggested_linear_tolerance_mm} mm.")
    recommended_precision_action = _recommended_precision_action(precision_index, calibration_reliability)

    return PipelineResult(
        image_width_px=original_w,
        image_height_px=original_h,
        processed_width_px=width_px,
        processed_height_px=height_px,
        threshold_ratio=round(threshold_ratio, 4),
        detected_line_count=len(geometry["lines"]),
        detected_contour_count=len(geometry["polylines"]),
        closed_shapes_count=geometry["closed_shapes_count"],
        detected_text_regions=len(text_boxes),
        recognized_text_count=len(ocr_items),
        estimated_scale_mm_per_px=mm_per_px,
        sheet_width_mm=sheet_w,
        sheet_height_mm=sheet_h,
        confidence_score=confidence,
        reference_mode=reference_mode,
        inferred_dpi=quality_metrics["inferred_dpi"],
        quality_band=quality_band,
        page_detected=page_detected,
        auto_upscaled=auto_upscaled,
        deskew_angle_deg=round(angle, 2) if page_detected else 0.0,
        assumptions=assumptions,
        warnings=warnings,
        insights=insights,
        quality_metrics=quality_metrics,
        geometry=geometry,
        output_files={
            "reconstruction_preview": str(reconstruction_path),
            "cleaned_image": str(cleaned_path),
            "presentation_image": str(presentation_path),
            "overlay_image": str(overlay_path),
            "original_preview": str(original_path),
        },
        sheet_orientation=sheet_orientation.upper(),
        preserved_aspect_ratio=True,
        document_orientation=document_orientation,
        calibration_summary=calibration_summary,
        calibration_distance_mm=calibration_distance_mm,
        calibration_pixel_span=calibration_pixel_span,
        ocr_engine=ocr_engine,
        detected_dimension_count=len(geometry.get("dimension_lines", [])),
        detected_title_block_count=len(geometry.get("title_blocks", [])),
        detected_label_count=len(rotulo_texts),
        detected_arrow_count=len(geometry.get("dimension_arrows", [])),
        review_text_count=len(geometry.get("review_texts", [])),
        line_segments_raw_count=geometry.get("line_segments_raw_count", 0),
        line_segments_after_cleanup_count=geometry.get("line_segments_after_cleanup_count", len(geometry["lines"])),
        duplicate_lines_removed_count=geometry.get("duplicate_lines_removed_count", 0),
        detected_symbol_count=len(geometry.get("electrical_symbols", [])) + len(geometry.get("sanitary_symbols", [])) + len(geometry.get("mechanical_symbols", [])),
        electrical_symbol_count=len(geometry.get("electrical_symbols", [])),
        sanitary_symbol_count=len(geometry.get("sanitary_symbols", [])),
        mechanical_symbol_count=len(geometry.get("mechanical_symbols", [])),
        discipline_guess=geometry.get("discipline_guess", "general"),
        suggested_block_count=len(geometry.get("symbol_blocks", [])),
        discipline_rule_count=len(geometry.get("discipline_rules", [])),
        precision_index=precision_index,
        calibration_reliability=calibration_reliability,
        geometry_stability=geometry_stability,
        text_separation_quality=text_separation_quality,
        recommended_precision_action=recommended_precision_action,
        precision_observations=precision_observations,
        precision_class=precision_class,
        expected_positional_error_mm=expected_positional_error_mm,
        suggested_linear_tolerance_mm=suggested_linear_tolerance_mm,
    )
