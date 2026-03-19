from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from reportlab.lib.colors import Color
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

SHEET_SIZES_MM: dict[str, tuple[int, int]] = {
    'A4': (210, 297),
    'A3': (297, 420),
    'A2': (420, 594),
    'A1': (594, 841),
}


def _resolve_sheet_size(sheet_size: str, orientation: str) -> tuple[float, float]:
    base_w, base_h = SHEET_SIZES_MM.get(str(sheet_size).upper(), SHEET_SIZES_MM['A3'])
    orientation = str(orientation).upper()
    if orientation == 'HORIZONTAL':
        page_w_mm, page_h_mm = max(base_w, base_h), min(base_w, base_h)
    elif orientation == 'VERTICAL':
        page_w_mm, page_h_mm = min(base_w, base_h), max(base_w, base_h)
    else:
        page_w_mm, page_h_mm = base_w, base_h
    return page_w_mm * mm, page_h_mm * mm


def _find_candidate_image(result: dict[str, Any]) -> Path | None:
    output_files = result.get('output_files', {}) if isinstance(result, dict) else {}
    for key in ('presentation_image', 'cleaned_full_image', 'cleaned_image', 'graphics_mask', 'original_preview'):
        raw = output_files.get(key)
        if raw and Path(raw).exists():
            return Path(raw)
    return None


def _crop_content(path: Path, padding_px: int = 24) -> Path:
    cropped_path = path.with_name(f'{path.stem}_cropped{path.suffix}')
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        inverted = ImageOps.invert(rgb.convert('L'))
        bbox = inverted.point(lambda p: 255 if p > 6 else 0).getbbox()
        if bbox is None:
            rgb.save(cropped_path)
            return cropped_path
        left, top, right, bottom = bbox
        left = max(0, left - padding_px)
        top = max(0, top - padding_px)
        right = min(rgb.width, right + padding_px)
        bottom = min(rgb.height, bottom + padding_px)
        rgb.crop((left, top, right, bottom)).save(cropped_path)
    return cropped_path


def build_report_pdf(output_path: Path, result: dict[str, Any], drawing_type: str, sheet_size: str, original_filename: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    orientation = str(result.get('sheet_orientation') or 'AUTO').upper()
    if orientation == 'AUTO':
        orientation = 'HORIZONTAL' if result.get('document_orientation') == 'apaisada' else 'VERTICAL'

    page_width, page_height = _resolve_sheet_size(sheet_size, orientation)
    page = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))

    frame_margin = 3.5 * mm
    footer_height = 6 * mm
    drawable_left = frame_margin
    drawable_bottom = frame_margin + footer_height
    drawable_width = page_width - 2 * frame_margin
    drawable_height = page_height - (2 * frame_margin + footer_height)

    line_color = Color(0.84, 0.87, 0.90)
    page.setStrokeColor(line_color)
    page.setLineWidth(0.5)
    page.rect(frame_margin, frame_margin, page_width - 2 * frame_margin, page_height - 2 * frame_margin)
    page.line(frame_margin, frame_margin + footer_height, page_width - frame_margin, frame_margin + footer_height)

    candidate = _find_candidate_image(result)
    cropped = _crop_content(candidate) if candidate else None
    if cropped and cropped.exists():
        with Image.open(cropped) as image:
            image_width, image_height = image.size
        if image_width > 0 and image_height > 0:
            scale = min(drawable_width / image_width, drawable_height / image_height)
            render_w = image_width * scale
            render_h = image_height * scale
            render_x = drawable_left + (drawable_width - render_w) / 2
            render_y = drawable_bottom + (drawable_height - render_h) / 2
            page.drawImage(str(cropped), render_x, render_y, width=render_w, height=render_h, preserveAspectRatio=True, mask='auto')

    page.setFillColor(Color(0.22, 0.27, 0.33))
    page.setFont('Helvetica-Bold', 6.8)
    page.drawString(frame_margin + 2.5 * mm, frame_margin + 5.6 * mm, 'TrazoCad | Tecno Logisti-K SA')

    page.setFont('Helvetica', 5.8)
    file_label = (original_filename or 'archivo_sin_nombre')[:64]
    page.drawString(frame_margin + 2.5 * mm, frame_margin + 2.0 * mm, f'Archivo: {file_label}')

    right_x = page_width - frame_margin - 2.5 * mm
    page.drawRightString(right_x, frame_margin + 5.6 * mm, f'Hoja: {sheet_size.upper()}')
    page.drawRightString(right_x, frame_margin + 2.0 * mm, f'Tipo: {drawing_type} | Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M")}')

    page.showPage()
    page.save()

    if cropped and cropped.exists() and cropped != candidate:
        try:
            cropped.unlink()
        except OSError:
            pass
    return output_path
