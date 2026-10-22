from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path

import pymupdf

from autodkp.config import Settings
from autodkp.domain.models import DealDraft, Person

from .errors import (
    FieldOverflowError,
    PdfRenderError,
    ProtectedRegionError,
    TemplateMismatchError,
)
from .layout import FieldLayout, TemplateLayout, load_layout, sha256_file
from .price_words import number_in_words


# A last-resort width reduction for unusually long one-line legal values.
# The font height stays readable; only the glyph width is reduced.  Normal
# values never reach this fallback and therefore keep their existing layout.
_MIN_HORIZONTAL_SCALE = 0.65


def _format_date(value: date) -> str:
    return value.strftime("%d.%m.%Y")


def _format_party(person: Person) -> str:
    return (
        f"{person.full_name}, {_format_date(person.birth_date)} года рождения, "
        f"место рождения: {person.birth_place}, адрес места жительства: {person.address}\n"
        f"паспорт: серия {person.passport_series} № {person.passport_number}, "
        f"выдан {person.passport_issued_by} {_format_date(person.passport_issue_date)}"
    )


def _required_render_value(field_name: str, value: str | None) -> str:
    if value is None or not value.strip():
        raise PdfRenderError(f"UNCONFIRMED_MISSING_FIELD: {field_name}")
    return value


_MONTHS_GENITIVE = (
    "",
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)


def _rects_intersect(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] <= right[0]
        or left[0] >= right[2]
        or left[3] <= right[1]
        or left[1] >= right[3]
    )


class PdfRenderer:
    def __init__(
        self,
        template_path: Path,
        layout_path: Path,
        font_path: Path,
    ) -> None:
        self.template_path = template_path
        self.layout = load_layout(layout_path)
        self.font_path = font_path

    @classmethod
    def from_defaults(cls, settings: Settings) -> "PdfRenderer":
        layout = Path(__file__).with_name("template_fields.json")
        return cls(settings.template_path, layout, settings.pdf_font_path)

    def _validate_template(self) -> None:
        actual_hash = sha256_file(self.template_path)
        if actual_hash != self.layout.template_sha256:
            raise TemplateMismatchError(
                f"Unexpected PDF template hash: {actual_hash}"
            )
        if not self.font_path.is_file():
            raise PdfRenderError(f"Cyrillic font not found: {self.font_path}")
        with pymupdf.open(self.template_path) as document:
            if len(document) != 1:
                raise TemplateMismatchError("Template must contain exactly one page")
            page = document[0]
            expected_width, expected_height = self.layout.page_size
            if (
                abs(page.rect.width - expected_width) > 0.02
                or abs(page.rect.height - expected_height) > 0.02
                or page.rotation != 0
            ):
                raise TemplateMismatchError("Unexpected template page geometry")

        for field in self.layout.fields:
            for region_name, protected in self.layout.protected_regions.items():
                if _rects_intersect(field.rect, protected):
                    raise ProtectedRegionError(
                        f"Field {field.name} intersects {region_name}"
                    )

    @staticmethod
    def _values(deal: DealDraft) -> dict[str, str]:
        vehicle = deal.vehicle
        price_digits = f"{deal.price_rubles:,}".replace(",", " ")
        buyer_lines = _format_party(deal.buyer).splitlines()
        seller_lines = _format_party(deal.seller).splitlines()
        return {
            "agreement_place": deal.agreement_place,
            "agreement_date": _format_date(deal.agreement_date),
            "buyer_details_line1": buyer_lines[0],
            "buyer_details_line2": buyer_lines[1],
            "seller_details_line1": seller_lines[0],
            "seller_details_line2": seller_lines[1],
            "vehicle_make_model": vehicle.make_model,
            "vehicle_vin": _required_render_value("vehicle.vin", vehicle.vin),
            "vehicle_type": vehicle.vehicle_type,
            "vehicle_year": str(vehicle.manufacture_year),
            "vehicle_mileage": str(vehicle.mileage_km),
            "vehicle_power": format(vehicle.engine_power_hp, "f"),
            "vehicle_displacement": str(vehicle.engine_displacement_cc),
            "vehicle_color": vehicle.body_color,
            "engine_model": _required_render_value("vehicle.engine_model", vehicle.engine_model),
            "engine_number": _required_render_value("vehicle.engine_number", vehicle.engine_number),
            "chassis_number": _required_render_value("vehicle.chassis_number", vehicle.chassis_number),
            "body_number": _required_render_value("vehicle.body_number", vehicle.body_number),
            "pts_series_number": vehicle.pts.series_number,
            "pts_issued_by": vehicle.pts.issued_by,
            "pts_issue_date": _format_date(vehicle.pts.issue_date),
            "registration_plate": _required_render_value(
                "vehicle.registration_plate", vehicle.registration_plate
            ),
            "sts_series": " ".join(vehicle.sts.series_number.split()[:-1]),
            "sts_number": vehicle.sts.series_number.split()[-1],
            "sts_issued_by": vehicle.sts.issued_by,
            "sts_issue_day": vehicle.sts.issue_date.strftime("%d"),
            "sts_issue_month": _MONTHS_GENITIVE[vehicle.sts.issue_date.month],
            "sts_issue_year": vehicle.sts.issue_date.strftime("%Y"),
            "price_digits": price_digits,
            "price_words": number_in_words(deal.price_rubles),
        }

    @staticmethod
    def _alignment(value: str) -> int:
        return {"left": 0, "center": 1, "right": 2}[value]

    def _insert_fitted(
        self,
        page: pymupdf.Page,
        field: FieldLayout,
        value: str,
        font_name: str,
    ) -> None:
        size = field.font_size_max
        font = pymupdf.Font(fontfile=str(self.font_path))
        while size + 1e-6 >= field.font_size_min:
            shape = page.new_shape()
            field_rect = pymupdf.Rect(field.rect)
            if field.vertical_align == "bottom":
                text_height = (font.ascender - font.descender) * size
                field_rect.y0 = field.rect[3] - text_height
                # PyMuPDF needs a little layout slack below the intended glyph
                # box; this does not move the text below the target baseline.
                field_rect.y1 = field.rect[3] + 2.5
            spare_height = shape.insert_textbox(
                field_rect,
                value,
                fontname=font_name,
                fontfile=str(self.font_path),
                fontsize=size,
                lineheight=field.line_height,
                align=self._alignment(field.align),
                color=(0, 0, 0),
            )
            if spare_height >= 0:
                shape.commit(overlay=True)
                return
            size -= 0.25

        if field.max_lines == 1 and self._insert_condensed_line(
            page, field, value, font_name, font
        ):
            return
        raise FieldOverflowError(field.name)

    def _insert_condensed_line(
        self,
        page: pymupdf.Page,
        field: FieldLayout,
        value: str,
        font_name: str,
        font: pymupdf.Font,
    ) -> bool:
        """Fit one exceptional line by narrowing glyphs without truncation."""
        if "\n" in value or not value:
            return False

        size = field.font_size_min
        natural_width = font.text_length(value, fontsize=size)
        available_width = field.rect[2] - field.rect[0]
        if natural_width <= 0:
            return False
        horizontal_scale = min(1.0, available_width / natural_width)
        if horizontal_scale + 1e-6 < _MIN_HORIZONTAL_SCALE:
            return False

        visual_width = natural_width * horizontal_scale
        if field.align == "center":
            x = field.rect[0] + (available_width - visual_width) / 2
        elif field.align == "right":
            x = field.rect[2] - visual_width
        else:
            x = field.rect[0]

        if field.vertical_align == "bottom":
            baseline = field.rect[3] + font.descender * size
        else:
            baseline = field.rect[1] + font.ascender * size

        shape = page.new_shape()
        shape.insert_text(
            pymupdf.Point(x, baseline),
            value,
            fontname=font_name,
            fontfile=str(self.font_path),
            fontsize=size,
            color=(0, 0, 0),
            morph=(
                pymupdf.Point(x, baseline),
                pymupdf.Matrix(horizontal_scale, 1.0),
            ),
        )
        shape.commit(overlay=True)
        return True

    def render(self, deal: DealDraft, output_path: Path) -> Path:
        self._validate_template()
        values = self._values(deal)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".pdf",
            dir=output_path.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            with pymupdf.open(self.template_path) as document:
                page = document[0]
                font_name = "autodkp-serif"
                for field in self.layout.fields:
                    self._insert_fitted(page, field, values[field.name], font_name)
                document.save(temporary_path, garbage=4, deflate=True)

            with pymupdf.open(temporary_path) as rendered:
                if len(rendered) != 1 or rendered[0].rotation != 0:
                    raise TemplateMismatchError("Generated PDF geometry is invalid")
                extracted = " ".join(rendered[0].get_text().split())
                for marker in (deal.buyer.full_name, deal.seller.full_name, vehicle_marker(deal)):
                    if " ".join(marker.split()) not in extracted:
                        raise PdfRenderError(f"Generated PDF is missing marker: {marker}")
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, output_path)
            return output_path
        finally:
            temporary_path.unlink(missing_ok=True)


def vehicle_marker(deal: DealDraft) -> str:
    return _required_render_value("vehicle.vin", deal.vehicle.vin)
