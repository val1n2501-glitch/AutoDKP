from pathlib import Path

import pymupdf
import pytest

from autodkp.config import Settings
from autodkp.domain.demo import make_demo_deal
from autodkp.pdf.layout import load_layout
from autodkp.pdf.renderer import PdfRenderer
from autodkp.pdf.errors import FieldOverflowError, PdfRenderError


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "resources/templates/dkp-agreement_new_logo.pdf"
LAYOUT = ROOT / "src/autodkp/pdf/template_fields.json"
FONT = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")


def _roi_bytes(page: pymupdf.Page, rect: tuple[float, ...], scale: float = 2) -> bytes:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    x0, y0, x1, y1 = (round(value * scale) for value in rect)
    channels = pix.n
    rows = []
    for y in range(y0, y1):
        start = (y * pix.width + x0) * channels
        end = (y * pix.width + x1) * channels
        rows.append(pix.samples[start:end])
    return b"".join(rows)


def test_demo_pdf_preserves_signature_regions(tmp_path: Path) -> None:
    output = tmp_path / "demo.pdf"
    renderer = PdfRenderer(TEMPLATE, LAYOUT, FONT)
    renderer.render(make_demo_deal(), output)
    layout = load_layout(LAYOUT)

    with pymupdf.open(TEMPLATE) as source, pymupdf.open(output) as result:
        assert len(result) == 1
        assert result[0].rect == source[0].rect
        assert result[0].rotation == 0
        catalog = result.pdf_catalog()
        assert result.xref_get_key(catalog, "AcroForm")[0] == "null"
        for name, rect in layout.protected_regions.items():
            assert _roi_bytes(source[0], rect) == _roi_bytes(result[0], rect), name

        text = " ".join(result[0].get_text().split())
        deal = make_demo_deal()
        assert deal.buyer.full_name in text
        assert deal.seller.full_name in text
        assert deal.vehicle.vin in text


def test_missing_vehicle_identifier_is_not_invented(tmp_path: Path) -> None:
    output = tmp_path / "missing-vin.pdf"
    deal = make_demo_deal()
    deal.vehicle.vin = None
    renderer = PdfRenderer(TEMPLATE, LAYOUT, FONT)
    with pytest.raises(PdfRenderError, match="UNCONFIRMED_MISSING_FIELD: vehicle.vin"):
        renderer.render(deal, output)
    assert not output.exists()


def test_long_sts_issuer_uses_dynamic_font_size(tmp_path: Path) -> None:
    output = tmp_path / "long-sts-issuer.pdf"
    deal = make_demo_deal()
    long_issuer = (
        "МРЭО ГИБДД ГУ МВД России по Московской области, "
        "регистрационный отдел № 3"
    )
    deal.vehicle.sts.issued_by = long_issuer
    PdfRenderer(TEMPLATE, LAYOUT, FONT).render(deal, output)
    with pymupdf.open(output) as document:
        extracted = " ".join(document[0].get_text().split())
        assert " ".join(long_issuer.split()) in extracted
        matching_sizes = [
            span["size"]
            for block in document[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
            if "регистрационный" in span["text"]
        ]
        assert matching_sizes and matching_sizes[0] < 7.8


def test_very_long_sts_issuer_is_condensed_without_truncation(tmp_path: Path) -> None:
    output = tmp_path / "very-long-sts-issuer.pdf"
    deal = make_demo_deal()
    long_issuer = (
        "Межрайонный регистрационно-экзаменационный отдел Государственной "
        "инспекции безопасности дорожного движения Главного управления МВД "
        "России по Московской области"
    )
    deal.vehicle.sts.issued_by = long_issuer

    PdfRenderer(TEMPLATE, LAYOUT, FONT).render(deal, output)

    with pymupdf.open(output) as document:
        page = document[0]
        extracted = " ".join(page.get_text().replace("\u00ad", "-").split())
        assert " ".join(long_issuer.split()) in extracted
        issuer_spans = [
            span
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
            if "Межрайонный" in span["text"]
        ]
        assert issuer_spans
        assert issuer_spans[0]["bbox"][2] <= 451.13 + 0.1


def test_extreme_sts_issuer_overflow_is_explicit(tmp_path: Path) -> None:
    output = tmp_path / "overflow-sts-issuer.pdf"
    deal = make_demo_deal()
    deal.vehicle.sts.issued_by = "ОЧЕНЬ ДЛИННОЕ НАЗВАНИЕ ПОДРАЗДЕЛЕНИЯ " * 20
    with pytest.raises(FieldOverflowError, match="sts_issued_by"):
        PdfRenderer(TEMPLATE, LAYOUT, FONT).render(deal, output)
    assert not output.exists()
