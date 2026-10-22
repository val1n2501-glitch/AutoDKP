from pathlib import Path

from autodkp.pdf.layout import load_layout, sha256_file


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "resources/templates/dkp-agreement_new_logo.pdf"
LAYOUT = ROOT / "src/autodkp/pdf/template_fields.json"


def intersects(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return not (
        left[2] <= right[0]
        or left[0] >= right[2]
        or left[3] <= right[1]
        or left[1] >= right[3]
    )


def test_template_fingerprint_is_expected() -> None:
    layout = load_layout(LAYOUT)
    assert sha256_file(TEMPLATE) == layout.template_sha256


def test_writable_fields_do_not_intersect_protected_regions() -> None:
    layout = load_layout(LAYOUT)
    for field in layout.fields:
        for protected in layout.protected_regions.values():
            assert not intersects(field.rect, protected), field.name
