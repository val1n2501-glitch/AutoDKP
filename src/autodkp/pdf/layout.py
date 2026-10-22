from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FieldLayout:
    name: str
    rect: tuple[float, float, float, float]
    font_size_max: float
    font_size_min: float
    max_lines: int
    align: str = "left"
    line_height: float = 1.05
    vertical_align: str = "top"


@dataclass(frozen=True)
class TemplateLayout:
    template_sha256: str
    page_size: tuple[float, float]
    fields: tuple[FieldLayout, ...]
    protected_regions: dict[str, tuple[float, float, float, float]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_layout(path: Path) -> TemplateLayout:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TemplateLayout(
        template_sha256=raw["template_sha256"],
        page_size=tuple(raw["page_size"]),
        fields=tuple(
            FieldLayout(
                name=name,
                rect=tuple(spec["rect"]),
                font_size_max=spec["font_size_max"],
                font_size_min=spec["font_size_min"],
                max_lines=spec["max_lines"],
                align=spec.get("align", "left"),
                line_height=spec.get("line_height", 1.05),
                vertical_align=spec.get("vertical_align", "top"),
            )
            for name, spec in raw["fields"].items()
        ),
        protected_regions={
            name: tuple(rect) for name, rect in raw["protected_regions"].items()
        },
    )

