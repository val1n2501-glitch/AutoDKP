class PdfRenderError(RuntimeError):
    """Base error for deterministic PDF generation failures."""


class TemplateMismatchError(PdfRenderError):
    """The template does not match the coordinate map."""


class FieldOverflowError(PdfRenderError):
    def __init__(self, field_name: str) -> None:
        super().__init__(f"FIELD_OVERFLOW: {field_name}")
        self.field_name = field_name


class ProtectedRegionError(PdfRenderError):
    """A configured field intersects a protected part of the template."""

