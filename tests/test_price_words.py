import pytest

from autodkp.pdf.price_words import rubles_in_words


@pytest.mark.parametrize(
    ("amount", "suffix"),
    [(1, "рубль"), (2, "рубля"), (5, "рублей"), (11, "рублей"), (21, "рубль")],
)
def test_ruble_declension(amount: int, suffix: str) -> None:
    assert rubles_in_words(amount).endswith(suffix)

