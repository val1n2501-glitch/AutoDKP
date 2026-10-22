from autodkp.domain.demo import make_demo_deal
from autodkp.pdf.renderer import _format_party


def test_demo_deal_uses_only_synthetic_values() -> None:
    deal = make_demo_deal()
    assert deal.buyer.passport_number == "345678"
    assert deal.seller.passport_number == "123456"
    assert deal.vehicle.vin == "XW7BF4FK90S123456"


def test_party_details_are_compact_across_two_template_lines() -> None:
    assert _format_party(make_demo_deal().buyer).count("\n") == 1

