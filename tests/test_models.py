from autodkp.domain.demo import make_demo_deal
from autodkp.domain.models import Vehicle
from autodkp.pdf.renderer import _format_party


def test_demo_deal_uses_only_synthetic_values() -> None:
    deal = make_demo_deal()
    assert deal.buyer.passport_number == "345678"
    assert deal.seller.passport_number == "123456"
    assert deal.vehicle.vin == "XW7BF4FK90S123456"


def test_party_details_use_all_four_template_lines() -> None:
    lines = _format_party(make_demo_deal().buyer).splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("Петров Пётр Петрович")
    assert lines[1].startswith("место рождения:")
    assert lines[2].startswith("адрес места жительства:")
    assert lines[3].startswith("паспорт:")


def test_body_number_is_normalized_without_spaces() -> None:
    vehicle_data = make_demo_deal().vehicle.model_dump()
    vehicle_data["body_number"] = "XW7 BF4 FK90 S123456"
    vehicle = Vehicle.model_validate(vehicle_data)
    assert vehicle.body_number == "XW7BF4FK90S123456"

