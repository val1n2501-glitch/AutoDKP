from datetime import date
from decimal import Decimal

from .models import DealDraft, Person, Vehicle, VehicleDocument


def make_demo_deal() -> DealDraft:
    return DealDraft(
        agreement_place="г. Москва",
        agreement_date=date(2026, 10, 21),
        buyer=Person(
            full_name="Петров Пётр Петрович",
            birth_date=date(1988, 2, 14),
            birth_place="г. Казань",
            address="г. Москва, ул. Примерная, д. 15, кв. 24",
            passport_series="4512",
            passport_number="345678",
            passport_issued_by="ОВМ ОМВД России по району Примерный г. Москвы",
            passport_issue_date=date(2018, 3, 20),
        ),
        seller=Person(
            full_name="Иванов Иван Иванович",
            birth_date=date(1985, 7, 9),
            birth_place="г. Тверь",
            address="г. Москва, пр-т Тестовый, д. 7, кв. 11",
            passport_series="4509",
            passport_number="123456",
            passport_issued_by="Отделом УФМС России по г. Москве",
            passport_issue_date=date(2015, 8, 12),
        ),
        vehicle=Vehicle(
            make_model="Toyota Camry",
            vin="XW7BF4FK90S123456",
            vehicle_type="легковой седан",
            manufacture_year=2020,
            mileage_km=54_321,
            engine_power_hp=Decimal("181"),
            engine_displacement_cc=2494,
            body_color="чёрный",
            engine_model="2AR",
            engine_number="A123456",
            chassis_number="отсутствует",
            body_number="XW7BF4FK90S123456",
            registration_plate="А123АА777",
            pts=VehicleDocument(
                series_number="77 УА 123456",
                issued_by="ЦЭП ГИБДД ГУ МВД России по г. Москве",
                issue_date=date(2020, 6, 15),
            ),
            sts=VehicleDocument(
                series_number="99 12 345678",
                issued_by="МО ГИБДД ТНРЭР № 1 ГУ МВД России по г. Москве",
                issue_date=date(2020, 6, 20),
            ),
        ),
        price_rubles=2_350_000,
    )

