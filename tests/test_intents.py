from tesla_fleet_chat.config import VehicleConfig
from tesla_fleet_chat.fleet import snapshot_fleet
from tesla_fleet_chat.intents import answer_question, detect_command
from tesla_fleet_chat.simulator import FakeTeslaClient, demo_vehicles

VEHICLES = [VehicleConfig(name="Model Y", vin="VINY"),
            VehicleConfig(name="Model 3", vin="VIN3")]


def setup():
    fake = FakeTeslaClient(demo_vehicles())
    return fake, snapshot_fleet(fake, VEHICLES)


def test_which_car_more_battery():
    fake, snap = setup()
    a = answer_question(fake, snap, "Which car has more battery?", False)
    assert "Model Y" in a.text and "82%" in a.text


def test_battery_single_car():
    fake, snap = setup()
    a = answer_question(fake, snap, "What's the Model 3 battery?", False)
    assert "47%" in a.text


def test_charging_question():
    fake, snap = setup()
    a = answer_question(fake, snap, "Is the Model 3 charging?", False)
    assert "Charging" in a.text


def test_start_charging_executes():
    fake, snap = setup()
    a = answer_question(fake, snap, "start charging the Model Y", False)
    assert fake.command_log == [("charge_start", "VINY")]
    assert a.commands_run == ["charge_start:Model Y"]


def test_start_charging_dry_run():
    fake, snap = setup()
    a = answer_question(fake, snap, "start charging the Model Y", True)
    assert fake.command_log == []
    assert "dry-run" in a.text


def test_command_without_car_asks_which():
    fake, snap = setup()
    a = answer_question(fake, snap, "start charging", False)
    assert "Which car" in a.text and fake.command_log == []


def test_precondition_maps_to_climate_start():
    fake, snap = setup()
    a = answer_question(fake, snap, "precondition the Model 3", False)
    assert fake.command_log == [("auto_conditioning_start", "VIN3")]


def test_sentry_on():
    fake, snap = setup()
    answer_question(fake, snap, "turn on sentry on the Model Y", False)
    assert fake.command_log == [("set_sentry_mode", "VINY")]
    assert fake.vehicles["VINY"].sentry_mode is True


def test_unknown_question_unmatched():
    fake, snap = setup()
    a = answer_question(fake, snap, "what's the meaning of life?", False)
    assert a.matched is False


def test_detect_command_variants():
    assert detect_command("please stop charging") == "charge_stop"
    assert detect_command("resume charging please") == "charge_start"
    assert detect_command("how much battery is left") is None
