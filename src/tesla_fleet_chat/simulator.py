"""Fake fleet for tests and demos. No API calls, no real cars."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeVehicle:
    vin: str
    name: str
    battery_level: int = 80
    charging_state: str = "Disconnected"
    inside_temp: float = 24.0
    outside_temp: float = 30.0
    climate_on: bool = False
    sentry_mode: bool = False
    locked: bool = True
    odometer: float = 12345.0


class FakeTeslaClient:
    """Stand-in for TeslaClient with the methods fleet-chat uses."""

    def __init__(self, vehicles: list[FakeVehicle]):
        self.vehicles = {v.vin: v for v in vehicles}
        self.command_log: list[tuple[str, str]] = []

    def vehicle_data(self, vin: str) -> dict:
        v = self.vehicles[vin]
        return {
            "display_name": v.name,
            "locked": v.locked,
            "sentry_mode": v.sentry_mode,
            "charge_state": {"battery_level": v.battery_level,
                             "charging_state": v.charging_state},
            "climate_state": {"inside_temp": v.inside_temp,
                              "outside_temp": v.outside_temp,
                              "is_climate_on": v.climate_on},
            "drive_state": {"shift_state": None, "odometer": v.odometer},
        }

    def charge_start(self, vin: str) -> dict:
        self.vehicles[vin].charging_state = "Charging"
        self.command_log.append(("charge_start", vin))
        return {"result": True}

    def charge_stop(self, vin: str) -> dict:
        self.vehicles[vin].charging_state = "Stopped"
        self.command_log.append(("charge_stop", vin))
        return {"result": True}

    def auto_conditioning_start(self, vin: str) -> dict:
        self.vehicles[vin].climate_on = True
        self.command_log.append(("auto_conditioning_start", vin))
        return {"result": True}

    def auto_conditioning_stop(self, vin: str) -> dict:
        self.vehicles[vin].climate_on = False
        self.command_log.append(("auto_conditioning_stop", vin))
        return {"result": True}

    def set_sentry_mode(self, vin: str, on: bool) -> dict:
        self.vehicles[vin].sentry_mode = on
        self.command_log.append(("set_sentry_mode", vin))
        return {"result": True}


def demo_vehicles() -> list[FakeVehicle]:
    return [
        FakeVehicle(vin="VINY", name="Model Y", battery_level=82,
                    charging_state="Disconnected", odometer=18420.0),
        FakeVehicle(vin="VIN3", name="Model 3", battery_level=47,
                    charging_state="Charging", odometer=32110.0),
    ]


DEMO_QUESTIONS = [
    "Which car has more battery?",
    "Is the Model 3 charging?",
    "start charging the Model Y",
    "What's the status of everything?",
]


def run_demo(settings) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """Run the demo Q&A through the real service against the fake fleet.
    Returns (question, answer, command_log) triples. LLM disabled."""
    from .config import VehicleConfig
    from .service import FleetChatService

    vehicles = demo_vehicles()
    settings.vehicles = [VehicleConfig(name=v.name, vin=v.vin) for v in vehicles]
    fake = FakeTeslaClient(vehicles)
    service = FleetChatService(settings, client_factory=lambda: fake,
                               llm_factory=lambda: None)
    results = []
    for q in DEMO_QUESTIONS:
        body = {"model": settings.model_id,
                "messages": [{"role": "user", "content": q}]}
        resp = service.chat_completions(body)
        results.append((q, resp["choices"][0]["message"]["content"],
                        list(fake.command_log)))
    return results
