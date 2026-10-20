"""Deterministic intent engine: answers questions and executes allowlisted
commands against the fleet snapshot. This is the offline brain — the app is
fully useful with the LLM backend disabled. The LLM layer only handles
questions the rules don't match (see service.py)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .fleet import FleetSnapshot, VehicleState

ALLOWED_COMMANDS = {"charge_start", "charge_stop", "climate_start",
                    "climate_stop", "sentry_on", "sentry_off"}

COMMAND_VERBS = ("charge_start", "charge_stop", "climate_start",
                 "climate_stop", "sentry_on", "sentry_off")


@dataclass
class Answer:
    text: str
    matched: bool = True
    commands_run: list[str] = field(default_factory=list)
    dry_run_notes: list[str] = field(default_factory=list)


def resolve_vehicle(snap: FleetSnapshot, text: str) -> VehicleState | None:
    """Pick the vehicle a user text refers to (name/model mention, else None)."""
    t = text.lower()
    for v in snap.vehicles:
        tokens = [v.name.lower()] + ([v.display_name.lower()] if v.display_name else [])
        for tok in tokens:
            if tok and tok in t:
                return v
    for v in snap.vehicles:
        for m in re.findall(r"model\s*[3yxs]", t):
            if m[-1] in v.name.lower():
                return v
    if len(snap.vehicles) == 1:
        return snap.vehicles[0]
    return None


def _fmt_temp(c: float | None) -> str:
    return f"{c:.0f}°C" if c is not None else "unknown"


def _status_line(v: VehicleState) -> str:
    if not v.reachable:
        return f"{v.name}: unreachable ({v.error})."
    return (f"{v.name}: battery {v.battery_level}%, {v.charging_state.lower()}, "
            f"climate {'on' if v.is_climate_on else 'off'} "
            f"(inside {_fmt_temp(v.inside_temp_c)}), "
            f"sentry {'on' if v.sentry_mode else 'off'}, "
            f"{'locked' if v.locked else 'unlocked'}, {v.odometer:.0f} mi.")


def detect_command(text: str) -> str | None:
    """Map imperative phrasing to an allowlisted command, or None."""
    t = text.lower()
    if re.search(r"\b(start|begin|resume)\b[^.]*\bcharg", t):
        return "charge_start"
    if re.search(r"\b(stop|pause|halt|end)\b[^.]*\bcharg", t):
        return "charge_stop"
    if re.search(r"\b(precondition|warm up|cool down|heat up|turn on|start)\b[^.]*\b(climate|a/?c|air|heat|precondition)", t) \
            or "precondition" in t:
        return "climate_start"
    if re.search(r"\b(turn off|stop)\b[^.]*\b(climate|a/?c|air|heat)", t):
        return "climate_stop"
    if re.search(r"\b(turn on|enable|arm)\b[^.]*\bsentry", t):
        return "sentry_on"
    if re.search(r"\b(turn off|disable|disarm)\b[^.]*\bsentry", t):
        return "sentry_off"
    return None


def execute_command(client, command: str, v: VehicleState, dry_run: bool) -> str:
    """Run one allowlisted command (already validated) against a vehicle."""
    if dry_run:
        return f"[dry-run] would run {command} on {v.name}"
    if command == "charge_start":
        client.charge_start(v.vin)
    elif command == "charge_stop":
        client.charge_stop(v.vin)
    elif command == "climate_start":
        client.auto_conditioning_start(v.vin)
    elif command == "climate_stop":
        client.auto_conditioning_stop(v.vin)
    elif command == "sentry_on":
        client.set_sentry_mode(v.vin, True)
    elif command == "sentry_off":
        client.set_sentry_mode(v.vin, False)
    else:  # pragma: no cover - guarded by ALLOWED_COMMANDS
        raise ValueError(f"unknown command {command}")
    return f"Ran {command} on {v.name}."


def answer_question(client, snap: FleetSnapshot, question: str, dry_run: bool) -> Answer:
    """Rules-first answering. Unmatched questions return matched=False so the
    service layer can escalate to the LLM (or the capability list)."""
    t = question.lower().strip()
    target = resolve_vehicle(snap, question)

    command = detect_command(question)
    if command:
        if target is None:
            names = ", ".join(v.name for v in snap.vehicles)
            return Answer(f"Which car? You have: {names}. Say e.g. "
                          f"'{command.replace('_', ' ')} the <name>'.")
        note = execute_command(client, command, target, dry_run)
        return Answer(note, commands_run=[] if dry_run else [f"{command}:{target.name}"],
                      dry_run_notes=[note] if dry_run else [])

    if any(w in t for w in ("which car", "which one", "more battery", "higher battery",
                            "most charge", "more charge")):
        ok = [v for v in snap.vehicles if v.reachable]
        if not ok:
            return Answer("I couldn't reach any of the cars right now.")
        best = max(ok, key=lambda v: v.battery_level)
        others = ", ".join(f"{v.name} at {v.battery_level}%" for v in ok if v is not best)
        return Answer(f"{best.name} has the most charge at {best.battery_level}%"
                      + (f" ({others})." if others else "."))

    if "battery" in t or "charge level" in t or "%" in t:
        if target:
            return Answer(f"{target.name} is at {target.battery_level}%.")
        return Answer(" ".join(f"{v.name}: {v.battery_level}%."
                               for v in snap.vehicles))

    if "charging" in t or "plugged" in t:
        if target:
            return Answer(f"{target.name}: charging state is {target.charging_state}.")
        return Answer(" ".join(f"{v.name}: {v.charging_state}."
                               for v in snap.vehicles))

    if "temperature" in t or "temp" in t or "hot" in t or "cold" in t:
        if target:
            return Answer(f"{target.name}: inside {_fmt_temp(target.inside_temp_c)}, "
                          f"outside {_fmt_temp(target.outside_temp_c)}.")
        return Answer(" ".join(f"{v.name}: inside {_fmt_temp(v.inside_temp_c)}."
                               for v in snap.vehicles))

    if "lock" in t:
        if target:
            return Answer(f"{target.name} is {'locked' if target.locked else 'unlocked'}.")
        return Answer(" ".join(f"{v.name}: {'locked' if v.locked else 'unlocked'}."
                               for v in snap.vehicles))

    if "sentry" in t:
        if target:
            return Answer(f"Sentry Mode is {'on' if target.sentry_mode else 'off'} "
                          f"on {target.name}.")
        return Answer(" ".join(f"{v.name}: sentry {'on' if v.sentry_mode else 'off'}."
                               for v in snap.vehicles))

    if "odometer" in t or "mileage" in t or "miles" in t:
        if target:
            return Answer(f"{target.name} has {target.odometer:.0f} miles.")
        return Answer(" ".join(f"{v.name}: {v.odometer:.0f} mi." for v in snap.vehicles))

    if any(w in t for w in ("status", "summary", "how is", "how are", "everything")):
        return Answer(" ".join(_status_line(v) for v in snap.vehicles))

    return Answer("", matched=False)
