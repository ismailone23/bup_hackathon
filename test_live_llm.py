"""Opt-in live model tests: 10 hard interpretation cases.

These call the real OpenAI model, so they are skipped unless
GRIDWISE_LIVE_TESTS=1 is set. The default pytest run stays offline.

Run:
    $env:GRIDWISE_LIVE_TESTS="1"; python -m pytest -q test_live_llm.py -v
"""

import asyncio
import os

import pytest

from main import OptimizeRequest, interpret_notes

pytestmark = pytest.mark.skipif(
    os.getenv("GRIDWISE_LIVE_TESTS") != "1",
    reason="set GRIDWISE_LIVE_TESTS=1 to run live model tests",
)

CAPACITY = 80.0


def request(notes: list[str]) -> OptimizeRequest:
    return OptimizeRequest.model_validate({
        "scenario_id": "live",
        "operator_notes": notes,
        "hours": [
            {"hour": h, "demand_kwh": 50.0, "solar_kwh": 30.0 if 9 <= h <= 16 else 0.0, "tariff_bdt_per_kwh": 4.0 if h < 18 else 12.0}
            for h in range(24)
        ],
        "battery": {
            "capacity_kwh": CAPACITY,
            "initial_energy_kwh": 40.0,
            "minimum_energy_kwh": 0.0,
            "max_charge_kwh_per_hour": 20.0,
            "max_discharge_kwh_per_hour": 20.0,
        },
    })


# notes, expected per-note (directive_type, hours set or None, numeric value or None)
CASES = [
    (
        "words-paraphrase-cross-midnight",
        ["Do not charge from ten at night until six in the morning."],
        [("no_charge_window", {0, 1, 2, 3, 4, 5, 22, 23}, None)],
    ),
    (
        "percentage-by",
        ["Solar output drops by 80% between 1 PM and 3 PM."],
        [("solar_reduction", {13, 14}, 0.2)],
    ),
    (
        "percentage-to",
        ["Keep solar at 75% from 11 AM to 1 PM."],
        [("solar_reduction", {11, 12}, 0.75)],
    ),
    (
        "fraction-reserve-half",
        ["Keep at least half the battery capacity in reserve from 6 PM to 9 PM."],
        [("minimum_battery_reserve", {18, 19, 20}, 40.0)],
    ),
    (
        "fraction-reserve-quarter",
        ["Reserve a quarter of the battery capacity from midnight to 4 AM."],
        [("minimum_battery_reserve", {0, 1, 2, 3}, 20.0)],
    ),
    (
        "grid-cap-paraphrase",
        ["Limit grid imports to 30 kWh during 7 PM to 9 PM."],
        [("max_grid_window", {19, 20}, 30.0)],
    ),
    (
        "no-discharge-paraphrase",
        ["Suspend battery discharge between 5 PM and 8 PM."],
        [("no_discharge_window", {17, 18, 19}, None)],
    ),
    (
        "distractor-only",
        ["The cafeteria will serve biryani on Friday."],
        [("no_op", None, None)],
    ),
    (
        "multi-note-mapping",
        [
            "Do not charge the battery from 2 AM until 5 AM.",
            "The library closes early this week.",
            "Reduce solar by 50% from 1 PM to 4 PM.",
        ],
        [
            ("no_charge_window", {2, 3, 4}, None),
            ("no_op", None, None),
            ("solar_reduction", {13, 14, 15}, 0.5),
        ],
    ),
    (
        "all-day-no-charge",
        ["The battery must not charge at all today."],
        [("no_charge_window", set(range(24)), None)],
    ),
]


def check(directives, expected):
    assert len(directives) == len(expected), f"got {len(directives)} directives for {len(expected)} notes"
    for directive, (kind, hours, value) in zip(directives, expected):
        assert directive.directive_type.value == kind, (
            f"note {directive.note_index}: expected {kind}, got {directive.directive_type.value}"
        )
        if kind == "no_op":
            assert directive.applies is False and directive.structured_adjustment is None
            continue
        assert directive.applies is True, f"note {directive.note_index}: applies must be true"
        assert set(directive.structured_adjustment.hours) == hours, (
            f"note {directive.note_index}: expected hours {sorted(hours)}, got {directive.structured_adjustment.hours}"
        )
        if value is not None:
            field = {"solar_reduction": "factor", "minimum_battery_reserve": "minimum_energy_kwh", "max_grid_window": "max_grid_kwh"}[kind]
            actual = getattr(directive.structured_adjustment, field)
            assert actual is not None and abs(actual - value) < 1e-6, (
                f"note {directive.note_index}: expected {field}={value}, got {actual}"
            )


@pytest.mark.parametrize("name,notes,expected", CASES, ids=[case[0] for case in CASES])
def test_live_hard_case(name, notes, expected):
    directives = asyncio.run(interpret_notes(request(notes)))
    check(directives, expected)
