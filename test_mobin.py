import asyncio
import os

import pytest

from main import (
    DirectiveInterpretation,
    OptimizeRequest,
    StructuredAdjustment,
    interpret_notes,
    solve,
    validate_directives,
)

NEEDS_KEY = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="needs live OpenAI model")


def _mobin_hours():
    return [{"hour": h, "demand_kwh": 50.0,
             "solar_kwh": 30.0 if 9 <= h <= 16 else 0.0,
             "tariff_bdt_per_kwh": 4.0 if h < 18 else 12.0} for h in range(24)]


def _mobin_battery():
    return {"capacity_kwh": 80.0, "initial_energy_kwh": 40.0,
            "minimum_energy_kwh": 0.0, "max_charge_kwh_per_hour": 20.0,
            "max_discharge_kwh_per_hour": 20.0}


def test_mobin_rejects_no_op_with_adjustment():
    item = DirectiveInterpretation(
        note_index=0,
        applies=False,
        directive_type="no_op",
        structured_adjustment=StructuredAdjustment(hours=[1]),
        explanation="invalid no-op",
    )
    with pytest.raises(ValueError):
        validate_directives([item], 1, 20)


@NEEDS_KEY
def test_mobin_cross_midnight_ten_pm_to_six_am():
    """A 10 PM to 6 AM window covers hours 22, 23, then 0 through 5."""
    req = OptimizeRequest.model_validate({
        "scenario_id": "MUST-01",
        "operator_notes": ["Do not charge from 10 PM to 6 AM."],
        "hours": _mobin_hours(),
        "battery": _mobin_battery(),
    })
    directives = asyncio.run(interpret_notes(req))
    first = directives[0]
    assert first.directive_type.value == "no_charge_window"
    assert set(first.structured_adjustment.hours) == {0, 1, 2, 3, 4, 5, 22, 23}
    resp = solve(req, directives)
    assert abs(resp.total_cost_bdt - 5920.0) <= 0.01
