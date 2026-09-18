from main import (
    DirectiveInterpretation,
    OptimizeRequest,
    StructuredAdjustment,
    replay,
    solve,
    validate_directives,
)
import pytest


def scenario():
    return OptimizeRequest.model_validate({
        "scenario_id": "test",
        "operator_notes": ["Do not charge from 2 AM to 5 AM."],
        "hours": [
            {"hour": hour, "demand_kwh": 10.0, "solar_kwh": 0.0, "tariff_bdt_per_kwh": float(1 if hour < 12 else 2)}
            for hour in range(24)
        ],
        "battery": {
            "capacity_kwh": 20.0,
            "initial_energy_kwh": 10.0,
            "minimum_energy_kwh": 0.0,
            "max_charge_kwh_per_hour": 5.0,
            "max_discharge_kwh_per_hour": 5.0,
        },
    })


def test_optimizer_and_replay():
    payload = scenario()
    directives = [DirectiveInterpretation(
        note_index=0,
        applies=True,
        directive_type="no_charge_window",
        structured_adjustment=StructuredAdjustment(hours=[2, 3, 4]),
        explanation="Charging is disabled.",
    )]
    validate_directives(directives, 1, 20)
    response = solve(payload, directives)
    replay(payload, directives, response.hourly_plan)
    assert len(response.hourly_plan) == 24
    assert response.hourly_plan[-1].battery_energy_after_kwh == 10
    assert all(response.hourly_plan[h].battery_action != "charge" for h in [2, 3, 4])


@pytest.mark.parametrize("directive", [
    {"directive_type": "no_charge_window", "structured_adjustment": {"hours": [2, 3, 4]}},
    {"directive_type": "no_discharge_window", "structured_adjustment": {"hours": [17, 18]}},
    {"directive_type": "solar_reduction", "structured_adjustment": {"hours": [11, 12, 13], "factor": 0.2}},
    {"directive_type": "minimum_battery_reserve", "structured_adjustment": {"hours": [18, 19], "minimum_energy_kwh": 10}},
    {"directive_type": "max_grid_window", "structured_adjustment": {"hours": [18, 19], "max_grid_kwh": 10}},
])
def test_directive_shapes_are_checked(directive):
    item = DirectiveInterpretation(
        note_index=0,
        applies=True,
        explanation="valid directive",
        **directive,
    )
    validate_directives([item], 1, 20)


def test_invalid_directive_hours_are_rejected():
    item = DirectiveInterpretation(
        note_index=0,
        applies=True,
        directive_type="no_charge_window",
        structured_adjustment=StructuredAdjustment(hours=[4, 2]),
        explanation="invalid ordering",
    )
    with pytest.raises(ValueError):
        validate_directives([item], 1, 20)
