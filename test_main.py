from main import (
    DirectiveInterpretation,
    HourPlan,
    OptimizeRequest,
    StructuredAdjustment,
    replay,
    solve,
    validate_directives,
    replay,
    health,
)
import pytest
from fastapi.testclient import TestClient
from main import app


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


def test_empty_directive_hours_are_rejected():
    with pytest.raises(ValueError):
        StructuredAdjustment(hours=[])


def test_strict_boolean_is_required():
    with pytest.raises(ValueError):
        DirectiveInterpretation.model_validate({
            "note_index": 0, "applies": "true", "directive_type": "no_op",
            "structured_adjustment": None, "explanation": "irrelevant",
        })
    with pytest.raises(ValueError):
        DirectiveInterpretation.model_validate({
            "note_index": 0, "applies": 1, "directive_type": "no_op",
            "structured_adjustment": None, "explanation": "irrelevant",
        })


def test_replay_rejects_bad_plan_shapes_and_values():
    payload = scenario()
    directives = [DirectiveInterpretation(
        note_index=0, applies=False, directive_type="no_op",
        structured_adjustment=None, explanation="irrelevant",
    )]
    valid = solve(payload, directives).hourly_plan
    for bad_plan in (valid[:-1], valid[:1] + valid[2:] + valid[1:2], valid[:1] + [valid[0]] + valid[1:-1]):
        with pytest.raises(Exception):
            replay(payload, directives, bad_plan)
    idle = valid[0].model_copy(update={"battery_action": "idle", "battery_kwh": 1.0})
    with pytest.raises(Exception):
        replay(payload, directives, [idle] + valid[1:])
    with pytest.raises(ValueError):
        HourPlan.model_validate({**valid[0].model_dump(), "grid_kwh": -1})
    with pytest.raises(ValueError):
        HourPlan.model_validate({**valid[0].model_dump(), "battery_kwh": -1})


def test_health_requires_model_configuration(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = TestClient(app).get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
