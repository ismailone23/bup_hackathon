import pytest

from main import DirectiveInterpretation, solve


def test_negative_tariff_has_a_physical_grid_bound():
    from test_main import scenario

    payload = scenario().model_copy(deep=True)
    payload.hours[0].tariff_bdt_per_kwh = -100
    directives = [DirectiveInterpretation(
        note_index=0,
        applies=False,
        directive_type="no_op",
        structured_adjustment=None,
        explanation="No applicable directive.",
    )]
    response = solve(payload, directives)
    assert response.hourly_plan[0].grid_kwh <= payload.hours[0].demand_kwh + payload.battery.max_charge_kwh_per_hour


@pytest.mark.parametrize("field", ["scenario_id", "battery", "hours"])
def test_malformed_api_payload_is_controlled(field):
    from fastapi.testclient import TestClient
    from main import app

    body = {
        "scenario_id": "test",
        "operator_notes": ["No-op"],
        "hours": [],
        "battery": {},
    }
    body.pop(field)
    response = TestClient(app).post("/optimize-energy", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
