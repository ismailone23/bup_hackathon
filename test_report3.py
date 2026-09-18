import json

import pytest
from fastapi.testclient import TestClient

import main


def body(note):
    return {
        "scenario_id": "report-3",
        "operator_notes": [note],
        "hours": [
            {
                "hour": h,
                "demand_kwh": 50.0,
                "solar_kwh": 30.0 if 9 <= h <= 16 else 0.0,
                "tariff_bdt_per_kwh": 4.0 if h < 18 else 12.0,
            }
            for h in range(24)
        ],
        "battery": {
            "capacity_kwh": 80.0,
            "initial_energy_kwh": 40.0,
            "minimum_energy_kwh": 0.0,
            "max_charge_kwh_per_hour": 20.0,
            "max_discharge_kwh_per_hour": 20.0,
        },
    }


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    finish_reason = "stop"

    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content):
        self.content = content

    async def create(self, **_kwargs):
        return FakeCompletion(self.content)


class FakeClient:
    def __init__(self, content):
        self.chat = type("Chat", (), {"completions": FakeCompletions(content)})()


def mock_llm(monkeypatch, interpretation):
    content = json.dumps({"interpretations": [interpretation]})
    monkeypatch.setattr(main, "_client", lambda: FakeClient(content))
    return TestClient(main.app)


def interpretation(directive_type, applies=True, adjustment=None):
    return {
        "note_index": 0,
        "applies": applies,
        "directive_type": directive_type,
        "structured_adjustment": adjustment,
        "explanation": "mocked interpretation",
    }


@pytest.mark.parametrize(
    "item",
    [
        interpretation("no_charge_window", adjustment={"hours": [23, 24], "factor": None, "minimum_energy_kwh": None, "max_grid_kwh": None}),
        interpretation("no_charge_window", adjustment={"hours": [4, 2, 3], "factor": None, "minimum_energy_kwh": None, "max_grid_kwh": None}),
        interpretation("no_op", applies=True, adjustment=None),
        interpretation("solar_reduction", adjustment={"hours": [12], "factor": 1.3, "minimum_energy_kwh": None, "max_grid_kwh": None}),
    ],
    ids=["F-08 hour 24", "F-09 unsorted", "F-10 no_op applies", "F-11 factor>1"],
)
def test_invalid_model_output_maps_to_4xx(monkeypatch, item):
    client = mock_llm(monkeypatch, item)
    response = client.post("/optimize-energy", json=body("Do not charge from 10 PM to 6 AM."))
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "INVALID_INTERPRETATION"


def test_model_hours_are_not_derived_from_note_text(monkeypatch):
    """The model is the sole interpreter; hours are never parsed from note wording."""
    item = interpretation(
        "no_charge_window",
        adjustment={"hours": [0, 1, 2, 3, 4, 5, 23], "factor": None, "minimum_energy_kwh": None, "max_grid_kwh": None},
    )
    client = mock_llm(monkeypatch, item)
    response = client.post("/optimize-energy", json=body("Do not charge from 10 PM to 6 AM."))
    assert response.status_code == 200, response.text
    payload = response.json()
    hours = payload["directive_interpretation"][0]["structured_adjustment"]["hours"]
    assert set(hours) == {0, 1, 2, 3, 4, 5, 23}
    assert len(payload["hourly_plan"]) == 24
